"""
OpenEnv Environment for API Integration Testing.

The agent interacts with a deliberately buggy REST API, discovering endpoints,
crafting requests, and finding bugs. Rewards are multi-signal: coverage,
validity, bug discovery, and exploration.
"""

import logging
import random
import time
import json
from typing import Any, Optional

from fastapi.testclient import TestClient
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import APITestAction, APITestObservation, APITestState
except ImportError:
    from models import APITestAction, APITestObservation, APITestState

from .buggy_api.database import Database
from .buggy_api.main import create_buggy_api
from .bug_detector import BugDetector
from .reward import RewardComputer
from .graders import TaskGrader

logger = logging.getLogger(__name__)

# Task definitions
TASKS = {
    "basic_validation": {
        "id": "basic_validation",
        "description": (
            "Test all CRUD endpoints with valid inputs and verify correct status codes. "
            "Find basic bugs like wrong status codes and missing field handling. "
            "Available endpoints: GET /tasks, POST /tasks, GET /tasks/{id}, PUT /tasks/{id}, "
            "DELETE /tasks/{id}, GET /users, POST /users, POST /auth/login. "
            "Try different methods on each endpoint and verify responses match the expected behavior."
        ),
        "difficulty": "easy",
        "max_steps": 25,
        "total_bugs": 3,
    },
    "edge_cases": {
        "id": "edge_cases",
        "description": (
            "Test boundary conditions, invalid inputs, and error responses. "
            "Send missing fields, wrong types, negative page numbers, huge limits. "
            "Test with non-existent resource IDs (e.g., /tasks/999999). "
            "Chain operations: create a resource, then read/update/delete it. "
            "Find bugs in input validation, pagination, and error handling."
        ),
        "difficulty": "medium",
        "max_steps": 35,
        "total_bugs": 9,
    },
    "security_workflows": {
        "id": "security_workflows",
        "description": (
            "Discover authorization flaws, injection vulnerabilities, and workflow bugs. "
            "Login as different users (alice/password, bob/password, charlie/password) and "
            "try accessing each other's resources. Test SQL injection patterns in input fields. "
            "Execute multi-step workflows: create -> modify -> verify -> delete -> re-fetch. "
            "Check if auth tokens properly scope access. Test with very long inputs."
        ),
        "difficulty": "hard",
        "max_steps": 45,
        "total_bugs": 13,
    },
}

# OpenAPI-like spec for the agent
API_SPEC = [
    {
        "method": "GET",
        "path": "/tasks",
        "summary": "List all tasks. Supports filtering by status, priority; pagination with page & limit; sorting with sort.",
        "parameters": [
            {"name": "status", "in": "query", "type": "string", "enum": ["pending", "in_progress", "done"]},
            {"name": "priority", "in": "query", "type": "string", "enum": ["low", "medium", "high"]},
            {"name": "sort", "in": "query", "type": "string", "enum": ["created_at", "updated_at", "title"]},
            {"name": "page", "in": "query", "type": "integer"},
            {"name": "limit", "in": "query", "type": "integer"},
        ],
    },
    {
        "method": "POST",
        "path": "/tasks",
        "summary": "Create a new task. Requires 'title' field. Optional: description, status, priority, assignee_email.",
        "request_body": {
            "required": ["title"],
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "assignee_email": {"type": "string", "format": "email"},
            },
        },
    },
    {
        "method": "GET",
        "path": "/tasks/{id}",
        "summary": "Get a specific task by ID.",
        "parameters": [{"name": "id", "in": "path", "type": "integer", "required": True}],
    },
    {
        "method": "PUT",
        "path": "/tasks/{id}",
        "summary": "Update a task. All fields optional.",
        "parameters": [{"name": "id", "in": "path", "type": "integer", "required": True}],
        "request_body": {
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "status": {"type": "string"},
                "priority": {"type": "string"},
                "assignee_email": {"type": "string", "format": "email"},
            },
        },
    },
    {
        "method": "DELETE",
        "path": "/tasks/{id}",
        "summary": "Delete a task by ID.",
        "parameters": [{"name": "id", "in": "path", "type": "integer", "required": True}],
    },
    {
        "method": "GET",
        "path": "/users",
        "summary": "List all users.",
    },
    {
        "method": "POST",
        "path": "/users",
        "summary": "Create a new user. Requires username, email, password.",
        "request_body": {
            "required": ["username", "email", "password"],
            "properties": {
                "username": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "password": {"type": "string"},
                "role": {"type": "string", "enum": ["user", "admin"]},
            },
        },
    },
    {
        "method": "GET",
        "path": "/users/{id}",
        "summary": "Get a specific user by ID.",
        "parameters": [{"name": "id", "in": "path", "type": "integer", "required": True}],
    },
    {
        "method": "POST",
        "path": "/auth/login",
        "summary": "Login and receive an auth token. Pre-seeded users: alice, bob, charlie (password: any string).",
        "request_body": {
            "required": ["username", "password"],
            "properties": {
                "username": {"type": "string"},
                "password": {"type": "string"},
            },
        },
    },
]


class APITestEnvironment(Environment):
    """OpenEnv environment for API integration testing.

    The agent tests a deliberately buggy REST API by sending HTTP requests
    and analyzing responses. It earns rewards for coverage, finding bugs,
    and exploring edge cases.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._db: Optional[Database] = None
        self._api: Optional[TestClient] = None
        self._bug_detector: Optional[BugDetector] = None
        self._reward_computer: Optional[RewardComputer] = None
        self._task: Optional[dict] = None
        self._found_bugs: set[str] = set()
        self._steps_taken: int = 0
        self._cumulative_reward: float = 0.0
        self._action_history: list[dict] = []
        self._auth_tokens: dict[str, str] = {}
        self._episode_id: str = ""

    def reset(self, seed=None, episode_id=None, **kwargs) -> APITestObservation:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for domain randomization. When provided, the
                  database is populated with different users, tasks, and data
                  so each training episode is unique. None = fixed default data.
            episode_id: Optional episode identifier for tracking.

        kwargs:
            task_id: str - one of "basic_validation", "edge_cases", "security_workflows"
        """
        task_id = kwargs.get("task_id", "basic_validation")
        if task_id not in TASKS:
            task_id = "basic_validation"

        self._task = TASKS[task_id]
        self._seed = seed
        self._episode_id = episode_id or f"ep_{int(time.time())}"

        # Reset database with seed for domain randomization
        # seed=None → fixed data (manual testing / Gradio)
        # seed=int  → randomized data (GRPO training)
        self._db = Database(seed=seed)
        buggy_app = create_buggy_api(self._db)
        self._api = TestClient(buggy_app, raise_server_exceptions=False)

        # Build dynamic task description that includes actual usernames
        user_names = self._db.user_names
        user_list = ", ".join(user_names)
        dynamic_description = (
            f"{self._task['description']} "
            f"Users in the system: {user_list} (use any password to login)."
        )

        # Reset tracking
        self._bug_detector = BugDetector(task_id)
        self._reward_computer = RewardComputer()
        self._found_bugs = set()
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._action_history = []
        self._auth_tokens = {}

        logger.info(f"Reset environment: task={task_id}, seed={seed}, episode={self._episode_id}")

        return APITestObservation(
            available_endpoints=API_SPEC,
            status_code=0,
            response_body=None,
            response_headers={},
            response_time_ms=0,
            feedback=(
                f"Environment reset. Task: {dynamic_description} "
                f"You have {self._task['max_steps']} steps. Start testing the API!"
            ),
            bugs_found_so_far=0,
            coverage_summary=self._reward_computer.coverage.summary(),
            known_resource_ids=self._reward_computer.created_ids,
            auth_tokens=self._auth_tokens,
            task_id=task_id,
            task_description=dynamic_description,
            steps_taken=0,
            max_steps=self._task["max_steps"],
            done=False,
            reward=0.0,
        )

    def step(self, action: APITestAction, timeout_s=None, **kwargs) -> APITestObservation:
        """Execute an API test action and return observation + reward."""
        self._steps_taken += 1

        # Forward request to buggy API
        method = action.method.value if hasattr(action.method, "value") else str(action.method)
        endpoint = action.endpoint
        headers = dict(action.headers) if action.headers else {}
        query_params = dict(action.query_params) if action.query_params else {}
        body = action.body

        # Make the request
        start_time = time.time()
        try:
            response = self._api.request(
                method=method.upper(),
                url=endpoint,
                headers=headers,
                params=query_params if query_params else None,
                json=body,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            response_status = response.status_code
            try:
                response_body = response.json()
            except Exception:
                response_body = response.text
            response_headers = dict(response.headers)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            response_status = 0
            response_body = {"error": str(e)}
            response_headers = {}

        # Track auth tokens from login responses
        if (
            endpoint == "/auth/login"
            and response_status == 200
            and isinstance(response_body, dict)
            and "token" in response_body
        ):
            username = body.get("username", "unknown") if body else "unknown"
            self._auth_tokens[username] = response_body["token"]

        # Check for bug detection
        detection = self._bug_detector.check(
            method=method,
            endpoint=endpoint,
            headers=headers,
            query_params=query_params,
            body=body,
            expected_status=action.expected_status,
            response_status=response_status,
            response_body=response_body,
            action_history=self._action_history,
            found_bugs=self._found_bugs,
        )

        bug_severity = None
        bug_id = None
        if detection:
            bug_severity = detection.bug.severity
            bug_id = detection.bug.id
            self._found_bugs.add(bug_id)

        # Compute reward
        reward_breakdown = self._reward_computer.compute(
            method=method,
            endpoint=endpoint,
            headers=headers,
            query_params=query_params,
            body=body,
            expected_status=action.expected_status,
            response_status=response_status,
            response_body=response_body,
            bug_found=bug_severity,
            bug_id=bug_id,
        )
        self._cumulative_reward += reward_breakdown.total

        # Record action in history
        self._action_history.append({
            "method": method,
            "endpoint": endpoint,
            "headers": headers,
            "query_params": query_params,
            "body": body,
            "response_status": response_status,
            "response_body": response_body,
        })

        # Generate feedback
        feedback_parts = [f"{method} {endpoint} -> {response_status}"]
        if detection:
            feedback_parts.append(f"BUG FOUND ({detection.bug.severity})! {detection.evidence}")
        if reward_breakdown.coverage > 0:
            feedback_parts.append(f"Coverage +{reward_breakdown.coverage:.2f}")
        if reward_breakdown.penalty < 0:
            feedback_parts.append("Repeated request penalty")

        done = self._steps_taken >= self._task["max_steps"]

        # Compute final grade if done
        if done:
            grade = TaskGrader.grade(
                task_id=self._task["id"],
                bugs_found=self._found_bugs,
                coverage_pct=self._reward_computer.coverage.summary()["coverage_pct"],
                endpoints_tested=len(self._reward_computer.coverage.endpoints_hit),
                total_endpoints=self._reward_computer.coverage.total_endpoints,
                method_endpoint_pairs=len(self._reward_computer.coverage.method_endpoint_pairs),
                status_codes_seen=self._reward_computer.coverage.status_codes_seen,
                action_history=self._action_history,
                created_resources=self._reward_computer.created_ids,
            )
            feedback_parts.append(
                f"\n=== EPISODE COMPLETE ===\n"
                f"Final Score: {grade.score:.4f}\n"
                f"Bugs Found: {len(self._found_bugs)}/{self._task['total_bugs']}\n"
                f"Grade Breakdown: {json.dumps(grade.breakdown, indent=2)}\n"
                f"Feedback: {grade.feedback}"
            )
            # Use grade score as final reward
            final_reward = grade.score
        else:
            final_reward = reward_breakdown.total

        return APITestObservation(
            available_endpoints=API_SPEC,
            status_code=response_status,
            response_body=response_body,
            response_headers={k: v for k, v in list(response_headers.items())[:20]},
            response_time_ms=round(elapsed_ms, 2),
            feedback=" | ".join(feedback_parts),
            bugs_found_so_far=len(self._found_bugs),
            coverage_summary=self._reward_computer.coverage.summary(),
            known_resource_ids=self._reward_computer.created_ids,
            auth_tokens=self._auth_tokens,
            task_id=self._task["id"],
            task_description=self._task["description"],
            steps_taken=self._steps_taken,
            max_steps=self._task["max_steps"],
            done=done,
            reward=final_reward,
            metadata={"reward_breakdown": reward_breakdown.as_dict()},
        )

    @property
    def state(self) -> APITestState:
        """Return current episode state."""
        if not self._task:
            return APITestState()

        coverage = self._reward_computer.coverage.summary() if self._reward_computer else {}
        return APITestState(
            episode_id=self._episode_id,
            step_count=self._steps_taken,
            task_id=self._task["id"],
            task_description=self._task["description"],
            difficulty=self._task["difficulty"],
            steps_taken=self._steps_taken,
            max_steps=self._task["max_steps"],
            bugs_found=len(self._found_bugs),
            total_bugs=self._task["total_bugs"],
            bugs_found_ids=list(self._found_bugs),
            coverage_pct=coverage.get("coverage_pct", 0.0),
            endpoints_tested=coverage.get("endpoints_tested", 0),
            total_endpoints=coverage.get("total_endpoints", 0),
            current_score=0.0,
            cumulative_reward=round(self._cumulative_reward, 4),
        )
