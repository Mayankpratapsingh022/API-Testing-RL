"""
Data models for the API Testing Environment.

Defines Action, Observation, State for API integration testing training.
An AI agent learns to test REST APIs intelligently — discovering endpoints,
crafting requests, validating responses, finding bugs, and handling edge cases.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class BugSeverity(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class APITestAction(Action):
    """What the agent sends each step — an HTTP request to test the API."""

    method: HTTPMethod = Field(..., description="HTTP method")
    endpoint: str = Field(..., min_length=1, description="API endpoint path, e.g. /tasks, /users/1")
    headers: dict[str, str] = Field(default_factory=dict, description="Request headers")
    query_params: dict[str, Any] = Field(default_factory=dict, description="URL query parameters")
    body: Optional[dict[str, Any]] = Field(default=None, description="Request JSON body")
    expected_status: Optional[int] = Field(
        default=None,
        description="What the agent expects the status code to be (used for bug detection)",
    )


class EndpointInfo(Action):
    """Information about a single API endpoint from the spec."""

    method: str = ""
    path: str = ""
    summary: str = ""
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    request_body_schema: Optional[dict[str, Any]] = None
    response_schema: Optional[dict[str, Any]] = None


class APITestObservation(Observation):
    """What the agent sees after each step."""

    # API spec info (provided on reset, updated each step)
    available_endpoints: list[dict[str, Any]] = Field(
        default_factory=list, description="Available API endpoints from the spec"
    )

    # Response from last request
    status_code: int = Field(default=0, description="HTTP status code of the response")
    response_body: Any = Field(default=None, description="Response body (JSON or text)")
    response_headers: dict[str, str] = Field(default_factory=dict, description="Response headers")
    response_time_ms: float = Field(default=0.0, description="Response time in milliseconds")

    # Feedback
    feedback: str = Field(default="", description="Human-readable feedback about the last action")
    bugs_found_so_far: int = Field(default=0, description="Number of bugs found so far")
    coverage_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Coverage stats: endpoints_tested, methods_used, status_codes_seen",
    )

    # Context from prior steps
    known_resource_ids: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Resource IDs created by POST requests, keyed by resource type",
    )
    auth_tokens: dict[str, str] = Field(
        default_factory=dict,
        description="Available auth tokens for different users/roles",
    )

    # Task info
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Description of the current task")
    steps_taken: int = Field(default=0, description="Steps taken in this episode")
    max_steps: int = Field(default=30, description="Maximum steps per episode")


class APITestState(State):
    """Episode metadata — internal state exposed via state() endpoint."""

    task_id: str = ""
    task_description: str = ""
    difficulty: str = "easy"
    steps_taken: int = 0
    max_steps: int = 30
    bugs_found: int = 0
    total_bugs: int = 0
    bugs_found_ids: list[str] = Field(default_factory=list)
    coverage_pct: float = 0.0
    endpoints_tested: int = 0
    total_endpoints: int = 0
    current_score: float = 0.0
    cumulative_reward: float = 0.0
