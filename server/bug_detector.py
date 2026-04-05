"""
Bug detection logic — checks if the agent's action/response pair reveals a planted bug.

Each bug has:
- A unique ID
- A severity level (easy/medium/hard)
- A detection function that checks action + response
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import re


@dataclass
class Bug:
    id: str
    severity: str  # "easy", "medium", "hard"
    description: str
    category: str  # "status_code", "validation", "security", "data_integrity"


@dataclass
class BugDetection:
    bug: Bug
    evidence: str  # Human-readable explanation of how the bug was detected


class BugDetector:
    """Detects planted bugs based on agent actions and API responses."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._build_bug_registry()

    def _build_bug_registry(self):
        """Define all bugs with their detection logic."""
        self.bugs: dict[str, Bug] = {}
        self.detectors: dict[str, Callable] = {}

        # === EASY BUGS ===

        self._register_bug(
            Bug("BUG_TASK_01", "easy", "GET /tasks/{id} returns 200 with null for non-existent task", "status_code"),
            self._detect_null_response_for_missing_task,
        )
        self._register_bug(
            Bug("BUG_TASK_02", "easy", "POST /tasks with missing title returns 500 instead of 400/422", "validation"),
            self._detect_missing_field_500,
        )
        self._register_bug(
            Bug("BUG_TASK_03", "easy", "GET /tasks?page=-1 returns 200 instead of 400", "validation"),
            self._detect_negative_page,
        )

        # === MEDIUM BUGS ===

        self._register_bug(
            Bug("BUG_TASK_04", "medium", "PUT /tasks/{id} accepts invalid email format for assignee_email", "validation"),
            self._detect_invalid_email_accepted,
        )
        self._register_bug(
            Bug("BUG_TASK_05", "medium", "DELETE /tasks/{id} returns 200 for non-existent task", "status_code"),
            self._detect_delete_nonexistent_200,
        )
        self._register_bug(
            Bug("BUG_TASK_06", "medium", "GET /tasks?limit=999999 has no pagination cap", "validation"),
            self._detect_no_pagination_cap,
        )
        self._register_bug(
            Bug("BUG_USER_01", "medium", "POST /users accepts invalid email format", "validation"),
            self._detect_user_invalid_email,
        )
        self._register_bug(
            Bug("BUG_USER_02", "medium", "POST /users response exposes password hash", "security"),
            self._detect_password_hash_exposed,
        )
        self._register_bug(
            Bug("BUG_AUTH_02", "medium", "Login with empty password succeeds", "security"),
            self._detect_empty_password_login,
        )

        # === HARD BUGS ===

        self._register_bug(
            Bug("BUG_TASK_07", "hard", "BOLA: User A can access User B's tasks without authorization check", "security"),
            self._detect_bola,
        )
        self._register_bug(
            Bug("BUG_TASK_08", "hard", "POST /tasks with very long title (>5000 chars) causes 500", "validation"),
            self._detect_long_input_crash,
        )
        self._register_bug(
            Bug("BUG_TASK_09", "hard", "SQL injection payload in title is stored verbatim (content injection)", "security"),
            self._detect_content_injection,
        )
        self._register_bug(
            Bug("BUG_AUTH_01", "hard", "Auth tokens not user-scoped: User A's token can modify User B's tasks", "security"),
            self._detect_broken_auth,
        )

    def _register_bug(self, bug: Bug, detector: Callable):
        self.bugs[bug.id] = bug
        self.detectors[bug.id] = detector

    def get_bugs_for_task(self) -> list[Bug]:
        """Return bugs relevant to the current task."""
        if self.task_id == "basic_validation":
            return [self.bugs[bid] for bid in ["BUG_TASK_01", "BUG_TASK_02", "BUG_TASK_03"]]
        elif self.task_id == "edge_cases":
            return [
                self.bugs[bid]
                for bid in [
                    "BUG_TASK_01", "BUG_TASK_02", "BUG_TASK_03",
                    "BUG_TASK_04", "BUG_TASK_05", "BUG_TASK_06",
                    "BUG_USER_01", "BUG_USER_02", "BUG_AUTH_02",
                ]
            ]
        else:  # security_workflows
            return list(self.bugs.values())

    def check(
        self,
        method: str,
        endpoint: str,
        headers: dict,
        query_params: dict,
        body: Optional[dict],
        expected_status: Optional[int],
        response_status: int,
        response_body: Any,
        action_history: list[dict],
        found_bugs: set[str],
    ) -> Optional[BugDetection]:
        """Check if this action/response reveals a bug.

        Returns the first new bug detected, or None.
        """
        ctx = {
            "method": method.upper(),
            "endpoint": endpoint,
            "headers": headers,
            "query_params": query_params,
            "body": body,
            "expected_status": expected_status,
            "response_status": response_status,
            "response_body": response_body,
            "action_history": action_history,
        }

        for bug_id, detector in self.detectors.items():
            if bug_id in found_bugs:
                continue
            # Only check bugs relevant to this task
            task_bugs = {b.id for b in self.get_bugs_for_task()}
            if bug_id not in task_bugs:
                continue
            result = detector(ctx)
            if result:
                return BugDetection(bug=self.bugs[bug_id], evidence=result)

        return None

    # === DETECTION FUNCTIONS ===

    def _detect_null_response_for_missing_task(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "GET"
            and re.match(r"^/tasks/\d+$", ctx["endpoint"])
            and ctx["response_status"] == 200
            and ctx["response_body"] is None
        ):
            task_id = ctx["endpoint"].split("/")[-1]
            return f"GET /tasks/{task_id} returned 200 with null body — should be 404"
        return None

    def _detect_missing_field_500(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "POST"
            and ctx["endpoint"] == "/tasks"
            and ctx["response_status"] == 500
            and ctx["body"] is not None
            and "title" not in ctx["body"]
        ):
            return "POST /tasks with missing 'title' returned 500 — should be 400 or 422"
        return None

    def _detect_negative_page(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "GET"
            and ctx["endpoint"] == "/tasks"
            and ctx["query_params"].get("page") is not None
        ):
            page = ctx["query_params"]["page"]
            try:
                page = int(page)
            except (ValueError, TypeError):
                return None
            if page < 1 and ctx["response_status"] == 200:
                return f"GET /tasks?page={page} returned 200 — should be 400 for invalid page"
        return None

    def _detect_invalid_email_accepted(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "PUT"
            and re.match(r"^/tasks/\d+$", ctx["endpoint"])
            and ctx["body"]
            and "assignee_email" in ctx["body"]
            and ctx["response_status"] in (200, 201)
        ):
            email = ctx["body"]["assignee_email"]
            if email and not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
                return f"PUT accepted invalid email '{email}' without validation"
        return None

    def _detect_delete_nonexistent_200(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "DELETE"
            and re.match(r"^/tasks/\d+$", ctx["endpoint"])
            and ctx["response_status"] == 200
        ):
            task_id = int(ctx["endpoint"].split("/")[-1])
            # Check if this task was never created (ID > 1000 is a safe bet for non-existent)
            if task_id > 100:
                return f"DELETE /tasks/{task_id} returned 200 for non-existent task — should be 404"
        return None

    def _detect_no_pagination_cap(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "GET"
            and ctx["endpoint"] == "/tasks"
            and ctx["response_status"] == 200
        ):
            limit = ctx["query_params"].get("limit")
            if limit is not None:
                try:
                    limit = int(limit)
                except (ValueError, TypeError):
                    return None
                if limit > 1000:
                    return f"GET /tasks?limit={limit} accepted without pagination cap — potential DoS"
        return None

    def _detect_user_invalid_email(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "POST"
            and ctx["endpoint"] == "/users"
            and ctx["body"]
            and "email" in ctx["body"]
            and ctx["response_status"] == 201
        ):
            email = ctx["body"]["email"]
            if email and not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
                return f"POST /users accepted invalid email '{email}'"
        return None

    def _detect_password_hash_exposed(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "POST"
            and ctx["endpoint"] == "/users"
            and ctx["response_status"] == 201
            and isinstance(ctx["response_body"], dict)
        ):
            if "password_hash" in ctx["response_body"]:
                return "POST /users response exposes password_hash field — security vulnerability"
        return None

    def _detect_empty_password_login(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "POST"
            and ctx["endpoint"] == "/auth/login"
            and ctx["body"]
            and ctx["response_status"] == 200
        ):
            password = ctx["body"].get("password", "NOTEMPTY")
            if password == "" or password is None:
                return "Login with empty password succeeded — authentication bypass"
        return None

    def _detect_bola(self, ctx: dict) -> Optional[str]:
        """Detect if agent tested cross-user resource access."""
        if (
            ctx["method"] == "GET"
            and re.match(r"^/tasks/\d+$", ctx["endpoint"])
            and ctx["response_status"] == 200
            and ctx["response_body"] is not None
            and isinstance(ctx["response_body"], dict)
            and ctx["headers"].get("Authorization")
        ):
            # Check if the agent logged in as a different user and accessed another's task
            for prev in reversed(ctx["action_history"]):
                if (
                    prev.get("method") == "POST"
                    and prev.get("endpoint") == "/auth/login"
                    and prev.get("response_status") == 200
                    and isinstance(prev.get("response_body"), dict)
                ):
                    login_user_id = prev["response_body"].get("user_id")
                    task_owner_id = ctx["response_body"].get("owner_id")
                    if (
                        login_user_id is not None
                        and task_owner_id is not None
                        and login_user_id != task_owner_id
                    ):
                        return (
                            f"User {login_user_id} accessed task owned by user {task_owner_id} "
                            f"— BOLA/IDOR vulnerability (no authorization check)"
                        )
        return None

    def _detect_long_input_crash(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "POST"
            and ctx["endpoint"] == "/tasks"
            and ctx["body"]
            and ctx["response_status"] == 500
        ):
            title = ctx["body"].get("title", "")
            if isinstance(title, str) and len(title) > 5000:
                return f"POST /tasks with title length {len(title)} caused 500 — no input length validation"
        return None

    def _detect_content_injection(self, ctx: dict) -> Optional[str]:
        if (
            ctx["method"] == "POST"
            and ctx["endpoint"] == "/tasks"
            and ctx["body"]
            and ctx["response_status"] == 201
            and isinstance(ctx["response_body"], dict)
        ):
            title = ctx["body"].get("title", "")
            injection_patterns = [
                "DROP TABLE", "DELETE FROM", "<script>", "javascript:",
                "'; --", "\" OR 1=1", "UNION SELECT",
            ]
            for pattern in injection_patterns:
                if pattern.lower() in str(title).lower():
                    stored_title = ctx["response_body"].get("title", "")
                    if pattern.lower() in str(stored_title).lower():
                        return (
                            f"Injection payload '{pattern}' in title was stored verbatim "
                            f"— no input sanitization (content injection)"
                        )
        return None

    def _detect_broken_auth(self, ctx: dict) -> Optional[str]:
        """Detect if agent successfully modified another user's task with their own token."""
        if (
            ctx["method"] in ("PUT", "DELETE")
            and re.match(r"^/tasks/\d+$", ctx["endpoint"])
            and ctx["response_status"] == 200
            and ctx["headers"].get("Authorization")
        ):
            for prev in reversed(ctx["action_history"]):
                if (
                    prev.get("method") == "POST"
                    and prev.get("endpoint") == "/auth/login"
                    and prev.get("response_status") == 200
                    and isinstance(prev.get("response_body"), dict)
                ):
                    login_user_id = prev["response_body"].get("user_id")
                    # Check if the task belonged to a different user
                    task_id = int(ctx["endpoint"].split("/")[-1])
                    if isinstance(ctx["response_body"], dict):
                        task_owner = ctx["response_body"].get("owner_id")
                        if (
                            login_user_id is not None
                            and task_owner is not None
                            and login_user_id != task_owner
                        ):
                            return (
                                f"User {login_user_id}'s token modified task owned by user {task_owner} "
                                f"— broken authorization"
                            )
                    break
        return None
