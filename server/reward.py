"""
Multi-signal reward function for the API Testing Environment.

Rewards are decomposed into:
1. Coverage reward — exploring new endpoints/methods/status codes
2. Validity reward — well-formed requests and proper dependency chaining
3. Bug discovery reward — the core goal, scaled by severity
4. Exploration bonus — trying novel actions
5. Penalties — for repeating exact requests or malformed input
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import re


@dataclass
class CoverageTracker:
    """Tracks API coverage across the episode."""

    endpoints_hit: set[str] = field(default_factory=set)
    method_endpoint_pairs: set[tuple[str, str]] = field(default_factory=set)
    status_codes_seen: set[int] = field(default_factory=set)
    total_endpoints: int = 10  # known endpoint patterns

    def record(self, method: str, endpoint: str, status_code: int) -> dict[str, bool]:
        """Record a request and return what's new."""
        normalized_endpoint = self._normalize_endpoint(endpoint)
        pair = (method.upper(), normalized_endpoint)

        is_new_endpoint = normalized_endpoint not in self.endpoints_hit
        is_new_pair = pair not in self.method_endpoint_pairs
        is_new_status = status_code not in self.status_codes_seen

        self.endpoints_hit.add(normalized_endpoint)
        self.method_endpoint_pairs.add(pair)
        self.status_codes_seen.add(status_code)

        return {
            "new_endpoint": is_new_endpoint,
            "new_method_endpoint": is_new_pair,
            "new_status_code": is_new_status,
        }

    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize /tasks/42 to /tasks/{id}."""
        normalized = re.sub(r"/(\d+)", "/{id}", endpoint)
        return normalized.rstrip("/") or "/"

    def summary(self) -> dict:
        return {
            "endpoints_tested": len(self.endpoints_hit),
            "total_endpoints": self.total_endpoints,
            "method_endpoint_pairs": len(self.method_endpoint_pairs),
            "status_codes_seen": sorted(self.status_codes_seen),
            "coverage_pct": round(len(self.endpoints_hit) / max(self.total_endpoints, 1) * 100, 1),
        }


@dataclass
class RewardBreakdown:
    coverage: float = 0.0
    validity: float = 0.0
    bug_discovery: float = 0.0
    exploration: float = 0.0
    penalty: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 4),
            "validity": round(self.validity, 4),
            "bug_discovery": round(self.bug_discovery, 4),
            "exploration": round(self.exploration, 4),
            "penalty": round(self.penalty, 4),
            "total": round(self.total, 4),
        }


class RewardComputer:
    """Computes multi-signal rewards for API testing actions."""

    def __init__(self):
        self.coverage = CoverageTracker()
        self.action_history: list[dict] = []
        self.found_bugs: set[str] = set()
        self.created_ids: dict[str, list[Any]] = {}  # resource type -> list of IDs

    def reset(self):
        self.coverage = CoverageTracker()
        self.action_history = []
        self.found_bugs = set()
        self.created_ids = {}

    def compute(
        self,
        method: str,
        endpoint: str,
        headers: dict,
        query_params: dict,
        body: Optional[dict],
        expected_status: Optional[int],
        response_status: int,
        response_body: Any,
        bug_found: Optional[str] = None,  # bug severity if found
        bug_id: Optional[str] = None,
    ) -> RewardBreakdown:
        """Compute reward for this step."""
        breakdown = RewardBreakdown()

        # 1. Coverage reward (0.0 - 0.3)
        coverage_info = self.coverage.record(method, endpoint, response_status)
        if coverage_info["new_endpoint"]:
            breakdown.coverage += 0.10
        if coverage_info["new_method_endpoint"]:
            breakdown.coverage += 0.05
        if coverage_info["new_status_code"]:
            breakdown.coverage += 0.05

        # 2. Validity reward (0.0 - 0.2)
        if response_status < 500:
            breakdown.validity += 0.03  # Non-crash request

        if self._used_dependency(method, endpoint, body, headers):
            breakdown.validity += 0.10  # Used a previously created resource ID or auth token

        if expected_status is not None and expected_status == response_status:
            breakdown.validity += 0.05  # Correctly predicted status code

        # Track created resources
        self._track_created_resources(method, endpoint, response_status, response_body)

        # 3. Bug discovery reward (0.0 - 0.4)
        if bug_found and bug_id:
            if bug_id not in self.found_bugs:
                self.found_bugs.add(bug_id)
                if bug_found == "easy":
                    breakdown.bug_discovery += 0.10
                elif bug_found == "medium":
                    breakdown.bug_discovery += 0.15
                elif bug_found == "hard":
                    breakdown.bug_discovery += 0.25
                # First discovery bonus
                breakdown.bug_discovery += 0.05

        # 4. Exploration bonus (0.0 - 0.1)
        action_sig = self._action_signature(method, endpoint, query_params, body)
        is_novel = all(
            self._action_signature(
                h.get("method", ""),
                h.get("endpoint", ""),
                h.get("query_params", {}),
                h.get("body"),
            )
            != action_sig
            for h in self.action_history
        )
        if is_novel:
            breakdown.exploration += 0.05

        # 5. Penalties
        # Exact duplicate request
        exact_match = any(
            h.get("method") == method
            and h.get("endpoint") == endpoint
            and h.get("query_params") == query_params
            and h.get("body") == body
            and h.get("headers") == headers
            for h in self.action_history
        )
        if exact_match:
            breakdown.penalty -= 0.08

        # Record this action in history
        self.action_history.append({
            "method": method,
            "endpoint": endpoint,
            "headers": headers,
            "query_params": query_params,
            "body": body,
            "response_status": response_status,
            "response_body": response_body,
        })

        # Total
        breakdown.total = max(
            breakdown.coverage + breakdown.validity + breakdown.bug_discovery + breakdown.exploration + breakdown.penalty,
            -0.1,  # Floor to prevent extreme negative rewards
        )
        breakdown.total = min(breakdown.total, 1.0)

        return breakdown

    def _used_dependency(self, method: str, endpoint: str, body: Optional[dict], headers: dict) -> bool:
        """Check if this request uses a resource ID or token from a previous step."""
        endpoint_str = str(endpoint)

        # Check if endpoint contains a known resource ID
        for resource_type, ids in self.created_ids.items():
            for rid in ids:
                if str(rid) in endpoint_str:
                    return True

        # Check if using an auth token obtained from login
        if headers.get("Authorization"):
            for prev in self.action_history:
                if (
                    prev.get("endpoint") == "/auth/login"
                    and prev.get("response_status") == 200
                    and isinstance(prev.get("response_body"), dict)
                    and "token" in prev["response_body"]
                ):
                    token = prev["response_body"]["token"]
                    if token in headers["Authorization"]:
                        return True
        return False

    def _track_created_resources(
        self, method: str, endpoint: str, status: int, body: Any
    ):
        """Track resource IDs from POST responses."""
        if method.upper() == "POST" and status == 201 and isinstance(body, dict):
            resource_id = body.get("id")
            if resource_id is not None:
                # Determine resource type from endpoint
                resource_type = endpoint.strip("/").split("/")[0]
                if resource_type not in self.created_ids:
                    self.created_ids[resource_type] = []
                self.created_ids[resource_type].append(resource_id)

    def _action_signature(
        self, method: str, endpoint: str, query_params: dict, body: Optional[dict]
    ) -> str:
        """Create a signature for an action to check novelty."""
        normalized = re.sub(r"/\d+", "/{id}", endpoint)
        body_keys = sorted(body.keys()) if body else []
        param_keys = sorted(query_params.keys()) if query_params else []
        return f"{method}:{normalized}:{param_keys}:{body_keys}"
