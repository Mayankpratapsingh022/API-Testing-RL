"""API Testing Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

from .models import APITestAction, APITestObservation, APITestState


class APITestEnv(
    EnvClient[APITestAction, APITestObservation, APITestState]
):
    """
    Client for the API Testing Environment.

    Example:
        >>> with APITestEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="basic_validation")
        ...     print(result.observation.feedback)
        ...     result = client.step(APITestAction(
        ...         method="GET", endpoint="/tasks", expected_status=200
        ...     ))
        ...     print(result.observation.status_code)
    """

    def __init__(self, base_url: str, **kwargs):
        kwargs.setdefault("message_timeout_s", 120.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: APITestAction) -> Dict:
        return {
            "method": action.method.value if hasattr(action.method, "value") else str(action.method),
            "endpoint": action.endpoint,
            "headers": action.headers or {},
            "query_params": action.query_params or {},
            "body": action.body,
            "expected_status": action.expected_status,
        }

    def _parse_result(self, payload: Dict) -> StepResult[APITestObservation]:
        obs_data = payload.get("observation", {})
        observation = APITestObservation(
            available_endpoints=obs_data.get("available_endpoints", []),
            status_code=obs_data.get("status_code", 0),
            response_body=obs_data.get("response_body"),
            response_headers=obs_data.get("response_headers", {}),
            response_time_ms=obs_data.get("response_time_ms", 0.0),
            feedback=obs_data.get("feedback", ""),
            bugs_found_so_far=obs_data.get("bugs_found_so_far", 0),
            coverage_summary=obs_data.get("coverage_summary", {}),
            known_resource_ids=obs_data.get("known_resource_ids", {}),
            auth_tokens=obs_data.get("auth_tokens", {}),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 30),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> APITestState:
        return APITestState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_description=payload.get("task_description", ""),
            difficulty=payload.get("difficulty", "easy"),
            steps_taken=payload.get("steps_taken", 0),
            max_steps=payload.get("max_steps", 30),
            bugs_found=payload.get("bugs_found", 0),
            total_bugs=payload.get("total_bugs", 0),
            bugs_found_ids=payload.get("bugs_found_ids", []),
            coverage_pct=payload.get("coverage_pct", 0.0),
            endpoints_tested=payload.get("endpoints_tested", 0),
            total_endpoints=payload.get("total_endpoints", 0),
            current_score=payload.get("current_score", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
