"""
Prompt formatting and action parsing for LLM-based API testing agents.

- SYSTEM_PROMPT: Instructions for the LLM on how to test APIs
- format_observation(): Converts environment observations into LLM prompts
- parse_action(): Extracts JSON actions from LLM text output
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import APITestAction, HTTPMethod


SYSTEM_PROMPT = """\
You are an expert API security tester. You are testing a REST API for bugs.

You will receive:
- The API specification (available endpoints)
- Results from your previous requests
- Coverage and bug discovery progress

Your job: find as many bugs as possible by sending HTTP requests.

RESPOND WITH EXACTLY ONE JSON ACTION per turn:
```json
{
  "method": "GET|POST|PUT|DELETE",
  "endpoint": "/path",
  "headers": {},
  "query_params": {},
  "body": null,
  "expected_status": 200
}
```

TESTING STRATEGIES:
- Test each endpoint with valid inputs first
- Try invalid inputs (missing fields, wrong types, boundary values)
- Test with non-existent resource IDs
- Login as different users and test cross-user access
- Try SQL injection patterns in text fields
- Test with very long inputs
- Chain operations: create -> read -> update -> delete
"""


def format_observation(obs) -> str:
    """Convert an observation into a human-readable prompt for the LLM."""
    parts = []

    if obs.steps_taken == 0:
        # First turn — show the full task and API spec
        parts.append(f"TASK: {obs.task_description}")
        parts.append(f"\nSTEPS REMAINING: {obs.max_steps}")
        parts.append("\nAVAILABLE ENDPOINTS:")
        for ep in obs.available_endpoints:
            line = f"  {ep['method']} {ep['path']} — {ep.get('summary', '')}"
            parts.append(line)
        parts.append("\nBegin testing. Send your first request as JSON.")
    else:
        # Subsequent turns — show last response + progress
        parts.append(f"STEP {obs.steps_taken}/{obs.max_steps}")
        parts.append(f"RESPONSE: HTTP {obs.status_code}")

        resp = obs.response_body
        if isinstance(resp, (dict, list)):
            resp_str = json.dumps(resp, indent=2)
            if len(resp_str) > 500:
                resp_str = resp_str[:500] + "\n... (truncated)"
        else:
            resp_str = str(resp)[:500]
        parts.append(f"BODY:\n{resp_str}")

        parts.append(f"\nFEEDBACK: {obs.feedback}")

        # Progress
        coverage = obs.coverage_summary
        parts.append(
            f"\nPROGRESS: Bugs found: {obs.bugs_found_so_far} | "
            f"Coverage: {coverage.get('coverage_pct', 0):.0f}% | "
            f"Endpoints tested: {coverage.get('endpoints_tested', 0)}/{coverage.get('total_endpoints', 0)}"
        )

        if obs.auth_tokens:
            parts.append(f"AUTH TOKENS: {list(obs.auth_tokens.keys())}")
        if obs.known_resource_ids:
            parts.append(f"CREATED RESOURCES: {dict(obs.known_resource_ids)}")

        parts.append("\nSend your next request as JSON.")

    return "\n".join(parts)


def parse_action(text: str) -> APITestAction | None:
    """Parse the LLM's text output into an APITestAction.

    Handles common LLM formatting: raw JSON, code blocks, extra text around JSON.
    """
    # Try to find JSON with "method" key
    json_match = re.search(r'\{[^{}]*"method"[^{}]*\}', text, re.DOTALL)
    if not json_match:
        # Try inside a code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            return None
    else:
        json_str = json_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    method = data.get("method", "GET").upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        method = "GET"

    endpoint = data.get("endpoint", "/tasks")
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return APITestAction(
        method=HTTPMethod(method),
        endpoint=endpoint,
        headers=data.get("headers", {}),
        query_params=data.get("query_params", {}),
        body=data.get("body"),
        expected_status=data.get("expected_status"),
    )
