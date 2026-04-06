"""
Prompt formatting and action parsing for LLM-based API testing agents.

- SYSTEM_PROMPT: Instructions for the LLM on how to test APIs
- format_observation(): Converts environment observations into LLM prompts
- parse_action(): Extracts a single JSON action from LLM text
- parse_test_plan(): Extracts a JSON array of actions (for GRPO training)
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import APITestAction, HTTPMethod


# =====================================================================
# System prompt for multi-turn evaluation (one action at a time)
# =====================================================================

SYSTEM_PROMPT = """\
You are an expert API security tester. You are testing a REST API for bugs.

You will receive:
- The API specification (available endpoints)
- Results from your previous requests
- Coverage and bug discovery progress

Your job: find as many bugs as possible by sending HTTP requests.

Think step by step about what to test next, then output your action as JSON.

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


# =====================================================================
# System prompt for GRPO training (full test plan in one shot)
# =====================================================================

PLAN_SYSTEM_PROMPT = """\
You are an expert API security tester. You will receive an API specification and must output a COMPLETE TEST PLAN as a JSON array of HTTP requests to execute in order.

Your goal: find as many bugs as possible through systematic testing.

OUTPUT FORMAT — a JSON array of actions to execute sequentially:
```json
[
  {"method": "GET", "endpoint": "/tasks", "headers": {}, "query_params": {}, "body": null, "expected_status": 200},
  {"method": "POST", "endpoint": "/auth/login", "headers": {}, "query_params": {}, "body": {"username": "alice", "password": "pass"}, "expected_status": 200},
  ...more actions...
]
```

OUTPUT EXACTLY ONE JSON ARRAY. No other text.

TESTING STRATEGY — follow this order:
1. DISCOVER: GET /tasks, GET /users to see what exists
2. AUTHENTICATE: Login as two different users (POST /auth/login)
3. CRUD: POST to create, GET to read, PUT to update, DELETE to remove
4. MISSING FIELDS: POST /tasks without required "title" field
5. NON-EXISTENT IDs: GET /tasks/999999 (expect 404 — if you get 200, that's a bug!)
6. BOUNDARY: GET /tasks?page=-1&limit=10 (negative page), GET /tasks?limit=999999 (huge limit)
7. INVALID DATA: PUT /tasks/1 with assignee_email="not-an-email"
8. SECURITY: Login as user B, then try to GET/PUT/DELETE user A's resources (BOLA test)
9. INJECTION: POST /tasks with title containing SQL injection like "'; DROP TABLE tasks;--"
10. EMPTY AUTH: POST /auth/login with empty password (should fail but might not)
11. DATA LEAKS: POST /users and check if response includes password_hash
12. STATE: DELETE a task, then GET it again (should be 404)
13. LONG INPUT: POST /tasks with a title of 6000+ characters

COMMON BUG PATTERNS TO TEST:
- API returns 200 with null body instead of 404 for missing resources
- API returns 500 instead of 400 for invalid input
- API accepts any password (even empty string) for login
- Users can access other users' resources (no authorization check)
- Response includes sensitive fields like password_hash
- No input length limits (very long strings crash the server)
- SQL/HTML injection payloads stored without sanitization
- DELETE returns 200 even for non-existent resources
- No pagination limit cap (limit=999999 accepted)

RULES:
- Output 15-25 actions
- Each action MUST have "method" and "endpoint"
- Vary your requests — never repeat the same action
- Use the usernames from the task description for login
"""


def format_observation(obs) -> str:
    """Convert an observation into a human-readable prompt for the LLM.
    Used in multi-turn evaluation (one action at a time).
    """
    parts = []

    if obs.steps_taken == 0:
        parts.append(f"TASK: {obs.task_description}")
        parts.append(f"\nSTEPS REMAINING: {obs.max_steps}")
        parts.append("\nAVAILABLE ENDPOINTS:")
        for ep in obs.available_endpoints:
            line = f"  {ep['method']} {ep['path']} — {ep.get('summary', '')}"
            parts.append(line)
        parts.append("\nBegin testing. Send your first request as JSON.")
    else:
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


def format_plan_prompt(obs) -> str:
    """Convert the initial observation into a prompt for generating a full test plan.
    Used in GRPO training (model outputs a complete plan in one completion).
    """
    parts = []
    parts.append(f"TASK: {obs.task_description}")
    parts.append(f"\nYou have {obs.max_steps} actions to find as many bugs as possible.")
    parts.append("\nAVAILABLE ENDPOINTS:")
    for ep in obs.available_endpoints:
        summary = ep.get("summary", "")
        parts.append(f"  {ep['method']} {ep['path']} — {summary}")

        # Show request body schema if available
        req_body = ep.get("request_body", {})
        if req_body:
            props = req_body.get("properties", {})
            required = req_body.get("required", [])
            if props:
                fields = []
                for fname, finfo in props.items():
                    req_mark = " (required)" if fname in required else ""
                    fields.append(f"{fname}: {finfo.get('type', 'any')}{req_mark}")
                parts.append(f"    Body: {', '.join(fields)}")

        # Show parameters if available
        params = ep.get("parameters", [])
        if params:
            param_strs = [f"{p['name']}: {p.get('type', 'any')}" for p in params]
            parts.append(f"    Params: {', '.join(param_strs)}")

    parts.append("\nOutput your complete test plan as a JSON array of actions.")
    return "\n".join(parts)


def parse_action(text: str) -> APITestAction | None:
    """Parse a single JSON action from LLM output.
    Used in multi-turn evaluation.
    """
    # Strip Qwen3 thinking blocks
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]

    json_match = re.search(r'\{[^{}]*"method"[^{}]*\}', text, re.DOTALL)
    if not json_match:
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

    return _dict_to_action(data)


def parse_test_plan(text: str) -> list[APITestAction]:
    """Parse a JSON array of actions from LLM output.
    Used in GRPO training where the model outputs a full test plan.
    """
    # Strip Qwen3 thinking blocks
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]

    # Try to find a JSON array
    # First: look for ```json [...] ```
    block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if block_match:
        json_str = block_match.group(1)
    else:
        # Find the outermost [ ... ] that contains "method"
        arr_match = re.search(r'\[.*"method".*\]', text, re.DOTALL)
        if arr_match:
            json_str = arr_match.group(0)
        else:
            # Fallback: try to find individual action objects and wrap them
            individual = re.findall(r'\{[^{}]*"method"[^{}]*\}', text, re.DOTALL)
            if individual:
                json_str = "[" + ",".join(individual) + "]"
            else:
                return []

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common issues: trailing commas
        cleaned = re.sub(r',\s*\]', ']', json_str)
        cleaned = re.sub(r',\s*\}', '}', cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        data = [data]

    actions = []
    for item in data:
        if isinstance(item, dict) and "method" in item:
            action = _dict_to_action(item)
            if action:
                actions.append(action)

    return actions


def _dict_to_action(data: dict) -> APITestAction | None:
    """Convert a dict to an APITestAction."""
    method = str(data.get("method", "GET")).upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        method = "GET"

    endpoint = data.get("endpoint", "/tasks")
    if not isinstance(endpoint, str):
        endpoint = str(endpoint)
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    headers = data.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}

    query_params = data.get("query_params") or {}
    if not isinstance(query_params, dict):
        query_params = {}

    body = data.get("body")
    if body is not None and not isinstance(body, dict):
        body = None

    expected = data.get("expected_status")
    if expected is not None:
        try:
            expected = int(expected)
        except (ValueError, TypeError):
            expected = None

    return APITestAction(
        method=HTTPMethod(method),
        endpoint=endpoint,
        headers=headers,
        query_params=query_params,
        body=body,
        expected_status=expected,
    )
