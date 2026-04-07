#!/usr/bin/env python3
"""
inference.py — Baseline inference script for the API Testing RL Environment.

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks and emits
structured [START] / [STEP] / [END] logs for automated scoring.

Required environment variables:
    API_BASE_URL  — LLM API base URL  (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME    — Model identifier  (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — API key (HuggingFace token or OpenAI key)

Optional environment variables:
    ENV_URL       — OpenEnv server base URL (default: http://localhost:8000)
"""

import json
import os
import re
import textwrap
from typing import Any, Optional

import httpx
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY:      str = os.environ.get("HF_TOKEN",     "")
ENV_URL:      str = os.environ.get("ENV_URL",      "http://localhost:8000").rstrip("/")

TEMPERATURE = 0.3
MAX_TOKENS  = 512
BENCHMARK   = "api_testing_env"
TASKS       = ["basic_validation", "edge_cases", "security_workflows"]

TASK_META = {
    "basic_validation":   {"total_bugs": 3,  "max_steps": 25},
    "edge_cases":         {"total_bugs": 9,  "max_steps": 35},
    "security_workflows": {"total_bugs": 13, "max_steps": 45},
}

# ── Structured log helpers ───────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = error if error else "none"
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert API security tester. At each step you must output a SINGLE JSON
    object representing one HTTP request to send to the API under test.

    === API ENDPOINTS ===
    GET    /tasks                    List tasks (params: status, priority, page, limit, sort)
    POST   /tasks                    Create task  (body: title*, description, status, priority, assignee_email)
    GET    /tasks/{id}               Get task by ID
    PUT    /tasks/{id}               Update task  (body: any task fields)
    DELETE /tasks/{id}               Delete task
    GET    /users                    List users
    POST   /users                    Create user  (body: username*, email*, password*, role)
    GET    /users/{id}               Get user by ID
    POST   /auth/login               Login        (body: username*, password*) → {"token": "..."}

    === ACTION FORMAT (output ONLY this JSON, no markdown, no explanation) ===
    {
        "method": "GET",
        "endpoint": "/tasks",
        "headers": {},
        "query_params": {},
        "body": null,
        "expected_status": 200
    }

    === TESTING STRATEGY ===
    Step 1-3:   GET /tasks, GET /users — discover existing resources and IDs
    Step 4-6:   POST /auth/login for alice, bob (password: "password") — get tokens
    Step 7-10:  POST /tasks with valid body (title required) → note the created ID
    Step 11-15: Test edge cases on that ID: GET, PUT, DELETE
    Step 16-20: Test bugs:
        - GET /tasks/999999        → expect 404, bug if 200
        - DELETE /tasks/999999     → expect 404, bug if 200
        - POST /tasks {}           → expect 400, bug if 500
        - GET /tasks?page=-1       → expect 400, bug if 200
        - GET /tasks?limit=999999  → expect 400, bug if 200
        - PUT /tasks/{id} {"assignee_email": "not-an-email"}  → expect 400, bug if 200
        - POST /auth/login {"username": "alice", "password": ""} → empty password auth bypass
    Step 21+:  Security tests (use bob's token to access alice's tasks, SQL injection):
        - Login as alice AND bob, then access each other's task IDs
        - POST /tasks {"title": "'; DROP TABLE tasks; --"}  → SQL injection
        - POST /tasks {"title": "A" * 6000}                → long input crash
        - POST /users {"username": "x", "email": "bad", "password": "p"}  → invalid email
        - GET /users — check if password_hash field is exposed in response

    Always set expected_status to what you EXPECT (not what the API returns) to trigger bug detection.
""").strip()


# ── Environment HTTP client ───────────────────────────────────────────────────

def env_reset(http: httpx.Client, task_id: str) -> dict[str, Any]:
    resp = http.post("/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(http: httpx.Client, action: dict[str, Any]) -> dict[str, Any]:
    resp = http.post("/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── LLM action generation ─────────────────────────────────────────────────────

_FALLBACK_SEQUENCE = [
    {"method": "GET",  "endpoint": "/tasks",      "headers": {}, "query_params": {}, "body": None, "expected_status": 200},
    {"method": "GET",  "endpoint": "/users",       "headers": {}, "query_params": {}, "body": None, "expected_status": 200},
    {"method": "POST", "endpoint": "/auth/login",  "headers": {}, "query_params": {}, "body": {"username": "alice", "password": "password"}, "expected_status": 200},
    {"method": "GET",  "endpoint": "/tasks/999999","headers": {}, "query_params": {}, "body": None, "expected_status": 404},
    {"method": "POST", "endpoint": "/tasks",       "headers": {}, "query_params": {}, "body": {},   "expected_status": 400},
    {"method": "GET",  "endpoint": "/tasks",       "headers": {}, "query_params": {"page": -1}, "body": None, "expected_status": 400},
]


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def get_action(
    llm: OpenAI,
    obs: dict[str, Any],
    history: list[str],
    step: int,
    fallback_idx: int,
) -> tuple[dict[str, Any], int]:
    """
    Ask the LLM for the next action. Returns (action_dict, next_fallback_idx).
    Falls back to a deterministic sequence if the LLM fails.
    """
    feedback      = obs.get("feedback", "")
    bugs_found    = obs.get("bugs_found_so_far", 0)
    coverage      = obs.get("coverage_summary", {})
    known_ids     = obs.get("known_resource_ids", {})
    auth_tokens   = obs.get("auth_tokens", {})
    task_desc     = obs.get("task_description", "")
    max_steps     = obs.get("max_steps", 30)

    history_block = "\n".join(history[-6:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Task: {task_desc}
        Step: {step} / {max_steps}
        Bugs found so far: {bugs_found}
        Coverage: {json.dumps(coverage)}
        Known IDs (from prior POSTs): {json.dumps(known_ids)}
        Auth tokens available for users: {list(auth_tokens.keys())}
        Last response: {feedback[:300]}

        Recent history:
        {history_block}

        Output your next API test action as a JSON object.
    """).strip()

    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        action = _extract_json(raw)

        # Validate required fields
        if "method" not in action or "endpoint" not in action:
            raise ValueError("Missing method or endpoint")

        action.setdefault("headers", {})
        action.setdefault("query_params", {})
        action.setdefault("body", None)
        action.setdefault("expected_status", 200)
        return action, fallback_idx

    except Exception as exc:
        print(f"[DEBUG] LLM/parse error at step {step}: {exc}", flush=True)
        # Rotate through fallback sequence
        fb_action = _FALLBACK_SEQUENCE[fallback_idx % len(_FALLBACK_SEQUENCE)]
        return fb_action, fallback_idx + 1


# ── Score extraction ───────────────────────────────────────────────────────────

def extract_final_score(feedback: str, bugs_found: int, total_bugs: int) -> float:
    """Parse 'Final Score: X.XXXX' from feedback, or estimate from bugs."""
    match = re.search(r"Final Score:\s*([\d.]+)", feedback)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return round(min(bugs_found / max(total_bugs, 1), 1.0), 3)


# ── Run a single task episode ─────────────────────────────────────────────────

def run_task(llm: OpenAI, http: httpx.Client, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    total_bugs = TASK_META[task_id]["total_bugs"]
    max_steps  = TASK_META[task_id]["max_steps"]

    # Reset environment
    try:
        result = env_reset(http, task_id)
    except Exception as exc:
        print(f"[DEBUG] env reset failed for {task_id}: {exc}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    obs       = result.get("observation", result)
    done      = result.get("done", False)
    rewards:  list[float] = []
    history:  list[str]   = []
    steps_taken   = 0
    final_score   = 0.0
    fallback_idx  = 0

    for step in range(1, max_steps + 1):
        if done:
            break

        action, fallback_idx = get_action(llm, obs, history, step, fallback_idx)
        action_str = json.dumps(action)

        error: Optional[str] = None
        reward = 0.0
        try:
            step_result = env_step(http, action)
            obs    = step_result.get("observation", step_result)
            reward = float(step_result.get("reward") or 0.0)
            done   = step_result.get("done", False)
        except Exception as exc:
            error = str(exc)
            done  = False

        rewards.append(reward)
        steps_taken = step

        # Build history entry for context
        feedback = obs.get("feedback", "") if isinstance(obs, dict) else ""
        history.append(
            f"Step {step}: {action.get('method')} {action.get('endpoint')} "
            f"→ {obs.get('status_code', '?') if isinstance(obs, dict) else '?'} | {feedback[:100]}"
        )

        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        if done:
            bugs_found = obs.get("bugs_found_so_far", 0) if isinstance(obs, dict) else 0
            final_score = extract_final_score(feedback, bugs_found, total_bugs)
            break

    # If episode ended without done=True (step limit hit), score from last obs
    if not done and isinstance(obs, dict):
        bugs_found  = obs.get("bugs_found_so_far", 0)
        final_score = extract_final_score(obs.get("feedback", ""), bugs_found, total_bugs)

    success = final_score > 0.0
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    llm  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)

    try:
        for task_id in TASKS:
            run_task(llm, http, task_id)
            print("", flush=True)
    finally:
        http.close()


if __name__ == "__main__":
    main()
