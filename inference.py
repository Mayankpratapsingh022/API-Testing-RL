#!/usr/bin/env python3
"""
inference.py — OpenEnv API Testing Environment baseline inference script.

Runs an LLM agent against the API Testing Environment for all 3 tasks
(basic_validation -> edge_cases -> security_workflows) and emits the
mandatory [START]/[STEP]/[END] stdout format used by the OpenEnv judging
pipeline.

Required env vars (per OpenEnv submission spec):
    API_BASE_URL   The OpenAI-compatible LLM endpoint
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Bearer token for the LLM endpoint (or API_KEY)

Optional env vars:
    IMAGE_NAME            Docker image to spin up the env via from_docker_image()
    LOCAL_IMAGE_NAME      Alias for IMAGE_NAME
    ENV_BASE_URL          URL of an already-running env server (e.g. http://localhost:8000)
    INFERENCE_TASKS       Comma-separated subset of tasks to run (default: all 3)
    INFERENCE_MAX_STEPS   Override max steps per task
    INFERENCE_TEMPERATURE Default 0.4
    INFERENCE_MAX_TOKENS  Default 4096 (plan completions need room for ~25 actions)

The script uses PLAN MODE: one LLM call per task produces a complete JSON
test plan, then the env executes each action sequentially. This matches the
GRPO training distribution and keeps total LLM cost to 3 calls per run, so
the script comfortably runs under 20 min on 2 vCPU / 8 GB RAM.

Usage:
    # Local in-process (no Docker, fastest)
    python inference.py

    # Against a built docker image
    IMAGE_NAME=api-testing-env:latest python inference.py

    # Against an already running server
    ENV_BASE_URL=http://localhost:8000 python inference.py

    # Against a deployed HF Space
    ENV_BASE_URL=https://your-user-api-testing-env.hf.space python inference.py
"""

import json
import os
import sys
import time
import traceback
from typing import Any, Optional

# Make sibling modules importable when run from the repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Auto-load .env file if present (for local development)
# Judges set env vars directly so this is harmless in production
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_THIS_DIR, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv is optional

from openai import OpenAI

from models import APITestAction, HTTPMethod  # noqa: E402
from training.prompts import (  # noqa: E402
    PLAN_SYSTEM_PROMPT,
    format_plan_prompt,
    parse_test_plan,
)


# ---------------------------------------------------------------------------
# Config (env vars per OpenEnv spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Default model: must be available on the HuggingFace Inference Router.
# Llama-3.3-70B-Instruct is reliable, follows JSON instructions well, and free.
# Override via: MODEL_NAME=other/model python inference.py
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not API_KEY:
    print(
        "[ERROR] No HF_TOKEN or API_KEY found in environment.\n"
        "  Set one of:\n"
        "    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
        "  Or create a .env file in this directory with:\n"
        "    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
        "  Get a token from: https://huggingface.co/settings/tokens\n"
        "  Make sure it has 'Make calls to Inference Providers' permission.",
        file=sys.stderr,
    )
    sys.exit(1)

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")

BENCHMARK = "api_testing_env"
DEFAULT_TASKS = ["basic_validation", "edge_cases", "security_workflows"]
TASKS = [t.strip() for t in os.getenv("INFERENCE_TASKS", ",".join(DEFAULT_TASKS)).split(",") if t.strip()]

TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("INFERENCE_MAX_TOKENS", "4096"))
_MAX_STEPS_OVERRIDE = os.getenv("INFERENCE_MAX_STEPS")
MAX_STEPS_OVERRIDE: Optional[int] = int(_MAX_STEPS_OVERRIDE) if _MAX_STEPS_OVERRIDE else None


# ---------------------------------------------------------------------------
# Strict stdout logging — these line formats are checked by the judge
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_str(action: APITestAction) -> str:
    """Compact human-readable action label for the [STEP] line."""
    method = action.method.value if hasattr(action.method, "value") else str(action.method)
    return f"{method}_{action.endpoint}"


# ---------------------------------------------------------------------------
# LLM call — plan mode (one completion per task)
# ---------------------------------------------------------------------------

def get_plan_from_llm(client: OpenAI, observation) -> str:
    """Ask the LLM for a complete JSON test plan for this task.

    Wraps the array in {"actions": [...]} so we can use OpenAI structured
    output mode (`response_format={"type": "json_object"}`), which forces
    the LLM to produce valid JSON. This is much more reliable than asking
    for a raw JSON array.
    """
    user_prompt = format_plan_prompt(observation)

    # Stronger system prompt for structured output mode
    system_prompt = (
        PLAN_SYSTEM_PROMPT
        + "\n\nIMPORTANT: Output a JSON object with a single key 'actions' "
        + "containing the array of actions:\n"
        + '{"actions": [{"method": "GET", "endpoint": "/tasks", "headers": {}, '
        + '"query_params": {}, "body": null, "expected_status": 200}, ...]}'
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},  # forces valid JSON
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM response length: {len(text)} chars", flush=True)
        if len(text) > 0:
            preview = text[:300].replace("\n", " ")
            print(f"[DEBUG] LLM response preview: {preview}...", flush=True)
        else:
            print(f"[DEBUG] LLM returned EMPTY string", flush=True)
            if hasattr(completion, "choices") and completion.choices:
                finish_reason = getattr(completion.choices[0], "finish_reason", None)
                print(f"[DEBUG] finish_reason: {finish_reason}", flush=True)
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] structured-output call failed ({type(exc).__name__}: {exc}), retrying without response_format...", flush=True)
        # Some providers don't support response_format — fall back to plain text
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            print(f"[DEBUG] fallback LLM response length: {len(text)} chars", flush=True)
            return text
        except Exception as exc2:  # noqa: BLE001
            print(f"[DEBUG] fallback LLM call failed: {type(exc2).__name__}: {exc2}", flush=True)
            return ""


# ---------------------------------------------------------------------------
# Per-task scoring helper — keeps the score in [0, 1]
# ---------------------------------------------------------------------------

def compute_task_score(state, total_step_reward: float) -> float:
    """Combine grader signals into a single normalized score in [0, 1].

    The server already runs `TaskGrader.grade(...)` at episode end and adds
    that score (already in [0, 1]) on top of the last step reward. We do
    NOT trust the raw step rewards — those are sums of partial signals and
    can exceed 1.0. Instead we derive the score from the published state:
        score = 0.7 * (bugs_found / total_bugs) + 0.3 * (coverage_pct / 100)
    which is bounded in [0, 1] and rewards both finding bugs and coverage.
    """
    bugs_found = getattr(state, "bugs_found", 0) or 0
    total_bugs = getattr(state, "total_bugs", 0) or 0
    coverage_pct = getattr(state, "coverage_pct", 0.0) or 0.0

    bug_ratio = (bugs_found / total_bugs) if total_bugs > 0 else 0.0
    coverage_ratio = max(0.0, min(1.0, coverage_pct / 100.0))

    score = 0.70 * bug_ratio + 0.30 * coverage_ratio
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Environment connector — supports docker / remote / in-process
# ---------------------------------------------------------------------------

class _EnvHandle:
    """Thin wrapper that exposes a uniform reset/step/state/close API.

    Three modes, picked automatically:
        1. IMAGE_NAME set         -> APITestEnv.from_docker_image(IMAGE_NAME)
        2. ENV_BASE_URL set       -> APITestEnv(base_url=ENV_BASE_URL)
        3. neither set (default)  -> APITestEnvironment() in-process
    """

    def __init__(self):
        self._mode: str = ""
        self._client = None        # remote/docker client
        self._env = None           # in-process env

    def open(self):
        if IMAGE_NAME:
            from client import APITestEnv
            self._mode = "docker"
            self._client = APITestEnv.from_docker_image(IMAGE_NAME)
        elif ENV_BASE_URL:
            from client import APITestEnv
            self._mode = "remote"
            self._client = APITestEnv(base_url=ENV_BASE_URL)
            if hasattr(self._client, "connect"):
                self._client.connect()
        else:
            from server.environment import APITestEnvironment
            self._mode = "local"
            self._env = APITestEnvironment()
        return self

    @property
    def mode(self) -> str:
        return self._mode

    def reset(self, task_id: str, seed: int = 42):
        if self._mode in ("docker", "remote"):
            result = self._client.reset(task_id=task_id, seed=seed)
            return result.observation, result
        obs = self._env.reset(seed=seed, task_id=task_id)
        return obs, None

    def step(self, action: APITestAction):
        if self._mode in ("docker", "remote"):
            result = self._client.step(action)
            return result.observation, result.reward or 0.0, result.done
        obs = self._env.step(action)
        return obs, (obs.reward or 0.0), obs.done

    def state(self):
        if self._mode in ("docker", "remote"):
            return self._client.state()
        return self._env.state

    def close(self):
        try:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[DEBUG] env close error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# One full episode (one task) -> emits [START] / [STEP]* / [END]
# ---------------------------------------------------------------------------

def run_task(env: _EnvHandle, client: OpenAI, task_id: str, seed: int = 42) -> dict:
    rewards: list[float] = []
    steps_taken = 0
    last_error: Optional[str] = None
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, _ = env.reset(task_id=task_id, seed=seed)
        max_steps = MAX_STEPS_OVERRIDE or getattr(obs, "max_steps", 25)

        # 1) Ask the LLM for a full plan
        plan_text = get_plan_from_llm(client, obs)
        actions = parse_test_plan(plan_text) if plan_text else []

        # Fallback: if parser failed but we have text, try a more lenient parse
        if not actions and plan_text:
            print(f"[DEBUG] {task_id}: parse_test_plan returned 0, trying lenient parse...", flush=True)
            try:
                import json as _json, re as _re
                # Try to find any JSON array of objects in the text
                cleaned = plan_text
                if "</think>" in cleaned:
                    cleaned = cleaned.split("</think>", 1)[-1]
                # Find first [ and last ]
                start = cleaned.find("[")
                end = cleaned.rfind("]")
                if start >= 0 and end > start:
                    arr_str = cleaned[start:end+1]
                    raw = _json.loads(arr_str)
                    if isinstance(raw, list):
                        from training.prompts import _dict_to_action
                        for item in raw:
                            if isinstance(item, dict) and "method" in item:
                                a = _dict_to_action(item)
                                if a:
                                    actions.append(a)
                        print(f"[DEBUG] {task_id}: lenient parse recovered {len(actions)} actions", flush=True)
            except Exception as exc:
                print(f"[DEBUG] {task_id}: lenient parse failed: {exc}", flush=True)
        if not actions:
            last_error = "no_plan_parsed"
            print(f"[DEBUG] {task_id}: model produced 0 valid actions", flush=True)

        actions = actions[:max_steps]

        # 2) Execute each action and emit one [STEP] line per env.step()
        done = False
        for i, action in enumerate(actions, start=1):
            if done:
                break
            try:
                obs, reward, done = env.step(action)
                rewards.append(float(reward))
                steps_taken = i
                log_step(step=i, action=_action_str(action), reward=reward, done=done, error=None)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                rewards.append(0.0)
                steps_taken = i
                log_step(step=i, action=_action_str(action), reward=0.0, done=False, error=last_error)

        # 3) Score from final state
        try:
            final_state = env.state()
            score = compute_task_score(final_state, sum(rewards))
        except Exception as exc:  # noqa: BLE001
            last_error = last_error or f"state_error: {exc}"
            score = 0.0

    except Exception as exc:  # noqa: BLE001
        last_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    success = score >= 0.20  # any meaningful progress counts as a successful episode
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
        "error": last_error,
    }


# ---------------------------------------------------------------------------
# Main — runs all 3 tasks sequentially against ONE env handle
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(
        f"[DEBUG] inference.py starting | model={MODEL_NAME} | "
        f"base_url={API_BASE_URL} | tasks={TASKS}",
        flush=True,
    )

    env = _EnvHandle().open()
    print(f"[DEBUG] env mode={env.mode}", flush=True)

    summary: list[dict] = []
    t0 = time.time()
    try:
        for task_id in TASKS:
            result = run_task(env, client, task_id=task_id, seed=42)
            summary.append(result)
    finally:
        env.close()

    elapsed = time.time() - t0
    avg_score = sum(r["score"] for r in summary) / max(len(summary), 1)
    print(
        f"[DEBUG] inference.py finished in {elapsed:.1f}s | "
        f"avg_score={avg_score:.3f}",
        flush=True,
    )
    print("[DEBUG] per-task scores: " + json.dumps(
        {r["task_id"]: round(r["score"], 3) for r in summary}
    ), flush=True)


if __name__ == "__main__":
    main()
