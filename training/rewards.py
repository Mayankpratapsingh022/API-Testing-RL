"""
Reward functions for GRPO training (v2 — plan-based).

The model outputs a FULL TEST PLAN (JSON array of actions).
Each reward function creates a FRESH environment, executes ALL actions,
and scores the result.

Three reward signals:
1. format_reward    — Valid JSON array with 3+ diverse actions? (+2 / -2)
2. plan_reward      — Execute plan, score on bugs + coverage + efficiency (0 to ~8)
3. diversity_reward — Variety of methods, endpoints, and request patterns (+0 to +2)
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import APITestAction, HTTPMethod
from server.environment import APITestEnvironment
from .prompts import parse_test_plan


def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward for valid JSON test plan format.

    +2.0 if output has 5+ diverse actions (a real plan)
    +1.0 if output has 3-4 actions (minimal plan)
    +0.0 if output has 1-2 actions (barely valid)
    -2.0 if it can't be parsed at all

    Also penalizes if all actions are identical.
    """
    rewards = []
    for text in completions:
        actions = parse_test_plan(text)
        if not actions:
            rewards.append(-2.0)
            continue

        n = len(actions)

        # Check diversity — are the actions actually different?
        unique_pairs = set()
        for a in actions:
            m = a.method.value if hasattr(a.method, "value") else str(a.method)
            ep = re.sub(r'/\d+', '/{id}', a.endpoint)
            unique_pairs.add((m, ep))

        diversity_ratio = len(unique_pairs) / max(n, 1)

        if n >= 5 and diversity_ratio >= 0.5:
            rewards.append(2.0)
        elif n >= 3:
            rewards.append(1.0)
        elif n >= 1:
            rewards.append(0.0)
        else:
            rewards.append(-2.0)

        # Penalty if all actions are the same
        if len(unique_pairs) <= 1 and n > 1:
            rewards[-1] = -1.0

    return rewards


def plan_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Execute the full test plan in a FRESH environment and return a balanced score.

    Score components:
    - Bug discovery:  min(bugs_found, 5) * 1.0  (capped at 5.0 to not dominate)
    - Coverage:       (coverage_pct / 100) * 2.0 (up to 2.0)
    - Efficiency:     if bugs > 0: +0.5 per bug found in first 10 actions
    - Crash penalty:  -0.1 per action that caused a 500 error

    Total range: roughly -2 to +8

    Each completion gets its OWN fresh environment — no state pollution.
    """
    prompts_meta = kwargs.get("prompts_meta", [])
    rewards = []

    for i, text in enumerate(completions):
        actions = parse_test_plan(text)
        if not actions:
            rewards.append(-1.0)
            continue

        # Get episode seed and task
        meta = prompts_meta[i % len(prompts_meta)] if prompts_meta else {}
        seed = meta.get("seed", 42)
        task_id = meta.get("task_id", "basic_validation")

        # Create a FRESH environment
        env = APITestEnvironment()
        env.reset(seed=seed, task_id=task_id)

        # Execute all actions, track results
        crashes = 0
        step_rewards = []
        for action in actions:
            try:
                obs = env.step(action)
                step_rewards.append(obs.reward or 0.0)
                if obs.status_code >= 500:
                    crashes += 1
            except Exception:
                step_rewards.append(0.0)
                crashes += 1

        state = env.state
        coverage = state.coverage_pct

        # Component 1: Bug discovery (capped to prevent domination)
        bug_score = min(state.bugs_found, 5) * 1.0

        # Component 2: Coverage (proportional, up to 2.0)
        coverage_score = (coverage / 100) * 2.0

        # Component 3: Efficiency — finding bugs early is better
        early_bug_bonus = 0.0
        early_steps = step_rewards[:10]
        for r in early_steps:
            if r > 0.2:  # High reward step = likely found a bug
                early_bug_bonus += 0.3

        # Component 4: Crash penalty
        crash_penalty = crashes * -0.1

        # Component 5: Step reward sum (small weight — mainly for gradient signal)
        step_sum = sum(step_rewards) * 0.2

        total = bug_score + coverage_score + early_bug_bonus + crash_penalty + step_sum
        rewards.append(round(total, 4))

    return rewards


def diversity_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward for diverse test plans — varied methods, endpoints, and strategies.

    Components:
    - Method variety:     up to +0.5 (using GET/POST/PUT/DELETE)
    - Endpoint variety:   up to +0.5 (testing different endpoints)
    - Strategy variety:   up to +0.5 (auth + invalid input + boundary + injection patterns)
    - Repetition penalty: up to -0.5
    """
    rewards = []
    for text in completions:
        actions = parse_test_plan(text)
        if not actions:
            rewards.append(0.0)
            continue

        methods = set()
        endpoints = set()
        unique_pairs = set()
        has_auth = False
        has_invalid_input = False
        has_boundary = False
        has_injection = False
        has_nonexistent_id = False

        for a in actions:
            m = a.method.value if hasattr(a.method, "value") else str(a.method)
            methods.add(m)
            norm_ep = re.sub(r'/\d+', '/{id}', a.endpoint)
            endpoints.add(norm_ep)
            unique_pairs.add((m, norm_ep))

            # Detect testing strategies
            if a.endpoint == "/auth/login":
                has_auth = True
            if a.body and not a.body.get("title") and a.method.value == "POST":
                has_invalid_input = True
            qp = a.query_params or {}
            if any(isinstance(v, (int, float)) and v < 0 for v in qp.values()):
                has_boundary = True
            if any(isinstance(v, (int, float)) and v > 10000 for v in qp.values()):
                has_boundary = True
            if a.body and any("DROP" in str(v).upper() or "script" in str(v).lower()
                              for v in (a.body or {}).values()):
                has_injection = True
            if re.search(r'/\d{4,}', a.endpoint):
                has_nonexistent_id = True

        # Method variety (max 4 methods = +0.5)
        method_score = min(len(methods) / 4, 1.0) * 0.5

        # Endpoint variety (max 7 endpoints = +0.5)
        endpoint_score = min(len(endpoints) / 7, 1.0) * 0.5

        # Strategy variety (each strategy = +0.1, max +0.5)
        strategies = sum([has_auth, has_invalid_input, has_boundary, has_injection, has_nonexistent_id])
        strategy_score = min(strategies * 0.1, 0.5)

        # Repetition penalty
        if len(actions) > 0:
            repeat_count = len(actions) - len(unique_pairs)
            repetition_penalty = min(repeat_count / len(actions), 1.0) * -0.5
        else:
            repetition_penalty = 0.0

        total = method_score + endpoint_score + strategy_score + repetition_penalty
        rewards.append(round(total, 3))

    return rewards
