#!/usr/bin/env python3
"""
Evaluation and rollout runner.

- run_rollout():        Run a single episode with a HuggingFace model
- run_baseline_local(): Run baseline agents against the local environment
- run_baseline():       Run baseline agents against a remote server
- main():              CLI for running baselines
"""

import argparse
import asyncio
import logging
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from models import APITestAction, HTTPMethod
from server.environment import APITestEnvironment
from .prompts import (
    PLAN_SYSTEM_PROMPT, format_plan_prompt,
    parse_action, parse_test_plan,
)
from .agents import AGENTS


def run_rollout(
    model,
    tokenizer,
    task_id: str = "basic_validation",
    seed: int = 42,
    max_steps: int | None = None,
) -> dict:
    """Run a single episode with a HuggingFace model.

    Uses PLAN mode: the model generates a full test plan (JSON array) in one shot,
    then all actions are executed sequentially. This matches how training works.

    Falls back to multi-turn mode if the model can't produce a valid plan.
    """
    import torch
    import time as _time

    # Force GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Move model to GPU if it's on CPU
        if next(model.parameters()).device.type == "cpu":
            logger.info("  Moving model to GPU...")
            model = model.to(device)
    else:
        device = next(model.parameters()).device

    env = APITestEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    actual_max = max_steps or obs.max_steps
    logger.info(f"  Rollout: {task_id} | max_steps={actual_max} | device={device}")

    # --- Try plan mode first (matches training) ---
    plan_prompt = format_plan_prompt(obs)
    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": plan_prompt},
    ]

    # Qwen3 thinking support
    chat_kwargs = {}
    if "qwen3" in str(getattr(model, "name_or_path", "") or "").lower():
        chat_kwargs["enable_thinking"] = True

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    gen_start = _time.time()
    print(f"  Generating test plan...", end="", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4096,  # Match training max_completion_length
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    gen_time = _time.time() - gen_start
    print(f" done ({gen_time:.1f}s, {len(completion)} chars)")

    # Parse the plan
    actions = parse_test_plan(completion)
    if actions:
        logger.info(f"  Plan generated: {len(actions)} actions")
    else:
        # Fallback: try single action parse
        single = parse_action(completion)
        if single:
            actions = [single]
            logger.info("  Plan parse failed, got 1 action from fallback")
        else:
            logger.warning("  Failed to parse any actions from model output")
            # Print first 500 chars of completion for debugging
            preview = completion[:500].replace("\n", " ")
            logger.warning(f"  Model output preview: {preview}...")
            actions = []

    # Limit to max_steps
    actions = actions[:actual_max]

    # Execute all actions
    total_reward = 0.0
    for i, action in enumerate(actions):
        try:
            obs = env.step(action)
            total_reward += obs.reward or 0.0
            method_str = action.method.value if hasattr(action.method, "value") else str(action.method)
            print(f"  Step {i+1}/{len(actions)}: {method_str} {action.endpoint} -> "
                  f"{obs.status_code} | reward={obs.reward:.3f} | bugs={obs.bugs_found_so_far}")
        except Exception as e:
            print(f"  Step {i+1}/{len(actions)}: ERROR - {e}")

    # If no actions were generated, show that
    if not actions:
        print("  (no valid actions generated)")

    state = env.state
    return {
        "task_id": task_id,
        "seed": seed,
        "steps": len(actions),
        "total_reward": round(total_reward, 4),
        "bugs_found": state.bugs_found,
        "total_bugs": state.total_bugs,
        "coverage_pct": state.coverage_pct,
        "bugs_found_ids": state.bugs_found_ids,
    }


def run_baseline_local(
    agent_name: str = "all",
    task_id: str = "all",
    seed: int = 42,
) -> list[dict]:
    """Run baseline agents against the local environment (no server needed).

    Args:
        agent_name: "random", "sequential", "smart", or "all"
        task_id: task ID or "all"
        seed: random seed

    Returns:
        List of result dicts with agent, task_id, total_reward, bugs_found, etc.
    """
    tasks = ["basic_validation", "edge_cases", "security_workflows"] if task_id == "all" else [task_id]
    agents = list(AGENTS.items()) if agent_name == "all" else [(agent_name, AGENTS[agent_name])]

    results = []
    for tid in tasks:
        for aname, agent_cls in agents:
            random.seed(seed)
            agent = agent_cls()
            env = APITestEnvironment()
            obs = env.reset(seed=seed, task_id=tid)

            total_reward = 0.0
            step = 0

            while not obs.done and step < obs.max_steps:
                obs_dict = {
                    "status_code": obs.status_code,
                    "response_body": obs.response_body,
                    "feedback": obs.feedback,
                    "bugs_found_so_far": obs.bugs_found_so_far,
                    "coverage_summary": obs.coverage_summary,
                    "known_resource_ids": obs.known_resource_ids,
                    "auth_tokens": obs.auth_tokens,
                    "steps_taken": obs.steps_taken,
                    "max_steps": obs.max_steps,
                }

                action = agent.act(obs_dict)
                obs = env.step(action)
                total_reward += obs.reward or 0.0
                step += 1

            state = env.state
            result = {
                "agent": aname,
                "task_id": tid,
                "seed": seed,
                "steps": step,
                "total_reward": round(total_reward, 4),
                "bugs_found": state.bugs_found,
                "total_bugs": state.total_bugs,
                "coverage_pct": state.coverage_pct,
                "bugs_found_ids": state.bugs_found_ids,
            }
            results.append(result)
            logger.info(
                f"  [{aname}] {tid}: reward={result['total_reward']:.4f}, "
                f"bugs={result['bugs_found']}/{result['total_bugs']}, "
                f"coverage={result['coverage_pct']:.1f}%"
            )

    return results


# =====================================================================
# Remote baseline runner (against server via WebSocket client)
# =====================================================================

async def run_episode(url: str, task_id: str, agent_cls, seed: int = 42) -> dict:
    """Run one baseline episode against a remote server."""
    from client import APITestEnv

    random.seed(seed)
    agent = agent_cls()

    async with APITestEnv(base_url=url) as env:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        logger.info(f"Starting {agent.name} agent on task '{task_id}'")

        total_reward = 0.0
        step = 0

        while not result.done:
            obs_dict = {
                "status_code": obs.status_code,
                "response_body": obs.response_body,
                "feedback": obs.feedback,
                "bugs_found_so_far": obs.bugs_found_so_far,
                "coverage_summary": obs.coverage_summary,
                "known_resource_ids": obs.known_resource_ids,
                "auth_tokens": obs.auth_tokens,
                "steps_taken": obs.steps_taken,
                "max_steps": obs.max_steps,
            }

            action = agent.act(obs_dict)
            result = await env.step(action)
            obs = result.observation
            total_reward += result.reward or 0

            step += 1
            method = action.method.value if hasattr(action.method, "value") else str(action.method)
            logger.info(
                f"  Step {step}: {method} {action.endpoint} -> "
                f"{obs.status_code} | reward={result.reward:.4f} | bugs={obs.bugs_found_so_far}"
            )

        state = await env.state()
        return {
            "task_id": task_id,
            "agent": agent.name,
            "total_reward": round(total_reward, 4),
            "bugs_found": state.bugs_found,
            "total_bugs": state.total_bugs,
            "coverage_pct": state.coverage_pct,
            "steps": step,
        }


async def main_async(args):
    tasks = ["basic_validation", "edge_cases", "security_workflows"] if args.task == "all" else [args.task]
    agents = list(AGENTS.values()) if args.agent == "all" else [AGENTS[args.agent]]

    results = []
    for task_id in tasks:
        for agent_cls in agents:
            try:
                result = await run_episode(args.url, task_id, agent_cls, seed=args.seed)
                results.append(result)
                logger.info(
                    f"\nRESULT: {result['agent']} on {result['task_id']}: "
                    f"reward={result['total_reward']}, bugs={result['bugs_found']}/{result['total_bugs']}, "
                    f"coverage={result['coverage_pct']:.1f}%"
                )
            except Exception as e:
                logger.error(f"Error running {agent_cls.name} on {task_id}: {e}", exc_info=True)

    if results:
        print("\n" + "=" * 80)
        print("BASELINE RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Agent':<15} {'Task':<25} {'Score':<10} {'Bugs':<10} {'Coverage':<10}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['agent']:<15} {r['task_id']:<25} "
                f"{r['total_reward']:<10.4f} "
                f"{r['bugs_found']}/{r['total_bugs']:<8} "
                f"{r['coverage_pct']:<10.1f}%"
            )
        print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline agents for API Testing Environment")
    parser.add_argument("--url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--task", default="all",
                        choices=["basic_validation", "edge_cases", "security_workflows", "all"])
    parser.add_argument("--agent", default="all", choices=["random", "sequential", "smart", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
