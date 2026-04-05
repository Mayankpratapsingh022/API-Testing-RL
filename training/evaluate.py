#!/usr/bin/env python3
"""
Evaluation and rollout runner.

- run_rollout():  Run a single episode with a HuggingFace model
- run_baseline(): Run baseline agents against the local environment
- main():         CLI for running baselines against a remote server
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
from .prompts import SYSTEM_PROMPT, format_observation, parse_action
from .agents import AGENTS


def run_rollout(
    model,
    tokenizer,
    task_id: str = "basic_validation",
    seed: int = 42,
    max_steps: int | None = None,
) -> dict:
    """Run a single episode with a HuggingFace model.

    Args:
        model: AutoModelForCausalLM instance
        tokenizer: AutoTokenizer instance
        task_id: which task to run
        seed: random seed for domain randomization

    Returns:
        dict with episode results (bugs, reward, coverage, etc.)
    """
    import torch

    env = APITestEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]

    total_reward = 0.0
    steps = 0
    actual_max = max_steps or obs.max_steps

    while not obs.done and steps < actual_max:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        action = parse_action(completion)
        if action is None:
            action = APITestAction(method=HTTPMethod.GET, endpoint="/tasks")

        obs = env.step(action)
        total_reward += obs.reward or 0.0
        steps += 1

        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": format_observation(obs)})

        method_str = action.method.value if hasattr(action.method, "value") else str(action.method)
        logger.info(
            f"  Step {steps}: {method_str} {action.endpoint} -> "
            f"{obs.status_code} | reward={obs.reward:.3f} | bugs={obs.bugs_found_so_far}"
        )

    state = env.state
    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "bugs_found": state.bugs_found,
        "total_bugs": state.total_bugs,
        "coverage_pct": state.coverage_pct,
        "bugs_found_ids": state.bugs_found_ids,
    }


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
