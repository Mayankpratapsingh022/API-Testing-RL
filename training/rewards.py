"""
Reward functions for GRPO training.

Two reward signals:
1. format_reward  — Does the LLM output valid JSON? (+1 / -1)
2. env_reward     — What does the environment say? (coverage + bugs, scaled by 5.0)
"""

from .prompts import parse_action


def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward for valid JSON output format.

    +1.0 if the completion contains valid parseable JSON action
    -1.0 if it doesn't (teaches the model to always output valid JSON)
    """
    rewards = []
    for text in completions:
        action = parse_action(text)
        rewards.append(1.0 if action is not None else -1.0)
    return rewards


def environment_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward from actually executing the action in the environment.

    For each completion:
    1. Parse the JSON action
    2. Step the environment
    3. Return the environment's reward (coverage + bugs + validity)

    Scaled by 5.0 to make it dominant over format reward.
    """
    envs = kwargs.get("envs", [])
    rewards = []

    for i, text in enumerate(completions):
        action = parse_action(text)
        if action is None:
            rewards.append(-0.5)
            continue

        env = envs[i % len(envs)] if envs else None
        if env is None:
            rewards.append(0.0)
            continue

        try:
            obs = env.step(action)
            rewards.append((obs.reward or 0.0) * 5.0)
        except Exception:
            rewards.append(-0.5)

    return rewards
