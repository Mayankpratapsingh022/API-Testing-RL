"""
Training module for the API Testing Environment.

Contains:
- prompts.py     — System prompt, observation formatting, action parsing
- rewards.py     — Reward functions for GRPO (format + environment)
- agents.py      — Baseline agents (random, sequential, smart)
- grpo.py        — GRPO training loop with TRL, HF Hub push, W&B logging
- evaluate.py    — Evaluation / rollout runner (local + remote)
"""
