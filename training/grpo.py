#!/usr/bin/env python3
"""
GRPO Training Script for the API Testing Environment.

Trains a small LLM (Qwen 1.7B) to become an intelligent API tester
using Group Relative Policy Optimization (GRPO).

The environment IS the dataset — each reset(seed=N) creates a unique
episode with different users, tasks, and data. No external dataset needed.

Usage:
    # Quick test (CPU, 2 minutes)
    python -m training.grpo --test-mode

    # Real training (GPU required)
    python -m training.grpo --model-id Qwen/Qwen3-1.7B --num-episodes 100

    # See what prompts look like (no GPU needed)
    SHOW_PROMPTS=1 python -m training.grpo

    # Resume from checkpoint
    python -m training.grpo --model-id ./checkpoints/step_50
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from server.environment import APITestEnvironment
from .prompts import SYSTEM_PROMPT, format_observation
from .rewards import format_reward_fn, environment_reward_fn
from .evaluate import run_rollout


def build_training_prompts(
    num_episodes: int = 50,
    task_ids: list[str] | None = None,
) -> list[dict]:
    """Generate training prompts by sampling environment episodes.

    Each prompt = one episode starting state with a unique seed.
    The actual training happens when GRPO rolls out actions and
    collects rewards from the environment.
    """
    if task_ids is None:
        task_ids = ["basic_validation", "edge_cases", "security_workflows"]

    prompts = []
    env = APITestEnvironment()

    for i in range(num_episodes):
        task_id = task_ids[i % len(task_ids)]
        seed = i * 1000 + 42

        obs = env.reset(seed=seed, task_id=task_id)
        user_message = format_observation(obs)

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        prompts.append({
            "prompt": prompt_messages,
            "task_id": task_id,
            "seed": seed,
        })

    logger.info(f"Generated {len(prompts)} training prompts across tasks: {task_ids}")
    return prompts


def train_grpo(args):
    """Run GRPO training with TRL."""
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}\n"
            "Install with: pip install trl transformers peft datasets torch"
        )
        sys.exit(1)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto" if not args.test_mode else None,
    )

    # Evaluate base model BEFORE training (establishes LLM baseline)
    logger.info("=" * 60)
    logger.info("Evaluating BASE model (before GRPO)...")
    logger.info("=" * 60)
    base_results = {}
    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        result = run_rollout(model, tokenizer, task_id=task_id, seed=9999)
        base_results[task_id] = result
        logger.info(
            f"  [BASE] {task_id}: reward={result['total_reward']:.3f}, "
            f"bugs={result['bugs_found']}/{result['total_bugs']}, "
            f"coverage={result['coverage_pct']:.1f}%"
        )

    # LoRA for parameter-efficient training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Generate training prompts from the environment
    logger.info(f"Generating {args.num_episodes} training episodes...")
    raw_prompts = build_training_prompts(num_episodes=args.num_episodes)

    # Format for TRL
    formatted_prompts = []
    for p in raw_prompts:
        text = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append({"prompt": text, "task_id": p["task_id"], "seed": p["seed"]})

    dataset = Dataset.from_list(formatted_prompts)

    # Create per-prompt environments for reward computation
    envs = []
    for p in raw_prompts:
        env = APITestEnvironment()
        env.reset(seed=p["seed"], task_id=p["task_id"])
        envs.append(env)

    # Combined reward function
    def combined_reward_fn(completions, **kwargs):
        format_rewards = format_reward_fn(completions)
        env_rewards = environment_reward_fn(completions, envs=envs)
        return [f + e for f, e in zip(format_rewards, env_rewards)]

    # GRPO config
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        temperature=0.8,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_funcs=[combined_reward_fn],
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    # Evaluate AFTER training and compare to base
    logger.info("=" * 60)
    logger.info("Evaluating TRAINED model (after GRPO)...")
    logger.info("=" * 60)
    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        result = run_rollout(model, tokenizer, task_id=task_id, seed=9999)
        base = base_results[task_id]
        reward_delta = result['total_reward'] - base['total_reward']
        bug_delta = result['bugs_found'] - base['bugs_found']
        cov_delta = result['coverage_pct'] - base['coverage_pct']
        logger.info(
            f"  [TRAINED] {task_id}: reward={result['total_reward']:.3f} ({reward_delta:+.3f}), "
            f"bugs={result['bugs_found']}/{result['total_bugs']} ({bug_delta:+d}), "
            f"coverage={result['coverage_pct']:.1f}% ({cov_delta:+.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for API Testing Agent")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B", help="Base model to fine-tune")
    parser.add_argument("--output-dir", default="./checkpoints/grpo_api_tester")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO parallel rollouts per prompt")
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-mode", action="store_true", help="Quick test with tiny config")
    args = parser.parse_args()

    if args.test_mode:
        logger.info("=== TEST MODE — quick sanity check ===")
        args.num_episodes = 3
        args.num_generations = 2
        args.max_steps = 5
        args.max_completion_length = 128

    if os.environ.get("SHOW_PROMPTS"):
        prompts = build_training_prompts(num_episodes=3)
        for p in prompts:
            print(f"\n{'='*60}")
            print(f"Task: {p['task_id']} | Seed: {p['seed']}")
            print(f"{'='*60}")
            for msg in p["prompt"]:
                print(f"[{msg['role']}]: {msg['content'][:300]}...")
        return

    train_grpo(args)


if __name__ == "__main__":
    main()
