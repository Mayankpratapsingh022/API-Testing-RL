#!/usr/bin/env python3
"""
GRPO Training Script for the API Testing Environment.

Trains a small LLM (Qwen 1.7B) to become an intelligent API tester
using Group Relative Policy Optimization (GRPO).

The environment IS the dataset — each reset(seed=N) creates a unique
episode with different users, tasks, and data. No external dataset needed.

Features:
    - Auto-push trained model weights to HuggingFace Hub
    - Weights & Biases logging for metrics, loss, rewards
    - Baseline agent evaluation before GRPO (random, sequential, smart)
    - Base model evaluation before GRPO for comparison
    - Post-training evaluation with delta reporting
    - Saves metrics, comparison tables, and plots to output dir

Usage:
    # Quick test (CPU, 2 minutes)
    python -m training.grpo --test-mode

    # Real training (GPU required)
    python -m training.grpo --model-id Qwen/Qwen3-1.7B --num-episodes 100

    # With HF Hub push
    python -m training.grpo --push-to-hub --hf-repo-id your-username/api-tester-grpo

    # With Weights & Biases
    python -m training.grpo --use-wandb --wandb-project api-testing-grpo

    # See what prompts look like (no GPU needed)
    SHOW_PROMPTS=1 python -m training.grpo

    # Resume from checkpoint
    python -m training.grpo --model-id ./checkpoints/step_50
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from server.environment import APITestEnvironment
from .prompts import SYSTEM_PROMPT, format_observation
from .rewards import format_reward_fn, environment_reward_fn
from .evaluate import run_rollout, run_baseline_local


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


def run_baseline_evaluation(seed: int = 9999) -> dict:
    """Run all baseline agents and return results for comparison.

    Returns:
        dict with structure: {agent_name: {task_id: result_dict}}
    """
    logger.info("=" * 60)
    logger.info("Running BASELINE AGENT evaluation...")
    logger.info("=" * 60)

    results = run_baseline_local(agent_name="all", task_id="all", seed=seed)

    # Organize by agent -> task
    organized = {}
    for r in results:
        agent = r["agent"]
        if agent not in organized:
            organized[agent] = {}
        organized[agent][r["task_id"]] = r

    # Print summary table
    print("\n" + "=" * 90)
    print("BASELINE AGENT RESULTS")
    print("=" * 90)
    print(f"{'Agent':<15} {'Task':<25} {'Reward':<10} {'Bugs':<12} {'Coverage':<10}")
    print("-" * 90)
    for agent_name in ["random", "sequential", "smart"]:
        if agent_name not in organized:
            continue
        for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
            r = organized[agent_name].get(task_id, {})
            print(
                f"{agent_name:<15} {task_id:<25} "
                f"{r.get('total_reward', 0):<10.4f} "
                f"{r.get('bugs_found', 0)}/{r.get('total_bugs', 0):<10} "
                f"{r.get('coverage_pct', 0):<10.1f}%"
            )
        print("-" * 90)
    print("=" * 90 + "\n")

    return organized


def save_metrics(
    output_dir: str,
    baseline_results: dict,
    base_model_results: dict,
    trained_model_results: dict,
    training_args: dict,
    training_time_s: float,
):
    """Save all metrics and comparison data to output_dir/metrics/."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Full results JSON
    all_results = {
        "training_args": training_args,
        "training_time_seconds": round(training_time_s, 1),
        "baseline_agents": {},
        "base_model": base_model_results,
        "trained_model": trained_model_results,
    }

    # Flatten baseline results
    for agent_name, tasks in baseline_results.items():
        all_results["baseline_agents"][agent_name] = {}
        for task_id, r in tasks.items():
            all_results["baseline_agents"][agent_name][task_id] = {
                "total_reward": r.get("total_reward", 0),
                "bugs_found": r.get("bugs_found", 0),
                "total_bugs": r.get("total_bugs", 0),
                "coverage_pct": r.get("coverage_pct", 0),
            }

    with open(os.path.join(metrics_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Comparison table as markdown
    md_lines = ["# Training Results\n"]
    md_lines.append(f"**Model**: {training_args.get('model_id', 'unknown')}")
    md_lines.append(f"**Training time**: {training_time_s / 60:.1f} minutes")
    md_lines.append(f"**Episodes**: {training_args.get('num_episodes', 0)}")
    md_lines.append(f"**Max steps**: {training_args.get('max_steps', 0)}\n")

    md_lines.append("## Comparison Table\n")
    md_lines.append("| Agent/Model | Task | Reward | Bugs | Coverage |")
    md_lines.append("|---|---|---|---|---|")

    # Baselines
    for agent_name in ["random", "sequential", "smart"]:
        if agent_name not in baseline_results:
            continue
        for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
            r = baseline_results[agent_name].get(task_id, {})
            md_lines.append(
                f"| {agent_name} | {task_id} | "
                f"{r.get('total_reward', 0):.4f} | "
                f"{r.get('bugs_found', 0)}/{r.get('total_bugs', 0)} | "
                f"{r.get('coverage_pct', 0):.1f}% |"
            )

    # Base model
    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        r = base_model_results.get(task_id, {})
        md_lines.append(
            f"| **base model** | {task_id} | "
            f"{r.get('total_reward', 0):.4f} | "
            f"{r.get('bugs_found', 0)}/{r.get('total_bugs', 0)} | "
            f"{r.get('coverage_pct', 0):.1f}% |"
        )

    # Trained model
    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        r = trained_model_results.get(task_id, {})
        base = base_model_results.get(task_id, {})
        delta = r.get("total_reward", 0) - base.get("total_reward", 0)
        md_lines.append(
            f"| **GRPO trained** | {task_id} | "
            f"{r.get('total_reward', 0):.4f} ({delta:+.4f}) | "
            f"{r.get('bugs_found', 0)}/{r.get('total_bugs', 0)} | "
            f"{r.get('coverage_pct', 0):.1f}% |"
        )

    md_lines.append("")
    with open(os.path.join(metrics_dir, "results.md"), "w") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Metrics saved to {metrics_dir}/")


def save_plots(output_dir: str, baseline_results: dict, base_model_results: dict, trained_model_results: dict):
    """Generate and save comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation. pip install matplotlib")
        return

    plots_dir = os.path.join(output_dir, "metrics", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    tasks = ["basic_validation", "edge_cases", "security_workflows"]
    task_labels = ["Basic", "Edge Cases", "Security"]

    # --- Plot 1: Reward comparison bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tasks))
    width = 0.15

    agents_to_plot = []
    for agent_name in ["random", "sequential", "smart"]:
        if agent_name in baseline_results:
            rewards = [baseline_results[agent_name].get(t, {}).get("total_reward", 0) for t in tasks]
            agents_to_plot.append((agent_name, rewards))

    base_rewards = [base_model_results.get(t, {}).get("total_reward", 0) for t in tasks]
    agents_to_plot.append(("Base Model", base_rewards))

    trained_rewards = [trained_model_results.get(t, {}).get("total_reward", 0) for t in tasks]
    agents_to_plot.append(("GRPO Trained", trained_rewards))

    colors = ["#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#2ecc71"]
    for i, (name, rewards) in enumerate(agents_to_plot):
        offset = (i - len(agents_to_plot) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rewards, width, label=name, color=colors[i % len(colors)])
        for bar, val in zip(bars, rewards):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Task")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Comparison: Baselines vs Base Model vs GRPO Trained")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "reward_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Bugs found comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, _) in enumerate(agents_to_plot):
        if name in baseline_results:
            bugs = [baseline_results[name].get(t, {}).get("bugs_found", 0) for t in tasks]
        elif name == "Base Model":
            bugs = [base_model_results.get(t, {}).get("bugs_found", 0) for t in tasks]
        else:
            bugs = [trained_model_results.get(t, {}).get("bugs_found", 0) for t in tasks]
        offset = (i - len(agents_to_plot) / 2 + 0.5) * width
        ax.bar(x + offset, bugs, width, label=name, color=colors[i % len(colors)])

    total_bugs = [base_model_results.get(t, {}).get("total_bugs", 0) or
                  trained_model_results.get(t, {}).get("total_bugs", 0) for t in tasks]
    ax.plot(x, total_bugs, "k--", marker="D", label="Total Bugs", linewidth=1.5)

    ax.set_xlabel("Task")
    ax.set_ylabel("Bugs Found")
    ax.set_title("Bug Discovery: Baselines vs Base Model vs GRPO Trained")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "bugs_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Coverage comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, _) in enumerate(agents_to_plot):
        if name in baseline_results:
            cov = [baseline_results[name].get(t, {}).get("coverage_pct", 0) for t in tasks]
        elif name == "Base Model":
            cov = [base_model_results.get(t, {}).get("coverage_pct", 0) for t in tasks]
        else:
            cov = [trained_model_results.get(t, {}).get("coverage_pct", 0) for t in tasks]
        offset = (i - len(agents_to_plot) / 2 + 0.5) * width
        ax.bar(x + offset, cov, width, label=name, color=colors[i % len(colors)])

    ax.set_xlabel("Task")
    ax.set_ylabel("Coverage %")
    ax.set_title("API Coverage: Baselines vs Base Model vs GRPO Trained")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "coverage_comparison.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots saved to {plots_dir}/")


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

    # --- W&B setup ---
    wandb_run = None
    report_to = "none"
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"grpo-{args.model_id.split('/')[-1]}-{int(time.time())}",
                config={
                    "model_id": args.model_id,
                    "num_episodes": args.num_episodes,
                    "num_generations": args.num_generations,
                    "max_steps": args.max_steps,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "max_completion_length": args.max_completion_length,
                    "lora_r": 16,
                    "lora_alpha": 32,
                },
            )
            report_to = "wandb"
            logger.info(f"W&B initialized: project={args.wandb_project}, run={wandb_run.name}")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging. pip install wandb")
            args.use_wandb = False

    training_args_dict = {
        "model_id": args.model_id,
        "num_episodes": args.num_episodes,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_completion_length": args.max_completion_length,
        "output_dir": args.output_dir,
        "test_mode": args.test_mode,
    }

    # --- Step 1: Run baseline agent evaluation ---
    baseline_results = run_baseline_evaluation(seed=9999)

    if args.use_wandb and wandb_run:
        import wandb
        for agent_name, tasks in baseline_results.items():
            for task_id, r in tasks.items():
                wandb.log({
                    f"baseline/{agent_name}/{task_id}/reward": r["total_reward"],
                    f"baseline/{agent_name}/{task_id}/bugs": r["bugs_found"],
                    f"baseline/{agent_name}/{task_id}/coverage": r["coverage_pct"],
                })

    # --- Step 2: Load model and tokenizer ---
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

    # --- Step 3: Evaluate base model BEFORE training ---
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
        if args.use_wandb and wandb_run:
            import wandb
            wandb.log({
                f"base_model/{task_id}/reward": result["total_reward"],
                f"base_model/{task_id}/bugs": result["bugs_found"],
                f"base_model/{task_id}/coverage": result["coverage_pct"],
            })

    # --- Step 4: LoRA config ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # --- Step 5: Generate training prompts ---
    logger.info(f"Generating {args.num_episodes} training episodes...")
    raw_prompts = build_training_prompts(num_episodes=args.num_episodes)

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

    # --- Step 6: GRPO training ---
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
        report_to=report_to,
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
    train_start = time.time()
    trainer.train()
    training_time = time.time() - train_start
    logger.info(f"Training completed in {training_time / 60:.1f} minutes")

    # --- Step 7: Save model locally ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    # --- Step 8: Push to HuggingFace Hub ---
    if args.push_to_hub:
        hf_repo = args.hf_repo_id
        if not hf_repo:
            logger.error("--hf-repo-id is required when using --push-to-hub")
        else:
            try:
                logger.info(f"Pushing model to HuggingFace Hub: {hf_repo}")
                trainer.push_to_hub(repo_id=hf_repo, commit_message="GRPO trained API testing agent")
                tokenizer.push_to_hub(repo_id=hf_repo, commit_message="GRPO trained API testing agent")
                logger.info(f"Model pushed to https://huggingface.co/{hf_repo}")
            except Exception as e:
                logger.error(f"Failed to push to HF Hub: {e}")
                logger.info("Make sure you're logged in: huggingface-cli login")

    # --- Step 9: Evaluate AFTER training ---
    logger.info("=" * 60)
    logger.info("Evaluating TRAINED model (after GRPO)...")
    logger.info("=" * 60)
    trained_results = {}
    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        result = run_rollout(model, tokenizer, task_id=task_id, seed=9999)
        trained_results[task_id] = result
        base = base_results[task_id]
        reward_delta = result["total_reward"] - base["total_reward"]
        bug_delta = result["bugs_found"] - base["bugs_found"]
        cov_delta = result["coverage_pct"] - base["coverage_pct"]
        logger.info(
            f"  [TRAINED] {task_id}: reward={result['total_reward']:.3f} ({reward_delta:+.3f}), "
            f"bugs={result['bugs_found']}/{result['total_bugs']} ({bug_delta:+d}), "
            f"coverage={result['coverage_pct']:.1f}% ({cov_delta:+.1f}%)"
        )
        if args.use_wandb and wandb_run:
            import wandb
            wandb.log({
                f"trained_model/{task_id}/reward": result["total_reward"],
                f"trained_model/{task_id}/bugs": result["bugs_found"],
                f"trained_model/{task_id}/coverage": result["coverage_pct"],
                f"delta/{task_id}/reward": reward_delta,
                f"delta/{task_id}/bugs": bug_delta,
                f"delta/{task_id}/coverage": cov_delta,
            })

    # --- Step 10: Print final comparison table ---
    print("\n" + "=" * 95)
    print("FINAL COMPARISON: All Agents & Models")
    print("=" * 95)
    print(f"{'Agent/Model':<18} {'Task':<25} {'Reward':<10} {'Bugs':<12} {'Coverage':<10}")
    print("-" * 95)

    for agent_name in ["random", "sequential", "smart"]:
        if agent_name in baseline_results:
            for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
                r = baseline_results[agent_name].get(task_id, {})
                print(
                    f"{agent_name:<18} {task_id:<25} "
                    f"{r.get('total_reward', 0):<10.4f} "
                    f"{r.get('bugs_found', 0)}/{r.get('total_bugs', 0):<10} "
                    f"{r.get('coverage_pct', 0):<10.1f}%"
                )
            print("-" * 95)

    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        r = base_results[task_id]
        print(
            f"{'Base Model':<18} {task_id:<25} "
            f"{r['total_reward']:<10.4f} "
            f"{r['bugs_found']}/{r['total_bugs']:<10} "
            f"{r['coverage_pct']:<10.1f}%"
        )
    print("-" * 95)

    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        r = trained_results[task_id]
        base = base_results[task_id]
        delta = r["total_reward"] - base["total_reward"]
        print(
            f"{'GRPO Trained':<18} {task_id:<25} "
            f"{r['total_reward']:<10.4f} "
            f"{r['bugs_found']}/{r['total_bugs']:<10} "
            f"{r['coverage_pct']:<10.1f}%  ({delta:+.4f})"
        )
    print("=" * 95)

    # --- Step 11: Save metrics & plots ---
    save_metrics(
        output_dir=args.output_dir,
        baseline_results=baseline_results,
        base_model_results=base_results,
        trained_model_results=trained_results,
        training_args=training_args_dict,
        training_time_s=training_time,
    )
    save_plots(
        output_dir=args.output_dir,
        baseline_results=baseline_results,
        base_model_results=base_results,
        trained_model_results=trained_results,
    )

    # --- Finalize W&B ---
    if args.use_wandb and wandb_run:
        import wandb
        # Log plots as artifacts
        plots_dir = os.path.join(args.output_dir, "metrics", "plots")
        if os.path.exists(plots_dir):
            for fname in os.listdir(plots_dir):
                if fname.endswith(".png"):
                    wandb.log({f"plots/{fname.replace('.png', '')}": wandb.Image(os.path.join(plots_dir, fname))})
        wandb.finish()
        logger.info("W&B run finalized")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for API Testing Agent")

    # Model & training
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B", help="Base model to fine-tune")
    parser.add_argument("--output-dir", default="./checkpoints/grpo_api_tester")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO parallel rollouts per prompt")
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-mode", action="store_true", help="Quick test with tiny config")

    # HuggingFace Hub
    parser.add_argument("--push-to-hub", action="store_true", help="Push trained model to HF Hub")
    parser.add_argument("--hf-repo-id", type=str, default=None,
                        help="HF Hub repo ID (e.g., your-username/api-tester-grpo)")

    # Weights & Biases
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="api-testing-grpo",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")

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
