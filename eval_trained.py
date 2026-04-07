#!/usr/bin/env python3
"""
Re-evaluate the trained GRPO model without re-training.

Usage:
    python eval_trained.py
    python eval_trained.py --checkpoint ./checkpoints/grpo_api_tester
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Suppress noisy logs
for _noisy in ["httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/grpo_api_tester",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-1.7B",
        help="Base model (needed if checkpoint is LoRA-only)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Max actions per task during evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="Random seed for evaluation",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Re-evaluating trained model")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Base model: {args.base_model}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Seed:       {args.seed}")
    print(f"{'='*60}\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("  WARNING: No GPU — eval will be slow")

    # Load tokenizer (from base model is fine)
    print(f"  Loading tokenizer from {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"  Loading base model {args.base_model}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Load LoRA adapter from checkpoint
    print(f"  Loading LoRA adapter from {args.checkpoint}...", flush=True)
    try:
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        # Merge LoRA into base for faster inference
        print(f"  Merging LoRA into base...", flush=True)
        model = model.merge_and_unload()
        print(f"  Model loaded successfully.", flush=True)
    except Exception as exc:
        print(f"  WARNING: Failed to load LoRA adapter: {exc}", flush=True)
        print(f"  Using base model without LoRA.", flush=True)
        model = base_model

    # Run evaluation on all 3 tasks
    from training.evaluate import run_rollout

    print(f"\n{'='*60}")
    print(f"  Running evaluation on all tasks...")
    print(f"{'='*60}\n")

    results = {}
    for task_id in ["basic_validation", "edge_cases", "security_workflows"]:
        print(f"\n--- Task: {task_id} ---")
        result = run_rollout(
            model, tokenizer,
            task_id=task_id,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        results[task_id] = result
        print(f"  reward={result['total_reward']:.3f}, "
              f"bugs={result['bugs_found']}/{result['total_bugs']}, "
              f"coverage={result['coverage_pct']:.1f}%")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Reward':<10} {'Bugs':<10} {'Coverage':<10}")
    print(f"{'-'*60}")
    for task_id, r in results.items():
        print(f"{task_id:<25} {r['total_reward']:<10.3f} "
              f"{r['bugs_found']}/{r['total_bugs']:<8} "
              f"{r['coverage_pct']:<10.1f}%")
    print(f"{'='*60}\n")

    avg = sum(r["total_reward"] for r in results.values()) / len(results)
    print(f"  Average reward: {avg:.3f}")


if __name__ == "__main__":
    main()
