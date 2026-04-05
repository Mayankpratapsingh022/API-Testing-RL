# Training Module

Everything related to training an AI agent to test APIs using GRPO (Group Relative Policy Optimization).

---

## Quick Start

```bash
cd api_testing_env

# 1. See what training prompts look like (no GPU needed)
SHOW_PROMPTS=1 python -m training.grpo

# 2. Quick sanity check (CPU, ~2 minutes)
python -m training.grpo --test-mode

# 3. Real training (GPU required)
pip install trl transformers peft torch datasets
python -m training.grpo --model-id Qwen/Qwen3-1.7B --num-episodes 50

# 4. Run baseline agents
python -m training.evaluate --task all --agent all --url http://localhost:8000
```

---

## How Training Works

There is **no external dataset**. The environment generates unique episodes on the fly.

```
                  ┌─────────────────────────────────────────────┐
                  │           GRPO Training Loop                │
                  │                                             │
  ┌───────────┐   │  1. env.reset(seed=N)                      │
  │           │   │     → unique users, tasks, data             │
  │  Qwen     │   │                                             │
  │  1.7B     │──▶│  2. LLM generates: {"method":"GET",...}     │
  │  + LoRA   │   │                                             │
  │           │◀──│  3. env.step(action) → reward               │
  └───────────┘   │     coverage + bugs + validity              │
                  │                                             │
                  │  4. GRPO: generate 4 attempts per prompt,   │
                  │     keep best, update model weights          │
                  │                                             │
                  │  5. Repeat with next seed                   │
                  └─────────────────────────────────────────────┘
```

### Why no dataset file?

Each `reset(seed=N)` creates a **unique database** with different users, tasks, and data:

| Seed | Users | Tasks |
|------|-------|-------|
| 42 | diana, alice, xander, ivan, hannah | 8 tasks |
| 99 | mike, george, tom, fiona | 6 tasks |
| 7 | priya, kevin, wendy | 4 tasks |

The agent can't memorize "login as alice" because alice might not exist. It must **read the observation and adapt** — that's the learning signal.

The bugs (13 planted flaws) are structural — same code flaws every episode — but the path to finding them changes because the data is different.

---

## File Guide

| File | Purpose | When to modify |
|------|---------|----------------|
| `prompts.py` | System prompt, `format_observation()`, `parse_action()` | Change how the LLM sees tasks or formats actions |
| `rewards.py` | `format_reward_fn()`, `environment_reward_fn()` | Tune reward scaling or add new reward signals |
| `agents.py` | `RandomAgent`, `SequentialAgent`, `SmartAgent` | Add new baseline strategies |
| `grpo.py` | `build_training_prompts()`, `train_grpo()` | Change training hyperparameters or model |
| `evaluate.py` | `run_rollout()`, remote baseline runner | Change evaluation logic |

### prompts.py

The bridge between the environment and the LLM.

**`SYSTEM_PROMPT`** — Instructions telling the LLM it's an API tester. Includes output format (JSON) and testing strategies.

**`format_observation(obs)`** — Converts an environment observation into text:
- First turn: full API spec + task description + available users
- Later turns: last response + feedback + progress stats + auth tokens

**`parse_action(text)`** — Extracts JSON from LLM output. Handles:
- Raw JSON: `{"method": "GET", "endpoint": "/tasks"}`
- Code blocks: `` ```json {...} ``` ``
- Extra text around JSON: `"I'll try: {...}"`

### rewards.py

Two reward functions that GRPO uses to score each LLM completion:

**`format_reward_fn`** — Binary: +1.0 if valid JSON action, -1.0 if not. Teaches the model to always output parseable actions.

**`environment_reward_fn`** — Runs the action in the environment and returns the actual reward (coverage + bugs + validity), scaled by 5.0 to dominate over format reward.

### agents.py

Three hand-coded baselines for comparison:

| Agent | Strategy | Expected Score |
|-------|----------|---------------|
| `RandomAgent` | Random method + random endpoint | ~0.10 |
| `SequentialAgent` | Fixed sequence: GET, POST, PUT, DELETE each endpoint | ~0.35 |
| `SmartAgent` | Multi-phase: discover → auth → CRUD → bug hunt → security | ~0.55 |

A GRPO-trained model should beat the SmartAgent.

### grpo.py

The main training script.

**`build_training_prompts(num_episodes)`** — Creates N prompts by resetting the environment with seeds 0..N. Each prompt is a chat message with system prompt + initial observation.

**`train_grpo(args)`** — Full GRPO loop:
1. Load model + tokenizer (Qwen 1.7B default)
2. Apply LoRA (r=16, alpha=32, targets q_proj + v_proj)
3. Generate prompts from environment
4. Create per-prompt environment instances for reward eval
5. Train with TRL's GRPOTrainer
6. Save model + run evaluation

### evaluate.py

**`run_rollout(model, tokenizer, task_id, seed)`** — Runs one full episode with a HuggingFace model. Multi-turn: LLM generates action → env steps → LLM sees result → repeats.

**`run_episode(url, task_id, agent_cls)`** — Runs a baseline agent against a remote server via WebSocket.

---

## Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-id` | `Qwen/Qwen3-1.7B` | Base model (any HF causal LM) |
| `--num-episodes` | 50 | Training prompts (more = more diverse episodes) |
| `--num-generations` | 4 | GRPO rollouts per prompt (higher = better but slower) |
| `--max-completion-length` | 256 | Max tokens per LLM response |
| `--max-steps` | 200 | Total training optimizer steps |
| `--learning-rate` | 2e-5 | AdamW learning rate |
| `--batch-size` | 1 | Per-device batch size |
| `--output-dir` | `./checkpoints/grpo_api_tester` | Where to save model |

### Hardware Requirements

| Setup | GPU | Time | Model |
|-------|-----|------|-------|
| Colab Free | T4 (16GB) | ~1-2 hours | Qwen 1.7B + 4-bit LoRA |
| Colab Pro | A100 (40GB) | ~30 min | Qwen 4B + LoRA |
| Local | Any 8GB+ | ~1-2 hours | Qwen 1.7B + 4-bit LoRA |
| CPU only | None | `--test-mode` only | Verifies pipeline works |

---

## Expected Results

### Before Training (base Qwen 1.7B, no fine-tuning)

The base model can output JSON sometimes, but has no API testing strategy:
```
basic_validation:    ~0.15 (random-level)
edge_cases:          ~0.08
security_workflows:  ~0.03
```

### After GRPO (50 episodes, 200 steps)

The model learns systematic testing patterns:
```
basic_validation:    ~0.55-0.65
edge_cases:          ~0.35-0.45
security_workflows:  ~0.25-0.35
```

### What the Model Learns

1. **Output format** — Always produce valid JSON (format reward)
2. **Coverage** — Test different endpoints, don't repeat the same request
3. **Dependency chaining** — POST to create, then GET/PUT/DELETE the created resource
4. **Bug patterns** — Try non-existent IDs, missing fields, invalid emails
5. **Auth workflows** — Login first, use tokens in subsequent requests
6. **Security testing** — Try cross-user access, injection payloads

---

## Extending the Training

### Add a new reward signal

Edit `rewards.py`:

```python
def efficiency_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward for concise, focused actions (penalize wasted steps)."""
    rewards = []
    for text in completions:
        action = parse_action(text)
        if action and action.expected_status:
            rewards.append(0.5)  # Bonus for predicting expected status
        else:
            rewards.append(0.0)
    return rewards
```

Then add it to the combined reward in `grpo.py`.

### Add a new baseline agent

Edit `agents.py`:

```python
class CoverageAgent:
    """Agent that prioritizes hitting every endpoint once."""
    name = "coverage"

    def __init__(self):
        self.tested = set()
        # ...
```

Then add it to the `AGENTS` dict.

### Use a different model

```bash
# Qwen 2.5 (smaller, faster)
python -m training.grpo --model-id Qwen/Qwen2.5-1.5B

# Llama 3 (if you have access)
python -m training.grpo --model-id meta-llama/Llama-3.2-1B
```

Any HuggingFace causal language model works — just make sure it supports chat templates.
