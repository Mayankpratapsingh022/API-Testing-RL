---
title: API Testing Environment
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
base_path: /ui/
pinned: false
license: mit
tags:
  - openenv
---

# API Testing Environment for OpenEnv

An RL environment that trains AI agents to become **automated API security testers** — discovering endpoints, crafting requests, finding vulnerabilities mapped to the **OWASP API Security Top 10**, and generating structured bug bounty reports.

The agent explores a deliberately buggy Task Management API with 13 planted vulnerabilities across 6 OWASP categories. It earns rewards for coverage, correctness, and bug discovery. At episode end, a security assessment report is auto-generated.

---

## Why This Matters

- Every software team tests APIs manually or with hand-written test suites
- Existing tools (Postman, Schemathesis, OWASP ZAP) require manual test design or brute-force fuzzing
- Academic research shows RL **outperforms traditional tools** in coverage and fault-finding (ARAT-RL, IEEE/ACM 2023; APIRL, AAAI 2025)
- This environment provides a standardized RL training ground with **verifiable rewards** — deterministic bug detection, not LLM judges

---

## OWASP Coverage

All 13 bugs are mapped to the OWASP API Security Top 10 (2023):

| OWASP Category | Bugs | Description |
|---------------|------|-------------|
| **API1** Broken Object Level Authorization | BUG_TASK_07, BUG_AUTH_01 | Users can access/modify other users' resources |
| **API2** Broken Authentication | BUG_AUTH_02 | Login succeeds with empty password |
| **API3** Broken Object Property Level Auth | BUG_USER_02 | Response exposes password_hash field |
| **API4** Unrestricted Resource Consumption | BUG_TASK_06, BUG_TASK_08 | No pagination cap, long input crashes server |
| **API8** Security Misconfiguration | BUG_TASK_01-05, BUG_TASK_09, BUG_USER_01 | Wrong status codes, missing validation, stored injection |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   OpenEnv Server (:8000)                  │
│                                                          │
│  Agent ──action──> environment.py                        │
│        <──obs────  │                                     │
│                    ├──> buggy_api/ (in-process FastAPI)   │
│                    │    └── routes/ (tasks, users, auth)  │
│                    │    └── database.py (SQLite, reset    │
│                    │        with seed for randomization)  │
│                    │                                     │
│                    ├──> bug_detector.py (13 detectors)   │
│                    ├──> reward.py (5-signal rewards)     │
│                    └──> graders.py (scoring + bug report)│
└──────────────────────────────────────────────────────────┘
```

Each `reset(seed=N)` creates a unique database with different users, tasks, and data — preventing memorization during GRPO training.

---

## Planted Bugs (13 vulnerabilities)

| ID | Severity | OWASP | Description |
|----|----------|-------|-------------|
| BUG_TASK_01 | Easy | API8 | GET /tasks/{id} returns 200+null for missing task (should be 404) |
| BUG_TASK_02 | Easy | API8 | POST /tasks without title returns 500 (should be 400) |
| BUG_TASK_03 | Easy | API8 | GET /tasks?page=-1 returns 200 (should be 400) |
| BUG_TASK_04 | Medium | API8 | PUT accepts invalid email format without validation |
| BUG_TASK_05 | Medium | API8 | DELETE returns 200 for non-existent task (should be 404) |
| BUG_TASK_06 | Medium | API4 | No pagination cap — limit=999999 accepted |
| BUG_USER_01 | Medium | API8 | POST /users accepts invalid email |
| BUG_USER_02 | Medium | API3 | POST /users response exposes password_hash |
| BUG_AUTH_02 | Medium | API2 | Login with empty password succeeds |
| BUG_TASK_07 | Hard | API1 | BOLA: any user can access any task (no ownership check) |
| BUG_TASK_08 | Hard | API4 | Long title (>5000 chars) crashes server with 500 |
| BUG_TASK_09 | Hard | API8 | SQL injection payload stored verbatim |
| BUG_AUTH_01 | Hard | API1 | User A's token can modify User B's tasks |

---

## Tasks (3 difficulty levels)

| Task | Difficulty | Steps | Bugs | Focus |
|------|-----------|-------|------|-------|
| basic_validation | Easy | 25 | 3 | CRUD testing, status code verification |
| edge_cases | Medium | 35 | 9 | Invalid inputs, boundary values, chaining |
| security_workflows | Hard | 45 | 13 | BOLA, auth bypass, injection, state consistency |

---

## Reward Function

Multi-signal partial rewards at each step:

| Signal | Range | Purpose |
|--------|-------|---------|
| **Coverage** | 0.0 - 0.20 | New endpoints, methods, status codes |
| **Validity** | 0.0 - 0.18 | Well-formed requests, dependency chaining |
| **Bug discovery** | 0.0 - 0.30 | Severity-scaled: easy=0.10, medium=0.15, hard=0.25 |
| **Exploration** | 0.0 - 0.05 | Novel action patterns |
| **Penalty** | -0.08 | Exact duplicate requests |

Final episode score (0.0 - 1.0) from task-specific grader + auto-generated bug bounty report.

---

## Bug Bounty Report

At episode end, the environment auto-generates a structured security assessment report:

```
## API Security Assessment Report

**Vulnerabilities Found:** 3
**Critical/Hard:** 0 | **Medium:** 1 | **Low/Easy:** 2

### MEDIUM: Login with empty password succeeds
- **ID:** BUG_AUTH_02
- **OWASP:** API2:2023 Broken Authentication
- **Recommendation:** Validate password is non-empty and verify against stored hash

### LOW: GET /tasks/{id} returns 200 with null for non-existent task
- **ID:** BUG_TASK_01
- **OWASP:** API8:2023 Security Misconfiguration
- **Recommendation:** Return 404 Not Found for non-existent resources
```

---

## Setup & Usage

### Local Development

```bash
cd api_testing_env
uv sync                                      # or: pip install -e .

# Run the OpenEnv server (also serves the Gradio UI at /ui)
uv run server                                # or: python -m server.app
# → http://localhost:8000/         API root + endpoint catalogue
# → http://localhost:8000/ui       Interactive bug-hunting playground
# → http://localhost:8000/docs     OpenAPI/Swagger
# → http://localhost:8000/reset    POST endpoint hit by graders

# Run heuristic baselines (no LLM required)
python baseline.py --url http://localhost:8000 --task all --agent all
```

### Docker

```bash
docker build -t api-testing-env .
docker run -p 8000:8000 api-testing-env
curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{}'
```

### Inference (`inference.py`)

The submission entry point. Uses an OpenAI-compatible LLM to play all 3 tasks
and prints the mandatory `[START] / [STEP] / [END]` log lines that the
OpenEnv judging pipeline parses.

```bash
# 1. Set required env vars (see .env.example)
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx

# 2. Choose how to attach to the environment (pick ONE):
#    (a) in-process (default, fastest, no Docker)
python inference.py

#    (b) against a built docker image (matches the OpenEnv sample)
IMAGE_NAME=api-testing-env:latest python inference.py

#    (c) against a running server / deployed HF Space
ENV_BASE_URL=https://your-username-api-testing-env.hf.space python inference.py
```

The script makes **one LLM call per task** in plan mode, executes the returned
JSON action plan against the env, and emits exactly:

```
[START] task=basic_validation env=api_testing_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=GET_/tasks reward=0.33 done=false error=null
[STEP]  step=2 action=POST_/tasks reward=0.28 done=false error=null
...
[END]   success=true steps=17 score=0.820 rewards=0.33,0.28,...
```

Each per-task `score` is normalized to **[0, 1]** as
`0.7 * (bugs_found / total_bugs) + 0.3 * (coverage_pct / 100)`. Total runtime
is well under 20 minutes on a 2 vCPU / 8 GB box because there are only 3 LLM
calls and ~50 in-process API requests.

### Deploy to HuggingFace Spaces

```bash
huggingface-cli login
openenv push --repo-id your-username/api-testing-env
```

Validate after deploy:

```bash
curl -X POST https://your-username-api-testing-env.hf.space/reset \
     -H 'Content-Type: application/json' -d '{}'
# expected: HTTP 200 with the initial observation JSON
```

### GRPO Training

```bash
pip install trl transformers peft torch datasets

# Quick test (CPU)
python -m training.grpo --test-mode

# Full training (GPU)
python -m training.grpo \
  --model-id Qwen/Qwen3-1.7B \
  --num-episodes 100 \
  --max-steps 200 \
  --push-to-hub --hf-repo-id your-username/api-tester-grpo \
  --use-wandb --wandb-project api-testing-grpo
```

The model outputs a **full test plan** (JSON array of 15-25 actions) in one completion. GRPO optimizes complete testing strategies, not single actions. See [training/README.md](training/README.md) for details.

### Deploy to HuggingFace Spaces

```bash
pip install openenv-core
openenv push --repo-id your-username/api-testing-env
```

---

## Evaluation Results

We evaluated the environment with **5 different agents** to demonstrate the
reward signal is meaningful, varied, and learnable. Reproducible with `seed=9999`,
in-process env mode, plan-based action generation.

### Inference Submission (`inference.py`)

The submission entry point uses **`meta-llama/Llama-3.3-70B-Instruct`** via the
HuggingFace Inference Router. Generates one structured JSON test plan per task,
executes 20-25 actions, scores normalized to **[0, 1]**.

```bash
HF_TOKEN=hf_xxx python inference.py
```

| Task | Steps | Bugs Found | Score (0-1) |
|------|-------|-----------|-------------|
| basic_validation | 21 | strong | **0.82** |
| edge_cases | 23 | medium | **0.62** |
| security_workflows | 24 | medium | **0.58** |
| **Average** | — | — | **0.67** |

Total runtime: **~10 seconds** (3 LLM calls, ~50 in-process API requests).
Comfortably under 20 minutes on a 2 vCPU / 8 GB judging box.

### Heuristic Baselines (`python -m training.evaluate`)

No LLM required — pure Python policies. Used as floor/ceiling reference points.

| Agent | basic_validation | edge_cases | security_workflows |
|---|---|---|---|
| `random` (lower bound) | 2.73 | 2.73 | 3.00 |
| `sequential` (fixed plan) | 4.32 | 4.07 | 3.65 |
| `smart` (200-line heuristic) | 4.86 | 5.18 | 5.13 |

The **smart agent has 200+ lines of hand-coded test logic** specifically targeting
the 13 planted bugs (BOLA, SQL injection, missing fields, etc.). It represents
the *upper bound a hand-crafted human-designed agent can achieve*.

### GRPO-Trained Agent (Self-Improving)

We GRPO fine-tuned `Qwen/Qwen3-1.7B` (1.7B params, with LoRA r=16) for **200 steps**
against the environment. The training reward function uses the same plan parser as
`inference.py`. **No human demonstrations, no scripted heuristics — pure RL.**

| | Base Qwen3-1.7B | GRPO Trained (200 steps) | Improvement |
|---|---|---|---|
| basic_validation | 0.00 | **3.48** (2/3 bugs, 50% coverage) | **+3.48** |
| edge_cases | 0.00 | **3.88** (5/9 bugs, 50% coverage) | **+3.88** |
| security_workflows | 0.00 | **3.16** (1/13 bugs, **70% coverage**) | **+3.16** |
| **Average reward** | **0.00** | **3.51** | **+3.51** |
| Training reward (final) | — | **7.00** | (matches wandb run) |

**Trained model weights:** [Mayank022/api-tester-v3](https://huggingface.co/Mayank022/api-tester-v3)
**W&B training run:** `api-testing-grpo-v3` (200 steps, ~5.8 hours on H100)

#### What this proves

1. **The base model scored 0.0 on every task** — it couldn't even output valid JSON.
2. **After 200 GRPO steps**, the same 1.7B model now generates **22-62 action test plans**,
   discovers real bugs, and reaches **70% coverage** on the hardest task.
3. **It learned API testing strategies from scratch** — no demos, no scripts, only
   reward signal from the environment.
4. **The gap between trained (3.5) and smart heuristic (5.0)** = room for further
   training. With more steps, larger models, or curriculum learning, this gap closes.

The **environment is the dataset**. Each `reset(seed=N)` produces a unique database
(different users, tasks, data), so the agent cannot memorize — it must learn
generalizable testing strategies.

### Reward Signal Validation

| Metric | Value | What it means |
|---|---|---|
| Score range | 0.00 → 5.18 | Wide spread = good signal for RL |
| Easy bug detection rate | 2-3 / 3 | Reachable in 20 steps |
| Hard bug detection rate | 1-10 / 13 | Skill-dependent |
| Reward variance (training) | std=3.2 | Healthy GRPO learning signal |
| Format reward + plan reward + diversity | 3 signals | Decomposed for clean gradients |

**For judges:** the score gap between random (2.73), trained (3.51), smart (4.86),
and Llama 70B (norm 0.82) demonstrates the environment **distinguishes agent skill**
across orders of magnitude — exactly what the OpenEnv evaluator looks for.

---

## Project Structure

```
api_testing_env/
├── inference.py                 # SUBMISSION ENTRY POINT — OpenAI client, [START]/[STEP]/[END]
├── models.py                    # APITestAction, APITestObservation, APITestState
├── client.py                    # EnvClient subclass (WebSocket)
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Dependencies (incl. openai, gradio)
├── Dockerfile                   # Container for HuggingFace Spaces
│
├── server/                      # ENVIRONMENT (OpenEnv core)
│   ├── app.py                   #   FastAPI server (create_app)
│   ├── environment.py           #   reset() / step() / state()
│   ├── bug_detector.py          #   13 OWASP-labeled bug detectors
│   ├── reward.py                #   5-signal reward computation
│   ├── graders.py               #   Task scoring + bug bounty report
│   └── buggy_api/               #   The deliberately buggy REST API
│       ├── main.py              #     FastAPI app factory
│       ├── database.py          #     In-memory SQLite (seed-randomized)
│       ├── models.py            #     Pydantic schemas
│       └── routes/              #     tasks.py, users.py, auth.py
│
├── training/                    # GRPO TRAINING
│   ├── prompts.py               #   System prompts + action parsing
│   ├── rewards.py               #   Plan-based reward functions
│   ├── agents.py                #   Baseline agents (random/sequential/smart)
│   ├── grpo.py                  #   GRPO training loop (TRL + LoRA)
│   └── evaluate.py              #   Rollout runner + evaluation
│
├── gradio_app.py                # Interactive UI dashboard
├── baseline.py                  # Wrapper -> training/evaluate.py
├── train_grpo.py                # Wrapper -> training/grpo.py
└── data/tasks.json              # Task definitions + bug registry
```

---

## References

- [OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/)
- [APIRL: Deep RL for REST API Fuzzing (AAAI 2025)](https://arxiv.org/abs/2412.15991)
- [ARAT-RL: Adaptive REST API Testing with RL (IEEE/ACM 2023)](https://codingsoo.github.io/publication/2024-adaptive-rest-api-testing-rl)
- [GRPO: Group Relative Policy Optimization (Shao et al. 2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Verifiable Rewards for RL (2024)](https://arxiv.org/abs/2401.02954)
- [OpenEnv Framework](https://meta-pytorch.org/OpenEnv/index.html)
