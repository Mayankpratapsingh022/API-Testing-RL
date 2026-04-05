# API Testing Environment 

An  RL environment where an AI agent learns to **test REST APIs intelligently** — discovering endpoints, crafting requests, validating responses, finding bugs, and handling edge cases.

The agent is given an API specification and must explore a deliberately buggy Task Management API, earning rewards for coverage, correctness, and bug discovery. Think of it as training an AI to become an automated QA engineer.

---

## Why This Matters

- Every software team tests APIs manually or with hand-written test suites
- Existing tools (Postman, Schemathesis) require manual test design or brute-force fuzzing
- Academic research (IEEE/ACM 2023-2024) shows RL **outperforms traditional tools** in coverage and fault-finding
- This environment provides a standardized training ground for intelligent API testing agents

---

## Environment Design

### Architecture

```
                                        ┌─────────────────────────┐
┌──────────────┐    APITestAction       │    OpenEnv Environment  │
│   AI Agent   │ ──────────────────────>│    (reset/step/state)   │
│   (learner)  │                        │         │               │
│              │ <──────────────────────│    ┌────▼────────────┐  │
└──────────────┘   APITestObservation   │    │  Buggy REST API │  │
                   + reward + feedback  │    │  (FastAPI+SQLite)│  │
                                        │    └─────────────────┘  │
                                        └─────────────────────────┘
```

The environment wraps a **deliberately buggy Task Management API** running in-process. When the agent sends an HTTP request (action), the environment forwards it to the buggy API, analyzes the response, checks for bug detection, computes multi-signal rewards, and returns the observation.

### Action Space

```python
class APITestAction(Action):
    method: HTTPMethod        # GET, POST, PUT, DELETE, PATCH
    endpoint: str             # /tasks, /users/1, /auth/login
    headers: dict[str, str]   # {"Authorization": "Bearer xxx"}
    query_params: dict        # {"page": 1, "limit": 10}
    body: dict | None         # {"title": "Test", "email": "a@b.com"}
    expected_status: int|None # What the agent expects (used for scoring)
```

### Observation Space

```python
class APITestObservation(Observation):
    # API specification
    available_endpoints: list[dict]      # Full endpoint catalog

    # Response from last request
    status_code: int                     # HTTP status
    response_body: Any                   # JSON response
    response_headers: dict               # Response headers
    response_time_ms: float              # Latency

    # Feedback & progress
    feedback: str                        # Human-readable feedback
    bugs_found_so_far: int               # Running bug count
    coverage_summary: dict               # Endpoints/methods/status codes tested

    # Context for chaining
    known_resource_ids: dict[str, list]  # IDs from POST responses
    auth_tokens: dict[str, str]          # Tokens from login

    # Episode info
    task_id: str
    steps_taken: int
    max_steps: int
    done: bool
    reward: float
```

---

## Tasks (Easy -> Medium -> Hard)

### Task 1: Basic Endpoint Validation (Easy)
**Goal:** Test all CRUD endpoints with valid inputs, verify status codes, find obvious bugs.

| Criteria | Weight | Description |
|----------|--------|-------------|
| GET coverage | 0.25 | Test all GET endpoints |
| POST testing | 0.20 | Create resources successfully |
| PUT/DELETE | 0.15 | Test update and delete |
| Bug discovery | 0.20 | Find 2+ easy bugs |
| Schema validation | 0.20 | See 4+ different status codes |

**Bugs to find:** 3 (wrong status codes, missing field handling)
**Max steps:** 25

### Task 2: Edge Cases & Error Handling (Medium)
**Goal:** Test boundary conditions, invalid inputs, chain CRUD operations.

| Criteria | Weight | Description |
|----------|--------|-------------|
| Missing fields | 0.15 | Test with missing required fields |
| Invalid types | 0.15 | Send wrong data types |
| Boundary values | 0.15 | Negative pages, huge limits |
| Non-existent resources | 0.15 | Test with invalid IDs |
| Bug discovery | 0.20 | Find 3+ edge case bugs |
| Dependency chaining | 0.20 | Create -> read -> update -> delete |

**Bugs to find:** 9 (validation gaps, pagination, email format)
**Max steps:** 35

### Task 3: Security & Multi-Step Workflows (Hard)
**Goal:** Find authorization flaws, injection vulnerabilities, workflow bugs.

| Criteria | Weight | Description |
|----------|--------|-------------|
| Cross-user auth | 0.20 | Login as 2+ users, test access |
| Injection testing | 0.20 | Try SQL injection patterns |
| State consistency | 0.20 | Create -> delete -> re-fetch |
| Security bugs | 0.20 | Find 2+ security vulnerabilities |
| Workflow coverage | 0.20 | 80%+ API coverage |

**Bugs to find:** 13 (BOLA, broken auth, content injection, long input crash)
**Max steps:** 45

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

Final episode score (0.0 - 1.0) computed by task-specific grader.

---

## Planted Bugs

| ID | Severity | Description |
|----|----------|-------------|
| BUG_TASK_01 | Easy | GET /tasks/{id} returns 200+null for missing task |
| BUG_TASK_02 | Easy | POST /tasks without title returns 500 |
| BUG_TASK_03 | Easy | GET /tasks?page=-1 returns 200 |
| BUG_TASK_04 | Medium | PUT accepts invalid email format |
| BUG_TASK_05 | Medium | DELETE returns 200 for non-existent task |
| BUG_TASK_06 | Medium | No pagination cap (limit=999999 works) |
| BUG_USER_01 | Medium | POST /users accepts invalid email |
| BUG_USER_02 | Medium | Response exposes password_hash |
| BUG_AUTH_02 | Medium | Empty password login succeeds |
| BUG_TASK_07 | Hard | BOLA: any user can access any task |
| BUG_TASK_08 | Hard | Long title (>5000 chars) causes 500 |
| BUG_TASK_09 | Hard | SQL injection payload stored verbatim |
| BUG_AUTH_01 | Hard | User A's token can modify User B's tasks |

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized execution)

### Local Development

```bash
# Clone and install
cd api_testing_env
pip install -e .

# Or with uv
uv sync

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or
uv run server
```

### Gradio UI (Interactive Dashboard)

```bash
# Install gradio
pip install gradio

# Launch the interactive UI
cd api_testing_env
python gradio_app.py

# Opens at http://localhost:7860
```

The Gradio UI provides:
- **Task selection** — choose easy/medium/hard tasks and reset
- **Request builder** — craft HTTP requests with method, endpoint, headers, body
- **Quick actions** — one-click bug-hunting actions (e.g., "GET /tasks/999999")
- **Live scoreboard** — step counter, bugs found, coverage progress bar
- **Reward breakdown** — per-step coverage/validity/bug/exploration/penalty signals
- **Activity log** — scrollable history of all requests and responses
- **Baseline agents** — watch random/sequential/smart agents run automatically
- **Bug tracker** — color-coded list of discovered bugs with severity

### Docker

```bash
# Build
docker build -t api-testing-env .

# Run
docker run -p 8000:8000 api-testing-env

# Health check
curl http://localhost:8000/health
```

### Using the Client

```python
from api_testing_env.client import APITestEnv
from api_testing_env.models import APITestAction, HTTPMethod

async with APITestEnv(base_url="http://localhost:8000") as env:
    # Reset with a task
    result = await env.reset(task_id="basic_validation")
    print(result.observation.feedback)

    # Send a test action
    result = await env.step(APITestAction(
        method=HTTPMethod.GET,
        endpoint="/tasks",
        expected_status=200,
    ))
    print(f"Status: {result.observation.status_code}")
    print(f"Reward: {result.reward}")
    print(f"Bugs found: {result.observation.bugs_found_so_far}")
```

### Running Baselines

```bash
# Start server first, then:
python baseline.py --url http://localhost:8000 --task all --agent all

# Run specific agent on specific task
python baseline.py --url http://localhost:8000 --task security_workflows --agent smart
```

---

## Baseline Scores

| Agent | Basic Validation | Edge Cases | Security Workflows |
|-------|-----------------|------------|-------------------|
| Random | ~0.15 | ~0.08 | ~0.03 |
| Sequential | ~0.45 | ~0.25 | ~0.10 |
| Smart (Heuristic) | ~0.65 | ~0.45 | ~0.35 |
| **Ideal (RL-trained)** | **1.0** | **1.0** | **1.0** |

---

## Project Structure

```
api_testing_env/
├── __init__.py
├── models.py                    # APITestAction, APITestObservation, APITestState
├── client.py                    # EnvClient subclass
├── openenv.yaml                 # Environment manifest
├── pyproject.toml               # Dependencies
├── Dockerfile                   # Container specification
├── baseline.py                  # Baseline inference script
├── gradio_app.py                # Interactive Gradio UI dashboard
├── README.md                    # This file
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI server (create_app)
    ├── environment.py           # OpenEnv Environment (reset/step/state)
    ├── bug_detector.py          # Bug detection logic
    ├── reward.py                # Multi-signal reward computation
    ├── graders.py               # Task-specific grading
    └── buggy_api/               # The deliberately buggy API
        ├── __init__.py
        ├── main.py              # FastAPI app factory
        ├── database.py          # In-memory SQLite
        ├── models.py            # Pydantic schemas
        └── routes/
            ├── __init__.py
            ├── tasks.py         # Task CRUD (10 planted bugs)
            ├── users.py         # User management (2 bugs)
            └── auth.py          # Authentication (2 bugs)
```

---

## Deployment to HuggingFace Spaces

```bash
# Install OpenEnv CLI
pip install openenv-core

# Push to HuggingFace
openenv push --repo-id your-username/api-testing-env
```

---

## References

- [Adaptive REST API Testing with RL (IEEE/ACM 2023)](https://dl.acm.org/doi/10.1109/ASE56229.2023.00218)
- [OpenEnv Framework](https://meta-pytorch.org/OpenEnv/index.html)
- [OWASP API Security Top 10](https://owasp.org/API-Security/)
- [REST API Testing Best Practices](https://www.code-intelligence.com/rest-api-testing)
