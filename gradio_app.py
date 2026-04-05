#!/usr/bin/env python3
"""
Gradio UI for the API Testing Environment.
"""

import json
import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional

import gradio as gr

from models import APITestAction, APITestObservation, HTTPMethod
from server.environment import APITestEnvironment, TASKS, API_SPEC


@dataclass
class SessionState:
    env: APITestEnvironment = field(default_factory=APITestEnvironment)
    initialized: bool = False
    task_id: str = ""
    step_log: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    last_obs: Optional[APITestObservation] = None


def new_session():
    return SessionState()


# =====================================================================
# Core logic
# =====================================================================

def reset_env(task_id, state):
    if not state:
        state = new_session()
    obs = state.env.reset(task_id=task_id)
    state.initialized = True
    state.task_id = task_id
    state.step_log = []
    state.total_reward = 0.0
    state.last_obs = obs
    t = TASKS[task_id]
    return (
        state,
        f"Environment reset. Task: **{task_id}** ({t['difficulty']})\n\nMax steps: {t['max_steps']} | Bugs to find: {t['total_bugs']}",
        obs.feedback,
        "",
        format_reward_display(0, 0, {}),
        f"0 / {t['total_bugs']}",
        format_coverage(obs.coverage_summary),
        "",
        f"0 / {t['max_steps']}",
        "No bugs found yet.",
        "No tokens acquired yet.",
        "No resources created yet.",
    )


def send_request(method, endpoint, headers_str, params_str, body_str, expected_status, state):
    if not state or not state.initialized:
        return (state, "Environment not initialized. Click 'Reset' first.", "", "", "", "", "", "", "", "", "")

    try:
        headers = json.loads(headers_str) if headers_str.strip() else {}
    except json.JSONDecodeError:
        return (state, "Invalid JSON in headers.", "", "", "", "", "", "", "", "", "")
    try:
        query_params = json.loads(params_str) if params_str.strip() else {}
    except json.JSONDecodeError:
        return (state, "Invalid JSON in query params.", "", "", "", "", "", "", "", "", "")
    try:
        body = json.loads(body_str) if body_str.strip() else None
    except json.JSONDecodeError:
        return (state, "Invalid JSON in body.", "", "", "", "", "", "", "", "", "")

    exp = int(expected_status) if expected_status.strip() else None
    action = APITestAction(
        method=HTTPMethod(method), endpoint=endpoint,
        headers=headers, query_params=query_params,
        body=body, expected_status=exp,
    )

    obs = state.env.step(action)
    reward = obs.reward or 0.0
    state.total_reward += reward
    state.last_obs = obs

    resp_body = obs.response_body
    if isinstance(resp_body, (dict, list)):
        resp_str = json.dumps(resp_body, indent=2)
    else:
        resp_str = str(resp_body)

    state.step_log.append({
        "step": obs.steps_taken, "method": method, "endpoint": endpoint,
        "status": obs.status_code, "reward": round(reward, 4), "bugs": obs.bugs_found_so_far,
    })

    breakdown = obs.metadata.get("reward_breakdown", {})
    reward_detail = format_reward_display(reward, state.total_reward, breakdown)

    t = TASKS[state.task_id]
    es = state.env.state

    status = ""
    if obs.done:
        status = (
            f"\n\n**EPISODE COMPLETE**\n\n"
            f"Final Score: {reward:.4f}\n"
            f"Bugs: {obs.bugs_found_so_far}/{t['total_bugs']}\n"
            f"Steps: {obs.steps_taken}/{obs.max_steps}"
        )

    return (
        state,
        obs.feedback + status,
        f"**{obs.status_code}** — {obs.response_time_ms:.1f}ms\n\n```json\n{resp_str}\n```",
        reward_detail,
        f"{obs.bugs_found_so_far} / {t['total_bugs']}",
        format_coverage(obs.coverage_summary),
        format_log(state.step_log),
        f"{obs.steps_taken} / {obs.max_steps}" + (" (DONE)" if obs.done else ""),
        format_bug_list(es.bugs_found_ids),
        format_auth_tokens(obs.auth_tokens),
        format_resources(obs.known_resource_ids),
    )


def apply_quick_action(action_name, _state):
    quick_actions = {
        "GET /tasks": ("GET", "/tasks", "{}", "{}", "", "200"),
        "GET /users": ("GET", "/users", "{}", "{}", "", "200"),
        "GET /tasks/1": ("GET", "/tasks/1", "{}", "{}", "", "200"),
        "GET /tasks/999999 (bug hunt)": ("GET", "/tasks/999999", "{}", "{}", "", "404"),
        "POST create task": ("POST", "/tasks", "{}", "{}", '{"title": "Test Task", "description": "Created via UI"}', "201"),
        "POST missing title (bug hunt)": ("POST", "/tasks", "{}", "{}", '{"description": "no title"}', "400"),
        "Login as alice": ("POST", "/auth/login", "{}", "{}", '{"username": "alice", "password": "pass"}', "200"),
        "Login as bob": ("POST", "/auth/login", "{}", "{}", '{"username": "bob", "password": "pass"}', "200"),
        "Login empty pwd (bug hunt)": ("POST", "/auth/login", "{}", "{}", '{"username": "alice", "password": ""}', "401"),
        "Negative page (bug hunt)": ("GET", "/tasks", "{}", '{"page": -1, "limit": 10}', "", "400"),
        "Huge limit (bug hunt)": ("GET", "/tasks", "{}", '{"limit": 999999}', "", "200"),
        "Invalid email PUT (bug hunt)": ("PUT", "/tasks/1", "{}", "{}", '{"assignee_email": "not-an-email"}', "422"),
        "DELETE non-existent (bug hunt)": ("DELETE", "/tasks/99999", "{}", "{}", "", "404"),
        "Create user invalid email (bug)": ("POST", "/users", "{}", "{}", '{"username": "baduser", "email": "nope", "password": "x"}', "422"),
        "SQL injection test": ("POST", "/tasks", "{}", "{}", '{"title": "test\'; DROP TABLE tasks;--"}', "201"),
        "Long title crash (bug hunt)": ("POST", "/tasks", "{}", "{}", '{"title": "' + "A" * 6000 + '"}', "400"),
    }
    if action_name and action_name in quick_actions:
        return quick_actions[action_name]
    return [gr.update()] * 6


def run_baseline_agent(agent_type, state):
    if not state or not state.initialized:
        yield state, "Environment not initialized.", "", "", "", "", "", "", "", "", ""
        return

    from training.agents import RandomAgent, SequentialAgent, SmartAgent
    agents = {"random": RandomAgent, "sequential": SequentialAgent, "smart": SmartAgent}
    agent = agents[agent_type]()
    t = TASKS[state.task_id]

    obs = state.env.reset(task_id=state.task_id)
    state.step_log = []
    state.total_reward = 0.0
    state.last_obs = obs

    while not obs.done:
        obs_dict = {
            "status_code": obs.status_code, "response_body": obs.response_body,
            "feedback": obs.feedback, "bugs_found_so_far": obs.bugs_found_so_far,
            "coverage_summary": obs.coverage_summary, "known_resource_ids": obs.known_resource_ids,
            "auth_tokens": obs.auth_tokens, "steps_taken": obs.steps_taken, "max_steps": obs.max_steps,
        }
        action = agent.act(obs_dict)
        obs = state.env.step(action)
        reward = obs.reward or 0.0
        state.total_reward += reward
        state.last_obs = obs

        ms = action.method.value if hasattr(action.method, "value") else str(action.method)
        state.step_log.append({
            "step": obs.steps_taken, "method": ms, "endpoint": action.endpoint,
            "status": obs.status_code, "reward": round(reward, 4), "bugs": obs.bugs_found_so_far,
        })

        resp_body = obs.response_body
        if isinstance(resp_body, (dict, list)):
            resp_str = json.dumps(resp_body, indent=2)
        else:
            resp_str = str(resp_body)

        breakdown = obs.metadata.get("reward_breakdown", {})
        reward_detail = format_reward_display(reward, state.total_reward, breakdown)

        es = state.env.state
        done_text = ""
        if obs.done:
            done_text = f"\n\n**EPISODE COMPLETE** — Final Score: {reward:.4f} | Bugs: {obs.bugs_found_so_far}/{t['total_bugs']}"

        yield (
            state,
            f"[{agent_type}] {ms} {action.endpoint} -> {obs.status_code}{done_text}",
            f"**{obs.status_code}**\n```json\n{resp_str[:500]}\n```",
            reward_detail,
            f"{obs.bugs_found_so_far} / {t['total_bugs']}",
            format_coverage(obs.coverage_summary),
            format_log(state.step_log),
            f"{obs.steps_taken} / {obs.max_steps}" + (" (DONE)" if obs.done else ""),
            format_bug_list(es.bugs_found_ids),
            format_auth_tokens(obs.auth_tokens),
            format_resources(obs.known_resource_ids),
        )
        time.sleep(0.3)


# =====================================================================
# Formatters
# =====================================================================

def format_reward_display(step_reward, cumulative, breakdown):
    """Render reward metrics as styled HTML with explanations."""
    components = [
        ("Coverage", breakdown.get("coverage", 0),
         "Reward for testing new endpoints and methods"),
        ("Validity", breakdown.get("validity", 0),
         "Reward for sending well-formed requests that return expected status codes"),
        ("Bug", breakdown.get("bug_discovery", 0),
         "Bonus for discovering a new bug in the API"),
        ("Explore", breakdown.get("exploration", 0),
         "Reward for trying new parameter combinations and edge cases"),
        ("Penalty", breakdown.get("penalty", 0),
         "Deduction for repeated or invalid requests"),
    ]
    bars = []
    for label, value, tip in components:
        val_color = "#16a34a" if value > 0 else "#dc2626" if value < 0 else "inherit"
        bars.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:2px 0;font-size:0.82em;" title="{tip}">'
            f'<span style="opacity:0.6;cursor:help;border-bottom:1px dotted currentColor;">'
            f'{label}</span>'
            f'<span style="color:{val_color};font-family:monospace;font-weight:600;">'
            f'{value:+.3f}</span></div>'
        )
    cum_color = "#16a34a" if cumulative > 0 else "#dc2626" if cumulative < 0 else "inherit"
    step_color = "#16a34a" if step_reward > 0 else "#dc2626" if step_reward < 0 else "inherit"
    return (
        f'<div style="display:flex;gap:16px;margin-bottom:8px;">'
        f'<div style="flex:1;text-align:center;padding:6px;background:rgba(128,128,128,0.1);'
        f'border-radius:8px;">'
        f'<div style="font-size:0.72em;opacity:0.55;">STEP REWARD</div>'
        f'<div style="font-size:1.3em;font-weight:700;color:{step_color};">'
        f'{step_reward:+.4f}</div></div>'
        f'<div style="flex:1;text-align:center;padding:6px;background:rgba(128,128,128,0.1);'
        f'border-radius:8px;">'
        f'<div style="font-size:0.72em;opacity:0.55;">CUMULATIVE</div>'
        f'<div style="font-size:1.3em;font-weight:700;color:{cum_color};">'
        f'{cumulative:.4f}</div></div></div>'
        f'<div style="border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:6px 10px;">'
        f'<div style="font-size:0.72em;opacity:0.5;margin-bottom:4px;">'
        f'REWARD BREAKDOWN '
        f'<span title="How the reward for the last step was calculated"'
        f' style="cursor:help;">&#9432;</span></div>'
        + "".join(bars)
        + "</div>"
    )


def format_coverage(summary):
    if not summary:
        return "No data"
    pct = summary.get("coverage_pct", 0)
    tested = summary.get("endpoints_tested", 0)
    total = summary.get("total_endpoints", 0)
    pairs = summary.get("method_endpoint_pairs", 0)
    codes = summary.get("status_codes_seen", [])
    color = "#dc2626" if pct < 30 else "#d97706" if pct < 70 else "#16a34a"
    bar_html = (
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
        f'<div style="flex:1;background:rgba(128,128,128,0.15);border-radius:6px;height:14px;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:6px;'
        f'transition:width 0.3s ease;"></div></div>'
        f'<span style="font-weight:700;min-width:48px;text-align:right;">{pct:.1f}%</span></div>'
    )
    code_pills = ""
    for c in codes:
        cc = "#16a34a" if 200 <= c < 300 else "#d97706" if 300 <= c < 400 else "#dc2626"
        code_pills += (
            f'<span style="background:{cc}18;color:{cc};padding:1px 7px;border-radius:10px;'
            f'font-size:0.78em;font-weight:600;margin-right:4px;">{c}</span>'
        )
    return (
        f"{bar_html}"
        f'<div style="display:flex;gap:10px;margin:6px 0;font-size:0.82em;">'
        f'<div style="flex:1;text-align:center;padding:4px;background:rgba(128,128,128,0.1);border-radius:6px;"'
        f' title="How many unique API endpoints have been called">'
        f'<div style="font-size:0.72em;opacity:0.5;">ENDPOINTS</div>'
        f'<div style="font-weight:700;">{tested}/{total}</div></div>'
        f'<div style="flex:1;text-align:center;padding:4px;background:rgba(128,128,128,0.1);border-radius:6px;"'
        f' title="Unique combinations of HTTP method + endpoint path tested">'
        f'<div style="font-size:0.72em;opacity:0.5;">METHOD+PATH</div>'
        f'<div style="font-weight:700;">{pairs}</div></div></div>'
        f'<div style="margin-top:4px;" title="HTTP status codes received from the API so far">'
        f'<span style="font-size:0.72em;opacity:0.5;">STATUS CODES SEEN </span>'
        f'{code_pills}</div>'
    )


def format_log(log):
    if not log:
        return (
            '<div style="opacity:0.55;font-size:0.85em;">'
            "Each row shows an API request the agent made, the HTTP status it got back, "
            "and the reward earned. Green = positive reward, red = penalty."
            "</div>"
        )
    method_colors = {
        "GET": "#2563eb", "POST": "#16a34a", "PUT": "#d97706",
        "DELETE": "#dc2626", "PATCH": "#9333ea",
    }
    rows = []
    for entry in log[-20:]:
        m = entry["method"]
        mcol = method_colors.get(m, "#6b7280")
        r = entry["reward"]
        rcol = "#16a34a" if r > 0 else "#dc2626" if r < 0 else "inherit"
        bug_tag = (
            '<span style="background:#92400e;color:#fef08a;padding:0 5px;border-radius:4px;'
            'font-size:0.7em;margin-left:4px;">BUG FOUND</span>'
        ) if r > 0.2 else ""
        status = entry["status"]
        scol = "#16a34a" if 200 <= status < 300 else "#d97706" if 300 <= status < 400 else "#dc2626"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
            f'border-bottom:1px solid rgba(128,128,128,0.1);font-size:0.82em;">'
            f'<span style="opacity:0.45;min-width:20px;text-align:right;">{entry["step"]}</span>'
            f'<span style="background:{mcol}18;color:{mcol};padding:1px 6px;border-radius:4px;'
            f'font-weight:600;font-size:0.8em;min-width:52px;text-align:center;">{m}</span>'
            f'<span style="flex:1;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;">{entry["endpoint"]}</span>'
            f'<span style="color:{scol};font-weight:600;min-width:28px;text-align:right;">{status}</span>'
            f'<span style="color:{rcol};min-width:52px;text-align:right;font-family:monospace;'
            f'font-size:0.85em;">{r:+.3f}</span>{bug_tag}</div>'
        )
    omitted = ""
    if len(log) > 20:
        omitted = (
            f'<div style="opacity:0.45;font-size:0.78em;padding:4px 0;text-align:center;">'
            f'... {len(log) - 20} earlier steps not shown</div>'
        )
    header = (
        '<div style="opacity:0.55;font-size:0.78em;margin-bottom:6px;">'
        "API requests made by the agent. Each row: step number, HTTP method, "
        "endpoint, status code, and reward earned.</div>"
        '<div style="display:flex;gap:6px;padding:2px 0 6px;border-bottom:1px solid rgba(128,128,128,0.2);'
        'font-size:0.75em;opacity:0.5;">'
        '<span style="min-width:20px;text-align:right;">#</span>'
        '<span style="min-width:52px;text-align:center;">Method</span>'
        '<span style="flex:1;">Endpoint</span>'
        '<span style="min-width:28px;text-align:right;">Status</span>'
        '<span style="min-width:52px;text-align:right;">Reward</span></div>'
    )
    return header + omitted + "\n".join(rows)


def format_bug_list(bug_ids):
    if not bug_ids:
        return "No bugs found yet."
    from server.bug_detector import BugDetector
    detector = BugDetector("security_workflows")
    severity_colors = {
        "easy": "#16a34a",
        "medium": "#d97706",
        "hard": "#dc2626",
    }
    cards = []
    for bid in sorted(bug_ids):
        bug = detector.bugs.get(bid)
        if bug:
            fg = severity_colors.get(bug.severity, "#6b7280")
            cards.append(
                f'<div style="border:1px solid {fg}40;border-radius:8px;padding:8px 10px;'
                f'margin-bottom:6px;background:{fg}0d;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-weight:700;font-size:0.85em;">{bid}</span>'
                f'<span style="background:{fg};color:#fff;padding:1px 8px;border-radius:10px;'
                f'font-size:0.75em;font-weight:600;">{bug.severity.upper()}</span></div>'
                f'<div style="margin-top:4px;font-size:0.85em;opacity:0.7;">'
                f'{bug.description}</div></div>'
            )
    return "\n".join(cards)


def format_auth_tokens(tokens):
    if not tokens:
        return (
            '<div style="opacity:0.5;font-size:0.85em;">'
            "No tokens yet. Login via <code>POST /auth/login</code> to get auth tokens "
            "for testing protected endpoints.</div>"
        )
    cards = []
    for user, token in tokens.items():
        cards.append(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
            f'border-bottom:1px solid rgba(128,128,128,0.1);font-size:0.85em;">'
            f'<span style="background:#2563eb18;color:#2563eb;padding:1px 8px;border-radius:10px;'
            f'font-weight:600;font-size:0.8em;">{user}</span>'
            f'<code style="opacity:0.55;font-size:0.82em;">{token[:20]}...</code></div>'
        )
    return (
        '<div style="font-size:0.72em;opacity:0.5;margin-bottom:4px;"'
        ' title="Auth tokens obtained by logging in. Use these in the Authorization header.">'
        "AUTHENTICATED USERS</div>"
        + "".join(cards)
    )


def format_resources(ids):
    if not ids:
        return (
            '<div style="opacity:0.5;font-size:0.85em;">'
            "No resources created. Use POST endpoints to create tasks or users "
            "and track their IDs here.</div>"
        )
    sections = []
    type_colors = {"tasks": "#d97706", "users": "#2563eb"}
    for rtype, id_list in ids.items():
        color = type_colors.get(rtype, "#6b7280")
        ids_str = ", ".join(str(i) for i in id_list) if isinstance(id_list, list) else str(id_list)
        sections.append(
            f'<div style="padding:4px 0;border-bottom:1px solid rgba(128,128,128,0.1);font-size:0.85em;">'
            f'<span style="background:{color}18;color:{color};padding:1px 8px;border-radius:10px;'
            f'font-weight:600;font-size:0.8em;text-transform:uppercase;">{rtype}</span>'
            f'<span style="margin-left:8px;opacity:0.7;">IDs: {ids_str}</span></div>'
        )
    return (
        '<div style="font-size:0.72em;opacity:0.5;margin-bottom:4px;"'
        ' title="Resources created during this episode. Use these IDs in GET/PUT/DELETE requests.">'
        "CREATED RESOURCES</div>"
        + "".join(sections)
    )


def format_endpoints():
    lines = []
    for ep in API_SPEC:
        lines.append(f"**{ep['method']}** `{ep['path']}` — {ep.get('summary', '')}")
    return "\n\n".join(lines)


# =====================================================================
# UI
# =====================================================================

def build_ui():
    with gr.Blocks(title="API Testing Environment") as demo:
        session = gr.State(value=new_session())

        gr.Markdown(
            "# API Testing Environment\n"
            "This is a reinforcement learning playground for training AI agents to test REST APIs. "
            "A simulated API server with **intentionally hidden bugs** is provided — the agent "
            "(or you, manually) sends HTTP requests and earns **rewards** for finding bugs, "
            "covering new endpoints, and sending valid requests. "
            "Use **Manual Testing** to craft requests yourself, or run a **Baseline Agent** to "
            "watch an automated strategy in action. The goal: maximize your score by discovering "
            "all bugs within the step limit."
        )

        with gr.Row():
            # ── Left Panel ──
            with gr.Column(scale=1):
                gr.Markdown("### Environment Control")
                task_dropdown = gr.Dropdown(choices=list(TASKS.keys()), value="basic_validation", label="Select Task")
                reset_btn = gr.Button("Reset Environment", variant="primary", size="lg")
                gr.Markdown(
                    '<span style="font-size:0.8em;opacity:0.55;">'
                    "Switch task or click Reset to start a fresh episode. "
                    "Resets all scores, bugs, and step count.</span>"
                )
                status_box = gr.Markdown("Initializing...")

                gr.Markdown("---")
                gr.Markdown("### Scoreboard")
                gr.Markdown(
                    '<span style="font-size:0.78em;opacity:0.55;">'
                    "Tracks your testing progress. Steps are API calls you've made; "
                    "bugs are issues discovered in the API; reward measures how well "
                    "the agent is testing.</span>"
                )
                with gr.Row():
                    step_display = gr.Markdown("0 / 25", label="Steps")
                    bug_display = gr.Markdown("0 / 3", label="Bugs")
                reward_display = gr.Markdown(format_reward_display(0, 0, {}), label="Reward")
                coverage_display = gr.Markdown("No data", label="Coverage")

                gr.Markdown("---")
                gr.Markdown("### Session Context")
                gr.Markdown(
                    '<span style="font-size:0.78em;opacity:0.55;">'
                    "Tokens and resources gathered during this episode. "
                    "Use tokens to test auth-protected endpoints and resource IDs for "
                    "GET/PUT/DELETE requests.</span>"
                )
                auth_display = gr.Markdown(format_auth_tokens({}))
                resource_display = gr.Markdown(format_resources({}))

                gr.Markdown("---")
                with gr.Accordion("API Specification", open=False):
                    gr.Markdown(format_endpoints())

            # ── Center Panel ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Manual Testing"):
                        gr.Markdown("### Craft Your Request")
                        with gr.Row():
                            method_input = gr.Dropdown(
                                choices=["GET", "POST", "PUT", "DELETE", "PATCH"],
                                value="GET", label="Method", scale=1,
                            )
                            endpoint_input = gr.Textbox(value="/tasks", label="Endpoint", placeholder="/tasks, /users/1, /auth/login", scale=3)
                            expected_input = gr.Textbox(value="200", label="Expected Status", placeholder="200", scale=1)

                        with gr.Row():
                            headers_input = gr.Textbox(value="{}", label="Headers (JSON)", placeholder='{"Authorization": "Bearer ..."}', lines=1)
                            params_input = gr.Textbox(value="{}", label="Query Params (JSON)", placeholder='{"page": 1, "limit": 10}', lines=1)

                        body_input = gr.Textbox(value="", label="Request Body (JSON)", placeholder='{"title": "My Task", "description": "..."}', lines=3)

                        send_btn = gr.Button("Send Request", variant="primary", size="lg")

                        gr.Markdown("### Quick Actions")
                        quick_actions = gr.Dropdown(
                            choices=[
                                "GET /tasks", "GET /users", "GET /tasks/1",
                                "GET /tasks/999999 (bug hunt)", "POST create task",
                                "POST missing title (bug hunt)", "Login as alice", "Login as bob",
                                "Login empty pwd (bug hunt)", "Negative page (bug hunt)",
                                "Huge limit (bug hunt)", "Invalid email PUT (bug hunt)",
                                "DELETE non-existent (bug hunt)", "Create user invalid email (bug)",
                                "SQL injection test", "Long title crash (bug hunt)",
                            ],
                            label="Quick Actions", value=None,
                        )
                        quick_btn = gr.Button("Load Quick Action", variant="secondary")

                    with gr.Tab("Run Baseline Agent"):
                        gr.Markdown("### Automated Agents\nWatch a baseline agent test the API step by step.")
                        agent_dropdown = gr.Dropdown(choices=["random", "sequential", "smart"], value="smart", label="Agent Type")
                        run_agent_btn = gr.Button("Run Agent", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Response")
                response_display = gr.Markdown("")

                gr.Markdown("### Feedback")
                feedback_display = gr.Markdown("")

            # ── Right Panel ──
            with gr.Column(scale=1):
                gr.Markdown("### Discovered Bugs")
                bug_list_display = gr.Markdown("No bugs found yet.")

                gr.Markdown("---")
                gr.Markdown("### Activity Log")
                log_display = gr.Markdown("No steps yet.")

        # ── Wiring ──
        reset_outputs = [
            session, status_box, feedback_display, response_display,
            reward_display, bug_display, coverage_display, log_display,
            step_display, bug_list_display, auth_display, resource_display,
        ]

        step_outputs = [
            session, feedback_display, response_display, reward_display,
            bug_display, coverage_display, log_display, step_display,
            bug_list_display, auth_display, resource_display,
        ]

        reset_btn.click(fn=reset_env, inputs=[task_dropdown, session], outputs=reset_outputs)

        send_btn.click(
            fn=send_request,
            inputs=[method_input, endpoint_input, headers_input, params_input, body_input, expected_input, session],
            outputs=step_outputs,
        )

        quick_btn.click(
            fn=apply_quick_action, inputs=[quick_actions, session],
            outputs=[method_input, endpoint_input, headers_input, params_input, body_input, expected_input],
        )

        run_agent_btn.click(fn=run_baseline_agent, inputs=[agent_dropdown, session], outputs=step_outputs)

        # Auto-reset on page load so users can start testing immediately
        demo.load(fn=reset_env, inputs=[task_dropdown, session], outputs=reset_outputs)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)
