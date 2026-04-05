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
        "0.0",
        f"0 / {t['total_bugs']}",
        format_coverage(obs.coverage_summary),
        "",
        f"0 / {t['max_steps']}",
        "No bugs found yet.",
        "No tokens acquired yet.",
        "No resources created yet.",
        format_endpoints(),
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
    reward_detail = (
        f"**Step reward:** {reward:.4f}\n"
        f"**Cumulative:** {state.total_reward:.4f}\n\n"
        f"Coverage: {breakdown.get('coverage', 0):.3f} | "
        f"Validity: {breakdown.get('validity', 0):.3f} | "
        f"Bug: {breakdown.get('bug_discovery', 0):.3f} | "
        f"Explore: {breakdown.get('exploration', 0):.3f} | "
        f"Penalty: {breakdown.get('penalty', 0):.3f}"
    )

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
        reward_detail = (
            f"**Step reward:** {reward:.4f}\n"
            f"**Cumulative:** {state.total_reward:.4f}\n\n"
            f"Coverage: {breakdown.get('coverage', 0):.3f} | "
            f"Validity: {breakdown.get('validity', 0):.3f} | "
            f"Bug: {breakdown.get('bug_discovery', 0):.3f} | "
            f"Explore: {breakdown.get('exploration', 0):.3f} | "
            f"Penalty: {breakdown.get('penalty', 0):.3f}"
        )

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

def format_coverage(summary):
    if not summary:
        return "No data"
    pct = summary.get("coverage_pct", 0)
    tested = summary.get("endpoints_tested", 0)
    total = summary.get("total_endpoints", 0)
    pairs = summary.get("method_endpoint_pairs", 0)
    codes = summary.get("status_codes_seen", [])
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    return (
        f"**{pct:.1f}%** [{bar}]\n\n"
        f"Endpoints: {tested}/{total} | Method+Endpoint pairs: {pairs}\n"
        f"Status codes seen: {', '.join(str(c) for c in codes)}"
    )


def format_log(log):
    if not log:
        return "No steps yet."
    lines = []
    for entry in log[-15:]:
        bug_marker = " 🐛" if entry.get("reward", 0) > 0.2 else ""
        lines.append(
            f"`{entry['step']:2d}` **{entry['method']}** {entry['endpoint']} "
            f"-> {entry['status']} (r={entry['reward']:+.3f}){bug_marker}"
        )
    if len(log) > 15:
        lines.insert(0, f"*... {len(log) - 15} earlier steps omitted*")
    return "\n".join(lines)


def format_bug_list(bug_ids):
    if not bug_ids:
        return "No bugs found yet."
    from server.bug_detector import BugDetector
    detector = BugDetector("security_workflows")
    lines = []
    for bid in sorted(bug_ids):
        bug = detector.bugs.get(bid)
        if bug:
            icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(bug.severity, "⚪")
            lines.append(f"{icon} **{bid}** ({bug.severity}): {bug.description}")
    return "\n".join(lines)


def format_auth_tokens(tokens):
    if not tokens:
        return "No tokens acquired yet."
    return "\n".join(f"**{user}**: `{token[:16]}...`" for user, token in tokens.items())


def format_resources(ids):
    if not ids:
        return "No resources created yet."
    return "\n".join(f"**{rtype}**: {id_list}" for rtype, id_list in ids.items())


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

        gr.Markdown("# API Testing Environment\nTrain an AI agent to test REST APIs — find bugs, earn rewards, maximize coverage.")

        with gr.Row():
            # ── Left Panel ──
            with gr.Column(scale=1):
                gr.Markdown("### Environment Control")
                task_dropdown = gr.Dropdown(choices=list(TASKS.keys()), value="basic_validation", label="Select Task")
                reset_btn = gr.Button("Reset Environment", variant="primary", size="lg")
                status_box = gr.Markdown("Click **Reset** to start.")

                gr.Markdown("---")
                gr.Markdown("### Scoreboard")
                with gr.Row():
                    step_display = gr.Markdown("0 / 25", label="Steps")
                    bug_display = gr.Markdown("0 / 3", label="Bugs")
                reward_display = gr.Markdown("0.0", label="Reward")
                coverage_display = gr.Markdown("No data", label="Coverage")

                gr.Markdown("---")
                gr.Markdown("### Discovered Bugs")
                bug_list_display = gr.Markdown("No bugs found yet.")

                gr.Markdown("---")
                gr.Markdown("### Session Context")
                auth_display = gr.Markdown("No tokens acquired yet.")
                resource_display = gr.Markdown("No resources created yet.")

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

                    with gr.Tab("API Specification"):
                        endpoint_spec = gr.Markdown(format_endpoints())

                gr.Markdown("---")
                gr.Markdown("### Response")
                response_display = gr.Markdown("")

                gr.Markdown("### Feedback")
                feedback_display = gr.Markdown("")

            # ── Right Panel ──
            with gr.Column(scale=1):
                gr.Markdown("### Activity Log")
                log_display = gr.Markdown("No steps yet.")

        # ── Wiring ──
        reset_outputs = [
            session, status_box, feedback_display, response_display,
            reward_display, bug_display, coverage_display, log_display,
            step_display, bug_list_display, auth_display, resource_display,
            endpoint_spec,
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

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)
