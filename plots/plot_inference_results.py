"""Visualize inference.py task scores and per-step rewards.

Generates matplotlib and plotly bar charts (PNG + SVG) under plots/.

Two figures are produced:
  1. inference_results_*  — LLM-only view: per-task final score + per-step rewards
  2. baseline_comparison_* — LLM vs random / sequential / smart baselines

LLM data is the inference.py run on 2026-04-08 against
meta-llama/Llama-3.3-70B-Instruct via the HF router. Baseline numbers come
from `python baseline.py --agent all --task all --seed 42` and are converted
to the same normalized score the LLM reports:
    score = 0.7 * (bugs_found / total_bugs) + 0.3 * (coverage_pct / 100)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["basic_validation", "edge_cases", "security_workflows"]
SCORES = [0.647, 0.772, 0.581]
STEPS = [18, 27, 29]
AVG_SCORE = 0.667

# --- Baseline rollout results (seed=42) ---
# Each entry: (bugs_found, total_bugs, coverage_pct, steps)
BASELINE_RAW = {
    "random": {
        "basic_validation":    (1, 3,  40.0, 25),
        "edge_cases":          (2, 9,  50.0, 35),
        "security_workflows":  (3, 13, 50.0, 45),
    },
    "sequential": {
        "basic_validation":    (3, 3,  50.0, 25),
        "edge_cases":          (4, 9,  50.0, 35),
        "security_workflows":  (4, 13, 50.0, 45),
    },
    "smart": {
        "basic_validation":    (3, 3,  50.0, 25),
        "edge_cases":          (9, 9,  50.0, 35),
        "security_workflows":  (12, 13, 50.0, 45),
    },
}


def normalized_score(bugs_found: int, total_bugs: int, coverage_pct: float) -> float:
    """Same formula as inference.compute_task_score — keeps everything in [0, 1]."""
    bug_ratio = (bugs_found / total_bugs) if total_bugs > 0 else 0.0
    cov_ratio = max(0.0, min(1.0, coverage_pct / 100.0))
    return max(0.0, min(1.0, 0.70 * bug_ratio + 0.30 * cov_ratio))


# Pre-compute normalized scores for each baseline + LLM
AGENT_LABELS = ["random", "sequential", "smart", "llm (Llama-3.3-70B)"]
LLM_SCORES_BY_TASK = dict(zip(TASKS, SCORES))

AGENT_SCORES: dict[str, list[float]] = {}
for agent_name, per_task in BASELINE_RAW.items():
    AGENT_SCORES[agent_name] = [
        normalized_score(*per_task[t][:3]) for t in TASKS
    ]
AGENT_SCORES["llm (Llama-3.3-70B)"] = [LLM_SCORES_BY_TASK[t] for t in TASKS]

AGENT_AVG = {a: sum(s) / len(s) for a, s in AGENT_SCORES.items()}

AGENT_COLORS = {
    "random":               "#9E9E9E",
    "sequential":           "#F4A261",
    "smart":                "#2A9D8F",
    "llm (Llama-3.3-70B)":  "#6A4C93",
}

PER_STEP_REWARDS = {
    "basic_validation": [
        0.33, 0.23, 0.28, 0.18, 0.13, 0.28, 0.25, 0.28, 0.28,
        0.18, 0.23, 0.33, 0.13, 0.03, 0.03, 0.13, -0.05, 0.03,
    ],
    "edge_cases": [
        0.33, 0.28, 0.28, 0.08, 0.18, 0.25, 0.48, 0.28, 0.33,
        0.08, 0.33, 0.03, 0.23, 0.33, 0.28, 0.18, 0.03, 0.08,
        0.08, 0.13, 0.13, 0.08, 0.13, 0.00, 0.33, 0.08, 0.00,
    ],
    "security_workflows": [
        0.33, 0.28, 0.28, 0.08, 0.03, 0.18, 0.48, 0.23, 0.28,
        0.25, 0.33, 0.33, 0.23, 0.33, 0.28, 0.08, 0.18, 0.03,
        0.13, 0.13, 0.13, 0.08, 0.00, 0.13, 0.00, -0.05, -0.05,
        0.03, -0.05,
    ],
}

COLORS = {
    "basic_validation": "#4C72B0",
    "edge_cases": "#55A868",
    "security_workflows": "#C44E52",
}


# ---------- matplotlib ----------
def plot_matplotlib() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # 1. Final scores per task
    ax = axes[0]
    bar_colors = [COLORS[t] for t in TASKS]
    bars = ax.bar(TASKS, SCORES, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.axhline(AVG_SCORE, color="#333", linestyle="--", linewidth=1.2,
               label=f"avg = {AVG_SCORE:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Final score")
    ax.set_title("Inference final score by task")
    ax.legend(loc="upper right", frameon=False)
    for bar, score, steps in zip(bars, SCORES, STEPS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.3f}\n({steps} steps)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.tick_params(axis="x", rotation=15)

    # 2. Per-step rewards (grouped over step index)
    ax = axes[1]
    max_len = max(len(v) for v in PER_STEP_REWARDS.values())
    width = 0.27
    x_base = list(range(1, max_len + 1))
    for i, task in enumerate(TASKS):
        rewards = PER_STEP_REWARDS[task]
        xs = [x + (i - 1) * width for x in range(1, len(rewards) + 1)]
        ax.bar(xs, rewards, width=width, color=COLORS[task],
               label=task, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="#666", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Per-step reward by task")
    ax.set_xticks(x_base[::2])
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "inference.py — meta-llama/Llama-3.3-70B-Instruct (avg score 0.667)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = OUT_DIR / "inference_results_matplotlib.png"
    svg_path = OUT_DIR / "inference_results_matplotlib.svg"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[matplotlib] wrote {png_path}")
    print(f"[matplotlib] wrote {svg_path}")


# ---------- plotly ----------
def plot_plotly() -> None:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Final score by task", "Per-step reward by task"),
    )

    # 1. Final scores
    fig.add_trace(
        go.Bar(
            x=TASKS,
            y=SCORES,
            marker_color=[COLORS[t] for t in TASKS],
            text=[f"{s:.3f}<br>({n} steps)" for s, n in zip(SCORES, STEPS)],
            textposition="outside",
            name="Final score",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=AVG_SCORE, line_dash="dash", line_color="#333",
        annotation_text=f"avg = {AVG_SCORE:.3f}",
        annotation_position="top left",
        row=1, col=1,
    )

    # 2. Per-step rewards (grouped bars)
    for task in TASKS:
        rewards = PER_STEP_REWARDS[task]
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(rewards) + 1)),
                y=rewards,
                name=task,
                marker_color=COLORS[task],
            ),
            row=1, col=2,
        )

    fig.update_yaxes(title_text="Final score", range=[0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_layout(
        title=dict(
            text="inference.py — meta-llama/Llama-3.3-70B-Instruct (avg score 0.667)",
            x=0.5, xanchor="center",
        ),
        barmode="group",
        bargap=0.2,
        template="plotly_white",
        width=1300,
        height=560,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=80, b=80, l=60, r=30),
    )

    png_path = OUT_DIR / "inference_results_plotly.png"
    svg_path = OUT_DIR / "inference_results_plotly.svg"
    fig.write_image(png_path, scale=2)
    fig.write_image(svg_path)
    print(f"[plotly] wrote {png_path}")
    print(f"[plotly] wrote {svg_path}")


# ---------- baseline comparison: matplotlib ----------
def plot_baselines_matplotlib() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # 1. Grouped bars per task
    ax = axes[0]
    n_agents = len(AGENT_LABELS)
    width = 0.2
    x = list(range(len(TASKS)))
    for i, agent in enumerate(AGENT_LABELS):
        offset = (i - (n_agents - 1) / 2) * width
        xs = [xi + offset for xi in x]
        bars = ax.bar(
            xs, AGENT_SCORES[agent], width=width,
            color=AGENT_COLORS[agent], label=agent,
            edgecolor="black", linewidth=0.4,
        )
        for bar, val in zip(bars, AGENT_SCORES[agent]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Normalized score")
    ax.set_title("Per-task score: baselines vs LLM")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # 2. Average score across all 3 tasks
    ax = axes[1]
    avgs = [AGENT_AVG[a] for a in AGENT_LABELS]
    colors = [AGENT_COLORS[a] for a in AGENT_LABELS]
    bars = ax.bar(AGENT_LABELS, avgs, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, avgs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean score (3 tasks)")
    ax.set_title("Average score across all tasks")
    ax.tick_params(axis="x", rotation=12)

    fig.suptitle(
        "Baseline agents vs LLM — score = 0.7·bug_ratio + 0.3·coverage_ratio",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png_path = OUT_DIR / "baseline_comparison_matplotlib.png"
    svg_path = OUT_DIR / "baseline_comparison_matplotlib.svg"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[matplotlib] wrote {png_path}")
    print(f"[matplotlib] wrote {svg_path}")


# ---------- baseline comparison: plotly ----------
def plot_baselines_plotly() -> None:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("Per-task score: baselines vs LLM", "Average score across all tasks"),
    )

    # 1. Grouped bars per task
    for agent in AGENT_LABELS:
        fig.add_trace(
            go.Bar(
                x=TASKS,
                y=AGENT_SCORES[agent],
                name=agent,
                marker_color=AGENT_COLORS[agent],
                text=[f"{v:.2f}" for v in AGENT_SCORES[agent]],
                textposition="outside",
                legendgroup=agent,
            ),
            row=1, col=1,
        )

    # 2. Average score
    avgs = [AGENT_AVG[a] for a in AGENT_LABELS]
    fig.add_trace(
        go.Bar(
            x=AGENT_LABELS,
            y=avgs,
            marker_color=[AGENT_COLORS[a] for a in AGENT_LABELS],
            text=[f"{v:.3f}" for v in avgs],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_yaxes(title_text="Normalized score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Mean score (3 tasks)", range=[0, 1.05], row=1, col=2)
    fig.update_layout(
        title=dict(
            text="Baseline agents vs LLM — score = 0.7·bug_ratio + 0.3·coverage_ratio",
            x=0.5, xanchor="center",
        ),
        barmode="group",
        bargap=0.18,
        template="plotly_white",
        width=1400,
        height=580,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=80, b=90, l=60, r=30),
    )

    png_path = OUT_DIR / "baseline_comparison_plotly.png"
    svg_path = OUT_DIR / "baseline_comparison_plotly.svg"
    fig.write_image(png_path, scale=2)
    fig.write_image(svg_path)
    print(f"[plotly] wrote {png_path}")
    print(f"[plotly] wrote {svg_path}")


if __name__ == "__main__":
    plot_matplotlib()
    plot_plotly()
    plot_baselines_matplotlib()
    plot_baselines_plotly()
