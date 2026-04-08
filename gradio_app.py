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


# =====================================================================
# Editorial blog-style documentation rendered below the playground.
# Uses an inline <style> block so it works without external CSS files,
# and adapts to both light and dark Gradio themes via CSS variables.
# Aesthetic: terminal-framed editorial zine — EB Garamond + JetBrains
# Mono, parchment cream over ink black, amber + forensic-green accents.
# =====================================================================

BLOG_HTML = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:opsz,wght,SOFT@9..144,200..400,30..100&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ─────────────────────────────────────────────────────────────────
   ElevenLabs-inspired editorial section.
   - Light Fraunces (used as Waldenburg-substitute) display
   - Warm stone surfaces with sub-0.1 shadow stacks
   - Pill buttons, generous whitespace
   - Adapts to both light and dark Gradio themes
   ───────────────────────────────────────────────────────────────── */

/* ─────────────────────────────────────────────────────────────────
   Theme variables.
   Strategy: ALWAYS default to LIGHT. Only flip to dark when an
   explicit `.dark` class is present on the body (Gradio sets this
   reliably based on ?__theme=dark URL param or system preference).
   We do NOT use `prefers-color-scheme` here because Gradio already
   reads it once and forwards the result via the body class — using
   the media query as well causes double-flipping in light mode when
   the OS is dark.
   ───────────────────────────────────────────────────────────────── */

.eleven {
  /* Default = LIGHT theme (always — overridden by .dark below) */
  --bg:           #ffffff;
  --bg-soft:      #f5f5f5;
  --stone:        rgba(245, 242, 239, 0.85);
  --stone-solid:  #f5f2ef;
  --fg:           #000000;
  --fg-2:         #4e4e4e;
  --fg-3:         #777169;
  --hairline:     rgba(0, 0, 0, 0.05);
  --border:       #e5e5e5;
  --warm-shadow:  rgba(78, 50, 23, 0.04);
  --inset-edge:   rgba(0, 0, 0, 0.075);
  --outline-ring: rgba(0, 0, 0, 0.06);
  --soft-elev:    rgba(0, 0, 0, 0.04);

  --accent-mint:  #2db97a;
  --accent-amber: #d97757;
  --accent-coral: #c4513a;
  --accent-blue:  #4a6fa5;

  --display: 'Fraunces', 'Iowan Old Style', Georgia, serif;
  --body:    'Inter', system-ui, sans-serif;
  --mono:    'IBM Plex Mono', 'SF Mono', Menlo, monospace;

  display: block;
  width: 100%;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--body);
  margin-top: 48px;
  border-top: 1px solid var(--hairline);
}

/* DARK theme — triggered by Gradio's .dark class on body / container.
   Gradio handles `?__theme=dark` and `prefers-color-scheme: dark` for
   us by setting this class, so a single rule covers all cases. */
.dark .eleven,
body.dark .eleven,
.gradio-container.dark .eleven,
html.dark .eleven {
  --bg:           #0a0a0a;
  --bg-soft:      #131313;
  --stone:        rgba(28, 24, 20, 0.9);
  --stone-solid:  #1c1814;
  --fg:           #f5f2ef;
  --fg-2:         #b8b3ad;
  --fg-3:         #8a847d;
  --hairline:     rgba(245, 242, 239, 0.06);
  --border:       rgba(245, 242, 239, 0.10);
  --warm-shadow:  rgba(0, 0, 0, 0.4);
  --inset-edge:   rgba(245, 242, 239, 0.08);
  --outline-ring: rgba(245, 242, 239, 0.08);
  --soft-elev:    rgba(0, 0, 0, 0.35);
}

/* ── Section container ── */
.eleven-section {
  padding: 96px 64px;
  max-width: 1280px;
  margin: 0 auto;
}
.eleven-section.alt {
  background: var(--bg-soft);
  max-width: none;
  padding: 96px 64px;
}
.eleven-section.alt > .eleven-section-inner {
  max-width: 1280px;
  margin: 0 auto;
}
@media (max-width: 900px) {
  .eleven-section,
  .eleven-section.alt { padding: 64px 24px; }
}

/* ── Eyebrow label ── */
.eleven-eyebrow {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--fg-3);
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 28px;
}
.eleven-eyebrow::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent-mint);
  box-shadow: 0 0 12px var(--accent-mint);
}

/* ── Hero masthead ── */
.eleven-hero {
  text-align: center;
  padding: 120px 24px 64px 24px;
  max-width: 960px;
  margin: 0 auto;
}
.eleven-hero h1 {
  font-family: var(--display);
  font-size: clamp(48px, 7vw, 96px);
  font-weight: 300;
  font-variation-settings: "SOFT" 50, "opsz" 144;
  line-height: 1.04;
  letter-spacing: -0.025em;
  color: var(--fg);
  margin: 0 0 24px 0;
}
.eleven-hero h1 em {
  font-style: italic;
  font-weight: 300;
  color: var(--fg);
  font-variation-settings: "SOFT" 100, "opsz" 144;
}
.eleven-hero .eleven-deck {
  font-family: var(--body);
  font-size: 20px;
  font-weight: 400;
  line-height: 1.55;
  letter-spacing: 0.1px;
  color: var(--fg-2);
  margin: 0 auto 40px auto;
  max-width: 640px;
}

/* ── Pill buttons (hero CTA row) ── */
.eleven-pills {
  display: inline-flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}
.eleven-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 22px;
  border-radius: 9999px;
  text-decoration: none;
  font-family: var(--body);
  font-size: 15px;
  font-weight: 500;
  letter-spacing: 0.1px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.eleven-pill.dark {
  background: var(--fg);
  color: var(--bg);
}
.eleven-pill.dark:hover {
  transform: translateY(-1px);
}
.eleven-pill.warm {
  background: var(--stone);
  color: var(--fg);
  box-shadow:
    var(--inset-edge) 0 0 0 0.5px inset,
    var(--warm-shadow) 0 6px 16px;
}
.eleven-pill.warm:hover {
  transform: translateY(-1px);
  box-shadow:
    var(--inset-edge) 0 0 0 0.5px inset,
    var(--warm-shadow) 0 8px 20px;
}
.eleven-pill .arrow {
  font-family: var(--body);
  font-weight: 400;
  font-size: 16px;
}

/* ── Two-column section: label + content ── */
.eleven-row {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 80px;
  align-items: start;
}
@media (max-width: 900px) {
  .eleven-row { grid-template-columns: 1fr; gap: 24px; }
}

.eleven-label {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--fg-3);
  position: sticky;
  top: 32px;
}
.eleven-label .num {
  display: block;
  font-family: var(--display);
  font-size: 56px;
  font-weight: 300;
  font-variation-settings: "SOFT" 100, "opsz" 144;
  letter-spacing: -0.04em;
  color: var(--fg);
  line-height: 1;
  margin-bottom: 12px;
}

.eleven-content h2 {
  font-family: var(--display);
  font-size: clamp(36px, 4.5vw, 56px);
  font-weight: 300;
  font-variation-settings: "SOFT" 50, "opsz" 144;
  line-height: 1.08;
  letter-spacing: -0.018em;
  color: var(--fg);
  margin: 0 0 28px 0;
}
.eleven-content h2 em {
  font-style: italic;
  font-weight: 300;
  font-variation-settings: "SOFT" 100, "opsz" 144;
  color: var(--fg);
}

.eleven-content p {
  font-family: var(--body);
  font-size: 18px;
  font-weight: 400;
  line-height: 1.62;
  letter-spacing: 0.18px;
  color: var(--fg-2);
  margin: 0 0 18px 0;
  max-width: 64ch;
}
.eleven-content p strong {
  color: var(--fg);
  font-weight: 600;
}

/* Inline tag chip */
.eleven-chip {
  display: inline-flex;
  align-items: center;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  background: var(--bg);
  color: var(--fg);
  padding: 2px 10px;
  border-radius: 9999px;
  margin: 0 2px;
  box-shadow:
    var(--inset-edge) 0 0 0 0.5px inset,
    var(--soft-elev) 0 1px 2px;
  vertical-align: 2px;
}

/* ── Pull quote (subtle, ElevenLabs-style — not loud) ── */
.eleven-quote {
  font-family: var(--display);
  font-size: clamp(28px, 3.5vw, 42px);
  font-weight: 300;
  font-variation-settings: "SOFT" 80, "opsz" 144;
  line-height: 1.2;
  letter-spacing: -0.012em;
  color: var(--fg);
  margin: 32px 0 16px 0;
  max-width: 28ch;
}
.eleven-quote em {
  font-style: italic;
  font-weight: 300;
  color: var(--fg-3);
  font-variation-settings: "SOFT" 100, "opsz" 144;
}

/* ── Reward cards (5 elegant cards instead of bar chart) ── */
.eleven-rewards {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin: 32px 0 24px 0;
}
@media (max-width: 700px) {
  .eleven-rewards { grid-template-columns: 1fr; }
}
.eleven-reward {
  background: var(--bg);
  border-radius: 16px;
  padding: 24px 26px;
  box-shadow:
    var(--inset-edge) 0 0 0 0.5px inset,
    var(--outline-ring) 0 0 0 1px,
    var(--soft-elev) 0 4px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.eleven-reward.featured {
  grid-column: 1 / -1;
  background: var(--stone);
  box-shadow:
    var(--inset-edge) 0 0 0 0.5px inset,
    var(--warm-shadow) 0 6px 16px;
}
.eleven-reward-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 4px;
}
.eleven-reward-name {
  font-family: var(--display);
  font-size: 22px;
  font-weight: 300;
  font-variation-settings: "SOFT" 80, "opsz" 144;
  letter-spacing: -0.005em;
  color: var(--fg);
}
.eleven-reward-val {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 600;
  color: var(--accent-mint);
  white-space: nowrap;
}
.eleven-reward-val.neg {
  color: var(--accent-coral);
}
.eleven-reward p {
  font-family: var(--body);
  font-size: 15px;
  line-height: 1.55;
  letter-spacing: 0.14px;
  color: var(--fg-2);
  margin: 0;
  max-width: none;
}

.eleven-r-foot {
  font-family: var(--display);
  font-size: 20px;
  font-style: italic;
  font-weight: 300;
  font-variation-settings: "SOFT" 100, "opsz" 144;
  line-height: 1.5;
  color: var(--fg-2);
  margin: 32px 0 0 0;
  max-width: 64ch;
  padding-left: 18px;
  border-left: 1px solid var(--border);
}

/* ── Steps as an editorial table ── */
.eleven-steps {
  margin: 40px 0 0 0;
  border-top: 1px solid var(--border);
  counter-reset: eleven-step;
}
.eleven-step {
  display: grid;
  grid-template-columns: 72px minmax(180px, 1.1fr) minmax(0, 2.4fr);
  column-gap: 32px;
  row-gap: 8px;
  align-items: start;
  padding: 32px 8px;
  border-bottom: 1px solid var(--border);
  counter-increment: eleven-step;
  transition: background 0.18s ease;
}
.eleven-step:hover {
  background: var(--bg-soft);
}
@media (max-width: 900px) {
  .eleven-step {
    grid-template-columns: 56px 1fr;
    column-gap: 16px;
    padding: 24px 4px;
  }
  .eleven-step-body { grid-column: 2 / -1; }
}
.eleven-step-num {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.18em;
  color: var(--fg-3);
  padding-top: 10px;
  position: relative;
}
.eleven-step-num::before {
  content: counter(eleven-step, decimal-leading-zero);
}
.eleven-step-num::after {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 18px;
  height: 2px;
  background: var(--accent-mint);
}
.eleven-step-title {
  font-family: var(--display);
  font-size: 24px;
  font-weight: 400;
  font-variation-settings: "SOFT" 80, "opsz" 144;
  letter-spacing: -0.012em;
  line-height: 1.2;
  color: var(--fg);
}
.eleven-step-body {
  font-family: var(--body);
  font-size: 15px;
  line-height: 1.65;
  color: var(--fg-2);
}
.eleven-step-body p {
  margin: 0;
}
.eleven-step-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 14px;
}
.eleven-step-chip {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
  padding: 5px 11px;
  border-radius: 999px;
  background: var(--bg-soft);
  color: var(--fg);
  border: 1px solid var(--border);
  white-space: nowrap;
}
.dark .eleven-step-chip {
  background: rgba(245, 242, 239, 0.04);
}
.eleven-step-chip.accent {
  background: rgba(45, 185, 122, 0.10);
  color: var(--accent-mint);
  border-color: rgba(45, 185, 122, 0.35);
}

/* ── Stack tiles (4 cards) ── */
.eleven-stack {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin: 32px 0 0 0;
}
@media (max-width: 700px) {
  .eleven-stack { grid-template-columns: 1fr; }
}
.eleven-tile {
  background: var(--bg);
  border-radius: 20px;
  padding: 32px 32px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow:
    var(--inset-edge) 0 0 0 0.5px inset,
    var(--outline-ring) 0 0 0 1px,
    var(--soft-elev) 0 4px 12px;
}
.eleven-tile-tag {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--fg-3);
}
.eleven-tile h3 {
  font-family: var(--display);
  font-size: 28px;
  font-weight: 300;
  font-variation-settings: "SOFT" 80, "opsz" 144;
  letter-spacing: -0.012em;
  line-height: 1.1;
  color: var(--fg);
  margin: 0;
}
.eleven-tile p {
  font-family: var(--body);
  font-size: 15px;
  line-height: 1.55;
  letter-spacing: 0.14px;
  color: var(--fg-2);
  margin: 0;
}
.eleven-tile p code {
  font-family: var(--mono);
  font-size: 12px;
  background: var(--bg-soft);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--fg);
  font-weight: 500;
  box-shadow: var(--inset-edge) 0 0 0 0.5px inset;
}

/* ── Link list (minimal underlined text) ── */
.eleven-links {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin: 32px 0 0 0;
}
.eleven-link {
  font-family: var(--mono);
  font-size: 14px;
  line-height: 1.55;
  color: var(--fg);
  text-decoration: underline;
  text-decoration-color: var(--border);
  text-underline-offset: 4px;
  text-decoration-thickness: 1px;
  word-break: break-all;
  transition: text-decoration-color 0.18s ease, color 0.18s ease;
}
.eleven-link:hover {
  text-decoration-color: var(--accent-mint);
  color: var(--accent-mint);
}
</style>

<div class="eleven">

  <!-- ─── HERO ─── -->
  <div class="eleven-hero">
    <h1>Where agents learn to <em>break APIs.</em></h1>
    <p class="eleven-deck">An OpenEnv reinforcement learning environment for API security testing. A live REST API with thirteen planted vulnerabilities, a verifiable reward function mapped to the OWASP API Security Top 10, and an episode that ends with a structured bug report.</p>
  </div>

  <!-- ─── 01 WHAT IS THIS ─── -->
  <div class="eleven-section">
    <div class="eleven-row">
      <div class="eleven-label"><span class="num">01</span>The premise</div>
      <div class="eleven-content">
        <h2>What <em>this is.</em></h2>
        <p>A Gradio playground for an OpenEnv RL environment that trains AI agents to test REST APIs the way a security engineer would. Behind the UI is a Task Management API with <strong>13 deliberately planted bugs</strong> covering 6 categories from the <strong>OWASP API Security Top 10</strong>.</p>
        <p>The agent connects, sends HTTP requests, earns rewards for finding bugs and covering endpoints, and generates a bug bounty report when the episode ends.</p>
        <div class="eleven-quote">Real API. Real bugs. Real OWASP categories — <em>verifiable end to end.</em></div>
      </div>
    </div>
  </div>

  <!-- ─── 02 WHY BOTHER ─── -->
  <div class="eleven-section alt">
    <div class="eleven-section-inner">
      <div class="eleven-row">
        <div class="eleven-label"><span class="num">02</span>The gap</div>
        <div class="eleven-content">
          <h2>Why <em>bother.</em></h2>
          <p>Every team ships APIs and every API has bugs. The usual tools <span class="eleven-chip">Postman</span> <span class="eleven-chip">Schemathesis</span> <span class="eleven-chip">OWASP&nbsp;ZAP</span> either need humans writing tests by hand or fall back to brute-force fuzzing.</p>
          <p>Recent papers — <em>APIRL</em> at AAAI 2025, <em>ARAT-RL</em> at ASE 2023 — show RL beats both. But there hasn't been a standard RL benchmark for it.</p>
          <div class="eleven-quote">This environment <em>is the benchmark.</em></div>
          <p>The agent doesn't get a written test plan. It reads the API spec, plans a campaign, runs it, and reports what broke. The reward function is verifiable — no LLM judge, no soft heuristics — and every signal maps to a real OWASP category, so episodes can be scored deterministically.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- ─── 03 REWARD ─── -->
  <div class="eleven-section">
    <div class="eleven-row">
      <div class="eleven-label"><span class="num">03</span>How reward works</div>
      <div class="eleven-content">
        <h2>Five signals,<br><em>one episode.</em></h2>
        <p>The reward function is verifiable — no LLM judge, no soft heuristics. Each step accumulates from five components and the task grader caps the episode with a terminal score in <span class="eleven-chip">[0, 1]</span>.</p>

        <div class="eleven-rewards">
          <div class="eleven-reward featured">
            <div class="eleven-reward-head">
              <div class="eleven-reward-name">Bug discovery</div>
              <div class="eleven-reward-val">+0.10 / +0.15 / +0.25</div>
            </div>
            <p>Finding a planted bug, scaled by severity. Easy bugs (status codes, missing fields) are worth 0.10. Medium (validation, auth) gets 0.15. Hard (BOLA, injection, broken auth chains) gets 0.25.</p>
          </div>
          <div class="eleven-reward">
            <div class="eleven-reward-head">
              <div class="eleven-reward-name">Coverage</div>
              <div class="eleven-reward-val">+0.20</div>
            </div>
            <p>Hitting endpoints, methods, and status codes the agent hasn't tried yet.</p>
          </div>
          <div class="eleven-reward">
            <div class="eleven-reward-head">
              <div class="eleven-reward-name">Validity</div>
              <div class="eleven-reward-val">+0.18</div>
            </div>
            <p>Well-formed requests, plus chaining IDs from previous responses.</p>
          </div>
          <div class="eleven-reward">
            <div class="eleven-reward-head">
              <div class="eleven-reward-name">Exploration</div>
              <div class="eleven-reward-val">+0.05</div>
            </div>
            <p>Trying genuinely novel action patterns the agent hasn't tried before.</p>
          </div>
          <div class="eleven-reward">
            <div class="eleven-reward-head">
              <div class="eleven-reward-name">Penalty</div>
              <div class="eleven-reward-val neg">−0.08</div>
            </div>
            <p>Repeating the same exact request twice — anti-spam, anti-loop.</p>
          </div>
        </div>

        <p class="eleven-r-foot">When the episode ends, the task grader adds a terminal score based on its own criteria — CRUD coverage, dependency chaining, security probing, that kind of thing.</p>
      </div>
    </div>
  </div>

  <!-- ─── 04 HOW TO USE ─── -->
  <div class="eleven-section alt">
    <div class="eleven-section-inner">
      <div class="eleven-row">
        <div class="eleven-label"><span class="num">04</span>How to use this</div>
        <div class="eleven-content">
          <h2>Five steps<br><em>to a verdict.</em></h2>

          <div class="eleven-steps">

            <div class="eleven-step">
              <div class="eleven-step-num"></div>
              <div class="eleven-step-title">Pick a task</div>
              <div class="eleven-step-body">
                <p>Three difficulty tiers in the dropdown on the left, from a CRUD smoke-test to a full BOLA + injection chain.</p>
                <div class="eleven-step-chips">
                  <span class="eleven-step-chip accent">basic_validation</span>
                  <span class="eleven-step-chip accent">edge_cases</span>
                  <span class="eleven-step-chip accent">security_workflows</span>
                </div>
              </div>
            </div>

            <div class="eleven-step">
              <div class="eleven-step-num"></div>
              <div class="eleven-step-title">Reset the environment</div>
              <div class="eleven-step-body">
                <p>Every reset spins up a fresh database with new users, new tasks, and randomized ownership, so the agent can't memorize answers between episodes.</p>
              </div>
            </div>

            <div class="eleven-step">
              <div class="eleven-step-num"></div>
              <div class="eleven-step-title">Run a baseline</div>
              <div class="eleven-step-body">
                <p>The Run Baseline Agent tab is open by default. Pick a strategy and watch it test the API step by step.</p>
                <div class="eleven-step-chips">
                  <span class="eleven-step-chip">random</span>
                  <span class="eleven-step-chip">sequential</span>
                  <span class="eleven-step-chip">smart</span>
                </div>
              </div>
            </div>

            <div class="eleven-step">
              <div class="eleven-step-num"></div>
              <div class="eleven-step-title">Or test manually</div>
              <div class="eleven-step-body">
                <p>Switch to Manual Testing. Quick Actions give one-click bug hunts, or craft your own request from scratch — method, endpoint, headers, body, expected status.</p>
              </div>
            </div>

            <div class="eleven-step">
              <div class="eleven-step-num"></div>
              <div class="eleven-step-title">Watch the panel</div>
              <div class="eleven-step-body">
                <p>Discovered Bugs and the Activity Log update live as the agent works. When the episode ends, expand the Bug Report (OWASP) drawer for the full structured findings, severities, and fix recommendations.</p>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ─── 05 STACK ─── -->
  <div class="eleven-section">
    <div class="eleven-row">
      <div class="eleven-label"><span class="num">05</span>Under the hood</div>
      <div class="eleven-content">
        <h2>Three <em>layers.</em></h2>
        <p>Self-contained, reproducible, and runs on a free-tier HuggingFace Space.</p>

        <div class="eleven-stack">
          <div class="eleven-tile">
            <span class="eleven-tile-tag">L1 · ENVIRONMENT</span>
            <h3>FastAPI + SQLite</h3>
            <p>A buggy Task Management API wrapped in OpenEnv's <code>step()</code> / <code>reset()</code> / <code>state()</code> contract. Runs in-process or as a Docker image, with seed-randomized data on every reset so episodes can't be memorized.</p>
          </div>
          <div class="eleven-tile">
            <span class="eleven-tile-tag">L2 · INFERENCE</span>
            <h3>OpenAI-compatible client</h3>
            <p><code>inference.py</code> talks to any HuggingFace-hosted model through the OpenAI SDK and structured JSON output. Plug in any model that follows the protocol — no environment-specific glue.</p>
          </div>
          <div class="eleven-tile">
            <span class="eleven-tile-tag">L3 · DEPLOY</span>
            <h3>Docker + HF Spaces</h3>
            <p>Containerized on top of the official <code>openenv-base</code> image and deployed as a public HuggingFace Space, so judges can hit it with a single HTTP call.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ─── 06 LINKS ─── -->
  <div class="eleven-section alt">
    <div class="eleven-section-inner">
      <div class="eleven-row">
        <div class="eleven-label"><span class="num">06</span>The artifacts</div>
        <div class="eleven-content">
          <h2>Everything <em>reproducible.</em></h2>
          <p>Source code, deployed environment, framework. Open and inspectable.</p>

          <div class="eleven-links">
            <a class="eleven-link" href="https://github.com/Mayankpratapsingh022/API-Testing-RL" target="_blank" rel="noopener">https://github.com/Mayankpratapsingh022/API-Testing-RL</a>
            <a class="eleven-link" href="https://meta-pytorch.org/OpenEnv/" target="_blank" rel="noopener">https://meta-pytorch.org/OpenEnv/</a>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>
"""


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

def _generate_report(bug_ids, action_history):
    """Generate OWASP bug bounty report from discovered bugs."""
    from server.graders import generate_bug_report
    return generate_bug_report(bug_ids, action_history)


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
        "No bugs found yet. Send requests to discover vulnerabilities.",
        "No tokens acquired yet.",
        "No resources created yet.",
    )


def send_request(method, endpoint, headers_str, params_str, body_str, expected_status, state):
    if not state or not state.initialized:
        return (state, "Environment not initialized. Click 'Reset' first.", "", "", "", "", "", "", "", "", "", "")

    try:
        headers = json.loads(headers_str) if headers_str.strip() else {}
    except json.JSONDecodeError:
        return (state, "Invalid JSON in headers.", "", "", "", "", "", "", "", "", "", "")
    try:
        query_params = json.loads(params_str) if params_str.strip() else {}
    except json.JSONDecodeError:
        return (state, "Invalid JSON in query params.", "", "", "", "", "", "", "", "", "", "")
    try:
        body = json.loads(body_str) if body_str.strip() else None
    except json.JSONDecodeError:
        return (state, "Invalid JSON in body.", "", "", "", "", "", "", "", "", "", "")

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
        _generate_report(es.bugs_found_ids, state.step_log),
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
        yield state, "Environment not initialized.", "", "", "", "", "", "", "", "", "", ""
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
            _generate_report(es.bugs_found_ids, state.step_log),
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
            owasp_badge = f' | {bug.owasp.split(" ")[0]}' if bug.owasp else ""
            cards.append(
                f'<div style="border:1px solid {fg}40;border-radius:8px;padding:8px 10px;'
                f'margin-bottom:6px;background:{fg}0d;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-weight:700;font-size:0.85em;">{bid}</span>'
                f'<span style="background:{fg};color:#fff;padding:1px 8px;border-radius:10px;'
                f'font-size:0.75em;font-weight:600;">{bug.severity.upper()}{owasp_badge}</span></div>'
                f'<div style="margin-top:4px;font-size:0.85em;opacity:0.7;">'
                f'{bug.description}</div>'
                f'<div style="margin-top:2px;font-size:0.78em;opacity:0.5;font-style:italic;">'
                f'{bug.owasp}</div></div>'
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

_GRADIO_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)


# Custom CSS injected into the Gradio app to highlight important interactive
# elements (primary buttons, active tabs, hover states) so the playground
# doesn't feel washed out. Works in both light and dark mode.
_GRADIO_CSS = """
/* ─── Mintlify-inspired green palette (flat, no gradients) ─── */
:root {
    --accent:        #18E299;   /* Brand Green */
    --accent-hover:  #0fa76e;   /* Brand Green Deep */
    --accent-soft:   #d4fae8;   /* Brand Green Light */
    --accent-border: rgba(15, 167, 110, 0.28);
    --ink:           #0d0d0d;
    --ink-muted:     #666666;
    --line:          #e5e5e5;
    --surface:       #fafafa;
    --success: #16a34a;
    --danger:  #dc2626;
    --info:    #2563eb;
}
.dark {
    --accent:        #18E299;
    --accent-hover:  #34efaa;
    --accent-soft:   rgba(24, 226, 153, 0.14);
    --accent-border: rgba(24, 226, 153, 0.35);
    --ink:           #f5f5f5;
    --ink-muted:     #a0a0a0;
    --line:          rgba(255, 255, 255, 0.10);
    --surface:       #141414;
}

/* ─── Primary buttons ─────────────────────────────────────────────
   Light mode: near-black surface, white text, green hover.
   Dark mode:  bright green surface, near-black text.
   Both flat (no gradients, no glow). */
button.primary,
.gr-button.primary,
button[class*="primary"],
.gr-button-primary {
    background: #0d0d0d !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid #0d0d0d !important;
    box-shadow: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}
button.primary:hover,
.gr-button.primary:hover,
button[class*="primary"]:hover,
.gr-button-primary:hover {
    background: #0fa76e !important;
    background-image: none !important;
    color: #ffffff !important;
    border-color: #0fa76e !important;
    box-shadow: none !important;
    transform: none !important;
    filter: none !important;
}
button.primary:active,
.gr-button.primary:active,
.gr-button-primary:active {
    background: #0a8a5a !important;
    border-color: #0a8a5a !important;
    transform: none !important;
    filter: none !important;
}
/* Dark-mode override: bright green CTA pops against the dark surface */
.dark button.primary,
.dark .gr-button.primary,
.dark button[class*="primary"],
.dark .gr-button-primary {
    background: #18E299 !important;
    color: #07301f !important;
    border: 1px solid #18E299 !important;
}
.dark button.primary:hover,
.dark .gr-button.primary:hover,
.dark button[class*="primary"]:hover,
.dark .gr-button-primary:hover {
    background: #34efaa !important;
    border-color: #34efaa !important;
    color: #07301f !important;
}

/* ─── Secondary buttons ──────────────────────────────────────────
   Light mode: white with dark border, fills near-black on hover.
   Dark mode:  ghost button with green border. */
button.secondary,
.gr-button.secondary,
.gr-button-secondary {
    background: #ffffff !important;
    background-image: none !important;
    border: 1px solid #0d0d0d !important;
    color: #0d0d0d !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
}
button.secondary:hover,
.gr-button.secondary:hover,
.gr-button-secondary:hover {
    background: #0d0d0d !important;
    color: #ffffff !important;
    border-color: #0d0d0d !important;
}
.dark button.secondary,
.dark .gr-button.secondary,
.dark .gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--accent-border) !important;
    color: var(--accent) !important;
}
.dark button.secondary:hover,
.dark .gr-button.secondary:hover,
.dark .gr-button-secondary:hover {
    background: var(--accent) !important;
    color: #07301f !important;
    border-color: var(--accent) !important;
}

/* ─── Tabs (selected tab uses brand green) ─── */
button[role="tab"][aria-selected="true"],
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    color: var(--accent-hover) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}
.dark button[role="tab"][aria-selected="true"],
.dark .tab-nav button.selected,
.dark .tab-nav button[aria-selected="true"] {
    color: var(--accent) !important;
}
button[role="tab"]:hover,
.tab-nav button:hover {
    color: var(--accent-hover) !important;
}
.dark button[role="tab"]:hover,
.dark .tab-nav button:hover {
    color: var(--accent) !important;
}

/* ─── Inputs (focus ring uses brand green, no glow) ─── */
.gr-dropdown,
.gr-input,
.gr-textbox,
input[type="text"],
input[type="number"],
textarea,
select {
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.gr-dropdown:focus-within,
.gr-input:focus-within,
.gr-textbox:focus-within,
input:focus,
textarea:focus,
select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
    outline: none !important;
}

/* ─── Section headings ─── */
h1, h2, h3 {
    letter-spacing: -0.01em !important;
}
h1, h2 {
    color: #0d0d0d !important;
}
.dark h1, .dark h2 {
    color: #f5f5f5 !important;
}
h3 {
    color: #0d0d0d !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--line) !important;
    padding-bottom: 6px !important;
    margin-bottom: 12px !important;
    position: relative !important;
}
/* small green accent bar before each section heading for brand identity */
h3::before {
    content: "" !important;
    display: inline-block !important;
    width: 4px !important;
    height: 14px !important;
    background: #18E299 !important;
    border-radius: 2px !important;
    margin-right: 8px !important;
    vertical-align: -2px !important;
}
.dark h3 {
    color: #f5f5f5 !important;
}

/* ─── Markdown links ─── */
.prose a, .markdown a, a {
    color: var(--accent-hover) !important;
    text-decoration: none !important;
    border-bottom: 1px solid var(--accent-border) !important;
}
.dark .prose a, .dark .markdown a, .dark a {
    color: var(--accent) !important;
}
.prose a:hover, .markdown a:hover, a:hover {
    border-bottom-color: var(--accent) !important;
}

/* ─── Accordion headers ─── */
.gr-accordion > button,
button[class*="accordion"] {
    color: var(--accent-hover) !important;
    font-weight: 600 !important;
}
.dark .gr-accordion > button,
.dark button[class*="accordion"] {
    color: var(--accent) !important;
}

/* ─── Card borders (Mintlify principle: borders, not shadows) ─── */
.gr-block.gr-box {
    border-color: var(--line) !important;
    box-shadow: none !important;
}

/* ─── Match the Gradio dark surface to the blog section ──────────
   The blog section below uses #0a0a0a as its background. Override
   Gradio's default slate so the page reads as one continuous canvas. */
.dark {
    --body-background-fill: #0a0a0a !important;
    --background-fill-primary: #0a0a0a !important;
    --background-fill-secondary: #131313 !important;
    --block-background-fill: #131313 !important;
    --panel-background-fill: #131313 !important;
    --input-background-fill: #131313 !important;
    --border-color-primary: rgba(255, 255, 255, 0.08) !important;
}
.dark,
.dark body,
.dark gradio-app,
.dark .gradio-container,
.dark .main,
.dark .wrap,
.dark .app,
.dark .contain {
    background: #0a0a0a !important;
    background-color: #0a0a0a !important;
}
/* Cards / blocks get a slightly lighter surface so they remain
   visually separated from the page background. */
.dark .gr-block,
.dark .gr-box,
.dark .gr-form,
.dark .gr-panel,
.dark .block,
.dark .form {
    background: #131313 !important;
    background-color: #131313 !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
}
"""


def build_ui():
    # Mintlify-inspired green Soft theme — adapts to ?__theme=light / ?__theme=dark
    # URL params on HuggingFace Spaces. The blog section below also reads the
    # .dark body class so the entire page adapts together.
    with gr.Blocks(title="API Testing Environment", theme=_GRADIO_THEME, css=_GRADIO_CSS) as demo:
        session = gr.State(value=new_session())

        gr.Markdown(
            "# API Testing Environment\n"
            "An OpenEnv RL environment that trains AI agents to become automated **API security testers**. "
            "A simulated API server with **13 hidden vulnerabilities** mapped to the **OWASP API Security Top 10** is provided. "
            "Send HTTP requests, earn rewards for finding bugs and covering endpoints, and generate a **bug bounty report** at episode end. "
            "Use **Manual Testing** to craft requests yourself, or run a **Baseline Agent** to watch an automated strategy."
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
                    with gr.Tab("Run Baseline Agent"):
                        gr.Markdown("### Automated Agents\nWatch a baseline agent test the API step by step. Pick a strategy and click Run Agent.")
                        agent_dropdown = gr.Dropdown(choices=["random", "sequential", "smart"], value="smart", label="Agent Type")
                        run_agent_btn = gr.Button("Run Agent", variant="primary", size="lg")

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

                gr.Markdown("---")
                gr.Markdown("### Response")
                response_display = gr.Markdown("")

                gr.Markdown("### Feedback")
                feedback_display = gr.Markdown("")

            # ── Right Panel ──
            # Stacked (no tabs) so Discovered Bugs and Activity Log are both
            # visible at once — users shouldn't have to click to see the log.
            with gr.Column(scale=1):
                gr.Markdown("### Discovered Bugs")
                bug_list_display = gr.Markdown("No bugs found yet.")

                gr.Markdown("### Activity Log")
                log_display = gr.Markdown("No steps yet.")

                with gr.Accordion("Bug Report (OWASP)", open=False):
                    gr.Markdown("*Auto-generated OWASP security report. Populates as bugs are found.*")
                    bug_report_display = gr.Markdown("No bugs found yet. Send requests to discover vulnerabilities.")

        # ── Editorial blog-style documentation below the app ──
        gr.HTML(BLOG_HTML)

        # ── Wiring ──
        reset_outputs = [
            session, status_box, feedback_display, response_display,
            reward_display, bug_display, coverage_display, log_display,
            step_display, bug_list_display, bug_report_display, auth_display, resource_display,
        ]

        step_outputs = [
            session, feedback_display, response_display, reward_display,
            bug_display, coverage_display, log_display, step_display,
            bug_list_display, bug_report_display, auth_display, resource_display,
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

    # Pass theme + css to both Blocks() (Gradio 5.x) and launch() (Gradio 6.0+)
    # so it works on whichever version the host runs.
    launch_kwargs = dict(server_name=args.host, server_port=args.port, share=args.share)
    try:
        build_ui().launch(theme=_GRADIO_THEME, css=_GRADIO_CSS, **launch_kwargs)
    except TypeError:
        # Older Gradio: launch() doesn't accept theme/css — Blocks() already has them
        build_ui().launch(**launch_kwargs)
