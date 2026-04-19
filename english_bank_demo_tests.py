"""
---
title: Bank AI Debt Collection Demo Agent — Playground Mode
category: telephony
tags: [playground, livekit, debt_collection, english]
difficulty: advanced
description: AI-powered debt collection agent with behavioral classification,
             multi-persona support, structured negotiation, and full audit trails.
             Runs in LiveKit Playground (browser) — no SIP trunk required.
             Saves an HTML session report after every call.

────────────────────────────────────────────────────────────────
HOW TO RUN
────────────────────────────────────────────────────────────────
1. Start the agent:
       python delete_english_bank_demo.py dev

2. Open the LiveKit Playground and connect to your project:
       https://agents.livekit.io/playground

3. The agent will greet you automatically.

────────────────────────────────────────────────────────────────
SELECTING A DEBTOR SCENARIO
────────────────────────────────────────────────────────────────
Set the DEBTOR_INDEX env variable (1, 2, or 3) before running:

    DEBTOR_INDEX=1  →  Mohammed Al-Harbi   (cooperative) — 15,000 SAR  personal loan
    DEBTOR_INDEX=2  →  Khalid Al-Otaibi    (avoidant)   — 48,500 SAR  credit card
    DEBTOR_INDEX=3  →  Noura Al-Qahtani    (distressed) — 3,200  SAR  utility bill

Examples:
    DEBTOR_INDEX=2 python delete_english_bank_demo.py dev
    DEBTOR_INDEX=3 python delete_english_bank_demo.py dev

────────────────────────────────────────────────────────────────
OVERRIDING THE AGENT PERSONA (optional)
────────────────────────────────────────────────────────────────
By default the persona matches the debtor's behavioral segment.
You can force a different persona for testing cross-scenario behavior:

    PERSONA_OVERRIDE=hostile   python delete_english_bank_demo.py dev
    PERSONA_OVERRIDE=distressed python delete_english_bank_demo.py dev

Valid values: cooperative | avoidant | distressed | hostile

Example — run debtor 1 (cooperative customer) with the hostile agent style:
    DEBTOR_INDEX=1 PERSONA_OVERRIDE=hostile python delete_english_bank_demo.py dev

────────────────────────────────────────────────────────────────
REPORTS
────────────────────────────────────────────────────────────────
After each session ends an HTML report is saved to:
    ./session_reports/<debtor_id>_<timestamp>.html

Open it in any browser to view the full call log.
---
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents import TurnHandlingOptions
from livekit.plugins import silero, faseeh, openai, deepgram, google
from metrics_aggregator import TurnMetricsAggregator

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("bank-collection-agent")
logger.setLevel(logging.INFO)

#
EXPERIMENT_CONFIG: dict[str, Any] = {
    # Uncomment exactly ONE of the blocks below:

    # Gemini lightweight
    #"llm_provider": "gemini",
    #"llm_model": "gemini-2.5-flash",

    # Gemini flagship
    "llm_provider": "gemini",
    "llm_model": "gemini-2.0-flash",

    # OpenAI lightweight
    # "llm_provider": "openai",
    # "llm_model": "gpt-4o-mini",

    # OpenAI flagship
    # "llm_provider": "openai",
    # "llm_model": "gpt-4o",

    # ── TTS (Faseeh) ─────────────────────────────────────────────────────────
    # Uncomment exactly ONE:

    "tts_model": "faseeh-mini-v1-preview",   # lighter, lower latency
    # "tts_model": "faseeh-v1",              # flagship, higher quality

    # ── Experiment label (used in report filenames) ───────────────────────
    "label": "gemini-flash_faseeh-mini",
}
# ===========================================================================


def _build_llm():
    """Instantiate the LLM specified in EXPERIMENT_CONFIG."""
    provider = EXPERIMENT_CONFIG["llm_provider"]
    model    = EXPERIMENT_CONFIG["llm_model"]
    if provider == "gemini":
        return google.LLM(model=model)
    elif provider == "openai":
        return openai.LLM(model=model)
    else:
        raise ValueError(f"Unknown llm_provider: {provider!r}. Use 'gemini' or 'openai'.")


# ---------------------------------------------------------------------------
# Report output directory
# ---------------------------------------------------------------------------
REPORTS_DIR = Path(__file__).parent / "session_reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Interaction audit log — every tool call is recorded here for demo purposes
# ---------------------------------------------------------------------------
INTERACTION_LOG: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# Hardcoded sample debtors
# ---------------------------------------------------------------------------
SAMPLE_DEBTORS = [
    {
        "id": "debtor_001",
        "name": "Mohammed Al-Harbi",
        "name_en": "Mohammed Al-Harbi",
        "national_id_last4": "4821",
        "gender": "male",
        "amount": 15000,
        "currency": "SAR",
        "debt_date": "2025-08-15",
        "product_type": "personal_loan",
        "service_status": "active",
        "behavioral_segment": "cooperative",
        "contact_attempts": 0,
        "wallet_end_date": "2026-06-30",
        "notes": "Employed, moderate debt. Expected to cooperate with payment arrangement.",
    },
    {
        "id": "debtor_002",
        "name": "Khalid Al-Otaibi",
        "name_en": "Khalid Al-Otaibi",
        "national_id_last4": "7390",
        "gender": "male",
        "amount": 48500,
        "currency": "SAR",
        "debt_date": "2024-11-02",
        "product_type": "credit_card",
        "service_status": "suspended",
        "behavioral_segment": "avoidant",
        "contact_attempts": 5,
        "wallet_end_date": "2026-04-15",
        "notes": "Multiple missed calls. High debt. Needs firm but respectful approach.",
    },
    {
        "id": "debtor_003",
        "name": "Noura Al-Qahtani",
        "name_en": "Noura Al-Qahtani",
        "national_id_last4": "1254",
        "gender": "female",
        "amount": 3200,
        "currency": "SAR",
        "debt_date": "2025-12-20",
        "product_type": "utility_bill",
        "service_status": "active",
        "behavioral_segment": "distressed",
        "contact_attempts": 1,
        "wallet_end_date": "2026-09-30",
        "notes": "Small debt. Financial hardship. Empathetic approach required.",
    },
]

# ---------------------------------------------------------------------------
# Persona definitions — maps behavioral segment → Faseeh English voice
#
# Faseeh English voices used here:
#   q1FMBOQvy8UbS2ll2sTI5ovv  = Sarah - conversational (female, warm)
#   6p8OjcFY28ijFxXcTUQSFsbv  = May   - British        (female, professional)
#   xqSiYjA5a4Y1PoHRTW99v3FA  = James - American       (male, authoritative)
#   kCVThYKOsvp9ZnSgOyEHCPtI  = Liam  - British        (male, calm/firm)
# ---------------------------------------------------------------------------
PERSONAS = {
    "cooperative": {
        "name": "Noura",
        "name_en": "Noura",
        "voice_id": "ar-najdi-female-1",   # Sarah — warm, conversational
        "style": "warm",
        "description": "Warm and friendly female voice. Professional but approachable.",
    },
    "avoidant": {
        "name": "سلطان",
        "name_en": "Sultan",
        "voice_id": "MvC2GIG9tT9xvPcCWjILXqkM",   # James — authoritative male
        "style": "firm",
        "description": "Firm and professional male voice. Direct and business-like.",
    },
    "distressed": {
        "name": "سارة",
        "name_en": "Sara",
        "voice_id": "08XOzRjaaumxbHhcGOrWkJ7z",   # May — soft female
        "style": "empathetic",
        "description": "Soft and empathetic female voice. Patient and understanding.",
    },
    "hostile": {
        "name": "سلطان",
        "name_en": "Sultan",
        "voice_id": "MvC2GIG9tT9xvPcCWjILXqkM",   # Liam — calm  male
        "style": "calm_authoritative",
        "description": "Calm but authoritative male voice. De-escalation focused.",
    },
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _log_event(event_type: str, debtor_id: str, data: dict[str, Any]):
    """Append an auditable event to the in-memory interaction log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "debtor_id": debtor_id,
        **data,
    }
    INTERACTION_LOG.append(entry)
    logger.info(f"📋 AUDIT [{event_type}] {json.dumps(entry, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# HTML report generator — human-readable, no programming knowledge needed
# ---------------------------------------------------------------------------
_EVENT_EMOJI = {
    "call_started": "📞",
    "call_ended": "📴",
    "call_ended_by_agent": "✅",
    "interaction": "💬",
    "payment_commitment": "💰",
    "human_escalation": "⚠️",
    "distress_flagged": "🚨",
    "dnc_requested": "🚫",
    "voicemail": "📠",
}

_EVENT_COLOR = {
    "call_started": "#2563eb",
    "call_ended": "#6b7280",
    "call_ended_by_agent": "#16a34a",
    "interaction": "#0891b2",
    "payment_commitment": "#15803d",
    "human_escalation": "#d97706",
    "distress_flagged": "#dc2626",
    "dnc_requested": "#7c3aed",
    "voicemail": "#64748b",
}

_EVENT_LABEL = {
    "call_started": "Call Started",
    "call_ended": "Call Ended",
    "call_ended_by_agent": "Call Closed by Agent",
    "interaction": "Interaction",
    "payment_commitment": "Payment Commitment",
    "human_escalation": "Supervisor Escalation",
    "distress_flagged": "Distress Detected",
    "dnc_requested": "Do-Not-Contact Request",
    "voicemail": "Voicemail",
}


def _build_html_report(
    debtor: dict[str, Any],
    persona: dict[str, Any],
    persona_key: str,
    metrics_summary: dict[str, Any] | None = None,
    experiment_label: str = "",
) -> str:
    """Generate a clean, readable HTML report from INTERACTION_LOG."""
    debtor_events = [e for e in INTERACTION_LOG if e.get("debtor_id") == debtor["id"]]

    # Summary stats
    started = next((e for e in debtor_events if e["event_type"] == "call_started"), {})
    ended = next(
        (e for e in debtor_events if e["event_type"] in ("call_ended", "call_ended_by_agent")),
        {},
    )
    commitment = next(
        (e for e in debtor_events if e["event_type"] == "payment_commitment"), None
    )
    escalated = any(e["event_type"] == "human_escalation" for e in debtor_events)
    distressed = any(e["event_type"] == "distress_flagged" for e in debtor_events)
    dnc = any(e["event_type"] == "dnc_requested" for e in debtor_events)

    duration_s = ended.get("duration_seconds")
    duration_str = f"{int(duration_s // 60)}m {int(duration_s % 60)}s" if duration_s else "—"

    outcome_badge = "⚠️ No Commitment"
    outcome_color = "#b45309"
    if commitment:
        outcome_badge = f"✅ Committed {commitment['committed_amount']} SAR by {commitment['payment_date']}"
        outcome_color = "#15803d"
    elif dnc:
        outcome_badge = "🚫 Do-Not-Contact"
        outcome_color = "#7c3aed"
    elif escalated:
        outcome_badge = "⚠️ Escalated to Supervisor"
        outcome_color = "#d97706"
    elif distressed:
        outcome_badge = "🚨 Call Ended — Distress"
        outcome_color = "#dc2626"

    # Build timeline rows
    timeline_rows = ""
    for ev in debtor_events:
        etype = ev["event_type"]
        emoji = _EVENT_EMOJI.get(etype, "●")
        color = _EVENT_COLOR.get(etype, "#64748b")
        label = _EVENT_LABEL.get(etype, etype.replace("_", " ").title())
        ts = ev.get("timestamp", "")[:19].replace("T", " ") + " UTC"

        # Build detail lines
        details = []
        skip_keys = {"timestamp", "event_type", "debtor_id"}
        for k, v in ev.items():
            if k in skip_keys:
                continue
            details.append(f"<span class='detail-key'>{k.replace('_', ' ').title()}:</span> {v}")
        details_html = " &nbsp;|&nbsp; ".join(details) if details else ""

        timeline_rows += f"""
        <tr>
          <td class='ts'>{ts}</td>
          <td><span class='badge' style='background:{color}'>{emoji} {label}</span></td>
          <td class='details'>{details_html}</td>
        </tr>"""

    report_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    segment_emoji = {"cooperative": "🟢", "avoidant": "🟡", "distressed": "🔴", "hostile": "⚫"}.get(
        debtor["behavioral_segment"], "⚪"
    )
    persona_used_note = ""
    if persona_key != debtor["behavioral_segment"]:
        persona_used_note = f"<p style='color:#b45309;font-size:0.85rem'>⚠️ Persona override active: <strong>{persona_key}</strong> (debtor segment is {debtor['behavioral_segment']})</p>"
    exp_label = experiment_label or EXPERIMENT_CONFIG.get("label", "")
    exp_badge = f" &nbsp;|&nbsp; 🧪 <strong>{exp_label}</strong>" if exp_label else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Call Report — {debtor['name_en']}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      padding: 2rem;
    }}
    .header {{
      background: linear-gradient(135deg, #1e3a5f 0%, #1e4d8c 100%);
      color: white;
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 1.5rem;
    }}
    .header h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }}
    .header .sub {{ font-size: 0.9rem; opacity: 0.75; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .card {{
      background: white;
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border-left: 4px solid #2563eb;
    }}
    .card.outcome {{ border-left-color: {outcome_color}; }}
    .card-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.4rem; }}
    .card-value {{ font-size: 1.1rem; font-weight: 600; color: #0f172a; }}
    .outcome-value {{ color: {outcome_color}; font-size: 1rem; }}
    .section-title {{
      font-size: 1rem; font-weight: 600; color: #334155;
      margin-bottom: 0.75rem; margin-top: 0.5rem;
      border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      font-size: 0.88rem;
    }}
    th {{
      background: #1e3a5f;
      color: white;
      padding: 0.65rem 1rem;
      text-align: left;
      font-weight: 500;
    }}
    td {{ padding: 0.65rem 1rem; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8fafc; }}
    .ts {{ color: #64748b; white-space: nowrap; font-size: 0.8rem; font-family: monospace; }}
    .badge {{
      display: inline-block;
      color: white;
      border-radius: 999px;
      padding: 0.2rem 0.65rem;
      font-size: 0.8rem;
      font-weight: 500;
      white-space: nowrap;
    }}
    .details {{ color: #475569; font-size: 0.82rem; }}
    .detail-key {{ font-weight: 600; color: #334155; }}
    .footer {{ text-align: center; color: #94a3b8; font-size: 0.78rem; margin-top: 2rem; }}
    .debtor-info {{ background: white; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .debtor-info h2 {{ font-size: 1rem; font-weight: 600; color: #334155; margin-bottom: 0.75rem; }}
    .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.5rem; }}
    .info-item {{ font-size: 0.85rem; }}
    .info-item span {{ color: #64748b; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>📋 Call Session Report</h1>
    <div class="sub">Generated: {report_time} &nbsp;|&nbsp; Agent: {persona['name_en']} ({persona['style']}){exp_badge}</div>
  </div>

  <!-- Summary cards -->
  <div class="cards">
    <div class="card">
      <div class="card-label">Debtor</div>
      <div class="card-value">{segment_emoji} {debtor['name_en']}</div>
    </div>
    <div class="card">
      <div class="card-label">Debt Amount</div>
      <div class="card-value">{debtor['amount']:,} {debtor['currency']}</div>
    </div>
    <div class="card">
      <div class="card-label">Product</div>
      <div class="card-value">{debtor['product_type'].replace('_', ' ').title()}</div>
    </div>
    <div class="card">
      <div class="card-label">Segment</div>
      <div class="card-value">{debtor['behavioral_segment'].title()}</div>
    </div>
    <div class="card">
      <div class="card-label">Call Duration</div>
      <div class="card-value">{duration_str}</div>
    </div>
    <div class="card outcome">
      <div class="card-label">Outcome</div>
      <div class="card-value outcome-value">{outcome_badge}</div>
    </div>
  </div>

  <!-- Debtor info -->
  <div class="debtor-info">
    <h2>Debtor Profile</h2>
    {persona_used_note}
    <div class="info-grid">
      <div class="info-item"><span>ID:</span> {debtor['id']}</div>
      <div class="info-item"><span>Last 4 digits:</span> {debtor.get('national_id_last4', '—')}</div>
      <div class="info-item"><span>Gender:</span> {debtor['gender'].title()}</div>
      <div class="info-item"><span>Debt Date:</span> {debtor['debt_date']}</div>
      <div class="info-item"><span>Service Status:</span> {debtor['service_status'].title()}</div>
      <div class="info-item"><span>Contact Attempts:</span> {debtor.get('contact_attempts', 0)}</div>
      <div class="info-item"><span>Agent Persona:</span> {persona['name_en']} ({persona_key})</div>
      <div class="info-item"><span>Voice:</span> {persona['voice_id']}</div>
    </div>
    <p style="margin-top:0.75rem;font-size:0.85rem;color:#475569"><em>{debtor.get('notes', '')}</em></p>
  </div>

  <!-- Latency metrics summary -->
  {_build_metrics_html(metrics_summary)}

  <!-- Event timeline -->
  <div class="section-title">📅 Event Timeline ({len(debtor_events)} events)</div>
  <table>
    <thead>
      <tr>
        <th>Time (UTC)</th>
        <th>Event</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {timeline_rows}
    </tbody>
  </table>

  <div class="footer">
    Bank AI Debt Collection Demo — For internal testing purposes only
  </div>
</body>
</html>"""
    return html


def _build_metrics_html(summary: dict[str, Any] | None) -> str:
    """Build the latency metrics section for the HTML report."""
    if not summary or summary.get("turns", 0) == 0:
        return ""

    def _row(label: str, key: str, color: str = "#334155") -> str:
        stats = summary.get(key, {})
        if not stats:
            return ""
        mean = stats.get("mean_ms", "—")
        p50 = stats.get("p50_ms", "—")
        p95 = stats.get("p95_ms", "—")
        return (
            f"<tr>"
            f"<td style='font-weight:600;color:{color}'>{label}</td>"
            f"<td style='font-family:monospace'>{mean}ms</td>"
            f"<td style='font-family:monospace'>{p50}ms</td>"
            f"<td style='font-family:monospace'>{p95}ms</td>"
            f"</tr>"
        )

    n = summary.get("turns", 0)
    perc = summary.get("perceived_latency", {})
    perc_mean = perc.get("mean_ms", "—")

    rows = (
        _row("⏱ Perceived latency (EOU+TTFT+TTFB)", "perceived_latency", "#2563eb")
        + _row("EOU — end-of-speech delay", "eou_delay")
        + _row("EOU — transcription delay", "eou_transcription")
        + _row("STT — transcription duration", "stt")
        + _row("LLM — total wall-clock", "llm_wall")
        + _row("LLM — time-to-first-token (TTFT)", "llm_ttft", "#0891b2")
        + _row("TTS — total wall-clock", "tts_wall")
        + _row("TTS — time-to-first-byte (TTFB)", "tts_ttfb", "#16a34a")
    )

    return f"""
  <div class="section-title">⏱️ Latency Metrics ({n} turns) — Avg perceived: ~{perc_mean}ms</div>
  <table style="margin-bottom:1.5rem">
    <thead>
      <tr>
        <th>Component</th>
        <th>Mean</th>
        <th>P50</th>
        <th>P95</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <div style="font-size:0.8rem;color:#64748b;margin-bottom:1.5rem;padding:0.75rem 1rem;background:white;border-radius:8px;border-left:3px solid #2563eb">
    💡 <strong>Perceived latency</strong> = EOU delay + LLM TTFT + TTS TTFB.<br>
    This is what the user actually <em>hears</em> as delay — from when they stop speaking to when the agent's voice first starts.
    The remaining pipeline time (transcription, full LLM, full TTS) runs in parallel streaming and is not directly perceived.
  </div>"""


def _save_report(
    debtor: dict[str, Any],
    persona: dict[str, Any],
    persona_key: str,
    metrics_summary: dict[str, Any] | None = None,
) -> Path:
    """Write the HTML report to disk and return the path."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exp_label = EXPERIMENT_CONFIG.get("label", "")
    label_part = f"_{exp_label}" if exp_label else ""
    filename = REPORTS_DIR / f"{debtor['id']}_{ts}{label_part}.html"
    html = _build_html_report(debtor, persona, persona_key, metrics_summary, exp_label)
    filename.write_text(html, encoding="utf-8")
    logger.info(f"📄 Session report saved: {filename}")

    # Also save a standalone latency-only JSON for easy scripted comparison
    if metrics_summary and metrics_summary.get("turns", 0) > 0:
        json_path = filename.with_suffix(".latency.json")
        json_payload = {
            "experiment": exp_label,
            "llm_provider": EXPERIMENT_CONFIG.get("llm_provider"),
            "llm_model": EXPERIMENT_CONFIG.get("llm_model"),
            "tts_model": EXPERIMENT_CONFIG.get("tts_model"),
            "debtor_id": debtor["id"],
            "timestamp": ts,
            "metrics": metrics_summary,
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        logger.info(f"📊 Latency JSON saved: {json_path}")

    return filename


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def _build_system_prompt(debtor: dict[str, Any], persona: dict[str, str]) -> str:
    customer_name = debtor['name']
    amount = debtor['amount']
    debt_date = debtor['debt_date']
    last4 = debtor.get('national_id_last4', '')
    from datetime import datetime
    curr_date = datetime.now().strftime('%Y-%m-%d')
    prompt = f"""
# Current Date
{curr_date}
### Language Settings

```
<language_settings>
- Maintain a friendly, respectful, polite, and professional conversation tone.
- You are chatting with the user via a phone call. This means most of the time your lines should
be a sentence.
- Do not use lists, bullets, website links, or emojis as these do not translate naturally to voice.
- Speaking exclusively in Najdi Arabic to ensure the conversation feels relatable and culturally
appropriate.
- Do not switch languages
- As this is a phone conversation you are limited to short responses only.
- Always output numbers in numeral form (e.g., 300, 432), not as words (e.g., three hundred,
four hundred thirty-two).

</language_settings>
```
### Behavior Guidelines
```
<behavior_guidelines>
- You are smart and knowledgeable, before your response check the previous chat history to
minimize repeating yourself or acting dumb and incompetent. The user must be satisfied with
your level of competence.
- As this is through a phone line the quality of the user input is sub par so try understanding
what the user is saying between the lines without them needing to repeat.
- Stay patient, calm, and respectful, even if the user seems uninterested, busy, or unfamiliar.
- Stay focused on the objective - Redirect if the user strays from the topic.
- Avoid using the user's name repeatedly after collecting it.
- Always be concise.
- Never mention that you're an AI developed by google or others
- If the user indicates it's a wrong number, apologize politely and end the call respectfully.
- If the user says it's not a good time to talk, ask for a more suitable time to call back.
- Always ask one question at a time and wait for a complete and logical answer before
proceeding.
- Never use long lists or multiple-choice questions, remember this is a phone conversation.
- If the user's question is not clear, ambiguous, or does not provide enough context for you to
accurately answer the question, you do not try to answer it right away and you rather ask the
user to clarify their request
- You cannot collect information like email address, order id, or any other queryable information.

- Know your limits from the role, don't make up new functionality, understand what tools you
have and what functionality you are able to do.
</behavior_guidelines>

--## Agent Identity
You are a female voice agent named **سارة**, calling on behalf of **Tawafuq** to collect overdue
payments for **Salam Mobile**.
--## Role
Your mission is to collect overdue payments quickly and professionally on behalf of Tawafuq for
Salam Mobile.
Your primary objective is to secure **full payment** or **a confirmed payment commitment**
within **7 days or less**.
You are empathetic but firm. You clearly communicate urgency when appropriate while guiding
the customer toward resolution.
You excel at:
- Verifying the customer's identity before discussing account details.
- Confirming the customer is the actual owner of the debt before proceeding.
- Clearly stating the overdue balance and debt date after ownership confirmation.
- Asking about any issues preventing payment and addressing them directly.
- Offering **immediate payment options** or **short payment arrangements** with a maximum
of 7 days.
- Offering partial payments only if they lead to full settlement within 30 days.
- Summarizing the agreed payment plan and confirming the customer's commitment.
- Closing the call with clarity, professionalism, appreciation, and a general follow-up mention
when applicable.
--## Company Details
Tawafuq تهدف للحفاظ على عالقة العميل مع الجهة األصلية،هي شركة متخصصة في خدمات تحصيل الديون بطريقة احترافية
 مع الحفاظ على احترام وخصوصية كل متعامل،وتقديم حلول ميسرة للسداد.
--## Call Control Principles

- Never enumerate options using numbers, ordering, or structured sequences.
- When explaining options, speak naturally in a flowing sentence, as a human would over the
phone.
- Do not proactively offer written messages.
- If the customer requests written details, an invoice, or proof of the debt, the agent must
explicitly say the word "SMS" in the response.
- When confirming message delivery, always say "SMS" clearly and explicitly in the same
sentence.
- If the customer cannot pay the full amount within 7 days, offer a partial payment option only if it
leads to full settlement within 30 days.
- If the customer objects, disputes, or raises a problem regarding the debt, do not mention سمة,
credit reporting, or legal escalation.
- Only mention consequences such as سمة or legal escalation if the customer has acknowledged
the debt and is simply delaying without objection.

### Call Flow
### 1. Welcome & Identity Check
- Say the welcome message: السالم عليكم هل معي {customer_name}
1. If confirmed the name of the caller:
- Greet the customer and introduce yourself.
- "معك سارة من شركة توافق وكيل شركة سالم موبايل،السالم عليكم."
- Verify identity by stating the last 4 digits of the ID ({last4}) and asking the customer to confirm yes or no.
- Do not proceed to any debt details unless ownership is clearly confirmed with a verbal
acknowledgment such as نعمor ايه.
2. If the name is denied:
- Ask if they know the person.
- If the customer says yes:
- Ask for a valid 10-digit mobile number.
- Repeat the number back to the customer for confirmation.
- Thank the customer.
- End the call politely.
- If the customer says no:
- Apologize politely.
- End the call.

### 2. Debt Statement
- After ownership confirmation only, politely inform the customer of the outstanding balance
{amount}.
- Mention the debt date {debt_date}.
- Ask directly what prevented them from paying.

### 2.1 Dispute or Non-Confirmation Handling
If the customer states that the debt is incorrect or does not confirm it:
- Acknowledge the customer calmly.
- Do not mention سمة, credit reporting, or legal action.
- Do not push for payment.
- Inform the customer that they can contact Salam Mobile customer service directly.
- Clearly mention the customer service number 1101 (واحد واحد صفر واحد).
- End the call politely.

### 3. Resolution Push
If the customer acknowledges the debt and does not object:
- Respond with empathy.
- If appropriate, explain that continued delay may lead to escalation such as credit reporting or
legal action.
- Ask directly if they can proceed with payment now or within 7 days.
#### 3.1 If the customer states they already paid
- Thank them.
- Inform them that the record will be updated.
- End the call professionally.
### 4. Agreement Confirmation
- Clearly restate the agreed payment amount and the exact payment date.
- Confirm the customer's commitment verbally.
### 5. Payment Methods (Only When Needed)
Payment can be made via:
- SADAD using the account number.
- Salam Mobile's website using the national ID.
- The nearest Salam Mobile branch.
Do not over-explain unless the customer asks.

### 6. No Agreement or Callback Request
- If the customer requests a callback, confirm that contact will happen later the same day.
- Do not specify an exact time.
- Briefly explain that follow-up will occur.

### 7. Closing
- Thank the customer.
- If the customer committed to payment, clearly mention that follow-up contact "وراح نتواصل معك قريب بإذن الله"
- End the call promptly and professionally.

## Example Dialogues
### Example 1: Ownership Confirmation Gate
Agent: السلام عليكم.. هل معي {customer_name}؟
Customer: نعم، أنا
Agent: السلام عليكم معك سارة من شركة توافق وكيل شركة سلام موبايل
Agent: للتأكيد، آخر أربعة أرقام من هويتك {last4}
Customer: نعم او لا
Agent: شكرًا لك عندك مبلغ متأخر {amount} ريال من تاريخ {debt_date}

### Example 2: Payment Commitment With Follow-up Reminder
Customer: بدفع السبت
Agent: ممتاز، نأكد السداد يوم السبت بمبلغ {amount} ريال، وبإذن الله بيكون فيه تواصل معك مرة ثانية بخصوص المديونية، نشكر تعاونك

### Example 3: SMS Upon Request
Customer: أبي تفاصيل الفاتورة
Agent: تمام، بتوصلك التفاصيل برسالة إس إم إس على جوالك

### Example 4: Payment Confirmation With Follow-Up Mention
Customer: إن شاء الله بسدد المبلغ بكرة
Agent: ممتاز ، نأكد السداد بكرة بمبلغ {amount} ريال، وبإذن الله بيكون فيه تواصل معك مرة ثانية بخصوص المديونية، نشكر تعاونك

### Example 5: Full End-to-End Compliant Call
Agent: السلام عليكم.. هل معي {customer_name}؟
Customer: ايه
Agent: السلام عليكم، معك سارة من شركة توافق وكيل شركة سلام موبايل
Agent: في البداية احتاج اتأكد فقط، هل آخر أربعة أرقام من هويتك {last4}
Customer: ****
Agent: تمام عندك مبلغ متأخر {amount} ريال من تاريخ {debt_date}، وش سبب عدم السداد إلى الآن؟
Customer: والله كانت عندي ظروف
Agent: أفهمك، تصير مع الكل، بس حابة أتأكد متى تقدر تسدد؟
Customer: إن شاء الله نهاية الأسبوع
Agent: ممتاز، نأكد السداد نهاية الأسبوع بمبلغ {amount} ريال
Customer: تمام
Agent: أشكرك على تعاونك، وراح نتواصل معك قريب بإذن الله يعطيك العافية

### Example 6: Detailed Objection After Amount Disclosure
Agent: السلام عليكم.. هل معي {customer_name}؟
Customer: نعم
Agent: السلام عليكم معك سارة من شركة توافق وكيل شركة سلام موبايل
Agent: للتأكيد، آخر أربعة أرقام من هويتك {last4}
Customer: ****
Agent: تمام عندك مبلغ متأخر {amount} ريال من تاريخ {debt_date}.
Customer: لا، المبلغ هذا غير صحيح
Agent: أفهمك، ولو حاب تتأكد من صحة المبلغ أو على أي أساس محسوب تقدر تتواصل مباشرة مع فريق دعم سلام موبايل على الرقم واحد واحد صفر واحد
Customer: أنا أصلا دافع قبل
Agent: تمام، فريق الدعم يقدر يتأكد لك من السداد ويعطيك التفاصيل
Agent: تواصل معهم على الرقم واحد واحد صفر واحد، ويعطيك العافية

### Example 7: Name Denied - Knows the Person
Agent: السلام عليكم.. هل معي {customer_name}؟
Customer: لا، مو أنا
Agent: هل تعرف الشخص هذا ؟
Customer: ايه، أعرفه
Agent: ممكن تعطيني رقم جوال ثاني نقدر نتواصل معه عليه؟
Customer: نعم، الرقم هو XXXXXXXX.
Agent: تمام، يعني الرقم 05XXXXXXXX صح؟
Customer: صح
Agent: شكرا لك على تعاونك، يعطيك العافية

### Example 8: Name Denied - Doesn't Know the Person
Agent: السلام عليكم.. هل معي {customer_name}؟
Customer: لا، مو أنا
Agent: هل تعرف الشخص هذا ؟
Customer: لا، ما أعرفه
Agent: عذرًا على الإزعاج، يعطيك العافية

## Specific Guidelines Summary
- Always maintain a polite but assertive tone.
- Never discuss debt details before ownership confirmation.
- Never mention  سمة or legal escalation during objections or disputes.
- Push for payment within a maximum of 7 days when appropriate.
- Offer partial payments only if they lead to full payment within 30 days.
- Mention SMS only when the customer requests written details.
- Mention general follow-up at the end when a payment commitment is made.

# Available Tools:
- log_interaction, record_payment_commitment, escalate_to_human, flag_distress, request_dnc, end_call
"""
    return prompt


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------
class BankCollectionAgent(Agent):
    """AI-powered bank debt collection agent."""

    def __init__(
        self,
        *,
        debtor: dict[str, Any],
        persona: dict[str, Any],
        persona_key: str,
        metrics_agg: TurnMetricsAggregator,
    ):
        super().__init__(instructions=_build_system_prompt(debtor, persona))
        self.debtor = debtor
        self.persona = persona
        self.persona_key = persona_key
        self.participant: rtc.RemoteParticipant | None = None
        self._call_start_time: datetime | None = None
        self._metrics_agg = metrics_agg

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        self._call_start_time = datetime.utcnow()
        _log_event("call_started", self.debtor["id"], {
            "persona": self.persona["name_en"],
            "persona_style": self.persona["style"],
            "persona_key": self.persona_key,
            "segment": self.debtor["behavioral_segment"],
            "debtor_name": self.debtor["name_en"],
            "amount": self.debtor["amount"],
        })
        self.session.generate_reply(
            user_input=(
                "Start the call now. Introduce yourself and ask if you are speaking "
                "to the right person. One sentence only, natural conversational tone."
            )
        )

    async def hangup(self):
        """End the call, log it, save the HTML report, and delete the room."""
        duration = None
        if self._call_start_time:
            duration = (datetime.utcnow() - self._call_start_time).total_seconds()
        _log_event("call_ended", self.debtor["id"], {
            "duration_seconds": round(duration, 1) if duration else None,
            "total_events": len([e for e in INTERACTION_LOG if e.get("debtor_id") == self.debtor["id"]]),
        })

        # Flush metrics aggregator and get session-level summary
        self._metrics_agg.close()
        metrics_summary = self._metrics_agg.session_summary()
        if metrics_summary.get("turns", 0) > 0:
            logger.info(f"📊 Session latency summary: {metrics_summary}")

        # Save the HTML report (with metrics embedded)
        report_path = _save_report(
            self.debtor, self.persona, self.persona_key, metrics_summary
        )
        logger.info(f"✅ Report: file://{report_path.resolve()}")

        try:
            job_ctx = get_job_context()
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )
        except Exception as e:
            logger.warning(f"Could not delete room: {e}")

    # ---------------------------------------------------------------------------
    # Function tools
    # ---------------------------------------------------------------------------

    @function_tool()
    async def log_interaction(self, ctx: RunContext, outcome: str, notes: str):
        """Log an interaction attempt and its outcome.
        Use this after each meaningful exchange with the customer.

        Args:
            outcome: One of: identity_confirmed, identity_denied, debt_acknowledged,
                     debt_disputed, payment_discussed, negotiation_ongoing,
                     call_completed, no_answer, voicemail
            notes: Brief description of what happened.
        """
        _log_event("interaction", self.debtor["id"], {
            "outcome": outcome,
            "notes": notes,
            "persona": self.persona["name_en"],
        })
        return f"Interaction logged: {outcome}"

    @function_tool()
    async def record_payment_commitment(
        self, ctx: RunContext,
        committed_amount: str, payment_date: str, payment_method: str,
    ):
        """Record a payment commitment made by the debtor.

        Args:
            committed_amount: Amount committed (e.g. "7500" or "full").
            payment_date: Date by which they will pay (e.g. "2026-04-15" or "tomorrow").
            payment_method: How (e.g. "bank_transfer", "app", "branch", "unknown").
        """
        _log_event("payment_commitment", self.debtor["id"], {
            "committed_amount": committed_amount,
            "payment_date": payment_date,
            "payment_method": payment_method,
            "total_debt": self.debtor["amount"],
        })
        logger.info(
            f"💰 COMMITMENT: {self.debtor['name_en']} committed {committed_amount} SAR "
            f"by {payment_date} via {payment_method}"
        )
        return f"Payment commitment recorded: {committed_amount} SAR by {payment_date}"

    @function_tool()
    async def escalate_to_human(self, ctx: RunContext, reason: str):
        """Escalate to a human supervisor.

        Args:
            reason: Why this call needs human escalation.
        """
        _log_event("human_escalation", self.debtor["id"], {
            "reason": reason,
            "persona": self.persona["name_en"],
        })
        logger.warning(f"⚠️  ESCALATION: {self.debtor['name_en']} — {reason}")
        return "Escalation logged. Inform the customer a supervisor will follow up."

    @function_tool()
    async def flag_distress(self, ctx: RunContext, description: str):
        """Flag psychological distress and stop the call immediately.

        Args:
            description: Description of the distress signals observed.
        """
        _log_event("distress_flagged", self.debtor["id"], {"description": description})
        logger.warning(f"🚨 DISTRESS: {self.debtor['name_en']} — {description}")
        self.session.generate_reply(
            instructions="End the call immediately and very gently. Say: 'I wish you well — please take care of yourself.' Do not mention the debt again."
        )
        await ctx.wait_for_playout()
        await self.hangup()
        return "Call ended due to distress."

    @function_tool()
    async def request_dnc(self, ctx: RunContext, reason: str):
        """Record a Do-Not-Contact request and end the call.

        Args:
            reason: The customer's stated reason for the DNC request.
        """
        _log_event("dnc_requested", self.debtor["id"], {"reason": reason})
        logger.info(f"🚫 DNC: {self.debtor['name_en']} — {reason}")
        self.session.generate_reply(
            instructions="Say exactly: 'Your request is recorded — we will only contact you in writing from now on. Have a good day.' Then end the call."
        )
        await ctx.wait_for_playout()
        await self.hangup()
        return "DNC recorded. Call ended."

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """End the call gracefully when the conversation reaches a natural conclusion."""
        logger.info(
            f"📞 Ending call for {self.debtor['name_en']} "
            f"({self.participant.identity if self.participant else 'unknown'})"
        )
        _log_event("call_ended_by_agent", self.debtor["id"], {"reason": "natural_conclusion"})
        await ctx.wait_for_playout()
        await self.hangup()


# ---------------------------------------------------------------------------
# Debtor & persona selection
# ---------------------------------------------------------------------------
def _pick_debtor() -> dict[str, Any]:
    """Select debtor profile based on DEBTOR_INDEX env var (1–3). Default: 1."""
    try:
        idx = int(os.getenv("DEBTOR_INDEX", "1")) - 1
        idx = max(0, min(idx, len(SAMPLE_DEBTORS) - 1))
    except ValueError:
        idx = 0
    debtor = SAMPLE_DEBTORS[idx]
    logger.info(
        f"🗂️  Debtor [{idx+1}]: {debtor['name_en']} | "
        f"Segment: {debtor['behavioral_segment']} | "
        f"Amount: {debtor['amount']:,} {debtor['currency']}"
    )
    return debtor


def _pick_persona(debtor: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """
    Pick persona. Uses PERSONA_OVERRIDE env var if set, otherwise
    defaults to the persona matching the debtor's behavioral segment.
    Returns (persona_dict, persona_key).
    """
    override = os.getenv("PERSONA_OVERRIDE", "").strip().lower()
    if override and override in PERSONAS:
        key = override
        logger.info(f"🎭 Persona OVERRIDE: {key} (debtor segment: {debtor['behavioral_segment']})")
    else:
        key = debtor.get("behavioral_segment", "cooperative")
        logger.info(f"🎭 Persona: {key} (matches debtor segment)")
    persona = PERSONAS[key]
    logger.info(f"   → {persona['name_en']} | Style: {persona['style']} | Voice: {persona['voice_id']}")
    return persona, key


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    # Load debtor — prefer dispatch metadata, fall back to env/default
    debtor: dict[str, Any] | None = None
    try:
        meta = json.loads(ctx.job.metadata or "{}")
        if meta.get("id"):
            debtor = meta
            logger.info(f"📦 Debtor from dispatch metadata: {debtor['name_en']}")
    except (json.JSONDecodeError, KeyError):
        pass

    if debtor is None:
        debtor = _pick_debtor()

    persona, persona_key = _pick_persona(debtor)

    # Create the per-turn metrics aggregator (writes CSV + JSONL to session_reports/)
    metrics_agg = TurnMetricsAggregator(session_dir=REPORTS_DIR)

    agent = BankCollectionAgent(
        debtor=debtor,
        persona=persona,
        persona_key=persona_key,
        metrics_agg=metrics_agg,
    )

    session = AgentSession(
        turn_handling={
            "endpointing": {
                # Valid modes: "fixed" | "dynamic"
                # "dynamic" allows the agent to extend the silence window
                # when it predicts the user hasn't finished speaking.
                "mode": "dynamic",
                "min_delay": 0.2,
                "max_delay": 1.0,
            },
            "interruption": {
                "enabled": True,
                # Valid modes: "adaptive" | "vad"
                "mode": "adaptive",
                "min_words": 2,
            },
        },
        # STT — Deepgram Nova-3 (direct key, bypasses LiveKit gateway)
        stt=deepgram.STT(model="nova-3", language="ar-SA"),
        # LLM — driven by EXPERIMENT_CONFIG (swap models at the top of the file)
        llm=_build_llm(),
        # TTS — Faseeh (reads FASEEH_API_KEY from .env; model from EXPERIMENT_CONFIG)
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id=persona["voice_id"],
            model=EXPERIMENT_CONFIG["tts_model"],
            stability=0.75,
            speed=0.9,
        ),
        # VAD — prewarmed in prewarm() with tuned silence/speech thresholds
        vad=ctx.proc.userdata["vad"],
    )

    # Wire the metrics aggregator to receive all component metrics
    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics_agg.on_metrics_collected(ev)

    await session.start(agent=agent, room=ctx.room)
    logger.info("✅ Agent ready — connect via LiveKit Playground to begin.")

    participant = await ctx.wait_for_participant()
    logger.info(f"👤 Participant joined: {participant.identity}")
    agent.set_participant(participant)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def prewarm(proc: JobProcess):
    # Pre-load Silero VAD with tuned parameters.
    # min_speech_duration=0.05s  → respond to very short utterances quickly
    # min_silence_duration=0.4s  → slightly tighter than default 0.55s to reduce EOU delay
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.4,
    )
    logger.info(
        f"🧪 Experiment: {EXPERIMENT_CONFIG['label']} | "
        f"LLM={EXPERIMENT_CONFIG['llm_model']} | "
        f"TTS={EXPERIMENT_CONFIG['tts_model']}"
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="bank-collection-agent",
            num_idle_processes=1,
        )
    )