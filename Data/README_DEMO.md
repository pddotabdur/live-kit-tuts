# Bank AI Debt Collection Demo

An AI-powered voice agent for automated debt collection, built on [LiveKit](https://livekit.io) with Arabic/multi-dialect support. This demo showcases the core capabilities needed for a bank's AI collection system RFP.

## What This Demo Shows

| Feature | How It Works |
|---------|-------------|
| **AI Voice Calls** | Outbound calls via LiveKit SIP + Twilio, natural Arabic conversation |
| **Identity Verification** | Name confirmation + last-4-digits of national ID verification |
| **Behavioral Classification** | Agent adapts persona, tone, and strategy per debtor segment |
| **Multi-Persona Agents** | 3 virtual agents (Noura/Sultan/Sara) with different voice IDs |
| **Structured Negotiation** | Debt presentation → reason inquiry → escalated negotiation flow |
| **Function Tools & Audit Trail** | Every interaction logged with timestamps via 7 callable tools |
| **Guardrails** | No hallucinated financials, DNC compliance, distress detection |
| **Human-in-the-Loop** | Escalation tool for critical cases requiring supervisor review |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Dispatch     │────▶│  LiveKit      │────▶│  SIP/Twilio  │
│  Script       │     │  Cloud        │     │  (Phone Call) │
│  (debtor data)│     │  (Room+Agent) │     │              │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │  Agent        │
                     │  Session      │
                     ├──────────────┤
                     │ STT: Deepgram │◀── LiveKit Inference
                     │ LLM: GPT-4o-m│◀── LiveKit Inference
                     │ TTS: Faseeh   │◀── Direct Plugin
                     │ VAD: Silero   │◀── Local
                     ├──────────────┤
                     │ Function Tools│
                     │ • log_interaction
                     │ • record_payment_commitment
                     │ • escalate_to_human
                     │ • flag_distress
                     │ • request_dnc
                     │ • end_call
                     │ • detected_answering_machine
                     └──────────────┘
```

## Files

| File | Description |
|------|-------------|
| `agent_bank_demo.py` | Main agent — prompt, personas, function tools, session config |
| `dispatch_bank_demo.py` | Dispatch script — sends debtor profile + triggers outbound call |
| `sample_debtors.json` | 3 demo debtor profiles (cooperative, avoidant, distressed) |
| `.env` | API keys for LiveKit, OpenAI, Deepgram, Faseeh, etc. |

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- LiveKit Cloud account with API keys
- SIP trunk configured (Twilio) with outbound trunk ID
- API keys: OpenAI, Deepgram, Faseeh (Munsit)

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Ensure your `.env` file has these keys:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
FASEEH_API_KEY=...
SIP_OUTBOUND_TRUNK_ID=ST_...
```

### 3. Start the Agent

```bash
# Activate venv and start in dev mode (auto-reload on changes)
source .venv/bin/activate
python agent_bank_demo.py dev
```

You should see:
```
INFO  livekit.agents  registered worker
                      {"agent_name": "bank-collection-agent", ...}
```

### 4. Dispatch a Call

In a **separate terminal**:

```bash
source .venv/bin/activate

# Call with cooperative debtor (default)
python dispatch_bank_demo.py

# Call with avoidant debtor (firm approach)
python dispatch_bank_demo.py --debtor 2

# Call with distressed debtor (empathetic approach)
python dispatch_bank_demo.py --debtor 3

# List all available demo debtors
python dispatch_bank_demo.py --list

# Override phone number
python dispatch_bank_demo.py --debtor 1 --phone +966501234567
```

## Demo Debtor Profiles

### 🟢 Debtor 1: Mohammed Al-Harbi (Cooperative)
- **Amount:** 15,000 SAR | **Product:** Personal Loan | **Status:** Active
- **Persona:** Noura (warm female voice)
- **Strategy:** Friendly approach, flexible payment arrangement

### 🟡 Debtor 2: Khalid Al-Otaibi (Avoidant)
- **Amount:** 48,500 SAR | **Product:** Credit Card | **Status:** Suspended
- **Persona:** Sultan (firm male voice)
- **Strategy:** Direct approach, early disclosure of consequences, 5 prior missed contacts

### 🔴 Debtor 3: Noura Al-Qahtani (Distressed)
- **Amount:** 3,200 SAR | **Product:** Utility Bill | **Status:** Active
- **Persona:** Sara (empathetic female voice)
- **Strategy:** Maximum empathy, minimum pressure, distress detection active

## Conversation Flow

```
1. Greeting → "السلام عليكم، هل أنا أكلم محمد الحربي؟"
2. Identity Check → Verify last 4 digits of national ID
3. Debt Presentation → Total amount (words only) + reason question
4. Wait for Response → No pressure before hearing the reason
5. Early Disclosure → Based on amount threshold (<200 / 200-500 / >500)
6. Negotiation → Full today → Half + rest → Flexible exception
7. Resolution → Payment commitment / Escalation / DNC / Close
```

## Audit Trail

Every interaction generates structured audit events logged to the terminal:

```
📋 AUDIT [call_started] {"timestamp": "...", "persona": "Noura", "segment": "cooperative", ...}
📋 AUDIT [interaction]  {"outcome": "identity_confirmed", "notes": "...", ...}
💰 COMMITMENT: محمد الحربي committed 7500 SAR by 2026-04-20 via bank_transfer
📋 AUDIT [call_ended]   {"duration_seconds": 245, "total_interactions": 5, ...}
```

## Key Technical Decisions

### Why LiveKit Inference for STT & LLM?
Using `inference.STT()` and `inference.LLM()` instead of direct plugin libraries routes through LiveKit's co-located infrastructure, reducing network hops and latency.

### Why Faseeh for TTS?
Faseeh provides the most natural Najdi Arabic voices. It's called as a direct plugin since it's not available through LiveKit Inference.

### Why Multi-Persona?
The bid requires 10+ virtual personas. This demo shows 3 with distinct voices and tones, selected automatically based on the debtor's behavioral classification. Production would expand this.

## Testing Without SIP

If SIP/Twilio is not configured, you can test the agent through the **LiveKit Playground**:

1. Start the agent: `python agent_bank_demo.py dev`
2. Go to [agents-playground.livekit.io](https://agents-playground.livekit.io)
3. Connect with your LiveKit Cloud credentials
4. The agent will attempt to work in room mode (no outbound dial)

> **Note:** The playground test won't trigger the SIP dialing flow but will let you interact with the agent's conversational capabilities directly.
