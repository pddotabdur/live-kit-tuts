# Collections call flow — نورا (Nora) outbound agent

This document describes the call flow implemented in `smart_agent.py` at the
current revision. The agent runs an outbound SIP call, drives the customer
through a stage machine, and produces a structured outcome.

## High-level state machine

```
                ┌──────────────────────┐
                │  SIP dial + connect  │
                └──────────┬───────────┘
                           ▼
        ┌─────── Stage 1: Right-party verify ────────┐
        │  customer == named person?                 │
        ├──────────┬─────────┬─────────┬─────────────┤
        ▼          ▼         ▼         ▼             ▼
     right      wrong       busy      DNC         deceased
       │          │           │         │             │
       │          ▼           ▼         ▼             ▼
       │   knows person?    schedule   close       close
       │     yes / no       callback   (DNC)       (death)
       │      │   └─► close (wrong)
       │      ▼
       │   collect mobile → close (referred)
       ▼
   Stage 1a: "Is now a good time?"
       ├── yes → Stage 1d: ID verify (last 4 digits, 2 attempts)
       │           ├── match    → Stage 2
       │           └── mismatch → close (id_mismatch)
       └── no  → schedule callback → close (busy_callback)

   Stage 2: Brief debt mention (one sentence, no "why?")
       ├── already_paid          → close (paid)
       └── proceed_to_negotiation → Stage 3

   Stage 3: Discovery-led negotiation (single agent — no ladder)
       │   "كم تقدر تدفع اليوم؟"  (NEVER name a number first)
       │
       ├── full_payment(when_iso)            → Stage 4 recap
       ├── partial_committed(initial_amt,    → Stage 4 recap
       │     initial_date, rest_amt,
       │     rest_date)
       ├── already_paid                      → close (paid)
       ├── vague_response (after 2 nudges)   → Reschedule callback → Stage 4 recap
       ├── refuses_payment                   → close (refusal)
       ├── disputes_debt                     → close (dispute)
       └── unclear                           → re-ask same question

   Stage 4: Recap & confirm  (one short sentence: amount + dates + "تمام كذا؟")
       ├── recap_confirmed              → close (ok)
       ├── recap_minor_correction(text) → close (ok)  (small detail tweak)
       └── wants_to_renegotiate         → BACK to Stage 3 (full reset)
```

## Stage-by-stage detail

### Stage 1 — Right-party verification
- Greets, identifies as نورا from شركة توافق representing بنك stc, asks for
  the named customer.
- Tools: `right_party`, `wrong_party`, `caller_busy`, `do_not_call`,
  `customer_deceased`.
- Wrong-party branch: ask if they know the person → if yes, collect a mobile
  number; if no, polite close.
- Busy / DNC / death branches close with appropriate parting line.

### Stage 1a — "Is now a good time?"
- One-line check before pulling account info.
- Tools: `good_time_now`, `bad_time_now` (→ schedule callback), `unclear`.

### Stage 1d — ID verification
- Asks for the last 4 digits of the national ID.
- Up to **2 attempts**: first mismatch is forgiven; second triggers
  `id_mismatch` close.
- Tools: `digits_provided(digits)`, `unclear`.

### Stage 2 — Brief debt mention
- Single sentence stating the outstanding amount. **No "why hasn't it been
  paid?", no SIMAH explanation, no preamble.**
- Tools: `already_paid`, `proceed_to_negotiation`.

### Stage 3 — Discovery-led negotiation
**Golden rule**: Nora **never names an amount first**. Always ask, then judge.

Internal thresholds (computed per call data, never read aloud):
- **Floor**  = 5 % of total (e.g. 500 SAR on 10 000)
- **Ideal**  = 10 % of total (e.g. 1 000 SAR on 10 000)

Conversation pattern (one short sentence per turn):

1. Open: *"كم تقدر تدفع اليوم؟"*
2. Customer offers `A`:
   - `A` covers full → ack, ask when, call `full_payment(when_iso)`
   - `A` ≥ ideal → ack, lock initial date, move on
   - floor ≤ `A` < ideal → soft push to ideal **once**, then accept
   - `A` < floor → "small vs the total, can you increase?" — no number
     from agent yet. If still under floor after his second number, agent
     **may** suggest a stretch range (only after he has named at least
     one number, and only once).
3. Lock initial amount + initial date.
4. Ask when the **remainder** will be paid; aim for ~14 days. If much later,
   ask once for sooner, then accept.
5. Call `partial_committed(initial_amount, initial_date_iso, rest_amount, rest_date_iso)`.

Other Stage 3 tools: `already_paid`, `vague_response`, `refuses_payment`,
`disputes_debt`, `unclear`.

### Stage 4 — Recap & confirm
- One sentence: amounts in Arabic words + dates in Arabic words + *"تمام كذا؟"*.
- Tools: `recap_confirmed`, `recap_minor_correction(text)`,
  `wants_to_renegotiate` (resets and re-enters Stage 3).

### Closing
- Parameterised by intent: `ok / paid / busy / busy_callback / dnc / death /
  wrong_party / referred / id_mismatch / refusal / dispute`.
- Each intent maps to a one-sentence parting line, then SIP hangup.

## Persona rules currently enforced

| Rule | Where it lives |
|---|---|
| ONE short sentence per turn (8–14 words) | PERSONA — ABSOLUTE RULES |
| Never say *والله* | PERSONA — ABSOLUTE RULES |
| *طال عمرك* only in opening greeting | PERSONA — ABSOLUTE RULES |
| Discovery, never dictate | PERSONA + Stage 3 task |
| Mid-call salaam → reply *أبشر*, do NOT reciprocate | PERSONA — ABSOLUTE RULES |
| One-word acks only (*أبشر / زين / تمام / طيب*) | PERSONA |
| No SIMAH / grace-period lecture | Stage 2 + Stage 3 task |
| No empathy monologue on hardship | PERSONA |

## Output / observability

For each call we log to stdout:

- `USER  '<final transcript>'`
- `AGENT '<full LLM output>'` — the script the TTS will read
- `TOOL  <name>  args=<json>`
- `EOU  <s>` — end-of-utterance + transcription delay
- `LLM  ttft <s> prompt=<n> (cached <n>, <%>) completion=<n>`
- `TTS  ttfb <s> audio <s> chars=<n>`
- `TURN total <s>  (EOU + TTFT + TTFB)` — what the user actually feels

After the call ends: room is deleted by `ClosingAgent.hangup()`.
`CallData.outcome` and `CallData.commitment` reflect the final state but are
**not yet persisted to disk** — that's the next gap if you want a per-call
audit trail.

## Plugins in use

- **STT**: `hamsa_livekit.STT(language="ar")` — non-streaming, adds ~2 s/turn.
  (Deepgram nova-3 ar-SA and Soniox are imported but not wired.)
- **LLM**: OpenAI gpt-4.1, temperature 0.4. Prompt cache hit climbs to ~95 %
  from turn 4 onward.
- **TTS**: Faseeh (Munsit) `faseeh-v1-preview`, voice
  `ybQaNl0nzt9TjN3Oh1zzyNgp`, stability 0.75, speed 0.9.
- **VAD**: silero, min_speech 50 ms, min_silence 400 ms.
- **Endpointing**: dynamic, 0.2–1.0 s.
- **SIP**: outbound trunk via `SIP_OUTBOUND_TRUNK_ID` env var.

## Batch testing

Use `batch_dispatch.py` with `customers.xlsx` (columns:
`phone_number, name, amount, debt_date, national_id_last4`):

```bash
uv run python batch_dispatch.py --init        # write a template xlsx
uv run python batch_dispatch.py               # sequential (default)
uv run python batch_dispatch.py --parallel    # fire all with --gap spacing
```

Sequential mode polls each room until the SIP participant has hung up before
dispatching the next call — avoids Faseeh rate-limit collisions.

## Known gaps

1. **STT latency** ~2 s/turn (Hamsa) — biggest single source of perceived
   slowness. Switching to Deepgram nova-3 streaming or Soniox is a one-line
   change.
2. **No per-call outcome persistence** — outcomes live only in process
   memory.
3. **No Faseeh 429 fallback** — if Munsit rate-limits, the call goes
   silent. Confirmed under 3 concurrent calls.
4. **No human-handoff path** — *"حولني لمشرف"* has no tool.
5. **No language-drift handling** — if customer answers in English /
   Egyptian, agent stays Najdi.
6. **No idle / silence timeout** — if customer says nothing for 15 s, no
   recovery prompt.
