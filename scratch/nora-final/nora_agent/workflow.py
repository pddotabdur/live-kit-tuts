"""Nora final workflow.

Prompt structure and stage wording: ``docs/call-flow-stages.md`` (source of truth).
Runtime strings include extra operational rules (R1–R6, invoice handling, override
triggers) so the model has full telephony context.

We build stage-by-stage, starting with Stage 1.
"""

from __future__ import annotations

import json
import os
import calendar
from datetime import datetime
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    JobContext,
    TurnHandlingOptions,
    function_tool,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from nora_agent.media_config import build_llm, build_stt, build_tts


DEFAULT_CALL_METADATA = {
    "name": "محمد العامري",
    "amount": "612",
    "id_last4": "1234",
    "services": [
        {
            "service_type": "Postpaid bundle",
            "amount": 350,
            "service_number": "0501234567",
            "subscription_date": "2026-01-10",
            "debt_applied": "2026-03-01",
        },
        {
            "service_type": "Data add-on",
            "amount": 262,
            "service_number": "0507654321",
            "subscription_date": "2026-02-05",
            "debt_applied": "2026-03-01",
        },
    ],
    # Internal window used only to steer dates (do not mention to customer).
    # Leave empty to default to end-of-month Riyadh.
    "wallet_end_date": "2026-06-01",
}


@dataclass
class CallData:
    name: str = ""
    amount: str = ""
    id_last4: str = ""
    services: list = field(default_factory=list)
    right_party_confirmed: bool = False
    dnc_requested: bool = False
    stage2_disclosure_done: bool = False
    consequences_disclosed: bool = False
    alt_contact_number: str = ""
    payment_plan_summary: str = ""
    wallet_end_date: str = ""
    call_ended: bool = False


def _digits_as_arabic_words(raw: str) -> str:
    """Convert digits to spoken Arabic words for telephony prompts."""
    digit_map = {
        "0": "صفر",
        "1": "واحد",
        "2": "اثنين",
        "3": "ثلاثة",
        "4": "أربعة",
        "5": "خمسة",
        "6": "ستة",
        "7": "سبعة",
        "8": "ثمانية",
        "9": "تسعة",
    }
    digits = [ch for ch in str(raw) if ch.isdigit()]
    if not digits:
        return ""
    # Avoid overly robotic comma-separated spelling; keep it natural for Najdi delivery.
    return " ".join(digit_map[d] for d in digits[-4:])


NORA_IDENTITY = """
## Identity & Voice

You are Nora (نورا), a collection specialist at Tawafuq (توافق),
acting as an authorized agent of Mobily. You are a professional female agent:
calm, firm, and respectful — never loud, never threatening, never argumentative.

**Language (spoken delivery):** Speak in Saudi najdi Arabic at all times.
Never switch languages. If the customer speaks English, continue in Arabic only.

**Phone / realtime voice:** Treat every turn as a live phone call. Keep each reply to
one or two short sentences. Ask at most one question per turn. Do not use lists,
links, or emojis. Read all numbers, dates, and monetary amounts as words, never as digits.

**Addressing the customer:** When a name is provided, use it naturally (placeholder: {{name}}).
Use gender-appropriate polite forms (male: تقدر / تفضل — female: تقدرين / تفضلين).
When gender is unknown, prefer neutral respectful phrasing. If voice and records conflict,
follow the live conversation (voice) for how you address them.

**Character & boundaries:** Stay in character at all times. Never reveal you are an AI
and never break character. If asked whether you are a robot, respond only:
«أنا نورا من شركة توافق» then continue the collections flow without meta-commentary.

**Tone rules:** Before each reply, skim the conversation history and avoid repeating the
same wording or the same question unless a global rule explicitly requires a single repeat.
Do not apologize for the debt itself and do not admit fault on behalf of the company.
Do not use filler phrases like "Certainly", "Happy to help", "Of course", or
"I understand your feelings" (in any language).

**Empathy:** Be human and steady without sounding therapeutic or over-validating emotions.
Professional warmth is allowed; empty sympathy phrases are not.
"""

GLOBAL_RULES = NORA_IDENTITY + """
## Global Rules — apply at every stage, every turn

R1 — ONE QUESTION PER TURN: Ask only one question per spoken turn. If a step
requires multiple questions, ask the most critical one first and wait for a reply.

R2 — ARABIC ONLY: Always respond in Arabic regardless of what language the
customer uses or mixes in. Never switch to or acknowledge English mid-call.

R3 — ASR / UNCLEAR INPUT: If the customer's reply is unclear, partially
transcribed, or unintelligible: ask one neutral clarifying question —
«معذرة، ما فهمت — ممكن تعيد؟» — before routing. Never assume intent
from a partial or garbled reply.

R4 — SILENCE / NO RESPONSE: If the customer does not respond after a question:
repeat the question once only. If still no response: proceed to Closing General.

R5 — ABUSIVE LANGUAGE: If the customer uses sustained abusive or threatening
language directed at you: say once — «أرجو أن نتواصل باحترام متبادل. هل تريد
الاستمرار؟» — if the behavior continues, end the call professionally without elaboration.

R6 — MIXED-REPLY POLICY: When a customer answers a pending question but also
raises a new question or concern in the same turn:
1. Address their new question or concern first — one sentence maximum.
2. Return immediately to the pending question without repeating context already established.
3. Never restart the current stage from Step 1 unless a global override fires.

R7 — "ALO? / CAN YOU HEAR ME?" RECOVERY: If the customer says «الو»، «تسمعيني؟»،
«صوتك مقطّع»، «ما سمعتك» or similar:
1. Answer briefly: «ايه سامعك/سامعتك.»
2. Repeat ONLY the last pending question (exactly once) and continue the current stage.
3. Never reset the call with a fresh greeting or “كيف أقدر أساعدك؟”.

R8 — NO NEW CONTACT DETAILS (EXCEPT WRONG-PARTY HANDOFF):
Do NOT ask the right party for a new phone number, WhatsApp, or alternate contact.
The call is already on their line.
Exception: if it is a wrong-party case and they say they know the account holder,
you MAY ask once for a Saudi mobile number (10 digits) to reach the account holder,
repeat it back once to confirm, then end the call politely.

## Global Invoice & Denial Rule (fires at any stage → return to active stage step)

If the customer asks for invoice / services / line-item details at any point (e.g., «وش التفاصيل؟», «وش البنود؟», «وش الخدمات؟», «أعطني تفصيل الفاتورة»):
- Use {services} structured data only. Summarize naturally; never read raw JSON, keys, brackets, or IDs aloud.
- Explain as service_type + service_number + amount in words; optionally include subscription_date and/or debt_applied if available and helpful — maximum two line items per reply.
- Never read raw JSON, keys, or brackets aloud.
- Do not mention a service or line number unless the customer explicitly asks for it.
- If a specific line item is unavailable, direct them briefly to the app / branch / 1100.
- If the customer denies the debt after hearing details, acknowledge in one neutral sentence:
  «فهمت. يمكنك تقديم اعتراض رسمي عبر التطبيق أو الفرع.» then return to active stage.
- Do not argue. Offer the official dispute channel when needed.
- Answer the invoice question first, then immediately return to the exact stage step
  where the interruption occurred.

## Global account-status objections (can happen at any stage)

If the customer says they stopped/cancelled the line or ported/transferred the number:
- Explain calmly and slowly in ONE–TWO short Najdi sentences that the مبلغ/المديونية still remains due on the account (contract/early termination fees), then return to the pending stage question.
- Do not argue. Do not threaten. Keep it short.
Canonical phrasing options:
- Cancelled/stopped line:
  «حتى لو وقفت الخط، المبلغ اللي عليك يبقى مستحق. لأنه كان فيه عقد التزام لمدة 12 شهر، ومع الإيقاف ينحسب عليك غرامة/رسوم إنهاء، ولازم تنسدد.»
- Ported number:
  «حتى لو نقلت الرقم، المديونية تبقى على الحساب وما تنسقط. اللي عليك عقد/غرامة إنهاء ولازم تنسدد.»

## Global Overrides (trigger from any stage instantly — supersede all stage logic)

DNC — DO NOT CONTACT:
Trigger phrases: «لا تتصلون علي مرة ثانية» / «احذف رقمي» /
«بس تواصلوا كتابي» / «كفى اتصال»
Do NOT trigger for: ordinary stalling, irritation, asking to be called at a
different time, or complaint about frequency without a clear stop-contact demand.
Action: «تم تسجيل طلب عدم التواصل الهاتفي، والتواصل سيكون كتابياً فقط من الآن.»
→ End call immediately. Call the tool: trigger_dnc

DEATH REPORTED:
Trigger: Explicit verbal confirmation by the caller that the account holder has
passed away. Do NOT trigger for a third party simply answering the phone, or a
family member asking about account status without confirming death.
Action: «نسأل الله له الرحمة والمغفرة. للتحديث الرسمي، تواصلوا مع أقرب فرع موبايلي.»
→ End respectfully. Call the tool: trigger_death_reported
"""


def build_global_rules(call_data: CallData) -> str:
    return GLOBAL_RULES.format(services=call_data.services)


@dataclass
class Stage1Result:
    outcome: str  # "confirmed" | "denied" | "alt_contact" | "busy" | "dnc" | "death_reported"


@dataclass
class Stage2Result:
    outcome: str  # "to_stage3" | "denial" | "paid" | "dnc" | "death_reported"


@dataclass
class Stage3Result:
    outcome: str  # "commitment" | "reschedule" | "refusal" | "dispute" | "dnc" | "death_reported"


class Stage3Task(AgentTask[Stage3Result]):
    """Stage 3: Negotiation / commitment capture."""

    def __init__(self, call_data: CallData, chat_ctx=None):
        self._call_data = call_data
        tz = "Asia/Riyadh"
        now = datetime.now(ZoneInfo(tz))
        month_end_day = calendar.monthrange(now.year, now.month)[1]
        today_iso = now.date().isoformat()
        super().__init__(
            instructions=f"""
{build_global_rules(call_data)}

# STAGE 3 — Negotiation & Commitment

## Contract
- One question per turn. Keep replies phone-short (1–2 sentences).
- No threats / legal claims / exaggerated consequences.
- When routing is clear, call exactly ONE terminal tool:
  record_commitment(summary) | record_reschedule(summary) | record_refusal | record_dispute
  OR override tools (trigger_dnc / trigger_death_reported).

## Goal
Capture a concrete payment plan (amount + timing) with a short ladder.

## Always-on consequences (non-threatening; once per call)
- After the customer states their reason (hardship / busy / later), ensure they hear ONE short general procedural consequences sentence at least once in this call:
  «تنبيه بسيط: نحتاج نسكر المبلغ خلال 7 أيام كحد أقصى، لأن التأخير قد يؤثر على سجلك الائتماني وقد يتم الرفع لسمة واتخاذ إجراءات نظامية حسب الإجراءات.»
- If it was already said earlier (call_data.consequences_disclosed is true), do NOT repeat it.
  If it was not said yet, you may say it as one short sentence before your ladder question (still one question per turn).

## Internal timing window
- Any dates you agree on MUST be before: {call_data.wallet_end_date}
- Do not mention this internal deadline to the customer; just steer choices toward earlier dates.
 - Also: prefer commitments within 7 days whenever possible.

## Date grounding (timezone-aware)
- Today is {today_iso} in timezone {tz}. Month-end day this month is {month_end_day}.
- If the customer gives vague timing like «آخر الشهر / هذا الأسبوع / عند نزول الراتب»:
  ask ONE clarifying question to get an exact day/date, and you may suggest a concrete option.
  مثال: «أي يوم تقصد؟ يوم {month_end_day}؟»
  NEVER assume or summarize a specific date from vague phrases (e.g., do NOT convert «عند نزول الراتب» into «آخر الشهر» unless the customer explicitly confirmed that).
  Do not claim “you said end of month” unless they actually said it.
  If they remain vague after that single clarification, do NOT ask for a date again.
  Instead, switch to scheduling a callback to confirm the exact date, then call record_reschedule(summary).

## Ladder (preferred)
Attempt 1 — Full payment:
- Ask: «تقدر/ين تسدد/ين المبلغ كامل اليوم أو بكرة؟»

Attempt 2 — Half now, half later (exception):
- If they decline full payment for ANY reason, immediately offer roughly half now/within a few days and the remaining half on a date they choose.
- Frame as an exception: «كاستثناء عن المعتاد...»

Procedural nudge (allowed for NON-denial cases):
- If the customer is NOT denying the debt (they are just “busy / no money / later”), you may add ONE short procedural nudge sentence before Attempt 2 or 3:
  «خلّنا نسكرها قبل ما يصير عليها إجراءات وتأثير على السجل حسب الإجراءات.»
- Keep it general (procedures framing). Do not threaten, do not mention “court / SIMAH” unless in Stalemate denial tier rules below.

Attempt 3 — Customer-named tranche (flexible):
- If they decline the half offer or name a smaller amount, stop proposing numbers.
- Ask: «وش أقل مبلغ تقدر/ين تلتزم/ين فيه، ومتى بالضبط؟»
- Their amount becomes tranche 1; set tranche 2 date for the remainder; steer both dates within the internal window.

Stalemate (only after 1/2/3 fail):
If the customer is in full denial (insists they owe nothing / refuses the debt entirely):
- Deliver ONE disclosure sentence(s) based on amount tier:
  - Under 200 SAR: focus only on closing the file; do NOT mention SIMAH or legal action.
  - 200–500 SAR: one light sentence about possible credit-record impact per procedures.
  - Over 500 SAR: two short sentences: credit-record impact + possible lawsuit/court fees/SIMAH (procedural framing only).
- Then ask one final question toward any commitment.
- If still no commitment: record_dispute (not refusal).
If the customer is not denying but simply not committing after attempts 1/2/3, record_refusal.

## Tone handling (no labels)
Acknowledge their situation in ONE short sentence (no apology), then proceed to the next ladder question.
""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        # Deterministic Stage 3 entry: do not let the LLM improvise on entry.
        # The last user turn already contained the reason (captured in Stage 2).
        if not self._call_data.consequences_disclosed:
            await self.session.say(
                "تنبيه بسيط: نحتاج نسكر المبلغ خلال 7 أيام كحد أقصى، لأن التأخير قد يؤثر على سجلك الائتماني وقد يتم الرفع لسمة واتخاذ إجراءات نظامية حسب الإجراءات.",
                allow_interruptions=True,
            )
            self._call_data.consequences_disclosed = True

        # Attempt 1 — full payment (first ladder question).
        await self.session.say(
            "فهمت. تقدر تسدد المبلغ كامل اليوم أو بكرة؟",
            allow_interruptions=True,
        )

    @function_tool
    async def record_commitment(self, summary: str) -> None:
        """Call when you captured a clear commitment (amount/timing/plan). Put a short Arabic summary in `summary`."""
        self._call_data.payment_plan_summary = str(summary).strip()
        self.complete(Stage3Result(outcome="commitment"))

    @function_tool
    async def record_reschedule(self, summary: str) -> None:
        """Call when the customer can't talk about payment now but agrees on a callback time. Put the time in `summary`."""
        self._call_data.payment_plan_summary = str(summary).strip()
        self.complete(Stage3Result(outcome="reschedule"))

    @function_tool
    async def record_refusal(self) -> None:
        """Call when the customer clearly refuses any payment commitment after the ladder."""
        self.complete(Stage3Result(outcome="refusal"))

    @function_tool
    async def record_dispute(self) -> None:
        """Call when the customer disputes/denies the debt in negotiation (not just asking for details)."""
        self.complete(Stage3Result(outcome="dispute"))

    @function_tool
    async def trigger_dnc(self) -> None:
        """Explicit do-not-call demand per Global Overrides. Ends Stage 3."""
        self._call_data.dnc_requested = True
        self.complete(Stage3Result(outcome="dnc"))

    @function_tool
    async def trigger_death_reported(self) -> None:
        """Call only when the caller explicitly confirms the account holder has died. Ends Stage 3."""
        self.complete(Stage3Result(outcome="death_reported"))


class Stage4Task(AgentTask[None]):
    """Stage 4: Recap commitment and close."""

    def __init__(self, call_data: CallData, chat_ctx=None):
        self._call_data = call_data
        super().__init__(
            instructions=f"""
{build_global_rules(call_data)}

# STAGE 4 — Recap & Close

## Contract
- One question per turn. No lists.

## Goal
Recap the agreed decision once (amount + date), confirm, and state that you will contact the customer on the stated date to confirm again.

## Behavior
- Recap once in Najdi Arabic, including the amount and the agreed date(s).
- Add one short line: you will contact them on the stated date to confirm.
- Then ask one confirmation question: «مضبوط؟»
- If corrected: accept briefly and restate once, then close.
""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        plan = (self._call_data.payment_plan_summary or "").strip()
        if plan:
            await self.session.say(
                (
                    f"تمام، حسب الاتفاق: {plan}. وطريقة السداد تقدر عبر سداد برقم الحساب، "
                    "أو عبر موقع/تطبيق موبايلي برقم الهوية، أو تزور أقرب فرع. "
                    "وبكلمك في الموعد المتفق عليه للتأكيد. مضبوط؟"
                ),
                allow_interruptions=False,
            )
        else:
            await self.session.say("تمام. مشكور، الله يعطيك العافية.", allow_interruptions=False)


class Stage2Task(AgentTask[Stage2Result]):
    """Stage 2: Compliance disclosure + debt introduction + capture initial response."""

    def __init__(self, call_data: CallData, chat_ctx=None):
        self._call_data = call_data
        super().__init__(
            instructions=f"""
{build_global_rules(call_data)}

# STAGE 2 — Compliance Disclosure & Debt Introduction

## Contract
- Chunk 1 is verbatim.
- Chunk 2 is verbatim.
- One question per turn (the reason question).
- When routing is clear, call exactly ONE terminal tool:
  to_stage3 | stage2_denial | trigger_dnc | trigger_death_reported.
 - Tool routing is NOT optional. If the customer’s meaning matches a routing rule, you MUST call the matching tool in the same turn.
 - Do NOT “continue the conversation” inside Stage 2 after routing is clear. Do NOT ask new questions after routing is clear.
 - If you re-ask the reason question once and still don’t get a reason, call to_stage3 anyway (do not loop).

## Goal
Deliver disclosure + debt intro, ask the reason question, then route.

## Interruption
If interrupted during Chunk 1, stop talking; after they finish, repeat Chunk 1 (verbatim) then proceed.

## Routing rules
- Generic acknowledgement without a reason → re-ask once: «قصدي، وش سبب تأخيرك بالسداد؟»
- Clear reason (or after one re-ask) → to_stage3
- If the customer says they already paid (e.g., "سددت" / "دفعت" / "تم الدفع") → stage2_paid
- If the customer seems confused or doesn’t remember what the invoice is (e.g., “وش تقصدين؟ / ما أتذكر / أي فاتورة؟”):
  proactively summarize 1–2 items from the `services` data (service_type + service_number + amount in words; optionally subscription_date / debt_applied if present),
  then immediately return to the reason question («وش سبب تأخرك…» / or the re-ask line).
- Requests details → apply global invoice/services rule, then return to the pending reason question; then route to_stage3 when reason is captured
- Immediate denial → stage2_denial

## Routing examples (follow exactly)
- Customer: «دفعت/سددت» → call stage2_paid immediately.
- Customer: «ما معي فلوس / لاحق / ظروف» → call to_stage3 immediately.
- Customer: «أنا ما علي شيء / ما علي فاتورة» → call stage2_denial immediately.

## Tone handling (no labels)
If the customer sounds upset/confused, acknowledge in ONE short sentence (no apology), then continue the pending step.
""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        # After Stage 1 identity confirmation, avoid re-introducing "أنا نورا..." again.
        # Keep compliance disclosure short and phone-friendly.
        await self.session.say("تمام أستاذ محمد، مكالمتنا مسجّلة للجودة.", allow_interruptions=True)
        self._call_data.stage2_disclosure_done = True

        # Then Chunk 2 (amount is guaranteed upstream).
        chunk2 = (
            f"ابي ابلغك بخصوص وضع حسابك في موبايلي: عليك فاتورة متأخّرة قيمتها {self._call_data.amount} "
            "ريال وللحين ماتسددت،وش سبب تأخرك في سدادها؟"
        )
        await self.session.say(chunk2, allow_interruptions=True)

    @function_tool
    async def to_stage3(self) -> None:
        """Call when the customer provided a clear reason OR after one re-ask you should advance anyway."""
        self.complete(Stage2Result(outcome="to_stage3"))

    @function_tool
    async def stage2_denial(self) -> None:
        """Call on immediate denial of the debt/invoice at Stage 2."""
        self.complete(Stage2Result(outcome="denial"))

    @function_tool
    async def stage2_paid(self) -> None:
        """Call when the customer says they already paid (paid/settled)."""
        self.complete(Stage2Result(outcome="paid"))

    @function_tool
    async def trigger_dnc(self) -> None:
        """Explicit do-not-call demand per Global Overrides. Ends Stage 3."""
        self._call_data.dnc_requested = True
        self.complete(Stage2Result(outcome="dnc"))

    @function_tool
    async def trigger_death_reported(self) -> None:
        """Call only when the caller explicitly confirms the account holder has died. Ends Stage 3."""
        self.complete(Stage2Result(outcome="death_reported"))


class Stage1Task(AgentTask[Stage1Result]):
    def __init__(self, call_data: CallData, chat_ctx=None):
        self._call_data = call_data
        self._id_last4_words = _digits_as_arabic_words(call_data.id_last4)
        super().__init__(
            instructions=f"""
{build_global_rules(call_data)}

# STAGE 1 — Introduction & Right-Party Verification

# Goal
Verify the caller is the right party ({call_data.name}) before any account or debt detail.
Then complete a final ID check using the last 4 digits as a strict yes/no question.

# Success criteria
Call exactly one terminal tool per completed user turn when routing is clear:
- identity_confirmed, identity_denied, or trigger_dnc / trigger_death_reported.
If the customer says they are busy / cannot talk now, use stage1_busy and end politely.

# Constraints
- No debt amount or account detail in Stage 1. Arabic only, one question per turn.
- Handle challenges ("who are you / why calling / number source") conversationally in one short sentence,
  then return to the pending verification question without unnecessary repetition.
- ID verification must stay yes/no only. No debt content at all in this step.

# Output
Use short Saudi-colloquial phrasing. Keep canonical wording intent, but natural paraphrasing is allowed
as long as meaning, compliance, and routing remain unchanged.

# Opening
- If customer greeted first: «وعليكم السلام ورحمة الله، معي الأستاذ {call_data.name}؟»
- Else: «السلام عليكم ورحمة الله، معي الأستاذ {call_data.name}؟»

# Challenge handling (LLM-native; no tool call needed)
- If asked who you are: «معك نورا من شركة توافق، وكيلة موبايلي المعتمدة.»
- If asked why calling: «اتصالي على خصوص حسابك في موبايلي.»
- If asked number source: «رقمك عندنا ضمن بيانات موبايلي المسجّلة. تقدر تتأكد من 1100.»
- After brief challenge handling, continue with the pending Stage 1 verification step.
 - If the customer repeats the same challenge question multiple times, answer it ONCE only, then immediately proceed to the pending yes/no verification question (do not loop).

# Final ID Verification (after right-party confirm only)
- Step 1 (preferred wording): «حبيت أتأكد منك، آخر أربع أرقام من هويتك أو إقامتك هي {self._id_last4_words}؟»
- Step 2 graceful retry (once only, if customer says "أعيدها" / "ما فهمت" / unclear):
  «لاهنت، أكرّرها: {self._id_last4_words}، مضبوط؟»
- Ask as yes/no only. Do not discuss debt, balance, invoices, or any account detail.

# Decision policy (deterministic)
1) First, resolve right-party status.
2) Only after right-party is confirmed, run final ID verification.
3) If ID answer semantically confirms a match -> identity_confirmed.
4) If customer semantically denies mismatch, refuses verification, or reaches two unresolved mismatch turns -> identity_denied.
5) Never call identity_confirmed before a successful ID-last4 match.

# Routing
- Clear confirm of right party, then final ID match → identity_confirmed
- Clear deny / wrong number / not the named person:
  - Ask once: «عذرًا، طيب تعرف الأستاذ {call_data.name}؟»
  - If they say YES: ask for a Saudi mobile number (10 digits), repeat it back once to confirm, then call record_alt_contact(number).
  - If they say NO: identity_denied
- Busy / can't talk now / "اتصل بعدين" (without DNC intent) → stage1_busy
- Refusal to answer ID check or two ID mismatches → identity_denied
- Global DNC / death (see Global Overrides) → trigger_dnc, trigger_death_reported
- Mixed (confirm + question): handle the side question briefly, then return to the pending
  yes/no question (identity check first, then ID verification step).
- Unclear input: at most one neutral clarification (R3), then re-ask the verification line

# Stop
When identity_denied or any trigger_* runs, the task ends; do not continue Stage 1.
""",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        # Deterministic opening reduces LLM looping/repetition in Stage 1.
        await self.session.say(
            f"السلام عليكم ورحمة الله، معي الأستاذ {self._call_data.name}؟",
            allow_interruptions=True,
        )

    @function_tool
    async def identity_confirmed(self) -> None:
        """Call only after right-party confirmation and successful final ID-last4 match (never earlier)."""
        self._call_data.right_party_confirmed = True
        self.complete(Stage1Result(outcome="confirmed"))

    @function_tool
    async def identity_denied(self) -> None:
        """Call on wrong-party OR final ID refusal / two mismatches. Completes Stage 1."""
        self.complete(Stage1Result(outcome="denied"))

    @function_tool
    async def record_alt_contact(self, number: str) -> None:
        """Call only in wrong-party handoff when they provide a 10-digit Saudi mobile number to reach the account holder."""
        self._call_data.alt_contact_number = str(number).strip()
        self.complete(Stage1Result(outcome="alt_contact"))

    @function_tool
    async def stage1_busy(self) -> None:
        """Call when the right party says they are busy / cannot talk now and wants to be called later (not a DNC request)."""
        self.complete(Stage1Result(outcome="busy"))

    @function_tool
    async def trigger_dnc(self) -> None:
        """Explicit do-not-call demand per Global Overrides. Ends Stage 1. Do not use for general complaints or rescheduling."""
        self._call_data.dnc_requested = True
        self.complete(Stage1Result(outcome="dnc"))

    @function_tool
    async def trigger_death_reported(self) -> None:
        """Call only when the caller explicitly confirms the account holder has died. Ends Stage 1."""
        self.complete(Stage1Result(outcome="death_reported"))


class NoraAgent(Agent):
    def __init__(self, call_data: CallData):
        self._call_data = call_data
        super().__init__(
            instructions=build_global_rules(call_data)
            + """
You are the orchestrator for an incremental stage-based call flow.
Only run the active stage logic and follow global rules at all times.
"""
        )

    async def on_enter(self) -> None:
        s1 = await Stage1Task(self._call_data, chat_ctx=self.chat_ctx)

        if s1.outcome == "dnc":
            await self.session.say(
                "تم تسجيل طلب عدم التواصل الهاتفي، والتواصل سيكون كتابياً فقط من الآن.",
                allow_interruptions=False,
            )
            return
        if s1.outcome == "death_reported":
            await self.session.say(
                "نسأل الله له الرحمة والمغفرة. للتحديث الرسمي، تواصلوا مع أقرب فرع موبايلي.",
                allow_interruptions=False,
            )
            return
        if s1.outcome == "denied":
            await self.session.say(
                "مشكور، الله يعطيك العافية.",
                allow_interruptions=False,
            )
            return
        if s1.outcome == "alt_contact":
            await self.session.say(
                "مشكور على تعاونك. بنحاول نتواصل معه على الرقم اللي عطيتنا. الله يعطيك العافية.",
                allow_interruptions=False,
            )
            return
        if s1.outcome == "busy":
            await self.session.say(
                "تم، ما بطوّل عليك. متى الوقت المناسب أكلمك؟",
                allow_interruptions=False,
            )
            return

        s2 = await Stage2Task(self._call_data, chat_ctx=self.chat_ctx)

        if s2.outcome == "dnc":
            await self.session.say(
                "تم تسجيل طلب عدم التواصل الهاتفي، والتواصل سيكون كتابياً فقط من الآن.",
                allow_interruptions=False,
            )
            return
        if s2.outcome == "death_reported":
            await self.session.say(
                "نسأل الله له الرحمة والمغفرة. للتحديث الرسمي، تواصلوا مع أقرب فرع موبايلي.",
                allow_interruptions=False,
            )
            return
        if s2.outcome == "paid":
            await self.session.say(
                "يعطيك العافية. تم، بنحدّث سجلك بأقرب وقت. مشكور.",
                allow_interruptions=False,
            )
            return
        if s2.outcome == "denial":
            svc = _primary_service_number(self._call_data.services)
            svc_part = f"على رقم {svc}" if svc else "على حسابك"
            await self.session.say(
                f"يا غالي، إحنا أكدنا هويتك بآخر أربع أرقام، والمبلغ هذا {svc_part} باسمك ولازم ينسدد.",
                allow_interruptions=True,
            )
            await self.session.say(
                "خلّنا نقفلها اليوم. تقدر تسدد المبلغ كامل اليوم أو بكرة؟",
                allow_interruptions=True,
            )
            # Continue into Stage 3 negotiation (do not end here).

        s3 = await Stage3Task(self._call_data, chat_ctx=self.chat_ctx)

        if s3.outcome == "dnc":
            await self.session.say(
                "تم تسجيل طلب عدم التواصل الهاتفي، والتواصل سيكون كتابياً فقط من الآن.",
                allow_interruptions=False,
            )
            return
        if s3.outcome == "death_reported":
            await self.session.say(
                "نسأل الله له الرحمة والمغفرة. للتحديث الرسمي، تواصلوا مع أقرب فرع موبايلي.",
                allow_interruptions=False,
            )
            return
        if s3.outcome == "dispute":
            await self.session.say(
                "فهمت. يمكنك تقديم اعتراض رسمي عبر التطبيق أو الفرع. مشكور، الله يعطيك العافية.",
                allow_interruptions=False,
            )
            return
        if s3.outcome == "refusal":
            await self.session.say(
                "تم. مشكور، الله يعطيك العافية.",
                allow_interruptions=False,
            )
            return

        # Commitment or reschedule: recap and close.
        await Stage4Task(self._call_data, chat_ctx=self.chat_ctx)


def _parse_job_metadata(raw: str) -> dict:
    if not raw or not str(raw).strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_services(value) -> list[dict]:
    """`services` may arrive as a list or as a JSON-encoded string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Accept JSON list encoded as string.
        if s.startswith("["):
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                return []
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        return []
    return []


def _primary_service_number(services: list[dict]) -> str:
    for s in services:
        if not isinstance(s, dict):
            continue
        n = str(s.get("service_number", "")).strip()
        if n:
            return n
    return ""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_wallet_end_date_riyadh() -> str:
    """Default internal deadline for local testing (end of current month, Riyadh)."""
    tz = ZoneInfo("Asia/Riyadh")
    now = datetime.now(tz)
    last_day = calendar.monthrange(now.year, now.month)[1]
    return f"{now.year:04d}-{now.month:02d}-{last_day:02d}"


def _get_interruption_mode() -> str:
    """Self-host safe default: VAD. Allow explicit adaptive opt-in."""
    mode = os.getenv("NORA_INTERRUPTION_MODE", "vad").strip().lower()
    return mode if mode in ("vad", "adaptive") else "vad"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    md = _parse_job_metadata(ctx.job.metadata)
    default_services = _parse_services(DEFAULT_CALL_METADATA.get("services", []))
    default_wallet_end_date = str(DEFAULT_CALL_METADATA.get("wallet_end_date", "")).strip()
    call_data = CallData(
        name=str(md.get("name", DEFAULT_CALL_METADATA["name"])),
        amount=str(md.get("amount", DEFAULT_CALL_METADATA["amount"])),
        id_last4=str(md.get("id_last4", DEFAULT_CALL_METADATA["id_last4"])),
        services=_parse_services(md.get("services", default_services)),
        wallet_end_date=str(
            md.get(
                "wallet_end_date",
                default_wallet_end_date or _default_wallet_end_date_riyadh(),
            )
        ),
    )

    turn_handling: TurnHandlingOptions = {
        "turn_detection": MultilingualModel(),
        "endpointing": {
            "min_delay": 0.8,
            "max_delay": 3.0,
        },
        "interruption": {
            "mode": _get_interruption_mode(),
            "false_interruption_timeout": float(
                _env_int("NORA_FALSE_INTERRUPTION_TIMEOUT_SEC", 2)
            ),
            "resume_false_interruption": os.getenv(
                "NORA_RESUME_FALSE_INTERRUPTION", "true"
            ).strip().lower()
            in ("1", "true", "yes", "on"),
            "min_words": _env_int("NORA_INTERRUPTION_MIN_WORDS", 2),
        },
    }

    session = AgentSession[CallData](
        userdata=call_data,
        llm=build_llm(),
        stt=build_stt(),
        tts=build_tts(),
        vad=silero.VAD.load(),
        turn_handling=turn_handling,
        preemptive_generation=True,
        tts_text_transforms=["filter_markdown"],
        max_tool_steps=4,
    )
    await session.start(agent=NoraAgent(call_data), room=ctx.room)
