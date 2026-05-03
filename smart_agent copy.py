"""
Arabic (Najdi) collections agent — system-prompt-driven, intelligent.

Departure from task_agent.py: instead of `session.say(STATIC_LINE)` per stage,
each agent here lets the LLM generate spoken Arabic naturally, anchored by a
shared PERSONA system prompt + per-stage task description. Tools handle the
hard state transitions matching the collections flowchart; off-topic /
clarification turns are answered by the LLM in-character without breaking
the flow.

Flowchart implemented:
    Stage 1 (right-party verify) →
        DNC | death | busy → end (with appropriate parting)
        wrong-party → knows-person? → collect-mobile → end
        right-party → ID-verify (last 4 digits) →
            mismatch → end wrong
            match → Stage 2
    Stage 2 (debt intro) →
        already-paid → end (paid update)
        reason / denial → Stage 3
    Stage 3 (negotiation: SLA + 7-day SIMAH context →
              L1 full today/tomorrow → L2 half → L3 instalment) →
        commit (any level) → Stage 4
        vague → reschedule → Stage 4
        refuse → end refusal
        dispute → end dispute
    Stage 4 (recap & confirm commitment) → end OK

STT: Deepgram nova-3 ar-SA. TTS: Faseeh ar-najdi-female-1.
The PERSONA + stage instructions stay in English — the model understands
them and generates Najdi Arabic in its replies.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice import ModelSettings
import hamsa_livekit
from livekit.plugins import deepgram, faseeh, openai, silero


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("smart-caller-ar")
logger.setLevel(logging.INFO)

logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.silero").setLevel(logging.WARNING)

_LIVEKIT_NOISE_PATTERNS = (
    "min endpointing delay updated",
    "max endpointing delay updated",
    "using preemptive generation",
    "reusing STT pipeline",
    "input stream attached",
    "start reading stream",
    "using audio io",
    "using transcript io",
    "max_tool_steps",
)


class _LiveKitNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _LIVEKIT_NOISE_PATTERNS)


logging.getLogger("livekit.agents").addFilter(_LiveKitNoiseFilter())

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")


# ---------- Shared state ----------

@dataclass
class CallData:
    customer_name: str = "محمد"
    amount: str = "10000"
    debt_date: str = "2023-01-01"
    national_id_last4: str = "1234"

    identity_confirmed: bool = False
    id_verified: bool = False
    outcome: str | None = None  # "paid" | "committed" | "rescheduled" | "refusal" | "dispute" | "dnc" | "death" | "busy" | "wrong_party"
    commitment: str | None = None  # free-form description set by the LLM
    callback_time: str | None = None
    referrer_mobile: str | None = None  # set when wrong-party knows the person

    sip_ready: asyncio.Event = field(default_factory=asyncio.Event)
    participant: rtc.RemoteParticipant | None = None


# ---------- Najdi pronunciation enforcement ----------
#
# Even though the LLM generates speech now, it still emits digits like
# "10000 ريال" sometimes. We normalize at the TTS node so the spoken output
# is always Najdi-correct regardless of the LLM's surface form.

_AR_UNITS = ["", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة",
             "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_AR_TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر",
             "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_AR_TENS = ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين",
            "ستين", "سبعين", "ثمانين", "تسعين"]
_AR_HUNDREDS = ["", "مئة", "مئتين", "ثلاث مئة", "أربع مئة", "خمس مئة",
                "ست مئة", "سبع مئة", "ثمان مئة", "تسع مئة"]
_AR_MONTHS = [
    "", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]
_AR_DIGITS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة",
              "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]

_AR_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _ar_below_1000(n: int) -> str:
    if n == 0:
        return ""
    parts: list[str] = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_AR_HUNDREDS[h])
    if rem:
        if rem < 10:
            parts.append(_AR_UNITS[rem])
        elif rem < 20:
            parts.append(_AR_TEENS[rem - 10])
        else:
            t, u = divmod(rem, 10)
            if u:
                parts.append(f"{_AR_UNITS[u]} و{_AR_TENS[t]}")
            else:
                parts.append(_AR_TENS[t])
    return " و".join(parts)


def _ar_amount_words(n: int) -> str:
    if n == 0:
        return "صفر"
    parts: list[str] = []
    millions, n = divmod(n, 1_000_000)
    thousands, n = divmod(n, 1_000)
    if millions:
        if millions == 1:
            parts.append("مليون")
        elif millions == 2:
            parts.append("مليونين")
        elif 3 <= millions <= 10:
            parts.append(f"{_AR_UNITS[millions]} ملايين")
        else:
            parts.append(f"{_ar_below_1000(millions)} مليون")
    if thousands:
        if thousands == 1:
            parts.append("ألف")
        elif thousands == 2:
            parts.append("ألفين")
        elif 3 <= thousands <= 10:
            parts.append(f"{_AR_UNITS[thousands]} آلاف")
        else:
            parts.append(f"{_ar_below_1000(thousands)} ألف")
    if n:
        parts.append(_ar_below_1000(n))
    return " و".join(parts)


def _ar_date_words(iso: str) -> str:
    try:
        y, m, d = (int(x) for x in iso.split("-"))
        if not (1 <= m <= 12):
            return iso
        day = _ar_below_1000(d) or "صفر"
        return f"{day} {_AR_MONTHS[m]} {_ar_amount_words(y)}"
    except (ValueError, IndexError):
        return iso


def _ar_digits_individual(s: str) -> str:
    return " ".join(_AR_DIGITS[int(c)] for c in s if c.isdigit())


_TERM_MAP = {
    "stc": "اس تي سي",
    "STC": "اس تي سي",
    "SAR": "ريال سعودي",
    "SIMAH": "سمة",
    "simah": "سمة",
}


def _najdi_normalize(text: str) -> str:
    text = re.sub(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        lambda m: _ar_date_words(m.group(1)),
        text,
    )
    text = re.sub(
        r"(\d+)\s*ريال",
        lambda m: f"{_ar_amount_words(int(m.group(1)))} ريال سعودي",
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d{4})(?!\d)",
        lambda m: _ar_digits_individual(m.group(1)),
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d+)(?!\d)",
        lambda m: _ar_amount_words(int(m.group(1))),
        text,
    )
    for term, repl in _TERM_MAP.items():
        text = re.sub(rf"\b{re.escape(term)}\b", repl, text)
    return text


# ---------- Persona (system-level prompt) ----------

PERSONA = """\
You are نورا (Nora), a polite female collections specialist at شركة توافق working
on behalf of بنك stc. You ALWAYS reply in Najdi Saudi Arabic; never English in
spoken output unless quoting a brand name (stc, simah).

Voice & tone:
- Warm, respectful, professional. Never aggressive, condescending, or pushy.
- Use natural Najdi politeness: طال عمرك، أبشر، ولا يهمك، تكفى، ما يخالف، حياك الله.
- Address the customer as الأستاذ + first name where appropriate.
- ONE or TWO short sentences per turn. Brevity matters — this is a phone call.

Speaking numbers and dates aloud:
- Amounts: Arabic words, never digits ("عشرة آلاف ريال", not "10000 ريال").
- ID / PIN / verification digits: spell one by one ("واحد اثنين ثلاثة أربعة").
- Dates: Arabic words ("أول يناير ألفين وثلاثة وعشرين").

Identity & off-topic:
- If asked who you are or why you're calling, briefly identify yourself and
  the purpose in ONE sentence (e.g. "أنا نورا من شركة توافق، نتواصل بخصوص
  حساب stc الخاص بك") then re-pose the current stage's question.
- If the customer goes off-topic, briefly acknowledge then redirect to the
  current question.
- Stay in role. If pressed about whether you're AI, acknowledge once politely
  ("نعم أنا مساعد آلي من توافق") and continue with the call.

Empathy & negotiation style (CRITICAL — this is what separates good agents from robotic ones):
- When the customer expresses hardship — money troubles, illness, family
  problems, lost job, salary delayed, ظروفي صعبة, ما عندي فلوس, تعبان —
  ACKNOWLEDGE the feeling FIRST in one short empathetic phrase, THEN steer
  toward a workable arrangement. Do not jump straight to the next ask.
  Examples:
  - "ما عندي فلوس" → "والله أتفهم وضعك طال عمرك، ولا يهمك، نشوف لك ترتيب
     يناسبك. كم تقدر تدفع شهرياً؟"
  - "ظروفي صعبة" → "الله يفرجها عليك، خلنا نرتب الموضوع بشكل أسهل عليك."
  - "والله متضايق" → "أكيد، ما أبغى أزعجك أكثر، خلنا نشوف حل سريع
     ومريح لك."
- Treat hardship as a SIGNAL to negotiate, not a refusal. Steer the
  conversation toward an arrangement the customer can actually keep.
- Never sound transactional, scripted, or robotic when hardship is mentioned.
- When the customer accepts a plan, reiterate it back clearly with specific
  amounts and specific dates so there is no ambiguity.

Time & date awareness:
- Use the dates context in the call data block to resolve relative time
  references. If the customer says "next week", "next month", "after my
  salary", "بكرا", "نهاية الشهر" — translate to a SPECIFIC ISO date and
  say it back to him in Arabic words so he can confirm.
- Never leave dates vague. Always commit to a concrete day.

Behavior rules:
- Use ONLY the call data provided in the next block. NEVER invent or guess
  amounts, dates, or digits. If you don't have something, say you'll check.
- When the customer's reply triggers a workflow transition described in
  the stage section, CALL THE CORRESPONDING TOOL — do not just keep talking.
- For clarifications and short side-questions, reply briefly in character
  without a tool, then redirect to the current stage question.
- Do not negotiate beyond the ladder defined for the current stage.
"""


_AR_WEEKDAYS = {
    "Monday": "الإثنين", "Tuesday": "الثلاثاء", "Wednesday": "الأربعاء",
    "Thursday": "الخميس", "Friday": "الجمعة", "Saturday": "السبت",
    "Sunday": "الأحد",
}


def call_context_block(data: CallData) -> str:
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    next_week = today + datetime.timedelta(days=7)
    next_month = today + datetime.timedelta(days=30)
    end_of_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
    one_year_out = today + datetime.timedelta(days=365)

    try:
        amount_int = int(str(data.amount).strip())
    except ValueError:
        amount_int = 0
    min_monthly = max(1, amount_int // 12) if amount_int else 0
    max_months = 12

    return (
        "Call data (use ONLY these values when speaking about the debt):\n"
        f"- Customer name: {data.customer_name}\n"
        f"- Outstanding amount: {data.amount} SAR\n"
        f"- Debt date: {data.debt_date}\n"
        f"- Last 4 digits of national ID on file: {data.national_id_last4}\n"
        "\n"
        "Dates context (resolve relative time references against these):\n"
        f"- Today: {today.isoformat()} ({_AR_WEEKDAYS.get(today.strftime('%A'), today.strftime('%A'))})\n"
        f"- Tomorrow: {tomorrow.isoformat()} ({_AR_WEEKDAYS.get(tomorrow.strftime('%A'), tomorrow.strftime('%A'))})\n"
        f"- One week from today: {next_week.isoformat()}\n"
        f"- End of this month: {end_of_month.isoformat()}\n"
        f"- One month from today: {next_month.isoformat()}\n"
        f"- One year from today (HARD limit for any payment plan): {one_year_out.isoformat()}\n"
        "\n"
        "Negotiation constraints (HARD limits — do not concede beyond these):\n"
        f"- Maximum instalment plan length: {max_months} months. The full debt MUST\n"
        f"  be cleared within 12 months of today.\n"
        f"- Therefore minimum monthly instalment: ~{min_monthly} SAR/month.\n"
        f"  If the customer offers less per month than this, COUNTER-OFFER\n"
        f"  with a higher monthly amount or a shorter cycle (e.g. weekly\n"
        f"  payments) so the debt clears within 12 months. Do NOT accept a\n"
        f"  plan that would take longer than 12 months.\n"
        f"- When proposing instalments, prefer a round number close to\n"
        f"  {min_monthly} SAR/month over fewer larger payments.\n"
    )


def stage_instructions(data: CallData, stage_task: str) -> str:
    return f"{PERSONA}\n\n{call_context_block(data)}\n\n{stage_task}"


# ---------- Base agent ----------

class BaseCallAgent(Agent):
    async def hangup(self):
        job_ctx = get_job_context()
        try:
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )
        except Exception as e:
            logger.warning(f"hangup error (room may already be gone): {e}")

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        async def normalized(stream: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in stream:
                yield _najdi_normalize(chunk)

        async for frame in Agent.default.tts_node(
            self, normalized(text), model_settings
        ):
            yield frame

    @function_tool()
    async def voicemail_detected(self, ctx: RunContext[CallData]):
        """Call this if you hear a voicemail / answering-machine greeting
        instead of a live person. Hangs up immediately."""
        logger.info("voicemail detected, hanging up")
        await self.hangup()


# ---------- Stage 1: Right-party verification ----------

STAGE1_TASK = """\
Current stage: 1 — Right-party verification.

You have just been connected. Greet the customer briefly, identify yourself
as نورا from شركة توافق on behalf of بنك stc, and ask whether you are speaking
with الأستاذ {customer_name}. Listen carefully to their reply.

Then call exactly one tool based on what you hear:
- right_party: customer confirms they ARE {customer_name}.
- wrong_party: customer says they are NOT the named person.
- caller_busy: it's not a good time / asks to call back later / مشغول / بعدين.
- do_not_call: asks not to be contacted again / DNC request / لا تتصلوا فيني.
- customer_deceased: informs you the named person has passed away / متوفى / الله يرحمه.

If the reply is a clarification or off-topic question, answer briefly in
character (one sentence) and re-ask the right-party question. Do NOT call
a tool unless one of the above conditions clearly applies.
"""


class Stage1RightPartyAgent(BaseCallAgent):
    def __init__(self, data: CallData) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE1_TASK.format(customer_name=data.customer_name),
            ),
        )
        self.data = data

    async def on_enter(self):
        # Wait for SIP participant to join before speaking; settle carrier.
        await self.data.sip_ready.wait()
        await asyncio.sleep(0.4)
        self.session.generate_reply(
            instructions=(
                "Open the call now. Greet warmly and identify yourself as "
                "نورا from شركة توافق on behalf of بنك stc, then ask whether "
                f"you are speaking with الأستاذ {self.data.customer_name}. "
                "ONE short sentence."
            )
        )

    @function_tool()
    async def right_party(self, ctx: RunContext[CallData]):
        """Customer confirmed they are the named person."""
        ctx.userdata.identity_confirmed = True
        return AskGoodTimeAgent(self.data, chat_ctx=None)

    @function_tool()
    async def wrong_party(self, ctx: RunContext[CallData]):
        """Customer says they are NOT the named person."""
        return WrongPartyKnowsAgent(self.data, chat_ctx=None)

    @function_tool()
    async def caller_busy(self, ctx: RunContext[CallData]):
        """Customer says it's not a good time / asked for callback,
        BEFORE we have confirmed identity. Route to scheduling so we
        capture a concrete callback time."""
        ctx.userdata.outcome = "busy"
        return ScheduleCallbackAgent(self.data, chat_ctx=None)

    @function_tool()
    async def do_not_call(self, ctx: RunContext[CallData]):
        """Customer requested not to be contacted again (DNC)."""
        ctx.userdata.outcome = "dnc"
        return ClosingAgent(self.data, intent="dnc", chat_ctx=None)

    @function_tool()
    async def customer_deceased(self, ctx: RunContext[CallData]):
        """Caller informed you the named person has passed away."""
        ctx.userdata.outcome = "death"
        return ClosingAgent(self.data, intent="death", chat_ctx=None)


# ---------- Stage 1a-bis: Is now a good time? ----------

ASK_GOOD_TIME_TASK = """\
Current stage: confirm-good-time.

You have just verified you are speaking with الأستاذ {customer_name}.
Before discussing the account, briefly ask whether NOW is a good time to
talk for a few minutes. ONE short sentence, friendly and unhurried.

Then call exactly one tool:
- good_time_now: customer says yes / تفضلي / أبشر / ما عندي مانع / it's fine.
- bad_time_now: customer says it's not a good moment / مشغول / في اجتماع /
  بعدين / مع العائلة. We will schedule a callback for him.
- unclear: ambiguous / asked to repeat / off-topic. Use whenever NOT confident.

Be empathetic if the customer hesitates — never pressure him to continue.
"""


class AskGoodTimeAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                ASK_GOOD_TIME_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Briefly ask the customer if NOW is a good time to talk "
                "for a few minutes. Friendly, unhurried, ONE short sentence."
            )
        )

    @function_tool()
    async def good_time_now(self, ctx: RunContext[CallData]):
        """Customer confirmed it is a good time to talk."""
        return IDVerifyAgent(self.data, chat_ctx=None)

    @function_tool()
    async def bad_time_now(self, ctx: RunContext[CallData]):
        """Customer says it's not a good moment — schedule callback."""
        ctx.userdata.outcome = "busy"
        return ScheduleCallbackAgent(self.data, chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply was ambiguous — re-ask gently."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask whether NOW is a good time for a few "
                "minutes. ONE short sentence."
            )
        )


# ---------- Schedule callback (busy at start) ----------

SCHEDULE_CALLBACK_TASK = """\
Current stage: schedule a callback for a busy customer.

The customer said now is not a good time. Apologize briefly and ask when
would be a better time to call back. Be specific — encourage him to name
a day and a rough time of day. ONE short sentence.

Use the dates context to translate any relative answer ("بكرا الصبح",
"الأسبوع الجاي", "بعد العصر") into a SPECIFIC ISO date and time-of-day,
then confirm it back to him verbally before recording.

Then call exactly one tool:
- callback_time(day_iso, time_of_day): customer named a time. Pass the
  ISO date (YYYY-MM-DD) and a short time-of-day label ('morning',
  'afternoon', 'evening', or HH:MM if specific).
- refuses_to_schedule: customer won't commit to any callback time.
- unclear: ambiguous / asked to repeat / off-topic. Use when NOT confident.
"""


class ScheduleCallbackAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, SCHEDULE_CALLBACK_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Apologize briefly for the bad timing and ask when would "
                "be a better time to call back — encourage him to name a "
                "day and a rough time of day. ONE short sentence."
            )
        )

    @function_tool()
    async def callback_time(
        self,
        ctx: RunContext[CallData],
        day_iso: str,
        time_of_day: str,
    ):
        """Customer named a callback day and time-of-day.

        Args:
            day_iso: ISO date (YYYY-MM-DD) translated from the customer's reply.
            time_of_day: short label — 'morning' / 'afternoon' / 'evening' /
                         or HH:MM if specific.
        """
        ctx.userdata.callback_time = f"{day_iso} {time_of_day}"
        ctx.userdata.outcome = "busy_callback"
        return ClosingAgent(self.data, intent="busy_callback", chat_ctx=None)

    @function_tool()
    async def refuses_to_schedule(self, ctx: RunContext[CallData]):
        """Customer refuses to commit to any callback time."""
        ctx.userdata.outcome = "busy"
        return ClosingAgent(self.data, intent="busy", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply ambiguous — re-ask the callback time."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask the customer to name a day and a rough "
                "time of day to call back. ONE short sentence."
            )
        )


# ---------- Stage 1b: Wrong-party — do you know them? ----------

WRONG_PARTY_KNOWS_TASK = """\
Current stage: 1b — wrong party, find out if they can refer us.

The person on the line is NOT الأستاذ {customer_name}. Politely apologize for
the mix-up, then ask if they happen to know الأستاذ {customer_name} and could
share a working number for him. ONE sentence.

Then call exactly one tool:
- knows_person: caller says yes, they know him / can help reach him.
- does_not_know_person: caller says no / never heard of him / can't help.

For clarifications, reply briefly then re-ask. Do not pressure.
"""


class WrongPartyKnowsAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                WRONG_PARTY_KNOWS_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                f"Apologize briefly for the mix-up and ask if they know "
                f"الأستاذ {self.data.customer_name} and could share a working "
                "number to reach him. One short sentence."
            )
        )

    @function_tool()
    async def knows_person(self, ctx: RunContext[CallData]):
        """Caller knows the named person and is willing to help."""
        return CollectMobileAgent(self.data, chat_ctx=None)

    @function_tool()
    async def does_not_know_person(self, ctx: RunContext[CallData]):
        """Caller does not know the named person."""
        ctx.userdata.outcome = "wrong_party"
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)


# ---------- Stage 1c: Collect mobile number from referrer ----------

COLLECT_MOBILE_TASK = """\
Current stage: 1c — collect a referral mobile number.

The caller knows الأستاذ {customer_name} and may share a number. Ask politely
for a working mobile number to reach him.

Then call exactly one tool:
- mobile_provided(number): caller spoke a number. Pass digits as a string,
  converting Arabic-word numbers to digit characters. Saudi mobile numbers
  start with 05 and are 10 digits — pass whatever digits the caller said.
- refuses_to_provide: caller declines / doesn't have a number / لا أعرف رقمه.

For clarifications, briefly answer and re-ask the number politely.
"""


class CollectMobileAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                COLLECT_MOBILE_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                f"Politely ask for a working mobile number to reach "
                f"الأستاذ {self.data.customer_name}. One short sentence."
            )
        )

    @function_tool()
    async def mobile_provided(self, ctx: RunContext[CallData], number: str):
        """Caller gave a mobile number.

        Args:
            number: digits-only string of the number they spoke.
        """
        clean = re.sub(r"\D", "", number.translate(_AR_INDIC_DIGITS))
        ctx.userdata.referrer_mobile = clean
        ctx.userdata.outcome = "wrong_party_referred"
        return ClosingAgent(self.data, intent="referred", chat_ctx=None)

    @function_tool()
    async def refuses_to_provide(self, ctx: RunContext[CallData]):
        """Caller declined to share a number."""
        ctx.userdata.outcome = "wrong_party"
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)


# ---------- Stage 1d: ID verification ----------

ID_VERIFY_TASK = """\
Current stage: 1d — verify last 4 digits of national ID.

You are speaking with الأستاذ {customer_name}. Politely ask for the last 4
digits of his national ID (آخر ٤ أرقام من الهوية الوطنية) so you can confirm
identity before discussing the account. ONE sentence.

Then call exactly one tool:
- digits_provided(digits): customer spoke 4 digits. Pass as a 4-character
  ASCII string. Convert Arabic-word numbers to digit characters
  ('واحد اثنين ثلاثة أربعة' → '1234'). If the customer says them as one
  number (e.g. 'ألف ومئتين وأربعة وثلاثين' or '١٢٣٤'), still pass '1234'.
- unclear: customer asked to repeat / refused / gave fewer than 4 digits /
  off-topic. Use whenever no clear 4-digit answer is given. Never guess.

The stored last-4 digits are in the call data; you do NOT speak them.
The system compares server-side and re-asks or ends if mismatched.
"""


class IDVerifyAgent(BaseCallAgent):
    # First mismatch is forgiven (single-digit STT mishears happen). Second
    # mismatch ends the call. Unclear replies do not consume an attempt.
    MAX_ATTEMPTS = 2

    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                ID_VERIFY_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data
        self._attempts = 0

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Politely ask the customer for the last 4 digits of his "
                "national ID to confirm identity. One short sentence."
            )
        )

    @function_tool()
    async def digits_provided(self, ctx: RunContext[CallData], digits: str):
        """Customer spoke 4 digits — verify against stored last-4.

        Args:
            digits: 4-character ASCII digit string, e.g. '1234'.
        """
        clean = re.sub(r"\D", "", digits.translate(_AR_INDIC_DIGITS))
        if len(clean) != 4:
            self.session.generate_reply(
                instructions=(
                    "You couldn't catch all 4 digits. Politely ask the "
                    "customer to repeat the last 4 digits of the national "
                    "ID, slowly. One short sentence."
                )
            )
            return

        if clean == self.data.national_id_last4:
            ctx.userdata.id_verified = True
            return Stage2DebtIntroAgent(self.data, chat_ctx=None)

        self._attempts += 1
        if self._attempts >= self.MAX_ATTEMPTS:
            ctx.userdata.outcome = "id_mismatch"
            return ClosingAgent(self.data, intent="id_mismatch", chat_ctx=None)

        self.session.generate_reply(
            instructions=(
                "The digits did not match. Apologize politely and ask the "
                "customer to repeat the last 4 digits one more time. "
                "One short sentence."
            )
        )

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Customer asked to repeat or didn't give a clear 4-digit reply."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask for the last 4 digits of the national ID, "
                "slowly. One short sentence."
            )
        )


# ---------- Stage 2: Debt intro & reason ----------

STAGE2_TASK = """\
Current stage: 2 — Introduce the debt and listen for paid claim or reason.

The customer's identity is verified. Briefly remind him there is an
outstanding amount of {amount} SAR from {debt_date} on his stc account that
has not been settled, and ask gently why it has not been paid. ONE or TWO
short sentences.

Then call exactly one tool based on his reply:
- already_paid: customer claims it's already been paid / مدفوع / سددته.
- proceed_to_negotiation: customer gives any reason, denial, or wants to
  discuss payment / hardship / forgot / disputes amount / will pay later.
  This is the default when the customer engages substantively.

For clarifications or short side-questions, reply briefly and re-ask why
the amount hasn't been settled.
"""


class Stage2DebtIntroAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE2_TASK.format(amount=data.amount, debt_date=data.debt_date),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                f"Mention there is an outstanding amount of {self.data.amount} "
                f"SAR from {self.data.debt_date} that has not been settled, "
                "and gently ask why. ONE or TWO short sentences."
            )
        )

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def proceed_to_negotiation(self, ctx: RunContext[CallData]):
        """Customer gave a reason / denial / wants to discuss payment."""
        return Stage3FullPaymentAgent(self.data, chat_ctx=None)


# ---------- Stage 3a: SLA explanation + L1 (full payment today/tomorrow) ----------

STAGE3_FULL_TASK = """\
Current stage: 3 — Negotiation, level 1 (full payment).

In ONE turn, briefly explain that there is a 7-day grace window after which
the case is reported to سمة (SIMAH) credit bureau, and ask if he can settle
the FULL amount today or tomorrow. Tone: factual, not threatening. Two short
sentences max.

Then call exactly one tool:
- already_paid: customer claims the debt is ALREADY paid / settled /
  cleared (مدفوع / سددته / دفعت / حولت / خلصت / paid already / I have
  already paid). This takes priority over every other tool — if he is
  saying it's paid, call this even if he also says other things.
- agrees_full: customer CLEARLY agrees to pay full today or tomorrow.
  Examples: نعم / موافق / أبشر / حول لي الرابط / إيه أبشر.
- cannot_full: customer CLEARLY declines full payment — too much, no
  liquidity, asks for partial / smaller / time. (We then offer half.)
- unclear: reply is ambiguous, off-topic, asks to repeat, or is a single
  noisy STT token you cannot confidently classify (e.g. 'في', 'هم',
  garbled fragments). Use this whenever NOT confident — never guess
  agreement from one ambiguous word.

Do NOT skip ahead to half-payment until the customer has clearly declined.
"""


class Stage3FullPaymentAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, STAGE3_FULL_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Briefly explain the 7-day grace window before SIMAH "
                "reporting, and ask if the customer can settle the FULL "
                f"amount of {self.data.amount} SAR today or tomorrow. "
                "Two short sentences."
            )
        )

    @function_tool()
    async def agrees_full(self, ctx: RunContext[CallData], when_iso: str):
        """Customer agrees to pay full today or tomorrow.

        Args:
            when_iso: ISO date (YYYY-MM-DD) of payment. Translate 'today' /
                      'tomorrow' / 'بكرا' to the concrete date using the
                      dates context in the call data block.
        """
        ctx.userdata.commitment = (
            f"full payment of {self.data.amount} SAR on {when_iso}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def cannot_full(self, ctx: RunContext[CallData]):
        """Customer cannot pay the full amount today or tomorrow."""
        return Stage3HalfPaymentAgent(self.data, chat_ctx=None)

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid / settled."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply is ambiguous, off-topic, or asks to repeat."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask whether the customer can settle the FULL "
                f"amount of {self.data.amount} SAR today or tomorrow. "
                "ONE short sentence."
            )
        )


# ---------- Stage 3b: L2 (half payment) ----------

STAGE3_HALF_TASK = """\
Current stage: 3 — Negotiation, level 2 (half payment exception).

The customer cannot pay the full amount. As an exception, offer to split:
half ({half} SAR) within a day or two, and the rest on a date he chooses.
ONE or TWO short sentences.

Then call exactly one tool:
- already_paid: customer claims the debt is ALREADY paid / settled
  (مدفوع / سددته / دفعت / حولت / paid already). Takes priority — call
  this even mid-negotiation if he asserts the debt is paid.
- agrees_half: customer CLEARLY agrees to pay half soon and the rest later.
- cannot_half: customer CLEARLY declines even half / no liquidity at all.
  (We will then offer an instalment plan.)
- unclear: reply is ambiguous, off-topic, asks to repeat, or is a noisy
  STT token you cannot confidently classify. Use whenever NOT confident.
"""


class Stage3HalfPaymentAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        try:
            self._half = max(1, int(str(data.amount).strip()) // 2)
        except ValueError:
            self._half = 0
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE3_HALF_TASK.format(half=self._half),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Briefly acknowledge with empathy that the full amount is "
                "difficult right now (one short empathetic phrase), THEN "
                f"as an exception (كاستثناء) offer to split: half "
                f"({self._half} SAR) within a day or two, and the rest on "
                "a date he chooses. TWO short sentences total."
            )
        )

    @function_tool()
    async def agrees_half(self, ctx: RunContext[CallData], rest_date: str | None = None):
        """Customer agrees to half-now-rest-later split.

        Args:
            rest_date: optional ISO date the customer named for the second half.
        """
        rest = f", remainder on {rest_date}" if rest_date else ""
        ctx.userdata.commitment = (
            f"half payment {self._half} SAR within 1-2 days{rest}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def cannot_half(self, ctx: RunContext[CallData]):
        """Customer cannot pay even half today."""
        return Stage3InstallmentAgent(self.data, chat_ctx=None)

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid / settled."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply is ambiguous or off-topic — re-ask the half offer once."""
        self.session.generate_reply(
            instructions=(
                f"Politely re-ask: split half ({self._half} SAR) within 1-2 "
                "days, the rest on a date the customer chooses. ONE short "
                "sentence."
            )
        )


# ---------- Stage 3c: L3 (instalment plan) ----------

STAGE3_INSTALLMENT_TASK = """\
Current stage: 3 — Negotiation, level 3 (instalment plan).

The customer cannot pay full or half. Open with a SHORT empathetic
acknowledgement, then propose a monthly instalment plan that clears the
debt within 12 months (see the negotiation constraints in the call data
block). Anchor with a concrete suggestion (e.g. the minimum monthly
amount over 12 months) instead of asking open-endedly.

Negotiate, don't just accept the customer's first number. If he proposes
a small amount that would take longer than 12 months, COUNTER-OFFER with
a higher monthly figure or shorter cycle (e.g. weekly) — explain it's
the only way to clear within the company's 12-month window. Be friendly
about it, never pushy.

Translate any relative date the customer gives (next week, end of the
month, after my salary) into a SPECIFIC ISO date using the dates context.

Then call exactly one tool:
- already_paid: customer claims the debt is ALREADY paid / settled
  (مدفوع / سددته / دفعت / حولت / I have paid already). Takes priority
  over every other tool — never push a plan on someone who is saying
  the debt is already paid.
- plan_agreed(monthly_amount, num_months, start_date_iso): customer
  agreed to a CONCRETE plan that fits within 12 months. The system will
  verify the plan covers the full debt within the cap and reject silently
  if not (then re-ask).
- vague_response: customer is willing in principle but won't name concrete
  numbers ("بشوف", "إن شاء الله", "يمكن"). We will schedule a callback.
- refuses_payment: customer flatly refuses to pay or commit anything
  ("ما أبغى أدفع", "ما تستحقون"). End as refusal.
- disputes_debt: customer disputes the debt / amount / claims fraud
  (مو ديني / غلط في الحساب). End for dispute review.
- unclear: reply is ambiguous, off-topic, asks to repeat, or is a noisy
  STT token you cannot confidently classify. Use whenever NOT confident.
"""


class Stage3InstallmentAgent(BaseCallAgent):
    MAX_MONTHS = 12

    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        try:
            self._amount_int = int(str(data.amount).strip())
        except ValueError:
            self._amount_int = 0
        self._min_monthly = max(1, self._amount_int // self.MAX_MONTHS)
        super().__init__(
            instructions=stage_instructions(data, STAGE3_INSTALLMENT_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Briefly acknowledge with empathy that things are tight "
                "(one short phrase), THEN propose a concrete instalment "
                f"plan: ~{self._min_monthly} SAR per month over "
                f"{self.MAX_MONTHS} months, starting next month, to clear "
                "the full amount. Ask if this works for him. TWO short "
                "sentences total. Anchor with a specific number — do NOT "
                "ask open-endedly."
            )
        )

    @function_tool()
    async def plan_agreed(
        self,
        ctx: RunContext[CallData],
        monthly_amount: float,
        num_months: int,
        start_date_iso: str,
    ):
        """Customer agreed to a concrete instalment plan.

        Args:
            monthly_amount: SAR per monthly instalment.
            num_months: total number of monthly instalments.
            start_date_iso: ISO date (YYYY-MM-DD) of the first instalment.
        """
        # Server-side validation: plan must clear the debt within 12 months.
        if num_months > self.MAX_MONTHS:
            self.session.generate_reply(
                instructions=(
                    f"The plan ({num_months} months) is longer than our "
                    f"{self.MAX_MONTHS}-month maximum. Politely counter-offer: "
                    f"the maximum we can extend is {self.MAX_MONTHS} months, "
                    f"which works out to about {self._min_monthly} SAR per "
                    "month. Ask if that works. TWO short sentences."
                )
            )
            return

        covered = monthly_amount * num_months
        if self._amount_int and covered < self._amount_int * 0.95:
            self.session.generate_reply(
                instructions=(
                    f"The instalments ({num_months} × {monthly_amount} = "
                    f"{covered:.0f} SAR) don't fully cover the outstanding "
                    f"{self._amount_int} SAR. Politely point this out and "
                    "ask the customer to adjust either the monthly amount "
                    "or the count. ONE short sentence."
                )
            )
            return

        # Compute end date for the recap.
        try:
            start = datetime.date.fromisoformat(start_date_iso)
            end = start + datetime.timedelta(days=30 * (num_months - 1))
            end_iso = end.isoformat()
        except ValueError:
            end_iso = "(unknown)"

        ctx.userdata.commitment = (
            f"instalment plan: {num_months} monthly payments of "
            f"{int(monthly_amount)} SAR, first on {start_date_iso}, "
            f"last on {end_iso}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def vague_response(self, ctx: RunContext[CallData]):
        """Customer is willing in principle but won't name a concrete plan."""
        return RescheduleAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_payment(self, ctx: RunContext[CallData]):
        """Customer flatly refuses to pay."""
        ctx.userdata.outcome = "refusal"
        return ClosingAgent(self.data, intent="refusal", chat_ctx=None)

    @function_tool()
    async def disputes_debt(self, ctx: RunContext[CallData]):
        """Customer disputes the debt — claims it isn't his or amount is wrong."""
        ctx.userdata.outcome = "dispute"
        return ClosingAgent(self.data, intent="dispute", chat_ctx=None)

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid / settled."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply is ambiguous or off-topic — re-ask the instalment offer."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask the customer how many monthly instalments "
                "and when to start. ONE short sentence."
            )
        )


# ---------- Reschedule (vague exit from L3) ----------

RESCHEDULE_TASK = """\
Current stage: reschedule callback.

The customer was vague about a plan. Politely ask when would be a better
time to follow up — within the next two weeks. ONE short sentence.

Then call exactly one tool:
- callback_scheduled(when): customer named a time. Pass a short description
  ('next Monday morning', 'tomorrow afternoon', or ISO date).
- refuses_to_schedule: customer won't commit to any callback time.

For clarifications, answer briefly and re-ask.
"""


class RescheduleAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, RESCHEDULE_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Politely ask when would be a good time to follow up "
                "within the next two weeks. One short sentence."
            )
        )

    @function_tool()
    async def callback_scheduled(self, ctx: RunContext[CallData], when: str):
        """Customer named a callback time.

        Args:
            when: short description of the time (e.g. 'next Monday morning').
        """
        ctx.userdata.callback_time = when
        ctx.userdata.outcome = "rescheduled"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_to_schedule(self, ctx: RunContext[CallData]):
        """Customer refuses to commit to any callback time."""
        ctx.userdata.outcome = "refusal"
        return ClosingAgent(self.data, intent="refusal", chat_ctx=None)


# ---------- Stage 4: Recap & confirm ----------

STAGE4_TASK = """\
Current stage: 4 — Recap and confirm the commitment.

You have a commitment from the customer. Briefly recap it back to him so he
can confirm — mention the amount and timing in Arabic words. ONE or TWO
short sentences. The exact commitment text is provided below.

Commitment to recap: {commitment}
Callback time (if rescheduled): {callback}

Then call exactly one tool:
- recap_confirmed: customer confirms the recap is correct.
- recap_minor_correction(correction): customer corrects a SMALL detail
  (different specific date, time of day, exact amount typo) but the
  OVERALL plan stays the same. Pass a short description.
- wants_to_renegotiate: customer is MATERIALLY changing the commitment —
  says he can't actually pay what was just agreed, asks to split the
  amount, asks for instalments, or asks for substantially more time.
  We will return to Stage 3 to find a workable plan. This is the
  correct tool when the customer is backing out of the commitment, even
  if politely.

For clarifications, answer briefly and re-pose the recap. Do NOT default
to recap_minor_correction when the customer is actually backing out.
"""


class Stage4RecapAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        commitment = data.commitment or "(no commitment recorded)"
        callback = data.callback_time or "(none)"
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE4_TASK.format(commitment=commitment, callback=callback),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        if self.data.outcome == "rescheduled":
            hint = (
                "Recap the agreed callback time clearly. State the day in "
                "Arabic words and the rough time. Ask the customer to "
                f"confirm. Callback: {self.data.callback_time}. "
                "ONE short sentence."
            )
        else:
            hint = (
                "Recap the commitment CLEARLY and SPECIFICALLY so there "
                "is no ambiguity. Speak amounts in Arabic words and dates "
                "in Arabic words. Reiterate the FULL plan structure: "
                "(1) total amount, (2) how many instalments and the value "
                "of each (if instalments), (3) the first payment date, "
                "(4) the final payment date if applicable. End by asking "
                f"the customer to confirm.\n\n"
                f"Commitment recorded: {self.data.commitment}.\n"
                "Use TWO short sentences max."
            )
        self.session.generate_reply(instructions=hint)

    @function_tool()
    async def recap_confirmed(self, ctx: RunContext[CallData]):
        """Customer confirms the recap is correct."""
        return ClosingAgent(self.data, intent="ok", chat_ctx=None)

    @function_tool()
    async def recap_minor_correction(
        self, ctx: RunContext[CallData], correction: str
    ):
        """Customer corrected a small detail but the overall plan stays.

        Args:
            correction: short description of what the customer corrected.
        """
        ctx.userdata.commitment = (
            f"{ctx.userdata.commitment or ''} | corrected: {correction}"
        ).strip(" |")
        return ClosingAgent(self.data, intent="ok", chat_ctx=None)

    @function_tool()
    async def wants_to_renegotiate(self, ctx: RunContext[CallData]):
        """Customer is materially backing out of the commitment.

        Reset the just-recorded commitment and route back into the
        negotiation ladder at the half-payment offer (which then cascades
        to instalment if declined).
        """
        ctx.userdata.commitment = None
        ctx.userdata.outcome = None
        return Stage3HalfPaymentAgent(self.data, chat_ctx=None)


# ---------- Closing (parameterized by intent) ----------

_CLOSING_HINTS = {
    "ok": (
        "Thank the customer warmly for his cooperation, wish him well, "
        "and say goodbye. ONE short sentence."
    ),
    "paid": (
        "Acknowledge his statement that the amount is paid, say you'll "
        "verify and update the record, apologize for the bother, and "
        "say goodbye. ONE or TWO short sentences."
    ),
    "busy": (
        "Politely acknowledge the bad timing, say you'll call back at a "
        "more convenient time, and say goodbye. ONE short sentence."
    ),
    "busy_callback": (
        "Confirm the callback time you just agreed on (use the date and "
        "time_of_day already recorded — speak the date in Arabic words), "
        "thank the customer, and say goodbye. ONE or TWO short sentences."
    ),
    "dnc": (
        "Acknowledge the do-not-call request, confirm it will be recorded, "
        "apologize for the disturbance, and say goodbye. ONE short sentence."
    ),
    "death": (
        "Express sincere condolences (الله يرحمه ويغفر له), thank the "
        "caller for letting you know, and say goodbye. ONE or TWO short "
        "sentences."
    ),
    "wrong_party": (
        "Apologize for the mix-up briefly and say goodbye. ONE short sentence."
    ),
    "referred": (
        "Thank the caller warmly for the help and say goodbye. ONE short "
        "sentence."
    ),
    "id_mismatch": (
        "Apologize that the data doesn't match this account, say you "
        "won't take more of his time, and say goodbye. ONE or TWO short "
        "sentences."
    ),
    "refusal": (
        "Acknowledge his decision politely, mention the case will proceed "
        "per company policy, and say goodbye respectfully. ONE or TWO "
        "short sentences. Do NOT threaten or pressure."
    ),
    "dispute": (
        "Acknowledge that he disputes the debt, say the case will be "
        "reviewed by the relevant team and someone will follow up, and "
        "say goodbye. ONE or TWO short sentences."
    ),
}


class ClosingAgent(BaseCallAgent):
    def __init__(
        self,
        data: CallData,
        *,
        intent: str,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        hint = _CLOSING_HINTS.get(intent, _CLOSING_HINTS["ok"])
        super().__init__(
            instructions=stage_instructions(
                data,
                f"Current stage: closing (intent={intent}).\n{hint}\n"
                "Do not call any tool. Just say the parting line.",
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data
        self.intent = intent

    async def on_enter(self):
        # Stop any in-flight speech / LLM turn so the goodbye is the next
        # thing the customer hears. Without this, a user reply (e.g. asking
        # a follow-up after recap_minor_correction) starts a new LLM turn
        # that races with hangup and the goodbye never plays.
        try:
            self.session.interrupt()
        except Exception:
            pass

        # Disable mic input so further user audio doesn't spawn a new turn
        # while we're playing the goodbye + tearing down the room.
        try:
            self.session.input.set_audio_enabled(False)
        except Exception:
            pass

        hint = _CLOSING_HINTS.get(self.intent, _CLOSING_HINTS["ok"])
        handle = self.session.generate_reply(
            instructions=hint,
            allow_interruptions=False,
        )
        try:
            await handle.wait_for_playout()
        except AttributeError:
            # Older API surface — fall back to awaiting the handle directly.
            try:
                await handle
            except Exception as e:
                logger.warning(f"closing speech await error: {e}")
        except Exception as e:
            logger.warning(f"closing speech playout error: {e}")

        # Even with the mic disabled, hangup can race with stream teardown.
        # Either way the call is ending — swallow the engine-closed error.
        try:
            await self.hangup()
        except Exception as e:
            logger.warning(f"hangup race ignored: {e}")


# ---------- Entrypoint ----------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05, min_silence_duration=0.4
    )


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name} via {LIVEKIT_URL}")
    await ctx.connect()

    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error("No valid phone_number in job metadata.")
        ctx.shutdown()
        return

    if not OUTBOUND_TRUNK_ID:
        logger.error("SIP_OUTBOUND_TRUNK_ID is not set.")
        ctx.shutdown()
        return

    data = CallData(
        customer_name=dial_info.get("name", "محمد"),
        amount=dial_info.get("amount", "10000"),
        debt_date=dial_info.get("debt_date", "2023-01-01"),
        national_id_last4=dial_info.get("national_id_last4", "1234"),
    )

    participant_identity = f"sip-{phone_number}"

    session = AgentSession[CallData](
        userdata=data,
        turn_handling={
            "endpointing": {
                "mode": "dynamic",
                "min_delay": 0.2,
                "max_delay": 1.0,
            },
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_words": 2,
            },
        },
        #stt=deepgram.STT(model="nova-3", language="ar-SA"),
        stt=hamsa_livekit.STT(language="ar"),
        llm=openai.LLM(model="gpt-4.1", temperature=0.4),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ybQaNl0nzt9TjN3Oh1zzyNgp",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.9,
        ),
        vad=ctx.proc.userdata["vad"],
    )

    @session.on("error")
    def _on_error(err):
        logger.error(f"session error: {err}")

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev):
        if getattr(ev, "is_final", True):
            logger.info(f"USER  {ev.transcript!r}")

    @session.on("function_tools_executed")
    def _on_tools(ev):
        for fc in ev.function_calls:
            args = (fc.arguments or "").strip()
            if args in ("", "{}"):
                logger.info(f"TOOL  {fc.name}")
            else:
                logger.info(f"TOOL  {fc.name}  args={args}")

    @session.on("conversation_item_added")
    def _on_conv_item(ev):
        item = ev.item
        if getattr(item, "type", None) != "message":
            return
        if getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", None)
        if not text:
            return
        logger.info(f"AGENT {text!r}")

    #@session.on("metrics_collected")
    #def _on_metrics(ev):
    #    logger.info(f"metrics: {ev.metrics}")

    # Per-turn latency rollup. We stash EOU + LLM TTFT and emit one summary
    # line per response when TTS finishes. VAD per-second pings and short STT
    # chunks are dropped — they were ~85% of log volume and added no signal.
    _turn_state: dict[str, float] = {}

    @session.on("metrics_collected")
    def _on_metrics(ev):
        m = ev.metrics
        mtype = getattr(m, "type", None) or m.__class__.__name__.lower()

        if mtype == "vad_metrics":
            return
        if mtype == "stt_metrics":
            # STT timing isn't actionable for us here — EOU already covers it.
            return

        if mtype == "eou_metrics":
            eou = getattr(m, "end_of_utterance_delay", 0.0)
            tdel = getattr(m, "transcription_delay", 0.0)
            _turn_state["eou"] = eou
            _turn_state["transcript"] = tdel
            logger.info(f"EOU  {eou:.2f}s  (transcript {tdel:.2f}s)")
            return

        if mtype == "llm_metrics":
            ttft = getattr(m, "ttft", 0.0)
            pt = getattr(m, "prompt_tokens", 0)
            pc = getattr(m, "prompt_cached_tokens", 0)
            ct = getattr(m, "completion_tokens", 0)
            _turn_state["ttft"] = ttft
            cache_pct = (pc / pt * 100) if pt else 0
            logger.info(
                f"LLM  ttft {ttft:.2f}s  "
                f"prompt={pt} (cached {pc}, {cache_pct:.0f}%)  "
                f"completion={ct}"
            )
            return

        if mtype == "tts_metrics":
            ttfb = getattr(m, "ttfb", 0.0)
            audio = getattr(m, "audio_duration", 0.0)
            chars = getattr(m, "characters_count", 0)
            logger.info(
                f"TTS  ttfb {ttfb:.2f}s  audio {audio:.1f}s  chars={chars}"
            )
            eou = _turn_state.pop("eou", None)
            ttft = _turn_state.pop("ttft", None)
            if eou is not None and ttft is not None:
                total = eou + ttft + ttfb
                _turn_state.clear()
                logger.info(
                    f"TURN total {total:.2f}s  "
                    f"(EOU {eou:.2f} + TTFT {ttft:.2f} + TTFB {ttfb:.2f})"
                )
            return

    try:
        logger.info(f"dialing {phone_number} via trunk {OUTBOUND_TRUNK_ID}")
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=OUTBOUND_TRUNK_ID,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                participant_name="Phone User",
                wait_until_answered=True,
            )
        )

        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")
        data.participant = participant
        data.sip_ready.set()

        await session.start(
            agent=Stage1RightPartyAgent(data),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                participant_identity=participant_identity,
            ),
        )

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("AGENT_NAME", "outbound-caller-smart2"),
            num_idle_processes=1,
        )
    )
