"""
Arabic (Najdi) test agent — modular, deterministic call flow.

Same multi-agent handoff structure as test_agent_en.py, but spoken output
is verbatim Najdi Arabic. STT: Deepgram nova-3 ar-SA. TTS: Faseeh
ar-najdi-female-1. The instructions given to the LLM stay in English
because the model only classifies replies into tool calls — it never
generates the user-facing audio (session.say does that verbatim).
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
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice import ModelSettings
from livekit.plugins import deepgram, faseeh, openai, silero


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("outbound-caller-ar")
logger.setLevel(logging.INFO)

# Quiet down chatty third-party loggers but keep the useful livekit.agents
# DEBUG lines (user transcripts, tool calls, agent state transitions).
# A pattern filter drops the per-frame noise that doesn't help debug a call.
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
    outcome: str | None = None  # "paid_now" | "scheduled" | "partial" | "denied"
    scheduled_date: str | None = None
    sip_ready: asyncio.Event = field(default_factory=asyncio.Event)
    participant: rtc.RemoteParticipant | None = None


# ---------- Najdi pronunciation enforcement ----------
#
# The TTS provider receives raw strings — if we hand it "10000 ريال" it tends
# to read each digit ("واحد صفر صفر صفر صفر"), which sounds like a foreign bot.
# A real Najdi debt collector would say "عشرة آلاف ريال سعودي" and would spell
# verification PINs digit-by-digit. We enforce that here, in the TTS layer,
# without touching the agent prompt strings — pure regex pass on each chunk
# that flows into Agent.default.tts_node, so no extra awaits, no added latency.

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
    """Integer → Arabic words, e.g. 10000 → 'عشرة آلاف'."""
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
    """YYYY-MM-DD → 'واحد يناير ألفين وثلاثة وعشرين'."""
    try:
        y, m, d = (int(x) for x in iso.split("-"))
        if not (1 <= m <= 12):
            return iso
        day = _ar_below_1000(d) or "صفر"
        return f"{day} {_AR_MONTHS[m]} {_ar_amount_words(y)}"
    except (ValueError, IndexError):
        return iso


def _ar_digits_individual(s: str) -> str:
    """Spell digits one-by-one — for PINs / ID last-4 / verification codes."""
    return " ".join(_AR_DIGITS[int(c)] for c in s if c.isdigit())


# Brand / acronym pronunciation. Without this, the TTS reads Latin letters
# in an unnatural way ("ess-tee-see" with English phonemes mid-Arabic sentence).
_TERM_MAP = {
    "stc": "اس تي سي",
    "STC": "اس تي سي",
    "SAR": "ريال سعودي",
}


def _najdi_normalize(text: str) -> str:
    # 1) ISO dates first, so their inner digits aren't reinterpreted as amounts.
    text = re.sub(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        lambda m: _ar_date_words(m.group(1)),
        text,
    )
    # 2) Amounts adjacent to "ريال" → words + explicit "ريال سعودي".
    text = re.sub(
        r"(\d+)\s*ريال",
        lambda m: f"{_ar_amount_words(int(m.group(1)))} ريال سعودي",
        text,
    )
    # 3) Bare 4-digit sequences → PIN-style, digit-by-digit.
    text = re.sub(
        r"(?<!\d)(\d{4})(?!\d)",
        lambda m: _ar_digits_individual(m.group(1)),
        text,
    )
    # 4) Any remaining integer → Arabic words.
    text = re.sub(
        r"(?<!\d)(\d+)(?!\d)",
        lambda m: _ar_amount_words(int(m.group(1))),
        text,
    )
    # 5) Brand / acronym fixes.
    for term, repl in _TERM_MAP.items():
        text = re.sub(rf"\b{re.escape(term)}\b", repl, text)
    return text


# ---------- Base agent (shared end_call + hangup) ----------

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
    async def detected_answering_machine(self, ctx: RunContext[CallData]):
        """Call this if you hear a voicemail greeting instead of a live person."""
        logger.info("voicemail detected, hanging up")
        await self.hangup()


# ---------- Step 1: Greeting ----------

GREETING_LINE = (
    "السلام عليكم، معك نورا من شركة توافق بالنيابة عن بنك stc، "
    "هل الوقت مناسب طال عمرك؟"
)
GREETING_REPEAT = (
    "أبشر، أعيد لك: معك نورا من قسم المتابعة المالية، "
    "هل الوقت مناسب طال عمرك؟"
)
BUSY_GOODBYE = (
    "ولا يهمك، بإذن الله نتواصل معك في وقت أنسب، "
    "الله يحفظك ومع السلامة."
)


class GreetingAgent(BaseCallAgent):
    def __init__(self, data: CallData) -> None:
        super().__init__(
            instructions=(
                "You just greeted the customer in Arabic and asked if it's a good time to talk. "
                "Listen to their Arabic reply and call exactly one tool: "
                "caller_available if they say yes / تفضلي / نعم / أبشر / إيه / صح / go ahead, "
                "caller_busy if they say no / مشغول / بعدين / لاحق / اتصلي بعدين, "
                "unclear_response if they say عيدي / ما سمعت / وش تقولين / صوتك مو واضح. "
                "Do not generate any free-form Arabic. Only call a tool."
            ),
        )
        self.data = data

    async def on_enter(self):
        # Wait for SIP participant to join before speaking.
        await self.data.sip_ready.wait()
        await asyncio.sleep(0.4)  # let carrier settle
        await self.session.say(GREETING_LINE, allow_interruptions=False)

    @function_tool()
    async def caller_available(self, ctx: RunContext[CallData]):
        """User confirmed it's a good time to talk."""
        return IdentityAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def caller_busy(self, ctx: RunContext[CallData]):
        """User said it's not a good time / asked to call back later."""
        await self.session.say(BUSY_GOODBYE, allow_interruptions=False)
        await self.hangup()

    @function_tool()
    async def unclear_response(self, ctx: RunContext[CallData]):
        """User did not hear or asked you to repeat."""
        await self.session.say(GREETING_REPEAT, allow_interruptions=False)


# ---------- Step 2: Identity confirmation ----------

WRONG_PERSON_GOODBYE = (
    "عذراً على الإزعاج، الله يعطيك العافية ومع السلامة."
)


def _identity_question(name: str) -> str:
    return f"من بعد إذنك، هل أكلم الأستاذ {name}؟"


class IdentityAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"You just asked the customer (in Arabic) if you are speaking with {data.customer_name}. "
                "Listen to their reply and call exactly one tool: "
                "identity_confirmed if they say yes / نعم / إيه / صح / هو / أنا هو, "
                "wrong_person if they say no / لا / مو أنا / غلطان / مو هو, "
                "unclear if they say عيدي / ما أسمع زين / وش تقولين / صوتك مو واضح or go off-topic. "
                "Do not skip this step. Do not discuss the debt yet."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            _identity_question(self.data.customer_name),
            allow_interruptions=False,
        )

    @function_tool()
    async def identity_confirmed(self, ctx: RunContext[CallData]):
        """User confirmed they are the named customer."""
        ctx.userdata.identity_confirmed = True
        return IDVerificationAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def wrong_person(self, ctx: RunContext[CallData]):
        """User says they are not the named customer."""
        await self.session.say(WRONG_PERSON_GOODBYE, allow_interruptions=False)
        await self.hangup()

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """User asked to repeat or replied off-topic."""
        await self.session.say(
            _identity_question(self.data.customer_name),
            allow_interruptions=False,
        )


# ---------- Step 3: ID verification ----------
#
# Demo flow: the agent reads back the customer's stored last-4 digits and
# asks for a yes/no confirmation, instead of asking the customer to recite
# digits. STT on spoken digits is unreliable enough to put the original flow
# into a repeat loop. Reading-and-confirming sidesteps that. The TTS layer
# already spells bare 4-digit sequences digit-by-digit (see _najdi_normalize),
# so "1234" comes out as "واحد اثنين ثلاثة أربعة".

ID_WRONG_DIGITS_GOODBYE = (
    "نعتذر على الإزعاج، يبدو أن البيانات ما تخصك، الله يعطيك العافية ومع السلامة."
)


def _id_confirm_question(last4: str) -> str:
    return f"للتأكد، آخر ٤ أرقام من هويتك الوطنية هي {last4}، صحيح؟"


def _id_reconfirm_question(last4: str) -> str:
    # Used after the first denial — STT mishears Najdi yes/no often, so we
    # treat the first 'no' as possibly a mis-transcription and re-ask once.
    return (
        "أعتذر طال عمرك، حاب أتأكد مرة ثانية: "
        f"آخر ٤ أرقام من هويتك الوطنية هي {last4}، صحيح؟"
    )


class IDVerificationAgent(BaseCallAgent):
    # First denial is treated as a possible STT mishear and re-asks. Only the
    # second consecutive denial ends the call. Field log showed a single
    # mis-transcribed "ألا" hanging up an otherwise valid call.
    MAX_DENIES = 2

    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You just read the customer's last 4 ID digits aloud and asked them to confirm "
                "with yes or no. Listen to their Arabic reply and call exactly one tool: "
                "id_correct if they confirm (نعم / إيه / صح / أيوه / تمام / yes / صحيح), "
                "id_incorrect if they clearly deny (لا / مو صح / غلط / مو هذي / لا مو صحيح), "
                "unclear if they ask you to repeat (عيدي / ما سمعت / وش تقولين / صوتك مو واضح) "
                "or reply off-topic. Prefer 'unclear' over 'id_incorrect' when the reply is "
                "a short ambiguous token like 'ألا' that could go either way."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data
        self._deny_count = 0

    async def on_enter(self):
        await self.session.say(
            _id_confirm_question(self.data.national_id_last4),
            allow_interruptions=False,
        )

    @function_tool()
    async def id_correct(self, ctx: RunContext[CallData]):
        """Customer confirmed the last 4 digits are theirs."""
        ctx.userdata.id_verified = True
        return DebtContextAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def id_incorrect(self, ctx: RunContext[CallData]):
        """Customer denied the digits.

        First denial: re-ask once (likely STT mishear on a short Najdi token).
        Second denial: treat as wrong account and end politely.
        """
        self._deny_count += 1
        if self._deny_count >= self.MAX_DENIES:
            await self.session.say(ID_WRONG_DIGITS_GOODBYE, allow_interruptions=False)
            await self.hangup()
            return
        await self.session.say(
            _id_reconfirm_question(self.data.national_id_last4),
            allow_interruptions=False,
        )

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """User asked to repeat or replied off-topic."""
        await self.session.say(
            _id_confirm_question(self.data.national_id_last4),
            allow_interruptions=False,
        )


# ---------- Step 4: Debt context ----------

PAYMENT_LINK_LINE = (
    "ممتاز، برسل لك رابط السداد الآن، وبعدها توصلك رسالة تأكيد. "
    "يعطيك العافية وشكراً جزيلاً لك."
)
DENIED_DEBT_LINE = (
    "ولا يهمك، خلني أتأكد من بياناتك بشكل أوضح، لحظات من فضلك. "
    "تم التأكد، فعلاً المبلغ تم سداده. "
    "نعتذر على الإزعاج وشكراً جزيلاً لك."
)
ANGRY_RESPONSE = (
    "أفهم شعورك طال عمرك، وهدفنا نسهّل الموضوع عليك بدون أي ضغط. "
    "هل تقدر تسدد المبلغ كامل اليوم، أو تقدر على جزء منه؟"
)
DEBT_CLARIFY = (
    "أعتذر، ودي أفهمك صح: تقدر تسدد المبلغ كامل اليوم، "
    "أو تقدر على جزء منه فقط، أو تحتاج وقت أكثر؟"
)


def _debt_question(amount: str, debt_date: str) -> str:
    return (
        f"حبيت أذكّرك بوجود مبلغ مستحق {amount} ريال بتاريخ {debt_date}، "
        "هل تقدر تسدد المبلغ كامل اليوم؟"
    )


class DebtContextAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"You just told the customer (in Arabic) about an outstanding amount of {data.amount} SAR "
                f"from {data.debt_date} and asked if they can pay the FULL amount today. "
                "Classify their Arabic reply and call exactly one tool. "
                "Critical rule: do NOT default to non-payment when the reply is ambiguous — "
                "if you cannot tell whether they are willing or unwilling, call unclear_response. "
                "Lean toward payment-friendly tools when the customer expresses ANY willingness. "
                "\n\n"
                "Tools:\n"
                "- will_pay_full_now: customer agrees to pay the FULL amount today. Examples: "
                "نعم / إيه / موافق / جاهز / بسدد / حول لي الرابط / ما عندي مشكلة / تمام كامل / "
                "أبشر / أسددها الحين / yes / pay now.\n"
                "- wants_partial: customer is willing to pay but only PART of the amount today, "
                "or says the full amount is too much. Examples: أقدر على جزء / مو كامل / "
                "بس شي بسيط / نص المبلغ / أقدر أدفع شي / part of it / not full / can pay some.\n"
                "- cant_pay_today: customer truly cannot pay anything today and asks to be "
                "contacted later. Examples: ما عندي شي / ما أقدر اليوم / بعدين / لاحق / بكرا / "
                "الأسبوع الجاي / ما معي ولا ريال / no money at all.\n"
                "- denies_debt: customer denies owing or claims already paid. Examples: "
                "مو علي / غلط / سددته / ما هو ديني / مدفوع.\n"
                "- is_angry: customer is insulting, frustrated, or accuses you of being a bot. "
                "Examples: أنتي AI / تستهبلين / امل عليّ / why are you calling.\n"
                "- unclear_response: reply is ambiguous, off-topic, or asks you to repeat. "
                "Examples: عيدي / ما فهمت / وش تقولين / صوتك مو واضح / hmm / يمكن / ماني عارف. "
                "Use this whenever you are NOT confident — never guess.\n"
                "\n"
                "Never invent a new amount or date."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            _debt_question(self.data.amount, self.data.debt_date),
            allow_interruptions=True,
        )

    @function_tool()
    async def will_pay_full_now(self, ctx: RunContext[CallData]):
        """Customer agrees to pay the FULL amount today."""
        ctx.userdata.outcome = "paid_now"
        await self.session.say(PAYMENT_LINK_LINE, allow_interruptions=False)
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def wants_partial(self, ctx: RunContext[CallData]):
        """Customer is willing to pay but only part of the amount today.

        Drop into the half-exception step first — proposing a concrete half
        is more effective than open-ended "how much can you pay?". If they
        decline half, HalfExceptionAgent falls back to PartialPaymentAgent.
        """
        return HalfExceptionAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def cant_pay_today(self, ctx: RunContext[CallData]):
        """Customer says they cannot pay today / needs time.

        Real collectors probe before scheduling — propose half today/within
        a couple of days as an exception, then fall back through PartialPayment
        and finally Schedule. The customer often underestimates what they can
        commit to today.
        """
        return HalfExceptionAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def denies_debt(self, ctx: RunContext[CallData]):
        """Customer denies the debt or says it's already paid."""
        ctx.userdata.outcome = "denied"
        await self.session.say(DENIED_DEBT_LINE, allow_interruptions=False)
        await self.hangup()

    @function_tool()
    async def is_angry(self, ctx: RunContext[CallData]):
        """Customer is angry, frustrated, or accusing you of being a bot."""
        await self.session.say(ANGRY_RESPONSE, allow_interruptions=True)

    @function_tool()
    async def unclear_response(self, ctx: RunContext[CallData]):
        """Customer's reply is ambiguous or off-topic — re-ask with the three options."""
        await self.session.say(DEBT_CLARIFY, allow_interruptions=True)


# ---------- Step 4.5: Half-exception offer ----------
#
# Lifted from the friend's negotiation ladder (Attempt 2). Before falling to
# open-ended "how much can you pay?", a real collector proposes a CONCRETE
# half-payment as an exception. The framing ("كاستثناء") signals goodwill,
# and a specific number is psychologically easier to accept than "any
# amount". If declined, we fall through to the open-ended PartialPaymentAgent.

def _half_exception_question(amount: str) -> str:
    try:
        half = max(1, int(str(amount).strip()) // 2)
    except ValueError:
        half = 0
    return (
        f"كاستثناء طال عمرك، تقدر تسدد نصف المبلغ {half} ريال خلال يومين، "
        "والباقي بتاريخ تختاره ونتواصل بخصوصه؟"
    )


def _half_accept_line(amount: str) -> str:
    try:
        half = max(1, int(str(amount).strip()) // 2)
    except ValueError:
        half = 0
    return (
        f"ممتاز، بنسجل لك دفع نصف المبلغ {half} ريال خلال يومين، "
        "والباقي ننسقه معك في موعد قريب. شاكر لك تعاونك."
    )


class HalfExceptionAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        try:
            self._half = max(1, int(str(data.amount).strip()) // 2)
        except ValueError:
            self._half = 0
        super().__init__(
            instructions=(
                "You just offered the customer a half-payment exception in "
                f"Arabic: half the debt ({self._half} SAR) within a day or two, "
                "the rest on a date they choose. Listen to their Arabic reply "
                "and call exactly one tool. Do NOT invent a different amount.\n"
                "\n"
                "- accepts_half: customer agrees to the half-today/rest-later "
                "split (تمام / موافق / إيه / ماشي / أبشر / ماني ضد / yes).\n"
                "- declines_half: customer cannot pay even half today "
                "(لا / ما أقدر / ما عندي / مو الحين / no, even half is too much).\n"
                "- unclear: customer asks to repeat, names a different amount, "
                "or replies off-topic. Use this when no clear yes/no — never guess."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            _half_exception_question(self.data.amount),
            allow_interruptions=True,
        )

    @function_tool()
    async def accepts_half(self, ctx: RunContext[CallData]):
        """Customer agreed to pay half within 1-2 days, rest later."""
        ctx.userdata.outcome = "half_committed"
        await self.session.say(
            _half_accept_line(self.data.amount),
            allow_interruptions=False,
        )
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def declines_half(self, ctx: RunContext[CallData]):
        """Customer cannot pay half — fall back to open-ended partial."""
        return PartialPaymentAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply ambiguous — re-ask the half exception offer once."""
        await self.session.say(
            _half_exception_question(self.data.amount),
            allow_interruptions=True,
        )


# ---------- Step 5a: Schedule a payment date ----------

SCHEDULE_PROMPT = "ما فيه مشكلة، متى الوقت المناسب لك للسداد؟"
TIME_TOO_FAR = (
    "نحاول نخليها خلال فترة أقصر طال عمرك، "
    "تقدر تحدد موعد أقرب خلال شهرين؟"
)
ESCALATION_WARNING = (
    "ودي نلاقي حل سريع، "
    "متى تقدر تحدد لنا موعد للسداد؟"
)
SCHEDULE_TO_PARTIAL_HANDOFF = (
    "ولا يهمك، خلنا نشوف حل ثاني يناسبك."
)


class ScheduleAgent(BaseCallAgent):
    # Cap the number of times we re-ask for a date before falling back to a
    # partial-payment offer. Without this the agent loops on the same
    # ESCALATION_WARNING line indefinitely (seen in field log).
    MAX_REFUSALS = 2

    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        # Compute fresh "today" each time the agent is constructed so a
        # long-lived worker process doesn't anchor on stale dates. We also
        # inject today and the horizon into the LLM instructions so the model
        # can't fall back on its training-cutoff prior (we saw it pass
        # 2023-11-01 in production).
        today = datetime.date.today()
        horizon = today + datetime.timedelta(days=60)
        super().__init__(
            instructions=(
                "You are negotiating a payment date with the customer in Arabic. "
                f"Today's date is {today.isoformat()}. "
                f"Acceptable window: any date from {today.isoformat()} up to "
                f"{horizon.isoformat()} (within the next 2 months). "
                "Never propose or accept a date before today. "
                "Listen to their reply and call exactly one tool: "
                "time_acceptable(date) if their proposed date is within the window — "
                "pass an ISO YYYY-MM-DD date that is on or after today. "
                "time_too_far if they propose more than 2 months out, OR a date in "
                "the past, OR an unparseable date. "
                "refuses_to_commit if they refuse to give any date or keep evading."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data
        self._today = today
        self._horizon = horizon
        self._refusal_count = 0

    async def on_enter(self):
        await self.session.say(SCHEDULE_PROMPT, allow_interruptions=True)

    @function_tool()
    async def time_acceptable(self, ctx: RunContext[CallData], date: str):
        """The customer proposed a date within the next 2 months.

        Args:
            date: The date they proposed, as ISO YYYY-MM-DD.
        """
        try:
            proposed = datetime.date.fromisoformat(date)
        except ValueError:
            # LLM passed something we can't parse — re-ask within window.
            await self.session.say(TIME_TOO_FAR, allow_interruptions=True)
            return
        if proposed < self._today or proposed > self._horizon:
            # Out of window (past date or too far) — defend against LLM bias.
            await self.session.say(TIME_TOO_FAR, allow_interruptions=True)
            return
        iso = proposed.isoformat()
        ctx.userdata.scheduled_date = iso
        ctx.userdata.outcome = "scheduled"
        await self.session.say(
            f"تمام، بنسجل لك تاريخ {iso} كموعد للسداد، "
            "ونتواصل معك بإذن الله قريب من الموعد.",
            allow_interruptions=False,
        )
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def time_too_far(self, ctx: RunContext[CallData]):
        """The customer's proposed date is more than 2 months away."""
        await self.session.say(TIME_TOO_FAR, allow_interruptions=True)

    @function_tool()
    async def refuses_to_commit(self, ctx: RunContext[CallData]):
        """Customer refuses to give any concrete date."""
        self._refusal_count += 1
        if self._refusal_count >= self.MAX_REFUSALS:
            # Two refusals is enough — pivot to a partial-payment offer
            # instead of looping the escalation line.
            await self.session.say(
                SCHEDULE_TO_PARTIAL_HANDOFF, allow_interruptions=True
            )
            return PartialPaymentAgent(self.data, chat_ctx=self.chat_ctx)
        await self.session.say(ESCALATION_WARNING, allow_interruptions=True)


# ---------- Step 5b: Partial payment offer ----------

PARTIAL_PROMPT = (
    "ما فيه مشكلة طال عمرك، نقدر نسهلها عليك. "
    "كم تقدر تدفع الآن؟ أي مبلغ يساعدنا نرتب لك خطة للباقي."
)
PARTIAL_REPROMPT = (
    "تكفى، أعطني مبلغ تقريبي تقدر تدفعه اليوم — حتى لو بسيط."
)


class PartialPaymentAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You just asked the customer (in Arabic) how much they can pay today, "
                "open-endedly. Do NOT suggest an amount yourself — wait for the customer to "
                "name one. Listen to their Arabic reply and call exactly one tool.\n"
                "\n"
                "- partial_amount(amount): customer named a concrete amount they can pay today. "
                "Pass digits only as a string. Examples: '50', '100', '500', '٢٠٠', "
                "خمسين / مية / مئتين / خمس مية / ألف / two hundred. Even very small amounts count.\n"
                "- cant_pay_anything: customer says they truly cannot pay any amount today. "
                "Examples: ولا ريال / ما معي شي / مفلس / ما أقدر أدفع أي مبلغ today.\n"
                "- unclear: reply is ambiguous, off-topic, or asks to repeat. "
                "Use this if no concrete amount and no clear refusal — never guess an amount.\n"
                "\n"
                "Critical: a customer who says 'I'll try' or 'maybe a little' WITHOUT a number "
                "is unclear, not partial_amount. Always require a concrete number."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(PARTIAL_PROMPT, allow_interruptions=True)

    @function_tool()
    async def partial_amount(self, ctx: RunContext[CallData], amount: str):
        """Customer named a concrete partial amount they can pay today.

        Args:
            amount: The amount as digits only, e.g. '500'.
        """
        ctx.userdata.outcome = "partial"
        await self.session.say(
            f"شاكر لك على تعاونك. برسل لك رابط السداد لمبلغ {amount} ريال الآن، "
            "وبنرتب لك خطة سداد للمبلغ المتبقي ونتواصل معك بإذن الله.",
            allow_interruptions=False,
        )
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def cant_pay_anything(self, ctx: RunContext[CallData]):
        """Customer truly cannot pay any amount today."""
        return ScheduleAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply was ambiguous or no concrete amount given — re-ask."""
        await self.session.say(PARTIAL_REPROMPT, allow_interruptions=True)


# ---------- Step 6: Closing ----------

CLOSING_LINE = "شاكر لك تعاونك، الله يجزاك خير ومع السلامة."


class ClosingAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions="The call is ending. Do not generate any reply.",
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(CLOSING_LINE, allow_interruptions=False)
        await self.hangup()


# ---------- Entrypoint ----------

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
        stt=deepgram.STT(model="nova-3", language="ar-SA"),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ybQaNl0nzt9TjN3Oh1zzyNgp",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.9,
        ),
        vad=silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.4),
    )

    @session.on("error")
    def _on_error(err):
        logger.error(f"session error: {err}")

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
            agent=GreetingAgent(data),
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
            agent_name=os.getenv("AGENT_NAME", "outbound-caller-ar"),
            num_idle_processes=1,
        )
    )
