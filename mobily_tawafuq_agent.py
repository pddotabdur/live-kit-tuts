"""
Tawafuq / Mobily debt-collection agent (Najdi Arabic, system-prompt-driven).

Implements the workflow in `mobily-debt-collection-logic.md`:
  Stage 1 — right-party verification + ID-last4 yes/no check
  Stage 2 — QA recording disclosure + debt intro + reason capture
  Stage 3 — negotiation ladder: full today/tomorrow → half exception → instalments
  Stage 4 — recap + payment methods (SADAD biller code 005) + close

STT is selectable via the STT_PROVIDER env var ("munsit" or "soniox") so we can
A/B latency. Munsit STT runs in streaming mode for the call. TTS is Faseeh.

Per-turn latency is logged tagged with the active STT provider; target is
total < 1.5 s (EOU + LLM TTFT + TTS TTFB).
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
from livekit.plugins import faseeh, munsit, openai, silero, soniox


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("mobily-tawafuq")
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
STT_PROVIDER = os.getenv("STT_PROVIDER", "soniox").strip().lower()


# ---------- Shared call state ----------

@dataclass
class CallData:
    name: str = "محمد"
    first_name: str | None = None
    gender: str | None = None  # "male" | "female" | None
    amount: str = "1500"
    national_id_last4: str = "1234"
    phone_number: str | None = None

    # Per-spec "services" array used when customer denies recognising the
    # debt — we explain 1-2 service items, last 4 digits of the service number.
    services: list[dict] = field(default_factory=list)

    # Stage state
    identity_confirmed: bool = False
    id_verified: bool = False
    id_yes_no_attempts: int = 0
    consequences_said: bool = False
    debt_clarification_given: bool = False
    ladder_attempt: int = 0  # 0=not started, 1=full, 2=half, 3=instalments
    vague_attempts: int = 0

    # Outcomes
    outcome: str | None = None
    commitment: str | None = None
    callback_time: str | None = None
    referrer_mobile: str | None = None
    payment_link_sent: bool = False

    sip_ready: asyncio.Event = field(default_factory=asyncio.Event)
    participant: rtc.RemoteParticipant | None = None


# ---------- Side-effect hooks (stub — wire to provider) ----------

def _emit_outcome(data: "CallData", kind: str, **fields) -> None:
    payload = {
        "customer": data.name,
        "phone": data.phone_number,
        "kind": kind,
        **fields,
    }
    logger.info(f"OUTCOME {json.dumps(payload, ensure_ascii=False)}")


def _send_sms_payment_link(data: "CallData", when: str | None = None) -> None:
    if not data.phone_number:
        return
    logger.info(
        f"SMS payment_link to={data.phone_number} amount={data.amount} when={when}"
    )
    data.payment_link_sent = True


# ---------- Najdi number/date pronunciation helpers ----------

_AR_UNITS = ["", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة",
             "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_AR_TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر",
             "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_AR_TENS = ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين",
            "ستين", "سبعين", "ثمانين", "تسعين"]
_AR_HUNDREDS = ["", "مية", "مئتين", "ثلاث مية", "أربع مية", "خمس مية",
                "ست مية", "سبع مية", "ثمان مية", "تسع مية"]
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


def _ar_digits_individual(s: str) -> str:
    return " ".join(_AR_DIGITS[int(c)] for c in s if c.isdigit())


_TERM_MAP = {
    "Mobily": "موبايلي",
    "MOBILY": "موبايلي",
    "mobily": "موبايلي",
    "Tawafuq": "توافق",
    "TAWAFUQ": "توافق",
    "tawafuq": "توافق",
    "SADAD": "سداد",
    "Sadad": "سداد",
    "SAR": "ريال سعودي",
    "SIMAH": "سمة",
    "simah": "سمة",
    "ATM": "الصراف الآلي",
}

_PRONUNCIATION_MAP = {
    "سمة": "سِمَه",
    "موبايلي": "مُوبَايلي",
    "توافق": "تَوَافُقْ",
    "أبشر": "أَبْشِرْ",
    "ابشر": "أَبْشِرْ",
    "تفضل": "تَفَضَّلْ",
    "للأسف": "للأَسَفْ",
    "بكرة": "بُكْرَهْ",
    "هلا": "هَلَا",
    "أمرني": "اَمْرُنِي",
    "دقيقة": "دَقِيقَة",
    "تسعة": "تِسْعَة",
    "عشرة": "عَشْرَة",
    "ثمانية": "ثمانيه",
    "سبعة": "سبعه",
    "خمسة": "خمسه",
    "اربعة": "اربعه",
    "ثلاثة": "ثلاثه",
    "الساعة": "الساعه",
}

_PRONUNCIATION_SORTED = sorted(_PRONUNCIATION_MAP.items(), key=lambda x: -len(x[0]))


def _apply_pronunciation(text: str) -> str:
    for original, replacement in _PRONUNCIATION_SORTED:
        text = text.replace(original, replacement)
    return text


def _najdi_normalize(text: str) -> str:
    # SADAD biller code 005 — speak as "صفر صفر خمسة" not "خمسة".
    text = re.sub(
        r"(?i)\b(?:biller code|كود المفوتر|الكود)\s*0*5\b",
        "كود المفوتر صفر صفر خمسة",
        text,
    )
    # Plain amounts → words + "ريال سعودي".
    text = re.sub(
        r"(\d+)\s*ريال",
        lambda m: f"{_ar_amount_words(int(m.group(1)))} ريال سعودي",
        text,
    )
    # 4-digit groups (ID, phone fragments) → digit-by-digit.
    text = re.sub(
        r"(?<!\d)(\d{4})(?!\d)",
        lambda m: _ar_digits_individual(m.group(1)),
        text,
    )
    # Other bare integers → words.
    text = re.sub(
        r"(?<!\d)(\d+)(?!\d)",
        lambda m: _ar_amount_words(int(m.group(1))),
        text,
    )
    for term, repl in _TERM_MAP.items():
        text = re.sub(rf"\b{re.escape(term)}\b", repl, text)
    text = _apply_pronunciation(text)
    return text


# ---------- Title / addressed-name helpers (gender-aware per spec) ----------

def _title_with_name(data: CallData) -> str:
    """Stage-1 form: full name with gendered title when known."""
    g = (data.gender or "").lower()
    if g == "male":
        return f"الأستاذ {data.name}"
    if g == "female":
        return f"الأستاذة {data.name}"
    return data.name


def _addressed_name(data: CallData) -> str:
    """Stage-2 form: short, gendered when known, first name only."""
    first = data.first_name or data.name.split()[0]
    g = (data.gender or "").lower()
    if g == "male":
        return f"أستاذ {first}"
    if g == "female":
        return f"أستاذة {first}"
    return first


# ---------- Persona (system prompt, all stages share this) ----------

PERSONA = """\
You are نورا (Nora), a female collections specialist at شركة توافق calling on
behalf of شركة موبايلي. You ALWAYS reply in Najdi Saudi Arabic; English only
for brand names (Mobily, Tawafuq, SIMAH, SADAD).

ABSOLUTE RULES:
- Brevity: ONE short sentence per turn (max ~14 words). No preambles, no
  explanations, no monologues.
- Ask ONE question at a time, wait for the answer.
- NEVER say "والله". NEVER use oaths or threats.
- Soft persuasion only — never commands or pressure.
- NEVER name an amount before the customer has named one (Stage 3).
- Mid-call salaam: if the customer says "السلام عليكم" mid-conversation,
  reply with just "أبشر" or "تفضل" and continue. Do NOT say "وعليكم
  السلام" — that sounds like restarting the call.
- Use the customer's name only on greeting and on closing.
- If asked if you are AI: "نعم أنا مساعد آلي من توافق" — one sentence,
  then back to the question.

STRICTLY FORBIDDEN:
- "لازم تسدد", "هذا إنذار", "إجراء قانوني", or any threatening language.
- Never shame, threaten, guilt, or pressure the customer.

CHALLENGE HANDLING (Stage 1 specific — answer in ONE short sentence):
- Who are you? → "معك نورا من توافق، نتصل بالنيابة عن موبايلي."
- Why are you calling? → "بخصوص حسابك في موبايلي."
- Where did you get my number? → "رقمك من السجلات الرسمية لدى موبايلي،
  وتقدر تتحقق عن طريق ١١٠٠."

EMPATHY (brief, genuine, EVERY stage where customer shares hardship):
- One short empathetic phrase, then move forward immediately.
- Templates: "مقدّر وضعك", "ولا يهمك", "أفهم شعورك".

NUMBERS / DATES / IDENTITY:
- Amounts in Arabic words ("خمس مية ريال").
- ID/PIN: spell digit-by-digit.
- Dates in Arabic words. Translate "بكرا", "نهاية الشهر" to a SPECIFIC
  ISO date and confirm.

COMPLIANCE (Stage 1):
- Until identity is verified, NEVER disclose the debt amount or any
  account detail. No threats, no legal scare language.

TOOLS:
- When the customer's reply triggers a transition described in the stage
  section, CALL THE TOOL. Do not keep talking.
- For clarifications, reply in ONE short sentence and re-pose the
  current question without a tool.
"""


_AR_WEEKDAYS = {
    "Monday": "الإثنين", "Tuesday": "الثلاثاء", "Wednesday": "الأربعاء",
    "Thursday": "الخميس", "Friday": "الجمعة", "Saturday": "السبت",
    "Sunday": "الأحد",
}


def call_context_block(data: CallData) -> str:
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    end_of_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
    next_week = today + datetime.timedelta(days=7)

    services_block = "\n".join(
        f"  - service #{i+1}: type={s.get('type','-')} "
        f"subscription_date={s.get('subscription_date','-')} "
        f"service_number_last4={s.get('service_number_last4','-')}"
        for i, s in enumerate(data.services or [])
    ) or "  (no service details on file)"

    return (
        "Call data (use ONLY these values when speaking about the debt):\n"
        f"- Customer name: {data.name}\n"
        f"- Title-with-name to use in Stage 1: {_title_with_name(data)}\n"
        f"- Short addressed name to use in Stage 2: {_addressed_name(data)}\n"
        f"- Outstanding amount: {data.amount} SAR\n"
        f"- Last 4 digits of national ID on file: {data.national_id_last4}\n"
        f"- Services on file (used ONLY if customer denies recognising the debt):\n"
        f"{services_block}\n"
        "\n"
        "Dates context (resolve relative time references against these):\n"
        f"- Today: {today.isoformat()} ({_AR_WEEKDAYS.get(today.strftime('%A'), today.strftime('%A'))})\n"
        f"- Tomorrow: {tomorrow.isoformat()} ({_AR_WEEKDAYS.get(tomorrow.strftime('%A'), tomorrow.strftime('%A'))})\n"
        f"- End of this month: {end_of_month.isoformat()}\n"
        f"- One week from today: {next_week.isoformat()}\n"
    )


def stage_instructions(data: CallData, stage_task: str) -> str:
    return f"{PERSONA}\n\n{call_context_block(data)}\n\n{stage_task}"


# ---------- Base agent (TTS normalization + hangup helper) ----------

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


# ============================================================
# Stage 1 — Right-party verification
# ============================================================

STAGE1_TASK = """\
Current stage: 1 — Right-party verification.

You just opened the call with the script-fixed line:
  "هلا، معي {title_with_name}؟"
Listen to the reply.

Then call EXACTLY one tool based on what you hear:
- right_party: customer confirms they ARE {title_with_name} (نعم / أيوه / أنا
  / صحيح). "تفضلي" alone is NOT confirmation — re-ask politely.
- wrong_party: caller is NOT the named person.
- caller_busy: bad time / asks for callback / مشغول / بعدين.
- do_not_call: asks not to be contacted again / DNC.
- customer_deceased: caller informs the named person has passed away.

Off-topic / clarification → answer in ONE short sentence then re-pose the
right-party question. Do NOT call a tool.
"""


class Stage1RightPartyAgent(BaseCallAgent):
    def __init__(self, data: CallData) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE1_TASK.format(title_with_name=_title_with_name(data)),
            ),
        )
        self.data = data

    async def on_enter(self):
        await self.data.sip_ready.wait()
        await asyncio.sleep(0.4)
        self.session.generate_reply(
            instructions=(
                f'Open the call with EXACTLY: "هلا، معي {_title_with_name(self.data)}؟" '
                "— nothing more, no greetings, no introductions."
            )
        )

    @function_tool()
    async def right_party(self, ctx: RunContext[CallData]):
        """Customer confirmed they are the named person."""
        ctx.userdata.identity_confirmed = True
        return Stage1IDYesNoAgent(self.data, chat_ctx=None)

    @function_tool()
    async def wrong_party(self, ctx: RunContext[CallData]):
        """Caller is NOT the named person."""
        return WrongPartyKnowsAgent(self.data, chat_ctx=None)

    @function_tool()
    async def caller_busy(self, ctx: RunContext[CallData]):
        """Customer says it's not a good time."""
        ctx.userdata.outcome = "busy"
        return ScheduleCallbackAgent(self.data, chat_ctx=None)

    @function_tool()
    async def do_not_call(self, ctx: RunContext[CallData]):
        """Customer requested DNC."""
        ctx.userdata.outcome = "dnc"
        return ClosingAgent(self.data, intent="dnc", chat_ctx=None)

    @function_tool()
    async def customer_deceased(self, ctx: RunContext[CallData]):
        """Caller reports the named person has passed away."""
        ctx.userdata.outcome = "death"
        return ClosingAgent(self.data, intent="death", chat_ctx=None)


# ---------- Stage 1: ID last4 (YES/NO question, per spec) ----------

ID_YES_NO_TASK = """\
Current stage: 1 — ID last4 yes/no verification.

You will read the last 4 digits of the customer's national ID and ask him to
confirm them as a YES/NO question. The digits to read (digit-by-digit in
Arabic) are stored in the call data.

First-time prompt:
  "للتأكيد، آخر ٤ أرقام من الهوية أو الإقامة هي [digits], صحيح؟"

If unclear / asks to repeat ONCE:
  "أعيد لك: [digits]، صحيح؟"

Then call EXACTLY one tool:
- id_confirmed: customer says yes / صح / نعم / أيوه / تمام.
- id_denied: customer says no / مو صحيح / غلط.
- unclear: ambiguous / off-topic — re-pose the same yes/no question once.
- refuses_to_verify: refuses to verify / "ما أعطيك معلومات".

Two unclear/refusal turns ⇒ end the call politely (handled by the tool).

Compliance: You may NOT mention the debt amount or any account detail
until the customer confirms the digits.
"""


class Stage1IDYesNoAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, ID_YES_NO_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        digits_words = _ar_digits_individual(self.data.national_id_last4)
        self.session.generate_reply(
            instructions=(
                f'Ask the yes/no verification question EXACTLY: '
                f'"للتأكيد، آخر ٤ أرقام من الهوية أو الإقامة هي {digits_words}، صحيح؟" '
                "Nothing more."
            )
        )

    @function_tool()
    async def id_confirmed(self, ctx: RunContext[CallData]):
        """Customer confirmed the last 4 digits are correct."""
        ctx.userdata.id_verified = True
        return Stage2DebtIntroAgent(self.data, chat_ctx=None)

    @function_tool()
    async def id_denied(self, ctx: RunContext[CallData]):
        """Customer says the last 4 digits are NOT correct."""
        ctx.userdata.outcome = "id_mismatch"
        _emit_outcome(ctx.userdata, "id_mismatch")
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply ambiguous — re-ask the SAME yes/no question once."""
        ctx.userdata.id_yes_no_attempts += 1
        if ctx.userdata.id_yes_no_attempts >= 2:
            ctx.userdata.outcome = "verify_failed"
            _emit_outcome(ctx.userdata, "verify_failed")
            return ClosingAgent(self.data, intent="verify_refused", chat_ctx=None)
        digits_words = _ar_digits_individual(self.data.national_id_last4)
        self.session.generate_reply(
            instructions=(
                f'Repeat the digits and ask yes/no once more: '
                f'"أعيد لك: {digits_words}، صحيح؟" Nothing more.'
            )
        )

    @function_tool()
    async def refuses_to_verify(self, ctx: RunContext[CallData]):
        """Customer refuses verification."""
        ctx.userdata.outcome = "verify_refused"
        _emit_outcome(ctx.userdata, "verify_refused")
        return ClosingAgent(self.data, intent="verify_refused", chat_ctx=None)


# ---------- Stage 1: wrong-party referral path ----------

WRONG_PARTY_KNOWS_TASK = """\
Current stage: 1b — wrong party, find out if they can refer us.

The person on the line is NOT {title_with_name}. Politely ask:
  "آسفة على الإزعاج — تعرف {title_with_name}؟"

Then call EXACTLY one tool:
- knows_person: caller says yes, knows him.
- does_not_know_person: no / never heard of him.
"""


class WrongPartyKnowsAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                WRONG_PARTY_KNOWS_TASK.format(title_with_name=_title_with_name(data)),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                f'Apologise and ask exactly: "آسفة على الإزعاج — تعرف '
                f'{_title_with_name(self.data)}؟" One short sentence.'
            )
        )

    @function_tool()
    async def knows_person(self, ctx: RunContext[CallData]):
        """Caller knows the named person."""
        return CollectMobileAgent(self.data, chat_ctx=None)

    @function_tool()
    async def does_not_know_person(self, ctx: RunContext[CallData]):
        """Caller does not know the named person."""
        ctx.userdata.outcome = "wrong_party"
        _emit_outcome(ctx.userdata, "contact_invalid")
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)


COLLECT_MOBILE_TASK = """\
Current stage: 1c — collect a referral mobile number.

Ask: "تقدر تشاركنا رقم جواله المكوّن من ١٠ أرقام للوصول إليه؟"
When digits arrive, repeat ONCE to confirm: "أعيد الرقم للتأكيد: [number]، صحيح؟"

Tools:
- mobile_provided(number): caller spoke a number. Pass digits-only string.
  Saudi mobiles start with 05 and are 10 digits. Pass whatever digits you heard.
- refuses_to_provide: caller declined.
"""


class CollectMobileAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, COLLECT_MOBILE_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                'Ask: "تقدر تشاركنا رقم جواله المكوّن من ١٠ أرقام للوصول إليه؟" '
                "One short sentence."
            )
        )

    @function_tool()
    async def mobile_provided(self, ctx: RunContext[CallData], number: str):
        """Caller gave a mobile number."""
        clean = re.sub(r"\D", "", number.translate(_AR_INDIC_DIGITS))
        ctx.userdata.referrer_mobile = clean
        ctx.userdata.outcome = "wrong_party_referred"
        return ClosingAgent(self.data, intent="referred", chat_ctx=None)

    @function_tool()
    async def refuses_to_provide(self, ctx: RunContext[CallData]):
        """Caller declined to share a number."""
        ctx.userdata.outcome = "wrong_party"
        _emit_outcome(ctx.userdata, "contact_invalid")
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)


# ---------- Schedule callback (busy at start) ----------

SCHEDULE_CALLBACK_TASK = """\
Current stage: schedule a callback for a busy customer.

Ask: "ولا يهمك — أي وقت يناسبك أكلمك فيه؟" — encourage him to name a day
and a rough time of day. Translate "بكرا الصبح" / "بعد العصر" to ISO date
+ time-of-day label, then confirm verbally.

Tools:
- callback_time(day_iso, time_of_day)
- refuses_to_schedule
- unclear  (re-ask once)
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
                'Ask: "ولا يهمك — أي وقت يناسبك أكلمك فيه؟" One short sentence.'
            )
        )

    @function_tool()
    async def callback_time(
        self, ctx: RunContext[CallData], day_iso: str, time_of_day: str,
    ):
        """Customer named a callback day + time-of-day.

        Args:
            day_iso: ISO date (YYYY-MM-DD).
            time_of_day: 'morning' / 'afternoon' / 'evening' / HH:MM.
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
        """Reply ambiguous — re-ask once."""
        self.session.generate_reply(
            instructions='Re-ask: "متى يناسبك أكلمك؟" One short sentence.'
        )


# ============================================================
# Stage 2 — QA disclosure + debt intro + reason capture
# ============================================================

STAGE2_TASK = """\
Current stage: 2 — QA disclosure + debt intro + reason.

You MUST run TWO short script-fixed turns in sequence (one turn each):

Turn A (QA disclosure):
  "شكراً {addressed_name}، هذه المكالمة قد تكون مسجّلة لأغراض الجودة."

Turn B (debt intro + reason question):
  "أكلمك بخصوص حسابك في موبايلي — عليك مبلغ متأخر قدره {amount} ريال
  ولم يُسدد. ايش سبب التأخير؟"

If the customer answers Turn B with bare "نعم/طيب/تمام/اوكي" without a reason,
re-ask ONCE: "أقصد، إيش الذي يمنعك من السداد؟"

If the customer says he ALREADY PAID (سددت / دفعت / تم الدفع):
  → call already_paid.

If the customer says he does NOT recognise / know the debt
("ما أعرف هالفاتورة", "أنا ما عليّ شي"):
  → call clarify_debt FIRST. After clarification, re-ask: "بعد التوضيح،
    إيش سبب التأخير؟" Do NOT treat as dispute yet.

If the customer disputes flatly after clarification (still says "مو حقي"):
  → call disputes_debt.

Otherwise, when a reason is captured (or he denies/refuses to engage):
  → call reason_captured.

TOOLS:
- already_paid
- clarify_debt: explain 1-2 service items from the services array using
  amount + subscription_date + last4 of service number, then re-ask reason.
  Call ONCE; after this, treat further denial as disputes_debt.
- disputes_debt
- reason_captured: any concrete reason / denial / hardship — go to Stage 3.
- unclear: re-pose the previous question.
"""


class Stage2DebtIntroAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE2_TASK.format(
                    addressed_name=_addressed_name(data),
                    amount=data.amount,
                ),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        # Turn A: QA disclosure (alone). The LLM will then be asked to run
        # turn B on the next user input by following the stage script.
        self.session.generate_reply(
            instructions=(
                f'Say EXACTLY: "شكراً {_addressed_name(self.data)}، هذه '
                'المكالمة قد تكون مسجّلة لأغراض الجودة." Then ask the '
                f'debt intro right after as a single connected reply: '
                f'"أكلمك بخصوص حسابك في موبايلي — عليك مبلغ متأخر قدره '
                f'{self.data.amount} ريال ولم يُسدد. ايش سبب التأخير؟"'
            )
        )

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims it's already paid."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def clarify_debt(self, ctx: RunContext[CallData]):
        """Customer says he doesn't recognise the debt — explain service
        details (amount, subscription date, last4 of service number) and
        re-ask the reason. Call this ONCE per call."""
        if ctx.userdata.debt_clarification_given:
            # Second denial after clarification — treat as dispute.
            ctx.userdata.outcome = "dispute"
            _emit_outcome(ctx.userdata, "dispute_open")
            return ClosingAgent(self.data, intent="dispute", chat_ctx=None)

        ctx.userdata.debt_clarification_given = True
        services = self.data.services[:2] or [{}]
        lines = []
        for s in services:
            stype = s.get("type", "خدمة")
            sub = s.get("subscription_date", "غير متوفر")
            last4 = s.get("service_number_last4", "")
            tail = f"وآخر ٤ أرقام {last4}" if last4 else ""
            lines.append(f"{stype} منذ {sub} {tail}".strip())
        explain = "؛ ".join(lines)
        self.session.generate_reply(
            instructions=(
                f'Explain briefly: "تفاصيل المبلغ: {explain}." Then ask: '
                f'"بعد التوضيح، إيش سبب التأخير؟" TWO short sentences max.'
            )
        )

    @function_tool()
    async def disputes_debt(self, ctx: RunContext[CallData]):
        """Customer disputes the debt after clarification."""
        ctx.userdata.outcome = "dispute"
        _emit_outcome(ctx.userdata, "dispute_open")
        return ClosingAgent(self.data, intent="dispute", chat_ctx=None)

    @function_tool()
    async def reason_captured(self, ctx: RunContext[CallData]):
        """A reason / denial / hardship was given — go to Stage 3."""
        return Stage3NegotiationAgent(self.data, chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Re-pose the reason question once."""
        self.session.generate_reply(
            instructions='Re-ask: "أقصد، إيش الذي يمنعك من السداد؟" One short sentence.'
        )


# ============================================================
# Stage 3 — Negotiation ladder (full → half → instalments)
# ============================================================

STAGE3_TASK = """\
Current stage: 3 — Negotiation ladder.

Outstanding amount: {amount} SAR.

DETERMINISTIC FIRST TURN (consequences line — say ONCE per call, not again):
  "للعلم، نحتاج تسوية الموضوع خلال ٧ أيام، وإلا قد يؤثر على سجلك الائتماني
  ويُرفع لـ سمة حسب الإجراءات."

After that line, immediately move into Attempt 1.

LADDER (one short question per turn, ack briefly between turns):

Attempt 1 — full today/tomorrow:
  "تقدر تسدد المبلغ كامل اليوم أو بكرا؟"
  • Yes (full today/tomorrow) → call commit_full(when_iso).
  • No / can't → ack briefly, then Attempt 2.

Attempt 2 — half exception:
  "كاستثناء، تقدر تسدد النصف اليوم أو خلال يومين، والباقي بتاريخ تختاره؟"
  • Yes → ask the date for the second half, then call commit_two_step(
    first_amount, first_date_iso, rest_amount, rest_date_iso).
  • No → Attempt 3.

Attempt 3 — customer-named instalments:
  "كم أقل مبلغ تقدر تلتزم فيه، وفي أي تاريخ بالضبط؟"
  Then: "والباقي متى تقدر تسدده؟"
  • Plan agreed → call commit_two_step(...).
  • Vague after ONE clarification → call commit_partial_then_callback(
    first_amount, first_date_iso) if at least the initial is set, else
    call vague_response.

VAGUE TIMING ("آخر الشهر / هذا الأسبوع / لما ينزل الراتب"):
  Ask ONE clarifying question to pin an exact day:
    "أي يوم بالضبط؟ مثلاً ٣٠؟"
  Never assume a date. If still vague after one clarification → callback.

OUTCOMES (call EXACTLY one tool):
- commit_full(when_iso)
- commit_two_step(first_amount, first_date_iso, rest_amount, rest_date_iso)
- commit_partial_then_callback(first_amount, first_date_iso) — initial
  agreed but rest is vague; we'll follow up.
- vague_response — no concrete number/date after offers.
- refuses_payment — flat refusal.
- disputes_debt — claims it's not his / wrong amount.
- already_paid — asserts already paid.
- unclear — ambiguous; re-ask the SAME question.

HARD RULES:
- ONE short sentence per turn (8-14 words).
- Acknowledgements are ONE word: "أبشر" / "زين" / "تمام" / "طيب".
- Hardship → ONE-word ack, then next question.
- Mention سمة ONLY in the consequences line above. Never repeat it.
"""


class Stage3NegotiationAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        try:
            self._amount_int = int(str(data.amount).strip())
        except ValueError:
            self._amount_int = 0
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE3_TASK.format(amount=data.amount),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        if not self.data.consequences_said:
            self.data.consequences_said = True
            self.session.generate_reply(
                instructions=(
                    'Say the consequences line ONCE then immediately ask '
                    'Attempt 1 (full today/tomorrow) as a single connected reply: '
                    '"للعلم، نحتاج تسوية الموضوع خلال ٧ أيام، وإلا قد يؤثر '
                    'على سجلك الائتماني ويُرفع لـ سمة حسب الإجراءات. تقدر '
                    'تسدد المبلغ كامل اليوم أو بكرا؟"'
                )
            )
        else:
            self.session.generate_reply(
                instructions='Ask Attempt 1: "تقدر تسدد المبلغ كامل اليوم أو بكرا؟"'
            )

    @function_tool()
    async def commit_full(self, ctx: RunContext[CallData], when_iso: str):
        """Customer commits to pay the FULL outstanding amount in one transfer.

        Args:
            when_iso: ISO date (YYYY-MM-DD) of the single payment.
        """
        ctx.userdata.commitment = (
            f"full payment of {self.data.amount} SAR on {when_iso}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def commit_two_step(
        self,
        ctx: RunContext[CallData],
        first_amount: float,
        first_date_iso: str,
        rest_amount: float,
        rest_date_iso: str,
    ):
        """Customer agreed to a two-step plan.

        Args:
            first_amount: SAR for the first payment.
            first_date_iso: ISO date of the first payment.
            rest_amount: SAR for the remainder.
            rest_date_iso: ISO date of the remainder.
        """
        ctx.userdata.commitment = (
            f"first {int(first_amount)} SAR on {first_date_iso}, "
            f"remainder {int(rest_amount)} SAR on {rest_date_iso}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def commit_partial_then_callback(
        self,
        ctx: RunContext[CallData],
        first_amount: float,
        first_date_iso: str,
    ):
        """Initial payment agreed but the remainder is vague — capture
        what we have and schedule a follow-up call.

        Args:
            first_amount: SAR for the initial good-faith payment.
            first_date_iso: ISO date of the first payment.
        """
        remainder = max(0, self._amount_int - int(first_amount))
        ctx.userdata.commitment = (
            f"initial {int(first_amount)} SAR on {first_date_iso}, "
            f"remainder {remainder} SAR TBD"
        )
        ctx.userdata.outcome = "committed_partial"
        return RescheduleAgent(self.data, chat_ctx=None)

    @function_tool()
    async def vague_response(self, ctx: RunContext[CallData]):
        """No concrete amount/date after offers — go to callback scheduling."""
        ctx.userdata.vague_attempts += 1
        return RescheduleAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_payment(self, ctx: RunContext[CallData]):
        """Flat refusal — close politely (no pressure)."""
        ctx.userdata.outcome = "refusal"
        return ClosingAgent(self.data, intent="refusal", chat_ctx=None)

    @function_tool()
    async def disputes_debt(self, ctx: RunContext[CallData]):
        """Customer disputes the debt."""
        ctx.userdata.outcome = "dispute"
        _emit_outcome(ctx.userdata, "dispute_open")
        return ClosingAgent(self.data, intent="dispute", chat_ctx=None)

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer asserts the debt is already paid."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply ambiguous — re-ask the same question."""
        self.session.generate_reply(
            instructions="Re-ask the SAME question, slightly rephrased. ONE short sentence."
        )


# ---------- Reschedule (callback for the rest) ----------

RESCHEDULE_TASK = """\
Current stage: reschedule callback.

You are scheduling a FOLLOW-UP CALL (not a payment). Ask:
  "متى يناسبك نتواصل معك لتأكيد التاريخ النهائي؟"
Use "نتواصل معك" / "نكلمك" — NEVER "تدفع".

Tools:
- callback_scheduled(when): pass short description ('next Monday morning'
  or ISO date).
- refuses_to_schedule.
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
                'Ask: "متى يناسبك نتواصل معك لتأكيد التاريخ النهائي؟" '
                "One short sentence."
            )
        )

    @function_tool()
    async def callback_scheduled(self, ctx: RunContext[CallData], when: str):
        """Customer named a callback time."""
        ctx.userdata.callback_time = when
        if not ctx.userdata.outcome:
            ctx.userdata.outcome = "rescheduled"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_to_schedule(self, ctx: RunContext[CallData]):
        """Customer won't commit to a callback."""
        ctx.userdata.outcome = "refusal"
        return ClosingAgent(self.data, intent="refusal", chat_ctx=None)


# ============================================================
# Stage 4 — Recap + payment methods + close
# ============================================================

STAGE4_TASK = """\
Current stage: 4 — Recap + payment methods + confirmation.

Deliver as a SINGLE concise reply (max ~3 short sentences, no monologue):

  "للتأكيد، الاتفاق هو: {plan_summary}. تقدر تسدد عبر سداد باستخدام رقم
  الهوية الوطنية وكود المفوتر صفر صفر خمسة، أو من تطبيق البنك، أو
  تطبيق موبايلي، أو أقرب فرع موبايلي، أو الصراف الآلي. بعد السداد،
  أرسل لنا إيصال الدفع للتأكيد. بنتواصل معك في التاريخ المتفق عليه.
  مضبوط؟"

If the agreement is a CALLBACK (no payment commitment), say instead:
  "للتأكيد، بنتواصل معك يوم {callback_summary}. مضبوط؟"

After the customer responds to "مضبوط؟" call EXACTLY one tool:
- recap_confirmed: customer says yes / مضبوط / تمام.
- recap_minor_correction(correction): small correction; plan stays the same.
- wants_to_renegotiate: customer materially backs out — go to reschedule.

For clarification questions about payment options, answer in ONE short
sentence and re-ask "مضبوط؟".
"""


class Stage4RecapAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        plan_summary = data.commitment or "(no commitment recorded)"
        callback_summary = data.callback_time or "(none)"
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE4_TASK.format(
                    plan_summary=plan_summary,
                    callback_summary=callback_summary,
                ),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        has_commitment = bool(self.data.commitment)
        has_callback = bool(self.data.callback_time)

        if has_commitment:
            hint = (
                'Say the recap as ONE concise reply ending with "مضبوط؟". '
                f'Plan: {self.data.commitment}. '
                "Mention SADAD with biller code 005, bank app, Mobily app, "
                "branch, ATM. Ask the customer to send the payment receipt. "
                'End with: "بنتواصل معك في التاريخ المتفق عليه. مضبوط؟"'
            )
        elif has_callback:
            hint = (
                'Say a short callback recap: "للتأكيد، بنتواصل معك يوم '
                f'{self.data.callback_time}. مضبوط؟" Use "بنتواصل معك" — '
                "NEVER 'بتدفع'. ONE short sentence."
            )
        else:
            hint = (
                'Restate whatever was agreed in ONE short sentence and '
                'ask "مضبوط؟".'
            )
        self.session.generate_reply(instructions=hint)

    @function_tool()
    async def recap_confirmed(self, ctx: RunContext[CallData]):
        """Customer confirms the recap."""
        return ClosingAgent(self.data, intent="ok", chat_ctx=None)

    @function_tool()
    async def recap_minor_correction(self, ctx: RunContext[CallData], correction: str):
        """Small correction; overall plan stands.

        Args:
            correction: short description of what changed.
        """
        ctx.userdata.commitment = (
            f"{ctx.userdata.commitment or ''} | corrected: {correction}"
        ).strip(" |")
        return ClosingAgent(self.data, intent="ok", chat_ctx=None)

    @function_tool()
    async def wants_to_renegotiate(self, ctx: RunContext[CallData]):
        """Customer is materially backing out — go to callback scheduling."""
        ctx.userdata.commitment = None
        ctx.userdata.outcome = None
        return RescheduleAgent(self.data, chat_ctx=None)


# ============================================================
# Closing
# ============================================================

_CLOSING_HINTS = {
    "ok": (
        'Thank the customer warmly: "شاكرة لك تعاونك، الله يجزاك خير. '
        'وبنتواصل معك قريب بإذن الله." TWO short sentences max.'
    ),
    "paid": (
        'Acknowledge: "شكراً لك. مدوّن، وراح يتم تحديث السجل في أقرب وقت. '
        'يعطيك العافية." ONE or TWO short sentences.'
    ),
    "busy": 'Say: "أبشر، بنتواصل معك في وقت مناسب لك بإذن الله." ONE short sentence.',
    "busy_callback": (
        "Confirm the callback time you just agreed on (date in Arabic words), "
        "thank the customer, and say goodbye. ONE or TWO short sentences."
    ),
    "dnc": (
        "Acknowledge the DNC request, confirm it will be recorded, apologise "
        "for the disturbance, say goodbye. ONE short sentence."
    ),
    "death": (
        'Express condolences (الله يرحمه ويغفر له), thank the caller for '
        "letting you know, say goodbye. ONE or TWO short sentences."
    ),
    "wrong_party": (
        'Apologise for the mix-up: "آسفة على الإزعاج، يعطيك العافية." '
        "ONE short sentence."
    ),
    "referred": (
        'Thank the caller warmly: "شاكرين لك تعاونك، يعطيك العافية." '
        "ONE short sentence."
    ),
    "refusal": (
        'Acknowledge respectfully: "أشكرك على وقتك، وبنتواصل معك في وقت '
        'مناسب لك بإذن الله." ONE short sentence. NO pressure.'
    ),
    "dispute": (
        "Acknowledge the dispute, say it will be reviewed by the relevant "
        "team and someone will follow up, say goodbye. ONE or TWO short sentences."
    ),
    "verify_refused": (
        "Apologise that we cannot continue without verification, invite him "
        "to call back on the official Mobily number, say goodbye. ONE or TWO "
        "short sentences. Do NOT mention any debt details."
    ),
}

_PAYMENT_LINK_INTENTS = {"ok"}


class ClosingAgent(BaseCallAgent):
    def __init__(
        self, data: CallData, *, intent: str, chat_ctx: ChatContext | None = None,
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
        try:
            self.session.interrupt()
        except Exception:
            pass
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
            try:
                await handle
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"closing speech playout error: {e}")

        if self.intent in _PAYMENT_LINK_INTENTS:
            _send_sms_payment_link(self.data, when=self.data.commitment)

        _emit_outcome(
            self.data,
            f"close_{self.intent}",
            outcome=self.data.outcome,
            commitment=self.data.commitment,
            callback_time=self.data.callback_time,
            referrer_mobile=self.data.referrer_mobile,
            payment_link_sent=self.data.payment_link_sent,
            id_verified=self.data.id_verified,
        )

        try:
            await self.hangup()
        except Exception as e:
            logger.warning(f"hangup race ignored: {e}")


# ============================================================
# STT factory — Munsit (streaming) or Soniox, selected by env var
# ============================================================

def _make_stt():
    """Build the STT instance based on STT_PROVIDER. Returns (stt, name).

    Munsit configuration notes (livekit-plugins-munsit==0.3.0):
    - Default is now SONIOX because Munsit streaming is unreliable against
      LiveKit's SIP audio path: the server intermittently rejects the
      first chunk of a stream with "Not a valid WAV file (missing RIFF
      header)" even though the plugin prepends a syntactically valid
      header (verified via direct WS probe). In server_diff endpointing
      this kills the whole session; in client_vad each failure silently
      drops one utterance — the customer keeps talking and gets no reply.
    - Set STT_PROVIDER=munsit to opt back in for A/B testing.
    - MUNSIT_MODE=batch uses HTTP POST per utterance (most reliable; adds
      1-2s of upload+server time, so total latency typically exceeds the
      1.5s budget).
    - MUNSIT_ENDPOINTING=client_vad|server_diff (only client_vad keeps
      the session alive when Munsit returns errors).
    """
    if STT_PROVIDER == "soniox":
        opts = soniox.STTOptions(language_hints=["ar"])
        logger.info("STT provider: SONIOX")
        return soniox.STT(params=opts), "soniox"

    mode = os.getenv("MUNSIT_MODE", "streaming").strip().lower()
    if mode == "batch":
        logger.info("STT provider: MUNSIT (batch)")
        return munsit.STT(mode="batch", model="munsit", language="ar"), "munsit"

    endpointing = os.getenv("MUNSIT_ENDPOINTING", "client_vad").strip().lower()
    logger.info(f"STT provider: MUNSIT (streaming, {endpointing})")
    return (
        munsit.STT(
            mode="streaming",
            model="munsit",
            endpointing=endpointing,
            finalize_after_silence_ms=500,
            vad_silence_ms=600,  # close WS quickly after the user stops
            interim_results=True,
            language="ar",
        ),
        "munsit",
    )


# ---------- Entrypoint ----------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.6,
        min_speech_duration=0.3,
        min_silence_duration=0.4,
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
        name=dial_info.get("name", "محمد"),
        first_name=dial_info.get("first_name"),
        gender=dial_info.get("gender"),
        amount=str(dial_info.get("amount", "1500")),
        national_id_last4=dial_info.get("national_id_last4", "1234"),
        phone_number=phone_number,
        services=dial_info.get("services", []),
    )

    participant_identity = f"sip-{phone_number}"

    stt_impl, stt_name = _make_stt()

    session = AgentSession[CallData](
        userdata=data,
        turn_handling={
            "endpointing": {
                "mode": "dynamic",
                "min_delay": 0.1,
                "max_delay": 0.6,
            },
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_words": 2,
            },
        },
        stt=stt_impl,
        llm=openai.LLM(model="gpt-4.1-mini", temperature=0.4),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-hijazi-female-2",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=1.0,
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

    # Per-turn latency rollup, tagged with the active STT provider so we can
    # A/B Munsit vs Soniox cleanly. Uses the new ChatMessage.metrics path —
    # the legacy metrics_collected event no longer carries EOU/transcription
    # delay (LiveKit prints a deprecation warning at startup).
    _last_user_metrics: dict[str, float] = {}

    @session.on("conversation_item_added")
    def _on_conv_item(ev):
        item = ev.item
        if getattr(item, "type", None) != "message":
            return
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        metrics = getattr(item, "metrics", None) or {}

        if role == "user":
            tdel = metrics.get("transcription_delay")
            eou = metrics.get("end_of_turn_delay")
            if tdel is not None:
                _last_user_metrics["transcript"] = float(tdel)
            if eou is not None:
                _last_user_metrics["eou"] = float(eou)
            if tdel is not None or eou is not None:
                logger.info(
                    f"USER_METRICS[{stt_name}]  transcript "
                    f"{(tdel or 0)*1000:.0f}ms  EOU {(eou or 0)*1000:.0f}ms"
                )
            if text:
                logger.info(f"USER  {text!r}")
            return

        if role == "assistant":
            if text:
                logger.info(f"AGENT {text!r}")
            ttft = metrics.get("llm_node_ttft") or metrics.get("llm_ttft")
            ttfb = metrics.get("tts_node_ttfb") or metrics.get("tts_ttfb")
            eou = _last_user_metrics.pop("eou", None)
            tdel = _last_user_metrics.pop("transcript", None)
            if ttft is not None and ttfb is not None and eou is not None:
                total = float(eou) + float(ttft) + float(ttfb)
                budget = 1.5
                tag = "OK" if total <= budget else "OVER"
                logger.info(
                    f"TURN[{stt_name}] total {total:.2f}s  "
                    f"(EOU {float(eou):.2f} + TTFT {float(ttft):.2f} + "
                    f"TTFB {float(ttfb):.2f})  transcript={float(tdel or 0):.2f}s  "
                    f"budget {budget:.2f}s [{tag}]"
                )

    # Legacy metrics_collected: still emits LLM/TTS/STT individually for
    # spot-checks, but EOU/transcript come via ChatMessage.metrics above.
    _turn_state: dict[str, float] = {}

    @session.on("metrics_collected")
    def _on_metrics(ev):
        m = ev.metrics
        mtype = getattr(m, "type", None) or m.__class__.__name__.lower()

        if mtype == "vad_metrics":
            return

        if mtype == "stt_metrics":
            # Some plugins surface streaming STT latency here. Keep one line
            # so we can attribute time spent inside the STT path itself.
            dur = getattr(m, "duration", None) or getattr(m, "audio_duration", None)
            if dur is not None:
                logger.info(f"STT[{stt_name}]  duration {dur:.2f}s")
            return

        if mtype == "eou_metrics":
            eou = getattr(m, "end_of_utterance_delay", 0.0)
            tdel = getattr(m, "transcription_delay", 0.0)
            _turn_state["eou"] = eou
            _turn_state["transcript"] = tdel
            logger.info(
                f"EOU[{stt_name}]  {eou:.2f}s  (transcript {tdel:.2f}s)"
            )
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
            tdel = _turn_state.pop("transcript", None)
            if eou is not None and ttft is not None:
                total = eou + ttft + ttfb
                budget = 1.5
                tag = "OK" if total <= budget else "OVER"
                logger.info(
                    f"TURN[{stt_name}] total {total:.2f}s  "
                    f"(EOU {eou:.2f} + TTFT {ttft:.2f} + TTFB {ttfb:.2f}) "
                    f"transcript={tdel or 0:.2f}s  budget {budget:.2f}s [{tag}]"
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
            agent_name=os.getenv("AGENT_NAME", "mobily-tawafuq-local"),
            num_idle_processes=1,
        )
    )
