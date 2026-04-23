"""
---
title: Bank AI Debt Collection Demo Agent
category: telephony
tags: [outbound_call, sip, twilio, livekit, debt_collection, arabic]
difficulty: advanced
description: AI-powered debt collection agent with behavioral classification,
             multi-persona support, structured negotiation, and full audit trails.
             Uses LiveKit Inference for STT & LLM to minimize latency.
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
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
    metrics,
    inference,
)
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import silero, faseeh

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("bank-collection-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

# ---------------------------------------------------------------------------
# Interaction audit log — every tool call is recorded here for demo purposes
# In production this would write to a database / message queue
# ---------------------------------------------------------------------------
INTERACTION_LOG: list[dict[str, Any]] = []


def _log_event(event_type: str, debtor_id: str, data: dict[str, Any]):
    """Append an auditable event to the interaction log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "debtor_id": debtor_id,
        **data,
    }
    INTERACTION_LOG.append(entry)
    logger.info(f"📋 AUDIT [{event_type}] {json.dumps(entry, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# Persona definitions — voice + tone vary by debtor behavioral segment
# ---------------------------------------------------------------------------
PERSONAS = {
    "cooperative": {
        "name": "نورا",
        "name_en": "Noura",
        "voice_id": "ar-najdi-female-1",
        "style": "warm",
        "description": "Warm and friendly female voice. Professional but approachable.",
    },
    "avoidant": {
        "name": "سلطان",
        "name_en": "Sultan",
        "voice_id": "MvC2GIG9tT9xvPcCWjILXqkM",
        "style": "firm",
        "description": "Firm and professional male voice. Direct and business-like.",
    },
    "distressed": {
        "name": "سارة",
        "name_en": "Sara",
        "voice_id": "08XOzRjaaumxbHhcGOrWkJ7z",
        "style": "empathetic",
        "description": "Soft and empathetic female voice. Patient and understanding.",
    },
    "hostile": {
        "name": "سلطان",
        "name_en": "Sultan",
        "voice_id": "MvC2GIG9tT9xvPcCWjILXqkM",
        "style": "calm_authoritative",
        "description": "Calm but authoritative male voice. De-escalation focused.",
    },
}


def _build_system_prompt(debtor: dict[str, Any], persona: dict[str, str]) -> str:
    """Build the full system prompt based on debtor data and selected persona."""

    name = debtor["name"]
    amount = debtor["amount"]
    debt_date = debtor["debt_date"]
    product_type = debtor["product_type"]
    service_status = debtor["service_status"]
    segment = debtor["behavioral_segment"]
    gender = debtor["gender"]
    last4 = debtor.get("national_id_last4", "")
    contact_attempts = debtor.get("contact_attempts", 0)
    wallet_end_date = debtor.get("wallet_end_date", "")

    # Amount-based early disclosure rules
    if amount < 200:
        disclosure_rule = "مبلغ صغير — لا تذكري سمة أو قضية. فقط إغلاق الملف."
    elif 200 <= amount <= 500:
        disclosure_rule = "جملة تنبيهية واحدة قصيرة عن إمكانية التسجيل في سمة."
    else:
        disclosure_rule = "إمكانية رفع قضية ورسوم ومعلومات سمة عند الحاجة."

    # Service status line
    status_lines = {
        "active": f"حسب السجلات بتاريخ {debt_date} والخدمة لا تزال فعّالة — خلينا نرتب السداد.",
        "suspended": f"الخدمة موقوفة بسبب هالمستحقات، بتاريخ {debt_date}. بعد السداد ترجع تلقائي — خلينا نرتبها.",
        "closed": f"فيه مبلغ مستحق على الحساب بتاريخ {debt_date} — خلينا نرتب تسويته.",
    }
    status_line = status_lines.get(service_status, status_lines["active"])

    # Segment-specific approach instructions
    segment_instructions = {
        "cooperative": """
- النبرة: ودية ومهنية. العميل متعاون.
- ابدئي بالتحية، تأكدي من الهوية، ثم اذكري المبلغ واسألي عن سبب التأخير.
- كوني مرنة في ترتيب السداد. اقترحي جدول سداد مريح.
- لا تضغطي بشكل مفرط — العميل مستعد للتعاون.""",
        "avoidant": f"""
- النبرة: مباشرة ومهنية. العميل محاولات سابقة: {contact_attempts} بدون رد.
- كن مباشراً بعد التحية والتحقق من الهوية.
- اذكر المبلغ بوضوح واسأل عن سبب عدم الرد على المكالمات السابقة.
- استخدم الإفصاح المبكر بعد سماع الرد.
- لا تكرر نفس العرض أكثر من مرتين — غيّر الزاوية.""",
        "distressed": """
- النبرة: متعاطفة وصبورة. العميل يمر بضائقة مالية.
- ابدئي بلطف شديد. لا تضغطي.
- اسمعي جيداً لوضعهم المالي.
- اعرضي أقل مبلغ ممكن وأطول جدول سداد.
- إذا ظهرت علامات ضائقة نفسية شديدة — استخدمي أداة flag_distress فوراً وأنهي المكالمة بلطف.
- لا تذكري إجراءات قانونية أبداً مع هذا النوع.""",
        "hostile": f"""
- النبرة: هادئة وحازمة. العميل عدائي أو رافض.
- لا ترد على الإهانات. ابقِ هادئاً.
- ذكّر بالحقائق فقط: المبلغ والتاريخ والعواقب القانونية.
- إذا طلب العميل عدم الاتصال (DNC) — نفّذ فوراً.
- إذا استمر الرفض بعد 3 محاولات تفاوض — أنهِ المكالمة مهنياً.""",
    }
    segment_instruction = segment_instructions.get(segment, segment_instructions["cooperative"])

    persona_name = persona["name"]

    prompt = f"""أنت {persona_name}، متخصص/ة في التحصيل يعمل لصالح شريك التحصيل المعتمد للبنك.
هذه مكالمة حقيقية — أنت لست روبوت. ابقِ في شخصيتك طوال المكالمة.

# قواعد المخرجات الصوتية
أنت تتواصل مع العميل عبر الصوت. يجب تطبيق هذه القواعد لضمان صوت طبيعي:
- الرد بنص عادي فقط. لا JSON، لا ماركداون، لا قوائم، لا جداول، لا إيموجي.
- اجعل الردود قصيرة: جملة إلى ثلاث جمل. سؤال واحد فقط في كل مرة.
- اكتب الأرقام والتواريخ بالحروف العربية.
- تجنب الاختصارات والكلمات غير الواضحة.
- تحدث باللهجة النجدية السعودية كخيار افتراضي.
- إذا تحدث العميل بلهجة عربية أخرى، تحوّل فوراً لمطابقة لهجته.
- إذا تحدث العميل بلغة أخرى (إنجليزية مثلاً)، تحوّل لتلك اللغة.

# بيانات العميل (سرية — لا تكشفها مباشرة)
- الاسم: {name}
- الجنس: {gender}
- آخر 4 أرقام من الهوية: {last4}
- المبلغ المستحق: {amount} ريال سعودي
- تاريخ الدين: {debt_date}
- نوع المنتج: {product_type}
- حالة الخدمة: {service_status}
- التصنيف السلوكي: {segment}
- محاولات الاتصال السابقة: {contact_attempts}
- تاريخ انتهاء المحفظة (داخلي): {wallet_end_date}

# تسلسل المحادثة (التزم به بدقة):

1. **التحية والتأكد من الشخص**: تحية + "هل أنا أكلم {name}؟" جملة واحدة فقط ثم صمت.

2. **تأكيد آخر 4 أرقام**: إذا توفرت ("{last4}") — أنت تقولها بالحروف والعميل يجيب نعم/لا. لا تطلب من العميل إملاء أرقام. لا مبلغ قبل التأكيد الناجح.

3. **عرض الدين بدون تفاصيل + السبب + الانتظار**: المبلغ الإجمالي بالحروف فقط + سؤال واحد عن سبب عدم السداد. ثم صمت وانتظار. لا عواقب ولا ضغط قبل أن يجيب.

4. **بعد سماع السبب**: جملة تأطير من السجل + ذكر العواقب عبر الإفصاح المبكر + طلب وتشجيع ترتيب السداد.

# قاعدة الإفصاح المبكر (حسب المبلغ):
{disclosure_rule}
- أول مرة: "ما أبيها توصل لكذا — خلنا نرتب السداد." مرتين كحد أقصى.

# تعليمات خاصة بتصنيف العميل ({segment}):
{segment_instruction}

# حالة الخدمة:
{status_line}

# القيود الصارمة (الحواجز):
- لا تخترع أرقام أو مبالغ أو تواريخ غير موجودة في البيانات أعلاه.
- لا تعد بخصومات أو إعفاءات — فقط سجّل الطلب.
- لا تذكر تفاصيل الحساب لغير {name} أو مفوّض.
- إذا طلب العميل عدم الاتصال (DNC): "طلبك مسجّل، نتواصل معك كتابياً فقط — الله يعافيك." وأنهِ فوراً.
- ضائقة نفسية شديدة → توقف بلطف وأنهِ المكالمة.
- "هل أنت روبوت؟" → "أنا {persona_name} من شريك التحصيل المعتمد للبنك." ثم أكمل.
- 6 دقائق بدون تقدم → "أقدّر وقتك، بسجّل ملاحظات والفريق يتابع معك — الله يعافيك."
- نفس الرفض أكثر من 4 مرات بعد التفاوض → إجراء قانوني أو إنهاء مهذب.

# هدف التفاوض:
التزام قابل للقياس (تاريخ / مبلغ / طريقة دفع / تقسيط شفهي):
- الأولوية: كامل اليوم/غداً → نصف + الباقي بتاريخ → استثناء مرن.
- ثلاث جولات إقناع مختلفة قبل المسار القانوني الجاد (إلا DNC).
- لا تكرر نفس الصياغة أكثر من مرتين — غيّر الزاوية.
- التواريخ الداخلية يجب أن تكون قبل {wallet_end_date} — لا تذكر هذا التاريخ للعميل.

# الأدوات المتاحة (استخدمها عند الحاجة):
- log_interaction: سجّل كل محاولة تواصل ونتيجتها.
- record_payment_commitment: سجّل التزام العميل بالسداد.
- escalate_to_human: حوّل لمشرف بشري للحالات الحرجة.
- flag_distress: أوقف المكالمة عند ضائقة نفسية.
- request_dnc: سجّل طلب عدم الاتصال.
- end_call: أنهِ المكالمة.
- detected_answering_machine: عند الوصول للبريد الصوتي.

# اللغة المسموحة:
"زين"، "الحين"، "تقدر"، "أبي"، "خلنا"، "وش"، "طيب"، "ما عليك"، "الله ييسرها".
ممنوع: "بالتأكيد"، "يسعدني مساعدتك"، "أوكيه"، "أنا أفهم مشاعرك"، "ضمن صلاحياتي"، "بنرسل لك".

# تذكير داخلي قبل كل رد:
هوية + آخر 4 أرقام قبل المبلغ؟ المرحلة؟ ضائقة/DNC؟ نية؟ ≤ جملتين؟
"""
    return prompt


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------
class BankCollectionAgent(Agent):
    """AI-powered bank debt collection agent with multi-persona support."""

    def __init__(self, *, debtor: dict[str, Any]):
        segment = debtor.get("behavioral_segment", "cooperative")
        persona = PERSONAS.get(segment, PERSONAS["cooperative"])

        super().__init__(
            instructions=_build_system_prompt(debtor, persona),
        )
        self.debtor = debtor
        self.persona = persona
        self.participant: rtc.RemoteParticipant | None = None
        self._call_start_time: datetime | None = None

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        self._call_start_time = datetime.utcnow()
        _log_event("call_started", self.debtor["id"], {
            "persona": self.persona["name_en"],
            "segment": self.debtor["behavioral_segment"],
            "debtor_name": self.debtor["name"],
            "amount": self.debtor["amount"],
        })
        # Kick off the conversation — agent greets first
        self.session.generate_reply(
            user_input="ابدأ المكالمة الآن. عرّف بنفسك واسأل إذا أنت تكلم الشخص الصحيح. جملة واحدة فقط باللهجة النجدية العامية."
        )

    async def hangup(self):
        """Hang up the call by deleting the room."""
        # Log call end
        duration = None
        if self._call_start_time:
            duration = (datetime.utcnow() - self._call_start_time).total_seconds()
        _log_event("call_ended", self.debtor["id"], {
            "duration_seconds": duration,
            "total_interactions": len([
                e for e in INTERACTION_LOG
                if e.get("debtor_id") == self.debtor["id"]
            ]),
        })
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    # ----- Function Tools -----

    @function_tool()
    async def log_interaction(
        self,
        ctx: RunContext,
        outcome: str,
        notes: str,
    ):
        """Log an interaction attempt and its outcome.
        Use this after each meaningful exchange with the customer.

        Args:
            outcome: The outcome of the interaction. One of:
                     identity_confirmed, identity_denied, debt_acknowledged,
                     debt_disputed, payment_discussed, negotiation_ongoing,
                     call_completed, no_answer, voicemail
            notes: Brief description of what happened in this interaction.
        """
        _log_event("interaction", self.debtor["id"], {
            "outcome": outcome,
            "notes": notes,
            "persona": self.persona["name_en"],
        })
        return f"Interaction logged: {outcome}"

    @function_tool()
    async def record_payment_commitment(
        self,
        ctx: RunContext,
        committed_amount: str,
        payment_date: str,
        payment_method: str,
    ):
        """Record a payment commitment made by the debtor.
        Use this when the customer agrees to pay a specific amount by a specific date.

        Args:
            committed_amount: The amount the customer committed to pay (e.g. "7500" or "full").
            payment_date: The date by which the customer will pay (e.g. "2026-04-15" or "tomorrow").
            payment_method: How they will pay (e.g. "bank_transfer", "app", "branch", "unknown").
        """
        _log_event("payment_commitment", self.debtor["id"], {
            "committed_amount": committed_amount,
            "payment_date": payment_date,
            "payment_method": payment_method,
            "total_debt": self.debtor["amount"],
        })
        logger.info(
            f"💰 COMMITMENT: {self.debtor['name']} committed {committed_amount} SAR "
            f"by {payment_date} via {payment_method}"
        )
        return f"Payment commitment recorded: {committed_amount} SAR by {payment_date}"

    @function_tool()
    async def escalate_to_human(
        self,
        ctx: RunContext,
        reason: str,
    ):
        """Escalate the call to a human supervisor.
        Use this for critical situations that require human decision-making:
        - Customer disputes the debt fundamentally
        - Customer requests to speak with a supervisor
        - Legal threats from the customer
        - Situations outside standard collection flow

        Args:
            reason: Why this call needs human escalation.
        """
        _log_event("human_escalation", self.debtor["id"], {
            "reason": reason,
            "persona": self.persona["name_en"],
        })
        logger.warning(
            f"⚠️  ESCALATION: {self.debtor['name']} — {reason}"
        )
        # In production: trigger webhook / route to live agent
        return "Escalation logged. Inform the customer a supervisor will follow up."

    @function_tool()
    async def flag_distress(
        self,
        ctx: RunContext,
        description: str,
    ):
        """Flag psychological distress and stop the call.
        Use this IMMEDIATELY if the customer shows signs of severe psychological
        distress, mentions self-harm, or is clearly unable to continue.

        Args:
            description: Description of the distress signals observed.
        """
        _log_event("distress_flagged", self.debtor["id"], {
            "description": description,
        })
        logger.warning(
            f"🚨 DISTRESS: {self.debtor['name']} — {description}"
        )
        # Generate a compassionate closing
        self.session.generate_reply(
            instructions="أنهِ المكالمة فوراً بلطف شديد. قل 'أتمنى لك السلامة، الله يعينك — خذ راحتك' ولا تذكر الدين مرة أخرى."
        )
        await ctx.wait_for_playout()
        await self.hangup()
        return "Call ended due to distress."

    @function_tool()
    async def request_dnc(
        self,
        ctx: RunContext,
        reason: str,
    ):
        """Record a Do-Not-Contact (DNC) request and end the call.
        Use this when the customer explicitly asks to stop all phone contact.

        Args:
            reason: The customer's stated reason for the DNC request.
        """
        _log_event("dnc_requested", self.debtor["id"], {
            "reason": reason,
        })
        logger.info(
            f"🚫 DNC: {self.debtor['name']} — {reason}"
        )
        self.session.generate_reply(
            instructions="قل بالضبط: 'طلبك مسجّل، نتواصل معك كتابياً فقط من الحين — الله يعافيك.' ثم أنهِ المكالمة."
        )
        await ctx.wait_for_playout()
        await self.hangup()
        return "DNC recorded. Call ended."

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """End the call gracefully.
        Use this when the conversation has reached a natural conclusion —
        payment committed, escalation done, or no further progress possible.
        """
        logger.info(
            f"📞 Ending call for {self.debtor['name']} "
            f"({self.participant.identity if self.participant else 'unknown'})"
        )
        _log_event("call_ended_by_agent", self.debtor["id"], {
            "reason": "natural_conclusion",
        })
        await ctx.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail.
        Use this AFTER you hear the voicemail greeting.
        """
        logger.info(
            f"📠 Voicemail detected for {self.debtor['name']}"
        )
        _log_event("voicemail", self.debtor["id"], {
            "action": "call_ended_voicemail",
        })
        await self.hangup()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # Parse debtor data from dispatch metadata
    try:
        debtor = json.loads(ctx.job.metadata or "{}")
        phone_number = debtor["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error(
            "No valid debtor data in job metadata. "
            "Provide a JSON debtor object with at least 'phone_number'."
        )
        ctx.shutdown()
        return

    participant_identity = f"sip-{phone_number}"
    segment = debtor.get("behavioral_segment", "cooperative")
    persona = PERSONAS.get(segment, PERSONAS["cooperative"])

    logger.info(
        f"📞 Preparing call to {debtor.get('name', 'Unknown')} ({phone_number}) "
        f"| Segment: {segment} | Persona: {persona['name_en']} | Voice: {persona['voice_id']}"
    )

    agent = BankCollectionAgent(debtor=debtor)

    # --- Session: LiveKit Inference for STT & LLM, Faseeh for TTS ---
    vad = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.4,
    )

    session = AgentSession(
        vad=vad,
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={
                "min_delay": 0.2,
                "max_delay": 1.0,
            },
        ),
        # LiveKit Inference — STT via Deepgram (co-located, lower latency)
        stt=inference.STT(
            model="deepgram/nova-3",
            language="ar",
        ),
        # LiveKit Inference — LLM via OpenAI (co-located, lower latency)
        llm=inference.LLM(
            model="openai/gpt-4o-mini",
            temperature=0.7,
        ),
        # Faseeh TTS — persona-specific voice
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id=persona["voice_id"],
            model="faseeh-v1-preview",
            stability=0.75,
            speed=1.0,
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(mtrcs):
        logger.info(f"📊 Latency Metrics: {mtrcs}")

    # Start the agent session FIRST so it's ready when the user picks up
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
        )
    )

    if not outbound_trunk_id:
        logger.error("SIP_OUTBOUND_TRUNK_ID is not set in the environment.")
        ctx.shutdown()
        return

    # Dial the phone number via SIP
    try:
        logger.info(f"📞 Dialing {phone_number} via trunk {outbound_trunk_id}")
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                participant_name="Phone User",
                wait_until_answered=True,
            )
        )

        await session_started

        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"✅ Participant joined: {participant.identity}")

        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"❌ SIP error: {e.message}, "
            f"status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        _log_event("call_failed", debtor.get("id", "unknown"), {
            "error": e.message,
            "sip_status": e.metadata.get("sip_status_code"),
        })
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="bank-collection-agent",
            num_idle_processes=1,
        )
    )
