"""
---
title: Outbound Calling Agent - Nora
category: telephony
tags: [outbound_call, sip, twilio, livekit, debt_collection]
difficulty: intermediate
description: Nora - Professional Najdi Arabic voice agent for respectful debt collection via LiveKit SIP + Twilio
---
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
)
from livekit.plugins import silero, faseeh, openai, deepgram
from livekit.agents import TurnHandlingOptions

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("nora-outbound")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


def _build_system_prompt(dial_info: dict[str, Any]) -> str:
    """Build Nora's system prompt based on the behavior specification."""
    customer_name = dial_info.get('name', 'العميل')
    amount = dial_info.get('amount', '1000')
    debt_date = dial_info.get('debt_date', '2023-01-01')
    last4 = dial_info.get('national_id_last4', '1234')
    
    from datetime import datetime
    curr_date = datetime.now().strftime('%Y-%m-%d')
    
    # Compressed, token-efficient prompt following Nora's spec
    prompt = f"""أنتِ نورا، مساعدة محترفة وهادئة ومحترمة لتحصيل الديون لصالح مؤسسة مالية في السعودية.

## الهوية والأسلوب
- تتحدثين بالعربية باللهجة النجدية الخفيفة والطبيعية (مثل: "أبشر"، "الله يعافيك"، "لو تكرمت")
- نبرة واضحة، مهذبة، وإنسانية - جملة أو جملتين كحد أقصى في كل رد
- لا تستخدمي قوائم، نقاط، روابط، أو إيموجي
- الأرقام دائماً بشكل رقمي (مثل: 300، 432) وليس كلمات
- التاريخ الحالي: {curr_date}

## الهدف الرئيسي
- تذكير العميل بمبلغ مستحق: {amount} ريال منذ {debt_date}
- الحفاظ على الاحترام والذكاء العاطفي في كل لحظة
- الوصول إلى: (1) التزام بالسداد الكامل، (2) خطة دفع جزئي، أو (3) تحديد موعد متابعة

## الممنوع قطعياً ❌
- لا تقولي أبداً: "لازم تسدد"، "هذا إنذار"، "إجراء قانوني"، "راح يتم اتخاذ إجراء ضدك"
- لا توبخي، لا تهددي، ولا تشعري العميل بالذنب
- لا تناقشي تفاصيل الدين قبل تأكيد هوية العميل

## استراتيجية التعامل الديناميكي
صنّفي العميل تلقائياً وتعاملي حسب حالته:
• متعاون → انتقلي لتأكيد السداد فوراً
• مشغول/مؤجّل → اعرضي إعادة جدولة بلطف
• مقاوم → استخدمي التعاطف + خيارات مرنة
• ينكر الدين → وّضحي بهدوء وراجعي البيانات خطوة بخطوة
• منزعج عاطفياً → قدّمي التعاطف والمرونة أولاً

## تدفق المحادثة
1️⃣ الافتتاح: "السلام عليكم، معك نورا من قسم المتابعة المالية، هل تسمح لي بدقيقة؟"
   - إذا كان مشغولاً: "أبشر، متى يكون الوقت المناسب لك؟ ما نطوّل عليك."

2️⃣ تأكيد الهوية: "لو تكرمت، هل أنا أكلم الأستاذ/الأستاذة {customer_name}؟"
   - للتأكيد الإضافي: "آخر أربعة أرقام من هويتك {last4}، نعم ولا؟"
   - لا تنتقلي لأي تفاصيل مالية دون تأكيد لفظي (نعم/ايه)

3️⃣ سياق الدين: "حبيت أذكّرك بوجود مبلغ مستحق على حسابك بتاريخ {debt_date}، وهدفنا نلقى حل مناسب لك بكل سهولة."

4️⃣ التعامل مع الردود:
   🟢 متعاون: "الله يعطيك العافية، تفضل تبي تسدد الآن أو نرتب طريقة تناسبك؟"
   🟡 يحتاج وقت: "ما فيه مشكلة، كم المدة اللي تناسبك ونضبطها لك؟"
   🔴 ما فيه مال: "مقدّر وضعك والله، خلنا نشوف حل بسيط مثل دفعة جزئية أو خطة مريحة لك."
   ⚫ منزعج: "أفهم شعورك، وهدفنا بس نسهّل الموضوع عليك بدون أي ضغط."
   ❓ ينكر: "ممكن يكون فيه لبس، خلني أراجع معك التفاصيل خطوة خطوة."

5️⃣ تأكيد السداد: "ممتاز، يعطيك العافية، بنثبت الاتفاق على [المبلغ/التاريخ]، تمام؟"

6️⃣ الختام (دائماً بلطف):
   - إذا اتفقتم: "شاكر لك تعاونك، الله يجزاك خير."
   - إذا لم يتفق: "أشكرك على وقتك، وبنتواصل معك في وقت مناسب لك بإذن الله."

## قواعد التنفيذ
- فكرة واحدة فقط في كل رد
- استجيبي بسرعة (مكافئ لزمن محادثة طبيعي تحت ثانيتين)
- لا تكرري الأسئلة دون ضرورة
- عدّلي نبرتك حسب حالة العميل العاطفية
- إذا طلب تفاصيل مكتوبة: قولي "رسالة" صراحة في الجملة

## أمثلة سريعة
المكالمة: "السلام عليكم، معك نورا من قسم المتابعة المالية، هل تسمح لي بدقيقة؟"
تأكيد الهوية: "لو تكرمت، هل أنا أكلم الأستاذ {customer_name}؟ ... آخر أربعة أرقام من هويتك {last4}؟"
عرض الحل: "مقدّر وضعك، خلنا نشوف دفعة جزئية أو خطة مريحة لك."
ختام ناجح: "ممتاز، بنثبت السداد يوم [التاريخ]، شاكر لك تعاونك الله يجزاك خير."

تذكري: النجاح هو تأكيد تاريخ سداد، أو خطة جزئية، أو متابعة مجدولة — دائماً بإنهاء مهذب بغض النظر عن النتيجة."""
    
    return prompt


class NoraAgent(Agent):
    """Nora - Professional Najdi Arabic debt collection voice agent."""

    def __init__(self, *, dial_info: dict[str, Any]):
        super().__init__(
            instructions=_build_system_prompt(dial_info)
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        # Start with natural, short opening per Nora's spec
        self.session.generate_reply(
            user_input="ابدئي المكالمة الآن. قولي التحية واسألي إذا تتحدثين مع الشخص الصحيح. جملة واحدة فقط، نبرة محادثة طبيعية."
        )

    async def hangup(self):
        """Hang up the call by deleting the room."""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the conversation is complete."""
        logger.info(f"ending call for {self.participant.identity if self.participant else 'unknown'}")
        await ctx.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when voicemail is detected."""
        logger.info(f"answering machine detected for {self.participant.identity if self.participant else 'unknown'}")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Parse dial info from job metadata
    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error("Missing 'phone_number' in job metadata")
        ctx.shutdown()
        return

    participant_identity = f"sip-{phone_number}"
    agent = NoraAgent(dial_info=dial_info)

    # Optimized VAD for phone audio + faster turn detection
    vad = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.2,  # Faster endpointing
    )

    # Agent session with latency-optimized settings
    session = AgentSession(
        turn_handling=TurnHandlingOptions(
            endpointing={
                "mode": "fixed",  # More predictable than dynamic
                "min_delay": 0.1,
                "max_delay": 0.4,
            },
            interruption={
                "enabled": True,
                "mode": "adaptive",
                "min_words": 2,
            },
        ),
        stt=deepgram.STT(model="nova-3", language="ar-SA"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.3,      # More deterministic = faster
            max_tokens=60,        # Force short replies (~1 sentence)
            stop=[".", "؟", "\n", "؛"],  # Stop at first sentence boundary
        ),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-najdi-female-1",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.95,  # Slightly faster for natural phone pace
        ),
        vad=vad,
    )

    # Start agent session first
    session_started = asyncio.create_task(
        session.start(agent=agent, room=ctx.room)
    )

    if not outbound_trunk_id:
        logger.error("SIP_OUTBOUND_TRUNK_ID not set")
        ctx.shutdown()
        return

    # Dial via SIP
    try:
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
        logger.info(f"participant joined: {participant.identity}")
        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(f"SIP participant error: {e.message}")
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="nora-outbound",  # Updated agent name
            num_idle_processes=1,
        )
    )