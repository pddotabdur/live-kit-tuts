"""
---
title: Outbound Calling Agent
category: telephony
tags: [outbound_call, sip, twilio, livekit]
difficulty: intermediate
description: Agent that makes outbound phone calls via LiveKit SIP + Twilio
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
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
    stt,
    tts,
    llm,
    metrics,
)
from livekit.plugins import silero, faseeh, openai, deepgram, google
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


def _build_system_prompt(dial_info: dict[str, Any]) -> str:
    customer_name = dial_info.get('name', 'العميل')
    amount = dial_info.get('amount', '10000')
    debt_date = dial_info.get('debt_date', '2023-01-01')
    last4 = dial_info.get('national_id_last4', '1234')

    prompt = f"""You are Nora, a real-time Voice Agent calling on behalf of **Tawafuq** to collect overdue
payments for **stc bank**.
Role: Professional, calm, respectful assistant.
Language: Arabic (primary), Najdi dialect (light, natural).
Keep responses very short (1-2 sentences max), conversational, and polite.

🎯 OBJECTIVE
Remind {customer_name} about {amount} SAR debt from {debt_date}.
Goal: Secure full payment, partial plan, or scheduled follow-up. 
incase of partial plan, only accept reasonable amount compare to the total (example: if total is 15000 accept 2000 minimum) 

🧭 RULES
- BE CALM & RESPECTFUL. Use soft persuasion, no pressure.
- NEVER say "لازم تسدد", "إنذار", "إجراء قانوني". Never threaten or shame.
- Use natural Najdi phrases (e.g., أبشر, الله يعافيك).
- Do not repeat questions. Adapt to sentiment.
- Never mention that you're an AI developed by google or others
- Verifying the customer's identity before discussing account details.
- Confirming the customer is the actual owner of the debt before proceeding.
- Clearly stating the overdue balance and debt date after ownership confirmation.

🗣️ CALL FLOW
1. OPENING: "السلام عليكم، معك نورا من قسم المتابعة المالية، هل تسمح لي بدقيقة؟"
   - If busy: "أبشر، متى الوقت المناسب؟"
2. ID CONFIRM: "لو تكرمت، هل أكلم الأستاذ/ة {customer_name}؟ وهل آخر ٤ أرقام من هويتك هي {last4}؟"
   - If unsure: "تمام، بس للتأكد، الموضوع بخصوص حساب مالي بسيط."
3. DEBT CONTEXT (Only after ID confirm): "حبيت أذكّرك بوجود مبلغ مستحق {amount} ريال بتاريخ {debt_date}، وهدفنا نلقى حل مناسب لك."
4. HANDLE RESPONSES:
   - Cooperative: "يعطيك العافية، تفضل تبي تسدد الآن أو نرتب طريقة تناسبك؟"
   - Needs time: "ما فيه مشكلة، كم المدة اللي تناسبك؟"
   - No money: "مقدّر وضعك، خلنا نشوف حل مثل دفعة جزئية."
   - Angry: "أفهم شعورك، هدفنا نسهّل الموضوع بدون ضغط."
   - Denial: "ممكن فيه لبس، خلني أراجع معك التفاصيل."
5. CONFIRMATION: "ممتاز، بنثبت الاتفاق على السداد، تمام؟"
6. CLOSING: "شاكر لك تعاونك، الله يجزاك خير." (or polite goodbye if no agreement).

Available Tools: end_call, detected_answering_machine
"""
    return prompt


class OutboundAgent(Agent):
    """Agent that handles outbound phone calls."""

    def __init__(self, *, dial_info: dict[str, Any]):
        super().__init__(
            instructions=_build_system_prompt(dial_info)
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        self.session.generate_reply(
            user_input=(
                "Start the call now in natural conversational tone."
            )
        )

    async def hangup(self):
        """Hang up the call by deleting the room."""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call."""
        logger.info(
            f"ending the call for {self.participant.identity if self.participant else 'unknown'}"
        )

        # let the agent finish speaking
        await ctx.wait_for_playout()

        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting."""
        logger.info(
            f"detected answering machine for {self.participant.identity if self.participant else 'unknown'}"
        )
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # Parse metadata passed via dispatch (contains phone_number)
    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
        if not dial_info:
            raise KeyError()
    except (json.JSONDecodeError, KeyError):
        logger.info("No metadata provided, using default demo data.")
        dial_info = {
            "name": "محمد",
            "amount": "1500",
            "debt_date": "2023-05-10",
            "national_id_last4": "5678"
        }

    agent = OutboundAgent(dial_info=dial_info)

    vad = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.4,
    )
    faseeh_tts = faseeh.TTS(
        voice_id="ar-najdi-female-1",           # Choose your voice
        model="faseeh-v1-preview",          # Use full model for best quality
        stability=0.75,                     # Balanced stability
        speed=1.0,                          # Normal speech speed (0.7-1.2)
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
        stt=deepgram.STT(model="nova-3", language="ar-SA"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7
        ),
        # llm=google.LLM(
        #     model="gemini-2.0-flash",
        #     temperature=0.7
        # ),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-najdi-female-1",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.9,
        ),
        vad=vad,
    )


    await session.start(agent=agent, room=ctx.room)
    logger.info("✅ Agent ready — connect via LiveKit Playground to begin.")

    participant = await ctx.wait_for_participant()
    logger.info(f"👤 Participant joined: {participant.identity}")
    agent.set_participant(participant)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
            num_idle_processes=1,
        )
    )