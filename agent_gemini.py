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
from livekit.agents import AgentSession, TurnHandlingOptions, inference
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundAgent(Agent):
    """Agent that handles outbound phone calls."""

    def __init__(self, *, dial_info: dict[str, Any]):
        customer_name = dial_info.get("name", "العميل")
        super().__init__(
            instructions=f"""أنتِ نورا، مساعدة صوتية ذكية لمكتب 'توافق' لتحصيل الديون. صوتكِ دافئ ونسائي.
مهمتكِ هي الاتصال بالعملاء لتحصيل المبالغ المستحقة عليهم.

القواعد الصارمة للتواصل:
1. ابدئي المكالمة باللهجة النجدية السعودية كخيار افتراضي.
2. استمعي جيداً للهجة العميل أو لغته. إذا كان يتحدث بلهجة عربية أخرى (مصرية، شامية، مغربية، إلخ)، حوّلي لهجتكِ فوراً لتطابق لهجته لتسهيل التواصل.
3. إذا بدأ العميل بالتحدث بلغة غير العربية (مثل الإنجليزية، الأوردو، إلخ)، حوّلي لغتكِ فوراً إلى تلك اللغة.
4. تحدثي فقط في سياق تحصيل الدين. لا تنجرفي في أحاديث جانبية.
5. إذا حاول العميل تغيير الموضوع، أعيديه بلباقة وحزم لموضوع الدين.
6. لا تقبلي أي مبلغ سداد يقل عن 100 ريال سعودي. إذا عرض العميل أقل، أخبريه أن الحد الأدنى للسداد هو 100 ريال.
7. كوني مهذبة ولكن حازمة.
8. ابدئي المكالمة بالتحية والتعريف بنفسك وبمكتب توافق، ثم اذكري اسم العميل والمبلغ المستحق.
9. تفاصيل العميل الحالي: الاسم: {customer_name}، المبلغ المستحق: 1000 ريال.
"""
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        self.session.generate_reply(
            user_input="ابدئي المكالمة بالتحية والتعريف بنفسك وبمكتب توافق، ثم اذكري اسم العميل والمبلغ المستحق."
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
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error("No valid phone_number provided in job metadata. Provide a JSON object with 'phone_number'.")
        ctx.shutdown()
        return

    participant_identity = f"sip-{phone_number}"

    logger.info(f"preparing to dial {phone_number}")

    agent = OutboundAgent(dial_info=dial_info)

    vad = silero.VAD.load()
    session = AgentSession(
        vad=vad,
        turn_handling=TurnHandlingOptions(
        turn_detection="vad",        
        endpointing={
            "min_delay": 0.25,      # how soon after user stops we consider turn ended
            "max_delay": 1.0,       # safety ceiling
        },
    ),
        stt=stt.FallbackAdapter(
            [
                deepgram.STT(model="nova-3", language="ar"),
                openai.STT(),
            ],
            vad=vad
        ),
        llm=inference.LLM(
            model="gemini-3.1-flash-lite-preview",
            extra_kwargs={
                "max_completion_tokens": 1000,
                "temperature": 0.7
            }
        ),
        tts = google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Zephyr",
            instructions="Speak in a friendly and engaging tone.",
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(mtrcs):
        logger.info(f"Latency Metrics: {mtrcs}")

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

    # Now dial the phone number via SIP
    try:
        logger.info(
            f"dialing {phone_number} via trunk {outbound_trunk_id}"
        )
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                participant_name="Phone User",
                # Block until the call is answered or fails
                wait_until_answered=True,
            )
        )

        # Wait for the agent session to finish starting
        await session_started

        # Wait for the SIP participant to fully join
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

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
            agent_name="outbound-caller",
            num_idle_processes=1,
        )
    )