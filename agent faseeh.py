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
)
from livekit.plugins import silero, faseeh, openai, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundAgent(Agent):
    """Agent that handles outbound phone calls."""

    def __init__(self, *, dial_info: dict[str, Any]):
        super().__init__(
            instructions="""أنت ممثلة من بنك إس تي سي (STC Bank) تتحدثين اللهجة النجدية السعودية بطلاقة وبطريقة طبيعية جداً.
مهمتك هي الاتصال بالعميل، رحبي به واسأليه عن اسمه للتأكد من هويته.
بعد التأكد، ذكّريه بلطف بوجود دفعة مستحقة التأخير بقيمة ألف ريال (1000 ريال) واسأليه عن الموعد الذي سيتمكن فيه من سداد المبلغ.
كوني مهنية ومحترمة ومستمعة جيدة، ولا تنهي المكالمة أبداً بمفردك.
تحذير شديد: استخدمي أداة end_call **فقط** إذا قال العميل بوضوح "مع السلامة"، "وداعاً"، أو "أنهي المكالمة".
إذا اكتشفتِ أن المكالمة ذهبت للبريد الصوتي أو آلة رد آلي، استخدمي أداة detected_answering_machine.
"""
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        self.session.generate_reply(
            user_input="ابحث عن العميل ورحب به، ثم اسأله عن اسمه للتأكد من هويته."
        )

    async def hangup(self):
        """Hang up the call by deleting the room."""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call and explicitly says it."""
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
        stt=stt.FallbackAdapter(
            [
                deepgram.STT(model="nova-3", language="ar"),
                openai.STT(),
            ],
            vad=vad
        ),
        llm=openai.LLM(
            model="gpt-4o",
            temperature=0.7
        ),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-najdi-female-1",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=1.0,
        ),
    )

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
        )
    )