"""
---
title: Outbound Calling Agent - English
category: telephony
tags: [outbound_call, sip, twilio, livekit, english]
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
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
    stt,
    llm,
)
from livekit.plugins import silero, openai, deepgram, cartesia
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("english-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class EnglishOutboundAgent(Agent):
    """Agent that handles outbound phone calls in English."""

    def __init__(self, *, dial_info: dict[str, Any]):
        super().__init__(
            instructions="""You are an empathetic, professional, and highly conversational banking representative from STC Bank.
Your task is to call the customer, warmly but very briefly introduce yourself, verify their identity by asking for their name, and delicately remind them of a $1,000 overdue payment. Ask them politely when they might be able to make this payment.
Speak naturally with advanced semantic turn-taking. Use excellent conversational skills, and occasionally employ brief filler words like 'hmm' or 'I see' to signal active listening.
Be polite and respectful. If they ask questions, answer them concisely based on your persona. Be polite but also firm like a bank representative.
CRITICAL WARNING: Do NOT hang up the call arbitrarily. You MUST wait until the conversation is completely finished. ONLY invoke the `end_call` tool when the customer explicitly says "Goodbye", "Have a good day", or clearly expresses they want to hang up.
If you detect a voicemail, answering machine, or automated greeting, use the `detected_answering_machine` tool.
"""
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        self.session.generate_reply(
            user_input="Introduce yourself warmly and ask for the customer's name to immediately verify their identity."
        )

    async def hangup(self):
        """Hang up the call by deleting the room."""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called ONLY when the user explicitly wants to end the call."""
        logger.info(
            f"ending the call for {self.participant.identity if self.participant else 'unknown'}"
        )

        await ctx.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches an answering machine or voicemail. Use this tool AFTER you hear the voicemail greeting."""
        logger.info(
            f"detected answering machine for {self.participant.identity if self.participant else 'unknown'}"
        )
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error("No valid phone_number provided in job metadata. Provide a JSON object with 'phone_number'.")
        ctx.shutdown()
        return

    participant_identity = f"sip-{phone_number}"

    logger.info(f"preparing to dial {phone_number}")

    agent = EnglishOutboundAgent(dial_info=dial_info)

    vad = silero.VAD.load()
    session = AgentSession(
        vad=vad,
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={"mode": "adaptive"},
        ),
        stt=stt.FallbackAdapter(
            [
                deepgram.STT(model="nova-3", language="en-US"),
                openai.STT(),
            ],
            vad=vad
        ),
        llm=openai.LLM(
            model="gpt-4o",
            temperature=0.7
        ),
        tts=cartesia.TTS(
            model="sonic-english",
            voice="6f84f4b8-58a2-430c-8c79-688dad597532",
            speed=1.0,
        ),
        turn_detection=MultilingualModel(),
    )

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
                wait_until_answered=True,
            )
        )

        await session_started

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
            agent_name="english-caller",
        )
    )
