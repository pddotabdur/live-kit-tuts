"""
Outbound voice agent.

Worker process: connects to a LiveKit server, waits for a dispatch from
`dispatch.py`, places a SIP call to the number in the dispatch metadata,
and runs the conversation through Deepgram (STT) → OpenAI (LLM + TTS).

Run on the same host as the LiveKit server (so audio publish goes over
loopback). See README.md for EC2 deploy steps.
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
)
from livekit.plugins import deepgram, openai, silero, faseeh


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")


def _build_system_prompt(dial_info: dict[str, Any]) -> str:
    customer_name = dial_info.get("name", "العميل")
    amount = dial_info.get("amount", "1000")
    debt_date = dial_info.get("debt_date", "2023-01-01")

    return f"""You are Nora, a real-time Voice Agent calling on behalf of **Tawafuq** to collect overdue
payments for **stc bank**.
Role: Professional, calm, respectful assistant.
Language: Arabic (primary), Najdi dialect (light, natural).
Keep responses very short (1-2 sentences max), conversational, and polite.

🎯 OBJECTIVE
Remind {customer_name} about {amount} SAR debt from {debt_date}.
Goal: Secure full payment, partial plan, or scheduled follow-up.
In case of partial plan, only accept reasonable amount compared to the total
(example: if total is 15000 accept 2000 minimum).

🧭 RULES
- BE CALM & RESPECTFUL. Use soft persuasion, no pressure.
- NEVER say "لازم تسدد", "إنذار", "إجراء قانوني". Never threaten or shame.
- Use natural Najdi phrases (e.g., أبشر, الله يعافيك).
- Do not repeat questions. Adapt to sentiment.
- Never mention that you're an AI developed by google or others.
- Verify the customer's identity before discussing account details.
- Confirm the customer is the actual owner of the debt before proceeding.
- Clearly state the overdue balance and debt date after ownership confirmation.

🗣️ CALL FLOW
1. OPENING: "السلام عليكم، معك نورا من قسم المتابعة المالية، هل تسمح لي بدقيقة؟"
   - If busy: "أبشر، متى الوقت المناسب؟"
2. ID CONFIRM: "لو تكرمت، هل أكلم الأستاذ/ة {customer_name}؟"
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


class OutboundAgent(Agent):
    def __init__(self, *, dial_info: dict[str, Any]):
        super().__init__(instructions=_build_system_prompt(dial_info))
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        self._sip_ready = asyncio.Event()

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant
        self._sip_ready.set()

    async def on_enter(self):
        # on_enter() fires while the phone is still ringing; wait for the SIP
        # participant to actually join before producing any audio.
        await self._sip_ready.wait()
        # Small settle so the carrier doesn't clip the first frames after answer.
        await asyncio.sleep(0.4)
        logger.info("SIP participant ready, driving call flow")
        self.session.generate_reply(
            user_input=(
                "The call has just been answered. Begin now by executing STEP 1 "
                "(OPENING) from your CALL FLOW in Najdi Arabic, then wait for the "
                "customer's reply and proceed through STEP 2 (ID CONFIRM), STEP 3 "
                "(DEBT CONTEXT after ID confirmed), STEP 4 (handle their response), "
                "STEP 5 (CONFIRMATION), STEP 6 (CLOSING). Do not skip steps. Keep "
                "each turn to 1-2 short sentences."
            )
        )

    async def hangup(self):
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call."""
        identity = self.participant.identity if self.participant else "unknown"
        logger.info(f"ending the call for {identity}")
        await ctx.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this AFTER you hear the voicemail greeting."""
        identity = self.participant.identity if self.participant else "unknown"
        logger.info(f"detected answering machine for {identity}")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name} via {LIVEKIT_URL}")
    await ctx.connect()

    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error(
            "No valid phone_number in job metadata. Expected JSON with 'phone_number'."
        )
        ctx.shutdown()
        return

    if not OUTBOUND_TRUNK_ID:
        logger.error("SIP_OUTBOUND_TRUNK_ID is not set in the environment.")
        ctx.shutdown()
        return

    participant_identity = f"sip-{phone_number}"
    agent = OutboundAgent(dial_info=dial_info)

    session = AgentSession(
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
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-najdi-female-1",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.9,
        ),,
        vad=silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.4),
    )

    @session.on("error")
    def _on_error(err):
        logger.error(f"Agent session error: {err}")

    @session.on("metrics_collected")
    def _on_metrics(ev):
        # surface STT/LLM/TTS latency so future "no audio" issues are diagnosable
        logger.info(f"metrics: {ev.metrics}")

    # Start the agent session FIRST so it's ready when the user picks up
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                participant_identity=participant_identity,
            ),
        )
    )

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
            agent_name="outbound-caller",
            num_idle_processes=1,
        )
    )
