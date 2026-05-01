"""
English test agent — modular, deterministic call flow.

Each step of the call script is a dedicated Agent subclass. Transitions
happen by returning the next agent from a function_tool. Verbatim lines
are delivered via session.say() (no LLM generation), and only branching
decisions go through the LLM as tool calls.

This is the pattern recommended by LiveKit for scripted workflows:
https://docs.livekit.io/agents/build/workflows/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from livekit.plugins import cartesia, deepgram, openai, silero


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("outbound-caller-en")
logger.setLevel(logging.INFO)

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")


# ---------- Shared state ----------

@dataclass
class CallData:
    customer_name: str = "Mohammed"
    amount: str = "10000"
    debt_date: str = "2023-01-01"
    national_id_last4: str = "1234"

    identity_confirmed: bool = False
    id_verified: bool = False
    outcome: str | None = None  # "paid_now" | "scheduled" | "partial" | "denied" | "no_agreement"
    scheduled_date: str | None = None
    sip_ready: asyncio.Event = field(default_factory=asyncio.Event)
    participant: rtc.RemoteParticipant | None = None


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

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext[CallData]):
        """Call this if you hear a voicemail greeting instead of a live person."""
        logger.info("voicemail detected, hanging up")
        await self.hangup()


# ---------- Step 1: Greeting ----------

GREETING_LINE = (
    "Hello, this is Nora calling from Tawafuq on behalf of STC Bank. "
    "Is now a good time to talk?"
)
GREETING_REPEAT = (
    "No problem. Again, this is Nora from Tawafuq's financial follow-up team. "
    "Is now a good time?"
)
BUSY_GOODBYE = (
    "No problem, we will call back at a more convenient time. Take care."
)


class GreetingAgent(BaseCallAgent):
    def __init__(self, data: CallData) -> None:
        super().__init__(
            instructions=(
                "You just greeted the customer and asked if it's a good time to talk. "
                "Listen to their reply and call exactly one tool: "
                "caller_available if they say yes / sure / go ahead, "
                "caller_busy if they say no / later / busy / call back, "
                "unclear_response if they didn't hear you or asked you to repeat. "
                "Do not say anything else. Do not generate free-form replies."
            ),
        )
        self.data = data

    async def on_enter(self):
        # Wait for SIP participant to actually join before speaking.
        await self.data.sip_ready.wait()
        await asyncio.sleep(0.4)  # small settle so carrier doesn't clip
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
    "Sorry to bother you, have a great day. Goodbye."
)


class IdentityAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"Ask exactly: 'Excuse me, am I speaking with {data.customer_name}?' "
                "Then listen and call exactly one tool: "
                "identity_confirmed if they say yes / that's me, "
                "wrong_person if they say no / wrong number / not me, "
                "unclear if they ask you to repeat or say something off-topic. "
                "Do not skip this step. Do not discuss the debt yet."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            f"With your permission, am I speaking with {self.data.customer_name}?",
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
            f"With your permission, am I speaking with {self.data.customer_name}?",
            allow_interruptions=False,
        )


# ---------- Step 3: ID verification ----------

class IDVerificationAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"Verify the customer's identity. Their last 4 ID digits should be {data.national_id_last4}. "
                "Listen for digits in their reply. Call exactly one tool: "
                f"id_correct if their digits match {data.national_id_last4}, "
                "id_incorrect if their digits do not match or they refuse, "
                "unclear if they asked to repeat. "
                "Never reveal the correct digits to them. Never proceed without confirmation."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            "Could you please confirm the last 4 digits of your national ID?",
            allow_interruptions=False,
        )

    @function_tool()
    async def id_correct(self, ctx: RunContext[CallData]):
        """The 4 digits the user spoke matched the expected value."""
        ctx.userdata.id_verified = True
        return DebtContextAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def id_incorrect(self, ctx: RunContext[CallData]):
        """The 4 digits did not match, or the user refused to provide them."""
        await self.session.say(
            "I'm sorry, those don't match our records. Could you try once more?",
            allow_interruptions=False,
        )

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """User asked you to repeat the question."""
        await self.session.say(
            "Could you please confirm the last 4 digits of your national ID?",
            allow_interruptions=False,
        )


# ---------- Step 4: Debt context ----------

PAYMENT_LINK_LINE = (
    "Excellent. I'll send you the payment link now, and you'll receive a "
    "confirmation message right after. Thank you very much."
)
DENIED_DEBT_LINE = (
    "No problem, let me double-check your records. "
    "Yes, I can confirm the amount has already been settled. "
    "We apologize for the inconvenience and thank you for your time."
)


class DebtContextAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"You just told the customer about an outstanding amount of {data.amount} SAR "
                f"from {data.debt_date} and asked if they can pay now. "
                "Classify their reply and call exactly one tool: "
                "will_pay_now if they agree to pay now / say yes / I can, "
                "needs_time if they ask for time / say later / not today, "
                "no_funds if they say they have no money / can't afford, "
                "denies_debt if they say it's not their debt / already paid / wrong amount, "
                "is_angry if they are frustrated, insulting, or accusing you of being a bot. "
                "Do not invent any new amount or date."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            f"I'd like to remind you of an outstanding amount of {self.data.amount} SAR "
            f"from {self.data.debt_date}. Are you able to settle it now?",
            allow_interruptions=True,
        )

    @function_tool()
    async def will_pay_now(self, ctx: RunContext[CallData]):
        """Customer agrees to pay now."""
        ctx.userdata.outcome = "paid_now"
        await self.session.say(PAYMENT_LINK_LINE, allow_interruptions=False)
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def needs_time(self, ctx: RunContext[CallData]):
        """Customer needs more time before paying."""
        return ScheduleAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def no_funds(self, ctx: RunContext[CallData]):
        """Customer says they have no money."""
        return PartialPaymentAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def denies_debt(self, ctx: RunContext[CallData]):
        """Customer denies the debt or says it's already paid."""
        ctx.userdata.outcome = "denied"
        await self.session.say(DENIED_DEBT_LINE, allow_interruptions=False)
        await self.hangup()

    @function_tool()
    async def is_angry(self, ctx: RunContext[CallData]):
        """Customer is angry, frustrated, or accusing you of being a bot."""
        await self.session.say(
            "I understand, and our goal is to make this as easy as possible "
            "for you with no pressure. Are you able to settle the amount today?",
            allow_interruptions=True,
        )


# ---------- Step 5a: Schedule a payment date ----------

ESCALATION_WARNING = (
    "Without a clear payment date, the case may be escalated. "
    "We always prefer to resolve this amicably. "
    "What's the soonest date that works for you within the next two months?"
)


class ScheduleAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You are negotiating a payment date with the customer. "
                "Acceptable window: within the next 2 months from today. "
                "Listen to their reply and call exactly one tool: "
                "time_acceptable(date) if their proposed date is within 2 months, "
                "time_too_far if they propose more than 2 months, "
                "refuses_to_commit if they refuse to give any date or keep evading. "
                "Pass dates as ISO YYYY-MM-DD when possible."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            "No problem. When would be a good time for you to settle the amount?",
            allow_interruptions=True,
        )

    @function_tool()
    async def time_acceptable(self, ctx: RunContext[CallData], date: str):
        """The customer proposed a date within the next 2 months.

        Args:
            date: The date they proposed, ideally ISO YYYY-MM-DD.
        """
        ctx.userdata.scheduled_date = date
        ctx.userdata.outcome = "scheduled"
        await self.session.say(
            f"Great, I've recorded {date} as your payment date. "
            "We'll be in touch closer to the time.",
            allow_interruptions=False,
        )
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def time_too_far(self, ctx: RunContext[CallData]):
        """The customer's proposed date is more than 2 months away."""
        await self.session.say(
            "We'd like to keep this within a shorter window. "
            "Could you commit to a date within the next two months?",
            allow_interruptions=True,
        )

    @function_tool()
    async def refuses_to_commit(self, ctx: RunContext[CallData]):
        """Customer refuses to give any concrete date."""
        await self.session.say(ESCALATION_WARNING, allow_interruptions=True)


# ---------- Step 5b: Partial payment offer ----------

class PartialPaymentAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You are offering a partial payment plan to a customer who said they have no funds. "
                "Listen to their reply and call exactly one tool: "
                "accepts_partial(amount) if they agree to pay any partial amount, "
                "refuses_partial if they cannot pay even a small amount."
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        await self.session.say(
            "No problem. Let's find a workable solution. "
            "Would you be able to pay a small amount, for example 100 SAR, today?",
            allow_interruptions=True,
        )

    @function_tool()
    async def accepts_partial(self, ctx: RunContext[CallData], amount: str):
        """Customer agreed to a partial payment.

        Args:
            amount: The partial amount they agreed to.
        """
        ctx.userdata.outcome = "partial"
        await self.session.say(
            f"Thank you. I'll send the link for {amount} SAR now.",
            allow_interruptions=False,
        )
        return ClosingAgent(self.data, chat_ctx=self.chat_ctx)

    @function_tool()
    async def refuses_partial(self, ctx: RunContext[CallData]):
        """Customer cannot pay any amount today."""
        return ScheduleAgent(self.data, chat_ctx=self.chat_ctx)


# ---------- Step 6: Closing ----------

CLOSING_LINE = (
    "Thank you for your cooperation. Have a great day, goodbye."
)


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
        customer_name=dial_info.get("name", "Mohammed"),
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
        stt=deepgram.STT(model="nova-3", language="en-US"),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
        tts=cartesia.TTS(
            model="sonic-2",
            voice="794f9389-aac1-45b6-b726-9d9369183238",  # default Cartesia female voice
            language="en",
        ),
        vad=silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.4),
    )

    @session.on("error")
    def _on_error(err):
        logger.error(f"Agent session error: {err}")

    @session.on("metrics_collected")
    def _on_metrics(ev):
        logger.info(f"metrics: {ev.metrics}")

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
            agent_name=os.getenv("AGENT_NAME", "outbound-caller-en"),
            num_idle_processes=1,
        )
    )
