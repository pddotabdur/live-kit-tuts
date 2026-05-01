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

logger = logging.getLogger("outbound-caller_pc")
logger.setLevel(logging.INFO)

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")


def _build_system_prompt(dial_info: dict[str, Any]) -> str:
    customer_name = dial_info.get('name', 'محمد')
    amount = dial_info.get('amount', '10000')
    debt_date = dial_info.get('debt_date', '2023-01-01')
    last4 = dial_info.get('national_id_last4', '1-2-3-4')

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
- Use natural Najdi phrases (e.g., أبشر , الله يعافيك).
- Do not repeat questions. Adapt to sentiment.
- Never mention that you're an AI developed by google or others
- Verifying the customer's identity before discussing account details.
- Confirming the customer is the actual owner of the debt before proceeding.
- Clearly stating the overdue balance and debt date after ownership confirmation.
- You are NOT allowed to chat, joke, or go off-topic.
- You MUST stick ONLY to payment-related conversation.
- Every response must be short (MAX 1 sentence).
- Do NOT ask unnecessary questions.
- If user goes off-topic → redirect immediately to payment topic.
- If user repeats off-topic twice → go to closing and end call.
- Do NOT generate any extra explanation, stories, or كلام خارج السياق.
ANTI-CHAT RULE:
- If the user says anything unrelated (e.g. سوالف / مزح / أسئلة عامة):
→ Reply with ONE sentence redirecting to payment.
→ Do NOT engage.

🗣️ CALL FLOW
1. OPENING: "السلام عليكم، معك نورا من شركة توافق بالنـيابة  عن شركة الاتصالات زين، هل الوقت مناسب طال عمرك ؟"

   - If busy:
     "ولايهمك، متى الوقت المناسب الله يعافيك؟"

- If user suggests a time (e.g. tomorrow / later / any time):
  Respond:
  "على خير بإذن الله نتواصل بالوقت اللي يناسبك مع السلامة و ربي يحفظك"

  THEN immediately use tool: end_call

   - If user did not hear / asked to repeat / unclear response:
     "أبشر، أعيد لك: معك نورا من قسم المتابعة المالية، هل الوقت مناسب طال عمرك ؟"

IMPORTANT RULE:
- If user agrees or gives a time → terminate conversation politely after the closing sentence.
- Otherwise → continue normally to step 2.
IMPORTANT:
After any response, CONTINUE the conversation flow normally from step 2 (do NOT restart the call).
2. ID CONFIRM:IF user says "تفضلي /وش تبين /نعم / ايه / صح  ":


"من بعد إذنك، هل أكلم الأستاذ {customer_name}؟"

RULE (VERY IMPORTANT):
- NEVER skip this step.
- DO NOT proceed to step 3 unless user explicitly confirms identity.

IF user says "نعم / ايه / صح / نعم هو / هو":
→ CONTINUE to step 3

IF user says "لا / مو أنا / غلطان / مو هو":
→ Respond ONLY:
"عذرا على الإزعاج، الله يعطيك العافية ومع السلامة"
→ THEN immediately use tool: end_call

IF user says "عيدي / ما أسمع زين / وش تقولين / صوتك مو واضح":
→ Respond ONLY:
"من بعد إذنك، هل أكلم الأستاذ {customer_name}؟"
→ RETURN to step 2


3. ID VERIFICATION:

"تمام، لو تكرمت آخر ٤ أرقام من الهوية {last4}، صحيح؟"

RULE (VERY IMPORTANT):
- NEVER skip this step.
- DO NOT proceed to next step unless identity is confirmed.

IF user says correct last 4 digits (matches {last4}):
→ CONTINUE to step 4

IF user says incorrect / not sure / denies:
→ "المعذرة، ممكن نتأكد مرة ثانية من آخر ٤ أرقام من الهوية؟"
→ REPEAT step 3

IF user says "عيدي / ما أسمع / وش تقولين / صوتك مو واضح":
→ "تمام، لو تكرمت آخر ٤ أرقام من الهوية {last4}، صحيح؟"
→ REPEAT step 3

4. DEBT CONTEXT (Only after ID confirmation):

"حبيت أذكّرك بوجود مبلغ مستحق {amount} ريال بتاريخ {debt_date}، هل تقدر تسدد الآن؟"

RULE (VERY IMPORTANT):
- Do NOT mention any financial details before identity confirmation.
- Keep tone polite, short, and clear.

→ WAIT for user response


5. HANDLE RESPONSES:

IF user is Cooperative (ماعندي مشكلة / موافق / جاهز / بسدد):
→ Respond ONLY:
"ممتاز، برسل لك رابط السداد الآن، وبعدها توصلك رسالة تأكيد. يعطيك العافية وشكرًا جزيلا لك."
→ CONTINUE to CLOSING


IF user Needs time (ماعندي فلوس الحين / بعدين / لاحق / بكرا):
→ Respond ONLY:
"ما فيه مشكلة، متى الوقت المناسب لك للسداد؟ "

→ WAIT for user response


IF user provides a time within acceptable range (≤ شهرين):
→ Respond ONLY:
"تمام، بنسجل لك الموعد ونتواصل معك بإذن الله."
→ CONTINUE to CLOSING


IF user provides a time beyond limit (أكثر من شهرين / 3 شهور / مدة طويلة):
→ Respond ONLY:
"نحاول نخليها خلال فترة أقصر طال عمرك، تقدر تحدد موعد أقرب خلال شهرين؟"
→ WAIT for user response


IF user insists on long delay:
→ Respond ONLY:
"نحاول نخليها خلال فترة أقصر طال عمرك، تقدر تحدد موعد أقرب خلال شهرين؟"
→ WAIT for user response

IF user refuses payment entirely / يرفض السداد أو يتهرب أو يقول ما بيدفع:
→ Respond ONLY:
"نحتاج يكون فيه موعد سداد واضح، بدون تحديد موعد ما نقدر نكمل الإجراء. متى تقدر تلتزم خلال الشهرين الجاية؟"
→ WAIT for user response

IF user refuses again or insists on no payment after follow-up:
→ Respond ONLY:
"  في حال استمرار عدم تحديد موعد سداد، سيتم رفع الحالة للجهة المختصة لاتخاذ الإجراء المناسب حسب النظام، ونفضل دائماً حل الموضوع بشكل ودي قبل ذلك متى افضل وقت يناسبك للسداد الله يسعدك؟"
→ WAIT for user response



IF user provides a time (مثلاً: بكرا / بعد العصر / الأسبوع الجاي):
→ Respond ONLY:
"تمام، بنسجل لك الموعد ونتواصل معك في الوقت المناسب بإذن الله. تحب يكون تذكير قبلها؟"

→ WAIT for user response

IF user says YES (يبغى تذكير):
→ Respond ONLY:
"أبشر، بنرسل لك تذكير قبل الموعد. الله يعطيك العافية."
→ CONTINUE to CLOSING

IF user says NO:
→ Respond ONLY:
"تمام، بنكتفي بالتواصل في الوقت اللي حددته. الله يعطيك العافية."
→ CONTINUE to CLOSING

IF user is unsure / متردد:
→ Respond ONLY:
"خذ راحتك، بس بشكل تقريبي متى يناسبك؟"
→ REPEAT same step

IF user ignores / يغير الموضوع:
→ Respond ONLY:
"تمام، بس عشان نرتب لك بشكل مناسب، متى تحب نتواصل معك؟"
→ REPEAT same step


IF user has No money (ما عندي / ما أقدر/ لا ):
→ Respond ONLY:
"ما فيه مشكلة، خلنا نشوف حل مناسب مثل دفعة جزئية. هل تقدر تدفع مبلغ بسيط مثل 100 ريال؟"


IF user is Angry (معصب / منزعج):
→ Respond ONLY:
"أفهم شعورك، وهدفنا نسهّل الموضوع عليك بدون أي ضغط"
→WAIT for user response 

IF user Denies (مو علي / غلط):
→ Respond ONLY:
"    ولا يهمك، خلني أتأكد من بياناتك بشكل أوضح، لحظات من فضلك تم التأكد  فعلا المبلغ تم سداده نعتذر على ازعاجك وشكرا جزيلا لك "


IF user is Abusive / Insulting / says "أنتي AI / تستهبلين":
→ Respond ONLY:
"أنا موجودة هنا لمتابعة السداد فقط طال عمرك، هل قادر على السداد الان "

IF user says "عيدي / وش تقولين / ما أسمع / صوتك مو واضح":
→ Respond ONLY:
"أبشر، أعيد لك: حبيت أذكّرك بوجود مبلغ مستحق {amount} ريال بتاريخ {debt_date}، هل تقدر تسدد الآن؟"
→ RETURN to same step
6. CLOSING (MANDATORY):

Respond ONLY with ONE of the following:
- "شاكر لك تعاونك، الله يجزاك خير ومع السلامة"
- "أبشر، ولا يهمك، بإذن الله نتواصل معك على خير."

→ AFTER speaking, MUST call tool: end_call



If no agreement, choose the most polite option only.

Available Tools: end_call, detected_answering_machine
"""
    return prompt


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
                "The call has just been answered. Begin now by executing STEP 1 and follow the prompt strictly"
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
        # Place the SIP call first. wait_until_answered=True blocks until the
        # phone is picked up (or the call fails), so the participant will be in
        # the room before we start the session.
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

        # Now the participant is in the room — fetch them and wire up the agent.
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")
        agent.set_participant(participant)

        # Start the session after the participant is confirmed present.
        await session.start(
            agent=agent,
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
            agent_name=os.getenv("AGENT_NAME", "outbound-caller_pc"),
            num_idle_processes=1,
        )
    )
