import asyncio
import logging
import os
from pathlib import Path

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
    stt,
    TurnHandlingOptions
)
from livekit.plugins import silero, openai, deepgram, cartesia, faseeh, google
import hamsa_livekit
from livekit.agents import AgentSession, TurnHandlingOptions, inference
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("terminal-agent")
logger.setLevel(logging.INFO)

# =======================
# PROMPTS & GREETINGS
# =======================

ENGLISH_PROMPT = """You are an empathetic, professional, and highly conversational banking representative from STC Bank.
Your task is to call the customer, warmly but very briefly introduce yourself, verify their identity by asking for their name, and delicately remind them of a $1,000 overdue payment. Ask them politely when they might be able to make this payment.
Speak naturally with advanced semantic turn-taking. Use excellent conversational skills, and occasionally employ brief filler words like 'hmm' or 'I see' to signal active listening.
Be polite and respectful. If they ask questions, answer them concisely based on your persona. Be polite but also firm like a bank representative.
CRITICAL WARNING: Do NOT hang up the call arbitrarily. You MUST wait until the conversation is completely finished. ONLY invoke the `end_call` tool when the customer explicitly says "Goodbye", "Have a good day", or clearly expresses they want to hang up."""

ENGLISH_GREETING = "Introduce yourself warmly and ask for the customer's name to immediately verify their identity."
ARABIC_NOURA_PROMPT = """أنتِ نورا، مساعدة صوتية ذكية لمكتب 'توافق' لتحصيل الديون. صوتكِ دافئ ونسائي.
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
9. تفاصيل العميل الحالي: الاسم: عبدالله، المبلغ المستحق: 1000 ريال."""

ARABIC_NOURA_GREETING = "ابدئي المكالمة بالتحية والتعريف بنفسك وبمكتب توافق، ثم اذكري اسم العميل والمبلغ المستحق."


# =======================
# UNIFIED AGENT CLASS
# =======================

class TerminalTestAgent(Agent):
    """Unified agent that handles phone calls based on text prompts without SIP."""

    def __init__(self, instructions: str, greeting: str):
        super().__init__(instructions=instructions)
        self.participant: rtc.RemoteParticipant | None = None
        self.greeting = greeting

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def on_enter(self):
        # We start the conversation locally when we enter the room and the agent is ready
        self.session.generate_reply(user_input=self.greeting)

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called ONLY when the user explicitly wants to end the call."""
        logger.info(f"ending the call for {self.participant.identity if self.participant else 'unknown'}")
        await ctx.wait_for_playout()
        # In a test terminal environment context without SIP, we can just log
        logger.info("Agent decided to terminate the conversation.")


# =======================
# WORKER ENTRYPOINT
# =======================

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # Note: We do NOT trigger SIP outbound here.
    # Instead, we wait for a participant to join through the frontend / sandbox.
    logger.info("Waiting for a user to join via LiveKit Sandbox or Frontend...")
    participant = await ctx.wait_for_participant()
    logger.info(f"participant joined: {participant.identity}")

    # Determine mode: english, arabic (hamsa), or faseeh
    # Examples:
    # AGENT_MODE=english python terminal_test_agent.py dev
    # AGENT_MODE=arabic python terminal_test_agent.py dev
    # AGENT_MODE=faseeh python terminal_test_agent.py dev
    mode = os.getenv("AGENT_MODE", "english").lower()

    vad = silero.VAD.load()

    if mode == "english":
        logger.info("Starting agent in ENGLISH mode (Cartesia + Deepgram)")
        agent = TerminalTestAgent(instructions=ENGLISH_PROMPT, greeting=ENGLISH_GREETING)
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
            llm=openai.LLM(model="gpt-4o", temperature=0.7),
            tts=cartesia.TTS(
                model="sonic-english",
                voice="6f84f4b8-58a2-430c-8c79-688dad597532",
                speed=1.0,
            ),
        )

    elif mode == "arabic" or mode == "hamsa":
        logger.info("Starting agent in ARABIC mode (Hamsa STT + Hamsa TTS)")
        agent = TerminalTestAgent(instructions=ARABIC_NOURA_PROMPT, greeting=ARABIC_NOURA_GREETING)
        session = AgentSessionsession = AgentSession(
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
        llm=google.LLM(model="gemini-3.1-flash-lite-preview", temperature=0.7),
        tts = google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Zephyr",
            instructions="Speak in a friendly and engaging tone.",
        ),
    )

    elif mode == "faseeh":
        logger.info("Starting agent in FASEEH mode (Deepgram STT + Faseeh TTS)")
        agent = TerminalTestAgent(instructions=ARABIC_NOURA_PROMPT, greeting=ARABIC_NOURA_GREETING)
        session = AgentSession(
            vad=vad,
            turn_handling=TurnHandlingOptions(
                turn_detection="vad",        
                endpointing={
                    "min_delay": 0.25,
                    "max_delay": 1.0,
                },
            ),
            stt=stt.FallbackAdapter(
                [
                    deepgram.STT(model="nova-3", language="ar"),
                    openai.STT(),
                ],
                vad=vad
            ),
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            tts=faseeh.TTS(
                base_url="https://api.munsit.com/api/v1",
                voice_id="ar-najdi-female-1",
                model="faseeh-v1-preview",
                stability=0.75,
                speed=1.0,
            ),
        )
    else:
        logger.error(f"Unknown AGENT_MODE: {mode}. Please set to 'english', 'arabic', or 'faseeh'.")
        ctx.shutdown()
        return

    agent.set_participant(participant)

    # Useful for tracking exact STT/LLM/TTS latencies in the terminal directly
    @session.on("metrics_collected")
    def _on_metrics_collected(mtrcs):
        logger.info(f"Latency Metrics: {mtrcs}")

    logger.info("Starting agent session...")
    await session.start(
        agent=agent,
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="terminal-tester",
        )
    )
