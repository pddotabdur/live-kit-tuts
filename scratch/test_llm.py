from livekit.plugins import openai
from dotenv import load_dotenv
import asyncio
load_dotenv()

async def main():
    try:
        llm = openai.LLM(model="gpt-4o-mini")
        from livekit.agents.llm import ChatContext
        ctx = ChatContext()
        ctx.add_message(role="user", content="hello")
        print("Sending chat to LLM...")
        stream = llm.chat(chat_ctx=ctx)
        async for chunk in stream:
            print("LLM chunk:", chunk)
    except Exception as e:
        print("LLM Exception:", type(e), e)

asyncio.run(main())
