from livekit.plugins import faseeh
from dotenv import load_dotenv
import asyncio
import os
load_dotenv()

async def main():
    try:
        tts = faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-najdi-female-1",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.9,
            api_key=os.getenv("FASEEH_API_KEY")
        )
        print("Synthesizing speech...")
        stream = tts.stream()
        stream.push_text("مرحبا")
        stream.end_input()
        async for chunk in stream:
            print("Audio chunk received, length:", len(chunk.data))
        print("Synthesis complete.")
    except Exception as e:
        print("TTS Exception:", type(e), e)
        cause = getattr(e, "__cause__", None)
        print("Cause:", type(cause), cause)
        if cause:
            cause2 = getattr(cause, "__cause__", None)
            print("Cause2:", type(cause2), cause2)

asyncio.run(main())
