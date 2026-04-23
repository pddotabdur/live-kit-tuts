import aiohttp
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    url = "https://api.munsit.com/api/v1/text-to-speech/faseeh-v1-preview"
    headers = {
        "x-api-key": os.getenv("FASEEH_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = {
        "voice_id": "ar-najdi-female-1",
        "text": "مرحبا",
        "stability": 0.75,
        "speed": 0.9,
        "streaming": False,
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                print("Status:", resp.status)
                if resp.status != 200:
                    print("Error:", await resp.text())
                else:
                    print("Success, reading chunks...")
                    async for chunk, _ in resp.content.iter_chunks():
                        if chunk:
                            print("Received bytes:", len(chunk))
                            break
        except Exception as e:
            print("Exception:", type(e), e)

asyncio.run(main())
