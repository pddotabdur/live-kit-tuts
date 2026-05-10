"""Probe: does Munsit STT streaming accept a hand-rolled first-chunk WAV?

We bypass the LiveKit plugin and talk to wss://api.munsit.com directly using
the documented protocol, sending a 16kHz mono 16-bit silence buffer with a
proper RIFF header on chunk #1. This isolates whether the "missing RIFF
header" failure is a server issue, a network issue, or a plugin bug.
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ["MUNSIT_API_KEY"]
URL = "wss://api.munsit.com/api/v1/websocket/speech-to-text?model=munsit"


def wav_header(sample_rate=16000, num_channels=1, bits=16) -> bytes:
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    data_size = 0xFFFFFFFF - 44
    riff_size = data_size + 36
    return (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<I", 16)
        + struct.pack("<H", 1) + struct.pack("<H", num_channels)
        + struct.pack("<I", sample_rate) + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align) + struct.pack("<H", bits)
        + b"data" + struct.pack("<I", data_size)
    )


async def main():
    silence = b"\x00\x00" * 1600  # 100ms of 16kHz silence
    first_payload = wav_header() + silence

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL, headers={"x-api-key": API_KEY}) as ws:
            print("WS connected")

            msg = {
                "event": "audio_chunk",
                "data": {"audioBuffer": list(first_payload)},
            }
            await ws.send_str(json.dumps(msg))
            print(f"sent first chunk ({len(first_payload)} bytes inc. WAV header)")

            for _ in range(5):
                cont = {
                    "event": "audio_chunk",
                    "data": {"audioBuffer": list(silence)},
                }
                await ws.send_str(json.dumps(cont))
                await asyncio.sleep(0.1)
            print("sent 5 follow-up chunks")

            try:
                async for raw in ws:
                    print("  ←", raw.type, str(raw.data)[:200])
                    if raw.type == aiohttp.WSMsgType.CLOSED:
                        break
            except Exception as e:
                print("recv error:", e)


asyncio.run(main())
