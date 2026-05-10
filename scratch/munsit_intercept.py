"""Monkey-patch the Munsit plugin to log what it actually sends as the
first chunk. Then call it the way LiveKit calls it (push a few audio
frames) and see the bytes that reach the server."""
from __future__ import annotations

import asyncio
import json
import os
import struct
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from livekit import rtc
from livekit.plugins import munsit
from livekit.plugins.munsit import stt as munsit_stt

# Patch send_audio_frame to dump first bytes.
orig = munsit_stt.SpeechStream._send_audio_frame  # type: ignore[attr-defined]

async def traced(self, frame):
    pcm = bytes(frame.data)
    print(f"  → frame: sr={frame.sample_rate} ch={frame.num_channels} "
          f"pcm_len={len(pcm)} first_chunk_sent={self._first_chunk_sent}")
    if not self._first_chunk_sent:
        from livekit.plugins.munsit._utils import build_wav_header
        h = build_wav_header(sample_rate=frame.sample_rate, num_channels=frame.num_channels)
        print(f"     header[0:8]={h[:8]!r}  full={len(h)} bytes")
        payload = h + pcm
        print(f"     payload first 8 bytes (as ints): {list(payload[:8])}")
    return await orig(self, frame)

munsit_stt.SpeechStream._send_audio_frame = traced


async def main():
    import aiohttp
    sess = aiohttp.ClientSession()
    stt = munsit.STT(
        mode="streaming",
        model="munsit",
        endpointing="server_diff",
        finalize_after_silence_ms=500,
        interim_results=True,
        language="ar",
        http_session=sess,
    )
    stream = stt.stream()

    # 100ms of silence at 16kHz mono int16
    # Try 10ms@8kHz frames to mimic SIP G.711 (no resample on our end).
    silence_8k_10ms = b"\x00\x00" * 80
    frame_8k = rtc.AudioFrame(
        data=silence_8k_10ms, sample_rate=8000, num_channels=1, samples_per_channel=80
    )
    print(f"=== TEST: 10ms@8kHz frames (SIP-like) ===")

    async def push():
        for _ in range(20):
            stream.push_frame(frame_8k)
            await asyncio.sleep(0.01)
        await asyncio.sleep(3)
        stream.end_input()

    async def consume():
        try:
            async for ev in stream:
                print("  ←", ev.type, getattr(ev.alternatives[0], "text", "") if ev.alternatives else "")
        except Exception as e:
            print("  ← error:", type(e).__name__, e)

    await asyncio.gather(push(), consume())


asyncio.run(main())
