"""
Outbound SIP call that plays a fixed audio clip twice and hangs up.

No STT/LLM/TTS — just dial, publish a track, play the WAV twice, end call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import JobContext, WorkerOptions, cli


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("audio-play-caller")
logger.setLevel(logging.INFO)

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")

AUDIO_CLIP_PATH = Path(__file__).parent / "Fasseh-fahad.wav"

PLAY_COUNT = 3
FRAME_MS = 10
GAP_BETWEEN_PLAYS_S = 0.5
HANGUP_TAIL_S = 0.5


async def _play_wav_once(source: rtc.AudioSource, path: Path) -> None:
    with wave.open(str(path), "rb") as wav:
        num_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise RuntimeError(
                f"Expected 16-bit PCM WAV, got sample_width={sample_width}"
            )

        samples_per_frame = sample_rate * FRAME_MS // 1000
        bytes_per_frame = samples_per_frame * num_channels * sample_width

        while True:
            chunk = wav.readframes(samples_per_frame)
            if not chunk:
                break
            if len(chunk) < bytes_per_frame:
                chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=samples_per_frame,
            )
            await source.capture_frame(frame)


def _wav_params(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnchannels()


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

    if not AUDIO_CLIP_PATH.exists():
        logger.error(f"audio clip not found: {AUDIO_CLIP_PATH}")
        ctx.shutdown()
        return

    sample_rate, num_channels = _wav_params(AUDIO_CLIP_PATH)

    source = rtc.AudioSource(sample_rate, num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("playback", source)
    publish_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.room.local_participant.publish_track(track, publish_options)

    participant_identity = f"sip-{phone_number}"

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

        await asyncio.sleep(0.4)

        for i in range(PLAY_COUNT):
            logger.info(f"playing clip ({i + 1}/{PLAY_COUNT})")
            await _play_wav_once(source, AUDIO_CLIP_PATH)
            if i < PLAY_COUNT - 1:
                await asyncio.sleep(GAP_BETWEEN_PLAYS_S)

        await asyncio.sleep(HANGUP_TAIL_S)
        logger.info("playback finished, ending call")

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
    finally:
        try:
            await ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=ctx.room.name)
            )
        except Exception as e:
            logger.error(f"error deleting room: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME", "audio-play-caller"),
            num_idle_processes=1,
        )
    )
