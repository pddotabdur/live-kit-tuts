"""Build LLM / STT / TTS from environment (ported from sara-agent).

- **LLM:** LiveKit Cloud Inference (``NORA_LLM_MODEL``; uses ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``).
- **STT:** Deepgram or Soniox.
- **TTS:** Faseeh (default), or Cartesia, ElevenLabs, or Fish Audio.

Each setting reads ``NORA_*`` first, then ``SARA_*`` so you can reuse a Sara ``.env``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("nora-agent")


def _env_first(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    raw = _env_first(*names, default="").lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(*names: str, default: int) -> int:
    raw = _env_first(*names, default="")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    raw = _env_first(*names, default="")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_language_hints(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_llm():
    """LiveKit Cloud Inference LLM (OpenAI-compatible; billed via LiveKit)."""
    from livekit.agents.inference import LLM

    model = _env_first("NORA_LLM_MODEL", "SARA_LLM_MODEL", default="openai/gpt-4.1-mini")
    logger.info("LLM (LiveKit inference): %s", model)
    return LLM.from_model_string(model)


def build_stt():
    """Speech-to-text: ``deepgram`` or ``soniox``."""
    provider = _env_first("NORA_STT_PROVIDER", "SARA_STT_PROVIDER", default="deepgram").lower()
    language = _env_first("NORA_STT_LANGUAGE", "SARA_STT_LANGUAGE", default="ar")

    if provider == "deepgram":
        from livekit.plugins import deepgram

        model = _env_first("NORA_DEEPGRAM_STT_MODEL", "SARA_DEEPGRAM_STT_MODEL", default="nova-3")
        logger.info("STT: Deepgram model=%s language=%s", model, language)
        return deepgram.STT(model=model, language=language)

    if provider == "soniox":
        from livekit.plugins import soniox

        hints_raw = _env_first("NORA_SONIOX_LANGUAGE_HINTS", "SARA_SONIOX_LANGUAGE_HINTS", default="ar")
        hints = _parse_language_hints(hints_raw) or ["ar"]
        model = _env_first("NORA_SONIOX_MODEL", "SARA_SONIOX_MODEL", default="stt-rt-v4")
        endpoint_ms = _env_int(
            "NORA_SONIOX_MAX_ENDPOINT_DELAY_MS",
            "SARA_SONIOX_MAX_ENDPOINT_DELAY_MS",
            default=500,
        )
        endpoint_ms = max(500, min(3000, endpoint_ms))
        opts = soniox.STTOptions(
            model=model,
            language_hints=hints,
            enable_language_identification=_env_bool(
                "NORA_SONIOX_ENABLE_LANGUAGE_IDENTIFICATION",
                "SARA_SONIOX_ENABLE_LANGUAGE_IDENTIFICATION",
                default=True,
            ),
            enable_speaker_diarization=_env_bool(
                "NORA_SONIOX_ENABLE_SPEAKER_DIARIZATION",
                "SARA_SONIOX_ENABLE_SPEAKER_DIARIZATION",
                default=False,
            ),
            max_endpoint_delay_ms=endpoint_ms,
        )
        logger.info("STT: Soniox model=%s language_hints=%s", model, hints)
        return soniox.STT(params=opts)

    raise ValueError(
        f"Unknown STT provider={provider!r}. Use 'deepgram' or 'soniox' (NORA_STT_PROVIDER / SARA_STT_PROVIDER)."
    )


def build_tts():
    """Text-to-speech: ``cartesia``, ``elevenlabs``, ``fishaudio``, or ``faseeh``."""
    provider = _env_first("NORA_TTS_PROVIDER", "SARA_TTS_PROVIDER", default="faseeh").lower()

    if provider == "cartesia":
        from livekit.plugins import cartesia

        model = _env_first("NORA_CARTESIA_MODEL", "SARA_CARTESIA_MODEL", default="sonic-3")
        lang = _env_first("NORA_TTS_LANGUAGE", "SARA_TTS_LANGUAGE", default="ar") or None
        voice_id = _env_first("NORA_TTS_VOICE_ID", "SARA_TTS_VOICE_ID", default="")
        kwargs: dict[str, Any] = {"model": model}
        if lang:
            kwargs["language"] = lang
        if voice_id:
            kwargs["voice"] = voice_id
        logger.info(
            "TTS: Cartesia model=%s voice=%s language=%s",
            model,
            voice_id or "(default)",
            lang,
        )
        return cartesia.TTS(**kwargs)

    if provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        model = _env_first("NORA_ELEVENLABS_MODEL", "SARA_ELEVENLABS_MODEL", default="eleven_turbo_v2_5")
        voice_id = _env_first(
            "NORA_TTS_VOICE_ID", "SARA_TTS_VOICE_ID", default=elevenlabs.DEFAULT_VOICE_ID
        )
        lang = _env_first("NORA_TTS_LANGUAGE", "SARA_TTS_LANGUAGE", default="") or None
        kwargs: dict[str, Any] = {"model": model, "voice_id": voice_id}
        if lang:
            kwargs["language"] = lang
        logger.info("TTS: ElevenLabs model=%s voice_id=%s", model, voice_id)
        return elevenlabs.TTS(**kwargs)

    if provider == "fishaudio":
        from livekit.plugins import fishaudio

        model = _env_first("NORA_FISH_MODEL", "SARA_FISH_MODEL", default="s1")
        ref = _env_first("NORA_FISH_REFERENCE_ID", "SARA_FISH_REFERENCE_ID", default="") or _env_first(
            "NORA_TTS_VOICE_ID", "SARA_TTS_VOICE_ID", default=""
        )
        latency = _env_first("NORA_FISH_LATENCY_MODE", "SARA_FISH_LATENCY_MODE", default="balanced")
        if latency not in ("normal", "balanced"):
            latency = "balanced"
        kwargs: dict[str, Any] = {"model": model, "latency_mode": latency}
        if ref:
            kwargs["reference_id"] = ref
        logger.info(
            "TTS: Fish Audio model=%s reference_id=%s latency_mode=%s",
            model,
            ref or "(default)",
            latency,
        )
        return fishaudio.TTS(**kwargs)

    if provider == "faseeh":
        from livekit.plugins import faseeh

        voice_id = _env_first("NORA_TTS_VOICE_ID", "SARA_TTS_VOICE_ID", default=faseeh.DEFAULT_VOICE_ID)
        model = _env_first("NORA_FASEEH_MODEL", "SARA_FASEEH_MODEL", default="faseeh-v1-preview")
        stability = _env_float("NORA_FASEEH_STABILITY", "SARA_FASEEH_STABILITY", default=0.5)
        speed = _env_float("NORA_FASEEH_SPEED", "SARA_FASEEH_SPEED", default=1.0)
        logger.info("TTS: Faseeh model=%s voice_id=%s", model, voice_id)
        return faseeh.TTS(
            voice_id=voice_id,
            model=model,
            stability=stability,
            speed=speed,
        )

    raise ValueError(
        f"Unknown TTS provider={provider!r}. "
        "Use 'cartesia', 'elevenlabs', 'fishaudio', or 'faseeh' (NORA_TTS_PROVIDER / SARA_TTS_PROVIDER)."
    )
