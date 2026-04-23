"""
metrics_aggregator.py
─────────────────────────────────────────────────────────────────────────────
Per-turn latency aggregator for the Bank Collection Demo Agent.

Tracks every component of every conversational turn:
  • VAD  — inference time (rolling avg per turn window)
  • EOU  — end-of-utterance delay + transcription delay
  • STT  — total transcription duration + audio duration
  • LLM  — wall-clock span + TTFT
  • TTS  — wall-clock span + TTFB

After each turn emits a single log line:
  TURN #n [speech_id] | VAD=12ms | EOU(eos/trans)=120ms/890ms
                      | STT=210ms | LLM=1200ms (TTFT=450ms)
                      | TTS=650ms (TTFB=120ms)
  PERCEIVED LATENCY:  EOU 120ms + LLM-TTFT 450ms + TTS-TTFB 120ms = ~690ms

Results are also written to:
  • <session_dir>/turns_<timestamp>.csv     — one row per turn, machine-readable
  • <session_dir>/turns_<timestamp>.jsonl   — full detail, one JSON object per turn

Usage
─────
    from metrics_aggregator import TurnMetricsAggregator

    agg = TurnMetricsAggregator(session_dir=Path("session_reports"))
    session.on("metrics_collected", agg.on_metrics_collected)
    # At session end:
    agg.close()
    summary = agg.session_summary()   # dict with p50/p95/mean per component

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from livekit.agents.metrics import (
    EOUMetrics,
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
)

logger = logging.getLogger("bank-collection-agent.metrics")


# ---------------------------------------------------------------------------
# Per-turn bucket
# ---------------------------------------------------------------------------
@dataclass
class TurnBucket:
    """Accumulates all component metrics for a single conversational turn."""

    speech_id: str
    turn_index: int

    # Filled in as each component reports
    vad_inference_ms: float | None = None        # avg inference time this window
    eou_delay_ms: float | None = None            # end-of-utterance delay (EOS→EOU decision)
    eou_transcription_ms: float | None = None    # transcription delay after speech ended
    stt_duration_ms: float | None = None         # STT request duration
    stt_audio_ms: float | None = None            # duration of audio pushed to STT
    llm_duration_ms: float | None = None         # total LLM wall-clock
    llm_ttft_ms: float | None = None             # LLM time-to-first-token
    llm_tokens: int = 0                          # completion tokens
    tts_duration_ms: float | None = None         # total TTS wall-clock
    tts_ttfb_ms: float | None = None             # TTS time-to-first-byte
    tts_chars: int = 0                           # characters synthesised

    completed: bool = False
    created_at: float = field(default_factory=time.monotonic)

    # ---- Derived ----
    @property
    def perceived_latency_ms(self) -> float | None:
        """
        The delay the human actually hears:
          EOU delay + LLM TTFT + TTS TTFB
        This is the time from when the user stopped speaking to
        when they first hear the agent's voice.
        """
        if None in (self.eou_delay_ms, self.llm_ttft_ms, self.tts_ttfb_ms):
            return None
        return self.eou_delay_ms + self.llm_ttft_ms + self.tts_ttfb_ms

    def is_complete(self) -> bool:
        """True once LLM + TTS have both reported (minimum for a meaningful turn line)."""
        return (
            self.eou_delay_ms is not None
            and self.llm_duration_ms is not None
            and self.tts_duration_ms is not None
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["perceived_latency_ms"] = self.perceived_latency_ms
        return d


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class TurnMetricsAggregator:
    """
    Subscribe to AgentSession `metrics_collected` events and produce
    per-turn latency summaries.
    """

    def __init__(self, session_dir: Path | None = None):
        self._buckets: dict[str, TurnBucket] = {}
        self._completed: list[TurnBucket] = []
        self._turn_counter = 0

        # Track last VAD state to compute per-turn delta
        self._last_vad: VADMetrics | None = None

        # Output files
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = (session_dir or Path(".")) / f"turns_{ts}"
        base.parent.mkdir(parents=True, exist_ok=True)
        self._csv_path = base.with_suffix(".csv")
        self._jsonl_path = base.with_suffix(".jsonl")

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "turn",
                "speech_id",
                "vad_inference_ms",
                "eou_delay_ms",
                "eou_transcription_ms",
                "stt_duration_ms",
                "stt_audio_ms",
                "llm_duration_ms",
                "llm_ttft_ms",
                "llm_tokens",
                "tts_duration_ms",
                "tts_ttfb_ms",
                "tts_chars",
                "perceived_latency_ms",
            ],
        )
        self._csv_writer.writeheader()
        self._jsonl_file = self._jsonl_path.open("w", encoding="utf-8")

        logger.info(f"📊 Metrics logs → {self._csv_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_metrics_collected(self, ev: Any) -> None:
        """
        Call this as:
            session.on("metrics_collected", agg.on_metrics_collected)

        The event object is a MetricsCollectedEvent with a `.metrics` field
        that is a union of VADMetrics | EOUMetrics | STTMetrics | LLMMetrics | TTSMetrics.
        """
        m = ev.metrics if hasattr(ev, "metrics") else ev
        mtype = getattr(m, "type", None)

        if mtype == "vad_metrics":
            self._handle_vad(m)
        elif mtype == "eou_metrics":
            self._handle_eou(m)
        elif mtype == "stt_metrics":
            self._handle_stt(m)
        elif mtype == "llm_metrics":
            self._handle_llm(m)
        elif mtype == "tts_metrics":
            self._handle_tts(m)
        # realtime / interruption metrics are ignored here

    def close(self) -> None:
        """Flush and close output files."""
        try:
            self._csv_file.flush()
            self._csv_file.close()
            self._jsonl_file.flush()
            self._jsonl_file.close()
        except Exception:
            pass

    def session_summary(self) -> dict[str, Any]:
        """Return aggregate statistics over all completed turns in this session."""
        if not self._completed:
            return {"turns": 0}

        def _stats(values: list[float]) -> dict:
            if not values:
                return {}
            sv = sorted(values)
            n = len(sv)
            return {
                "mean_ms": round(sum(sv) / n, 1),
                "p50_ms": round(sv[n // 2], 1),
                "p95_ms": round(sv[int(n * 0.95)], 1),
                "min_ms": round(sv[0], 1),
                "max_ms": round(sv[-1], 1),
            }

        def _collect(attr: str) -> list[float]:
            return [
                getattr(b, attr)
                for b in self._completed
                if getattr(b, attr) is not None
            ]

        return {
            "turns": len(self._completed),
            "eou_delay": _stats(_collect("eou_delay_ms")),
            "eou_transcription": _stats(_collect("eou_transcription_ms")),
            "stt": _stats(_collect("stt_duration_ms")),
            "llm_wall": _stats(_collect("llm_duration_ms")),
            "llm_ttft": _stats(_collect("llm_ttft_ms")),
            "tts_wall": _stats(_collect("tts_duration_ms")),
            "tts_ttfb": _stats(_collect("tts_ttfb_ms")),
            "perceived_latency": _stats(
                [b.perceived_latency_ms for b in self._completed if b.perceived_latency_ms is not None]
            ),
        }

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _get_or_create_bucket(self, speech_id: str) -> TurnBucket:
        if speech_id not in self._buckets:
            self._turn_counter += 1
            self._buckets[speech_id] = TurnBucket(
                speech_id=speech_id, turn_index=self._turn_counter
            )
        return self._buckets[speech_id]

    def _handle_vad(self, m: VADMetrics) -> None:
        """VAD metrics are periodic aggregates — compute avg inference per frame."""
        if self._last_vad is None:
            self._last_vad = m
            return
        delta_count = m.inference_count - self._last_vad.inference_count
        delta_total = m.inference_duration_total - self._last_vad.inference_duration_total
        if delta_count > 0:
            avg_ms = round((delta_total / delta_count) * 1000, 2)
            # Stamp the most recent open bucket (if any), else store for next turn
            self._pending_vad_ms = avg_ms
        self._last_vad = m

    def _handle_eou(self, m: EOUMetrics) -> None:
        speech_id = m.speech_id or f"_eou_{int(m.timestamp * 1000)}"
        bucket = self._get_or_create_bucket(speech_id)
        bucket.eou_delay_ms = round(m.end_of_utterance_delay * 1000, 1)
        bucket.eou_transcription_ms = round(m.transcription_delay * 1000, 1)
        # Attach any pending VAD reading to this turn
        if hasattr(self, "_pending_vad_ms"):
            bucket.vad_inference_ms = self._pending_vad_ms
            del self._pending_vad_ms
        self._try_complete(bucket)

    def _handle_stt(self, m: STTMetrics) -> None:
        # STT doesn't carry speech_id directly; we match by recency
        # Find the most recently created incomplete bucket
        bucket = self._latest_incomplete_bucket()
        if bucket is None:
            return
        bucket.stt_duration_ms = round(m.duration * 1000, 1)
        bucket.stt_audio_ms = round(m.audio_duration * 1000, 1)
        self._try_complete(bucket)

    def _handle_llm(self, m: LLMMetrics) -> None:
        speech_id = m.speech_id or f"_llm_{int(m.timestamp * 1000)}"
        bucket = self._get_or_create_bucket(speech_id)
        bucket.llm_duration_ms = round(m.duration * 1000, 1)
        bucket.llm_ttft_ms = round(m.ttft * 1000, 1)
        bucket.llm_tokens = m.completion_tokens
        self._try_complete(bucket)

    def _handle_tts(self, m: TTSMetrics) -> None:
        speech_id = m.speech_id or f"_tts_{int(m.timestamp * 1000)}"
        bucket = self._get_or_create_bucket(speech_id)
        bucket.tts_duration_ms = round(m.duration * 1000, 1)
        bucket.tts_ttfb_ms = round(m.ttfb * 1000, 1)
        bucket.tts_chars = m.characters_count
        self._try_complete(bucket)

    def _latest_incomplete_bucket(self) -> TurnBucket | None:
        """Return the most recent bucket that hasn't received STT data yet."""
        candidates = [
            b for b in self._buckets.values()
            if not b.completed and b.stt_duration_ms is None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda b: b.created_at)

    # ------------------------------------------------------------------
    # Completion & output
    # ------------------------------------------------------------------

    def _try_complete(self, bucket: TurnBucket) -> None:
        if bucket.completed or not bucket.is_complete():
            return
        bucket.completed = True
        self._completed.append(bucket)
        self._emit_turn_line(bucket)
        self._write_csv(bucket)
        self._write_jsonl(bucket)

    def _emit_turn_line(self, b: TurnBucket) -> None:
        """Print the formatted turn latency line to the logger."""

        def _fmt(v: float | None) -> str:
            return f"{v:.0f}ms" if v is not None else "—"

        def _fmtpair(a: float | None, b_: float | None, label: str) -> str:
            return f"{label}({_fmt(a)}/{_fmt(b_)})"

        sid_short = (b.speech_id or "?")[:8]
        perc = b.perceived_latency_ms

        line1 = (
            f"TURN #{b.turn_index:>2} [{sid_short}]  "
            f"VAD={_fmt(b.vad_inference_ms)}  "
            f"EOU(eos/trans)={_fmt(b.eou_delay_ms)}/{_fmt(b.eou_transcription_ms)}  "
            f"STT={_fmt(b.stt_duration_ms)}  "
            f"LLM={_fmt(b.llm_duration_ms)} (TTFT={_fmt(b.llm_ttft_ms)})  "
            f"TTS={_fmt(b.tts_duration_ms)} (TTFB={_fmt(b.tts_ttfb_ms)})"
        )
        if perc is not None:
            line2 = (
                f"         └─ PERCEIVED LATENCY: "
                f"EOU {_fmt(b.eou_delay_ms)} + "
                f"LLM-TTFT {_fmt(b.llm_ttft_ms)} + "
                f"TTS-TTFB {_fmt(b.tts_ttfb_ms)} = ~{_fmt(perc)}"
            )
        else:
            line2 = None

        logger.info("─" * 80)
        logger.info(f"⏱️  {line1}")
        if line2:
            logger.info(line2)
        logger.info("─" * 80)

    def _write_csv(self, b: TurnBucket) -> None:
        try:
            row = {
                "turn": b.turn_index,
                "speech_id": b.speech_id,
                "vad_inference_ms": b.vad_inference_ms,
                "eou_delay_ms": b.eou_delay_ms,
                "eou_transcription_ms": b.eou_transcription_ms,
                "stt_duration_ms": b.stt_duration_ms,
                "stt_audio_ms": b.stt_audio_ms,
                "llm_duration_ms": b.llm_duration_ms,
                "llm_ttft_ms": b.llm_ttft_ms,
                "llm_tokens": b.llm_tokens,
                "tts_duration_ms": b.tts_duration_ms,
                "tts_ttfb_ms": b.tts_ttfb_ms,
                "tts_chars": b.tts_chars,
                "perceived_latency_ms": b.perceived_latency_ms,
            }
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception as e:
            logger.warning(f"CSV write failed: {e}")

    def _write_jsonl(self, b: TurnBucket) -> None:
        try:
            self._jsonl_file.write(json.dumps(b.to_dict()) + "\n")
            self._jsonl_file.flush()
        except Exception as e:
            logger.warning(f"JSONL write failed: {e}")
