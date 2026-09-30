"""Per-utterance timing across pipeline stages, all on one clock (time.perf_counter, ms).

Stages, in order (they add up exactly to end_to_end):

    detect       last sign's hand motion stopped -> segmenter closed the segment
                 (the closing pause, motion smoothing, and tracking the closing frame)
    classify     -> classifier returned the sign
    buffer       -> utterance judged complete (waiting to be sure the sentence is over)
    queue        -> speech thread picked it up (waits while the previous sentence plays)
    translate    -> sentence ready
    tts          -> audio rendered
    audio_start  -> sound reaches the speaker (stream start + device output latency)

end_to_end = "gesture completes" (last hand motion of the last sign) -> "audio starts".
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ("detect", "classify", "buffer", "queue", "translate", "tts", "audio_start")
NAN = float("nan")


def now_ms() -> float:
    # perf_counter, not monotonic: on Windows time.monotonic() only ticks every ~15.6 ms.
    return time.perf_counter() * 1000.0


@dataclass
class SignEvent:
    label: str
    confidence: float
    gesture_end_ms: float  # last frame with hand motion
    detected_ms: float  # segmenter closed the segment
    classified_ms: float
    duration_s: float  # trimmed sign length, for pace
    intensity: float  # facial expression intensity, for emphasis
    alternatives: list = field(default_factory=list)  # [(label, probability), ...]


@dataclass
class UtteranceTrace:
    index: int
    signs: list[SignEvent]
    flushed_ms: float
    flush_reason: str  # "gap", "hands_down", "max_signs" or "end"
    sentence: str = ""
    translate_start_ms: float = NAN
    translate_end_ms: float = NAN
    tts_end_ms: float = NAN
    audio_start_ms: float = NAN
    audio_seconds: float = 0.0
    audio_played: bool = False
    pace: float = 1.0
    emphasized: list[str] = field(default_factory=list)
    error: str | None = None

    def stages(self) -> dict[str, float]:
        last = self.signs[-1]
        return {
            "detect": last.detected_ms - last.gesture_end_ms,
            "classify": last.classified_ms - last.detected_ms,
            "buffer": self.flushed_ms - last.classified_ms,
            "queue": self.translate_start_ms - self.flushed_ms,
            "translate": self.translate_end_ms - self.translate_start_ms,
            "tts": self.tts_end_ms - self.translate_end_ms,
            "audio_start": self.audio_start_ms - self.tts_end_ms,
            "end_to_end": self.audio_start_ms - last.gesture_end_ms,
        }

    def describe(self) -> str:
        stages = self.stages()
        parts = " + ".join(f"{name} {stages[name]:.0f}" for name in STAGES)
        return f'#{self.index} "{self.sentence}"  end-to-end {stages["end_to_end"]:.0f} ms = {parts}'


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class TraceLog:
    """One JSON line per utterance: every timestamp plus the stage breakdown."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: UtteranceTrace) -> None:
        record = asdict(trace) | {"stages_ms": trace.stages()}
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(_clean(record)) + "\n")


def summarize(traces: list[UtteranceTrace], fps: float, warmup: dict[str, float], target_ms: float = 1500) -> str:
    lines = [f"Capture loop: {fps:.1f} frames/s."]
    if warmup:
        lines.append("Warm-up before the first frame (not in any latency): " + ", ".join(f"{k} {v:.0f} ms" for k, v in warmup.items()))
    measured = [t for t in traces if t.flush_reason != "end" and t.error is None]
    skipped = len(traces) - len(measured)
    if not measured:
        lines.append("No utterances completed during the run, so there is no latency to report.")
        return "\n".join(lines)

    reasons = {}
    for trace in measured:
        reasons[trace.flush_reason] = reasons.get(trace.flush_reason, 0) + 1
    lines.append(
        f"{len(measured)} utterances measured (sentence end detected by: "
        + ", ".join(f"{k} {v}" for k, v in reasons.items())
        + ")"
        + (f"; {skipped} excluded (cut off at the end of the run or failed)." if skipped else ".")
    )
    table = {name: [t.stages()[name] for t in measured] for name in (*STAGES, "end_to_end")}
    median_total = statistics.median(table["end_to_end"])
    lines.append(f"\n  {'stage':<13}{'median':>9}{'min':>8}{'max':>8}{'share':>8}")
    for name, values in table.items():
        median = statistics.median(values)
        share = "" if name == "end_to_end" else f"{median / median_total:7.0%}"
        lines.append(f"  {name:<13}{median:8.0f}ms{min(values):7.0f}{max(values):8.0f}{share:>8}")
    met = sum(value < target_ms for value in table["end_to_end"])
    lines.append(f"\n  Under {target_ms:.0f} ms end-to-end: {met} of {len(measured)} utterances.")
    return "\n".join(lines)
