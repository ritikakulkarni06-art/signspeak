"""Text -> speech with prosody: speaking rate follows signing pace, and words
signed with an intense facial expression are emphasized.

    from src.tts import synthesize
    audio = synthesize("I think this idea is really important.", pace=1.3, emphasis_regions=[(5, 7, 0.9)])
    audio.save("out.wav"); audio.play()
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np

from .engines import Audio, OneCoreEngine, SapiEngine, create_engine
from .prosody import EmphasisRegion, build_ssml, rate_percent
from .signals import emphasis_for_sentence, expression_intensity, pace_from_durations

_engines: dict[str | None, object] = {}


def synthesize(
    text: str,
    pace: float = 1.0,
    emphasis_regions: Iterable = (),
    *,
    voice: str | None = None,
    engine=None,
    trim: bool = True,
) -> Audio:
    """Speak `text` with prosody.

    pace: speaking-rate multiplier from the signing speed (1.0 normal, 1.3 = 30%
        faster, clamped to 0.6..1.6).
    emphasis_regions: words to emphasize, by whitespace-separated word index:
        EmphasisRegion(start, end, strength), (start, end[, strength]) or a
        single index. Strength 0..1 scales the pitch/volume lift.
    voice: part of a voice name ("Mark", "Zira"); default from SIGNSPEAK_TTS_VOICE
        or the system voice. engine: an explicit engine (OneCoreEngine/SapiEngine).
    trim: drop the silence the engine pads around the speech (on by default).
    """
    if not text.strip():
        return Audio(np.zeros(0, np.float32), 22050)
    if engine is None:
        voice = voice or os.environ.get("SIGNSPEAK_TTS_VOICE")
        if voice not in _engines:
            _engines[voice] = create_engine(voice)
        engine = _engines[voice]
    audio = engine.render(build_ssml(text, pace, emphasis_regions))
    return audio.trim_silence() if trim else audio


__all__ = [
    "Audio",
    "EmphasisRegion",
    "OneCoreEngine",
    "SapiEngine",
    "build_ssml",
    "create_engine",
    "emphasis_for_sentence",
    "expression_intensity",
    "pace_from_durations",
    "rate_percent",
    "synthesize",
]
