"""Speech signals -> SSML.

Only markup the local Windows engines were measured to honor is used (probing
SAPI5 David/Zira and OneCore Mark/Zira with the same sentence):

  * <prosody rate="+-N%">                  changes duration on every engine
                                           (-30% -> ~36% longer, +30% -> ~21% shorter)
  * <prosody pitch="+N%" volume="+N%">     on a word raises its pitch 8-14% and
                                           loudness 5-8 dB on every engine
  * <emphasis>                             ignored by SAPI5, unpredictable on
                                           OneCore -> not used
  * named volumes ("loud")                 made words *quieter* -> not used

So emphasis is written as explicit relative pitch, volume and a slight slowdown
on the flagged words, optionally preceded by a short pause.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from xml.sax.saxutils import escape

PACE_RANGE = (0.6, 1.6)  # pace is clamped here: -40% .. +60% rate stays intelligible
EMPHASIS_PITCH = (8, 30)  # % pitch raise, at strength 0 -> 1
EMPHASIS_VOLUME = (10, 40)  # % louder
EMPHASIS_SLOWDOWN = 15  # % slower at full strength
BREAK_FROM_STRENGTH = 0.7  # strong emphasis gets a short lead-in pause
BREAK_MS = 120


@dataclass(frozen=True)
class EmphasisRegion:
    """Words [start, end) of the sentence, by whitespace-separated word index,
    emphasized with `strength` in 0..1 (e.g. facial expression intensity)."""

    start: int
    end: int
    strength: float = 1.0


def rate_percent(pace: float) -> int:
    """Speaking-rate change for a pace signal: 1.0 = normal, 1.3 = 30% faster."""
    low, high = PACE_RANGE
    return round((min(max(pace, low), high) - 1.0) * 100)


def as_region(region) -> EmphasisRegion:
    """Accepts EmphasisRegion, a word index, (start, end) or (start, end, strength)."""
    if isinstance(region, EmphasisRegion):
        return region
    if isinstance(region, int):
        return EmphasisRegion(region, region + 1)
    if isinstance(region, (tuple, list)) and len(region) in (2, 3):
        return EmphasisRegion(*region)
    raise TypeError(f"not an emphasis region: {region!r}")


def word_strengths(word_count: int, regions: Iterable) -> list[float]:
    """Emphasis strength per word; overlapping regions keep the stronger value,
    and indices outside the sentence are ignored."""
    strengths = [0.0] * word_count
    for region in map(as_region, regions):
        strength = min(max(float(region.strength), 0.0), 1.0)
        for i in range(max(region.start, 0), min(region.end, word_count)):
            strengths[i] = max(strengths[i], strength)
    return strengths


def build_ssml(text: str, pace: float = 1.0, emphasis_regions: Iterable = (), language: str = "en-US") -> str:
    words = text.split()
    strengths = word_strengths(len(words), emphasis_regions)
    parts, i = [], 0
    while i < len(words):  # group consecutive words with the same strength
        j = i
        while j < len(words) and strengths[j] == strengths[i]:
            j += 1
        chunk = escape(" ".join(words[i:j]))
        parts.append(_emphasize(chunk, strengths[i]) if strengths[i] > 0 else chunk)
        i = j
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">'
        f'<prosody rate="{rate_percent(pace):+d}%">{" ".join(parts)}</prosody></speak>'
    )


def _emphasize(chunk: str, strength: float) -> str:
    pitch = round(EMPHASIS_PITCH[0] + (EMPHASIS_PITCH[1] - EMPHASIS_PITCH[0]) * strength)
    volume = round(EMPHASIS_VOLUME[0] + (EMPHASIS_VOLUME[1] - EMPHASIS_VOLUME[0]) * strength)
    slower = round(EMPHASIS_SLOWDOWN * strength)
    pause = f'<break time="{BREAK_MS}ms"/>' if strength >= BREAK_FROM_STRENGTH else ""
    return f'{pause}<prosody pitch="+{pitch}%" volume="+{volume}%" rate="-{slower}%">{chunk}</prosody>'
