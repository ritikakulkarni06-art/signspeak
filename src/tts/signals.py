"""Signals from the capture stage -> prosody parameters for synthesize().

  * Signing speed -> pace: a signer producing signs faster than usual gets
    faster speech.
  * Facial expression intensity (the non-manual markers: brow raise, eye and
    mouth opening) -> emphasis on the words those signs became.

The constants are starting points, not calibrated values: tune TYPICAL_SIGN_S
and the neutral face per signer (e.g. from a few seconds of neutral signing).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from src.capture.features import LAYOUT, NMM_NAMES

from .prosody import EmphasisRegion

# Median length of a trimmed ASL Citizen training clip (data/processed, 15 fps;
# quartiles 1.00-1.53 s): an isolated sign signed at a comfortable pace.
TYPICAL_SIGN_S = 1.27
PACE_LIMITS = (0.6, 1.6)

# Webcam medians of a neutral face (measured on this setup) and how far each
# marker has to move, in head-frame units, to count as full intensity.
NEUTRAL_NMM = np.array([0.25, 0.24, 0.089, 0.082, 0.02, 0.53], np.float32)
NMM_FULL_SCALE = np.array([0.08, 0.08, 0.04, 0.04, 0.10, 0.10], np.float32)
assert len(NEUTRAL_NMM) == len(NMM_FULL_SCALE) == len(NMM_NAMES)


def pace_from_durations(sign_durations_s: Sequence[float], typical_s: float = TYPICAL_SIGN_S) -> float:
    """Pace from how long recent signs took (e.g. trimmed segment lengths):
    1.0 at the typical duration, >1 when signing faster."""
    durations = [d for d in sign_durations_s if d > 0]
    if not durations:
        return 1.0
    low, high = PACE_LIMITS
    return float(np.clip(typical_s / float(np.median(durations)), low, high))


def expression_intensity(sequence: np.ndarray, neutral: np.ndarray = NEUTRAL_NMM) -> float:
    """0..1 intensity of the face during a sign: how far the non-manual markers
    move from the neutral face, taken near the peak of the sign (90th percentile
    over frames) so one noisy frame doesn't count. 0 if the face wasn't tracked."""
    face_seen = sequence[:, LAYOUT["face.present"]][:, 0] > 0
    if not face_seen.any():
        return 0.0
    markers = sequence[face_seen][:, LAYOUT["face.nmm"]]
    deviation = np.abs(markers - neutral) / NMM_FULL_SCALE
    return float(np.clip(np.percentile(deviation.max(axis=1), 90), 0.0, 1.0))


def emphasis_for_sentence(
    sentence: str, gloss_intensity: Sequence[tuple[str, float]], threshold: float = 0.4
) -> list[EmphasisRegion]:
    """Emphasis regions for the words that signs with an intense expression
    became. Translation reorders words, so each sign is found in the sentence by
    word (a gloss of several words matches them consecutively). Signs the
    translation paraphrased away are skipped."""
    words = [re.sub(r"[^\w']", "", word).lower() for word in sentence.split()]
    regions = []
    for gloss, intensity in gloss_intensity:
        if intensity < threshold:
            continue
        target = gloss.lower().split()
        for start in range(len(words) - len(target) + 1):
            if all(words[start + k].startswith(part) for k, part in enumerate(target)):
                regions.append(EmphasisRegion(start, start + len(target), intensity))
                break
    return regions
