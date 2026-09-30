"""Sequence-level steps shared by dataset preparation and live recognition, so
training clips and live segments go through exactly the same preparation."""

from __future__ import annotations

import numpy as np

from .features import LAYOUT, mirror_features

HAND_MARGIN_FRAMES = 2  # frames kept on each side of the first/last frame with a hand
MIN_FRAMES = 3


def trim_to_hands(sequence: np.ndarray, margin: int = HAND_MARGIN_FRAMES) -> np.ndarray | None:
    """Drop leading/trailing frames without any hand. None if no hand is ever
    seen or fewer than MIN_FRAMES remain."""
    has_hand = (sequence[:, LAYOUT["left_hand.present"]] > 0) | (sequence[:, LAYOUT["right_hand.present"]] > 0)
    with_hands = np.flatnonzero(has_hand.ravel())
    if len(with_hands) == 0:
        return None
    start, end = max(0, with_hands[0] - margin), min(len(sequence), with_hands[-1] + 1 + margin)
    trimmed = sequence[start:end]
    return trimmed if len(trimmed) >= MIN_FRAMES else None


def center_crop(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """Keep the middle max_frames frames, where the sign is."""
    if len(sequence) <= max_frames:
        return sequence
    start = (len(sequence) - max_frames) // 2
    return sequence[start : start + max_frames]


def left_hand_dominant(sequence: np.ndarray) -> bool:
    """True if the signer's left hand does more of the moving (or is the only
    hand seen). MediaPipe labels hands by the signer's side, so left-handed
    signers, and mirrored recordings, put the dominant hand in the left slot."""
    motion, seen = [], []
    for hand in ("left_hand", "right_hand"):
        present = sequence[:, LAYOUT[f"{hand}.present"]][:, 0] > 0
        steps = np.linalg.norm(np.diff(sequence[:, LAYOUT[f"{hand}.location"]][:, :2], axis=0), axis=1)
        motion.append(steps[present[1:] & present[:-1]].sum())
        seen.append(present.sum())
    if motion[0] == motion[1]:
        return seen[0] > seen[1]
    return motion[0] > motion[1]


def to_right_dominant(sequence: np.ndarray) -> tuple[np.ndarray, bool]:
    """Mirror the sequence if the left hand is dominant, so the dominant hand is
    always in the right_hand slot. Returns (sequence, whether it was mirrored)."""
    if left_hand_dominant(sequence):
        return mirror_features(sequence), True
    return sequence, False
