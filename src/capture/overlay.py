"""Draw tracked landmarks on a BGR frame, for checking tracking quality by eye."""

from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python.vision import (
    FaceLandmarksConnections,
    HandLandmarksConnections,
    PoseLandmarksConnections,
)

from .landmarks import FrameLandmarks


def _pairs(connections) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in connections]


_Face = FaceLandmarksConnections
_HAND = _pairs(HandLandmarksConnections.HAND_CONNECTIONS)
_POSE_UPPER_BODY = [(a, b) for a, b in _pairs(PoseLandmarksConnections.POSE_LANDMARKS) if a <= 24 and b <= 24]
_FACE_OVAL = _pairs(_Face.FACE_LANDMARKS_FACE_OVAL)
_FACE_EYES = _pairs(_Face.FACE_LANDMARKS_LEFT_EYE + _Face.FACE_LANDMARKS_RIGHT_EYE)
_FACE_BROWS = _pairs(_Face.FACE_LANDMARKS_LEFT_EYEBROW + _Face.FACE_LANDMARKS_RIGHT_EYEBROW)
_FACE_LIPS = _pairs(_Face.FACE_LANDMARKS_LIPS)

# BGR
LEFT_HAND_COLOR = (80, 220, 80)
RIGHT_HAND_COLOR = (255, 160, 40)
POSE_COLOR = (210, 210, 210)
FACE_OVAL_COLOR = (130, 130, 130)
EYE_COLOR = (230, 230, 0)
BROW_COLOR = (0, 215, 255)
LIP_COLOR = (200, 80, 255)

_MIN_POSE_VISIBILITY = 0.5


def draw_landmarks(image: np.ndarray, frame: FrameLandmarks, mirror: bool = False) -> None:
    """Draw pose, hands and the face regions used for non-manual markers onto
    image, in place. Pass mirror=True when image was flipped horizontally for
    display; the landmarks are flipped to match."""
    height, width = image.shape[:2]

    def to_px(points: np.ndarray) -> list[list[int]]:
        x = 1.0 - points[:, 0] if mirror else points[:, 0]
        return np.stack([x * width, points[:, 1] * height], axis=1).round().astype(int).tolist()

    if frame.face is not None:
        px = to_px(frame.face)
        _lines(image, px, _FACE_OVAL, FACE_OVAL_COLOR, 1)
        _lines(image, px, _FACE_EYES, EYE_COLOR, 1)
        _lines(image, px, _FACE_BROWS, BROW_COLOR, 2)
        _lines(image, px, _FACE_LIPS, LIP_COLOR, 2)

    if frame.pose is not None:
        px = to_px(frame.pose)
        visible = frame.pose[:, 3] >= _MIN_POSE_VISIBILITY
        _lines(image, px, [(a, b) for a, b in _POSE_UPPER_BODY if visible[a] and visible[b]], POSE_COLOR, 2)
        for i in np.flatnonzero(visible[:25]):
            cv2.circle(image, tuple(px[i]), 4, POSE_COLOR, -1, cv2.LINE_AA)

    for hand, color, label in (
        (frame.left_hand, LEFT_HAND_COLOR, "L"),
        (frame.right_hand, RIGHT_HAND_COLOR, "R"),
    ):
        if hand is None:
            continue
        px = to_px(hand)
        _lines(image, px, _HAND, color, 2)
        for point in px:
            cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
        wrist_x, wrist_y = px[0]
        cv2.putText(image, label, (wrist_x - 8, wrist_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def _lines(image: np.ndarray, px: list[list[int]], pairs, color, thickness: int) -> None:
    for a, b in pairs:
        cv2.line(image, tuple(px[a]), tuple(px[b]), color, thickness, cv2.LINE_AA)
