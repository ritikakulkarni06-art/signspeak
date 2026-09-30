"""MediaPipe Holistic tracker: BGR webcam frame in, FrameLandmarks out."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .landmarks import FrameLandmarks

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)


class HolisticTracker:
    """Runs MediaPipe Holistic in VIDEO mode: synchronous, one result per frame,
    with tracking carried between frames. Use as a context manager or call close().
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, min_confidence: float = 0.5):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Holistic model not found at {model_path}. Download it from {MODEL_URL}")
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=min_confidence,
            min_face_landmarks_confidence=min_confidence,
            min_pose_detection_confidence=min_confidence,
            min_pose_landmarks_confidence=min_confidence,
            min_hand_landmarks_confidence=min_confidence,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> FrameLandmarks:
        """Track one frame. Pass the raw (unmirrored) camera frame so left/right
        hand labels match the signer's actual hands."""
        if timestamp_ms is None:
            timestamp_ms = int(time.perf_counter() * 1000)  # high-resolution; shares the clock with pipeline timing
        # VIDEO mode rejects timestamps that don't strictly increase.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        height, width = frame_bgr.shape[:2]
        return FrameLandmarks(
            image_size=(width, height),
            pose=_to_array(result.pose_landmarks, with_visibility=True),
            left_hand=_to_array(result.left_hand_landmarks),
            right_hand=_to_array(result.right_hand_landmarks),
            face=_to_array(result.face_landmarks),
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HolisticTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _to_array(landmarks, with_visibility: bool = False) -> np.ndarray | None:
    if not landmarks:
        return None
    if with_visibility:
        rows = [(p.x, p.y, p.z or 0.0, 1.0 if p.visibility is None else p.visibility) for p in landmarks]
    else:
        rows = [(p.x, p.y, p.z or 0.0) for p in landmarks]
    return np.array(rows, np.float32)
