"""Landmark container passed from the tracker to feature extraction and drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameLandmarks:
    """One frame of Holistic output, as arrays in MediaPipe's normalized image
    coordinates: x and y in [0, 1] of image width and height, z on x's scale.

    A part is None when it wasn't tracked this frame. Left/right are the
    signer's sides, not the viewer's.
    """

    image_size: tuple[int, int]  # (width, height) in pixels
    pose: np.ndarray | None = None  # (33, 4): x, y, z, visibility
    left_hand: np.ndarray | None = None  # (21, 3)
    right_hand: np.ndarray | None = None  # (21, 3)
    face: np.ndarray | None = None  # (478, 3): 468 mesh points + 10 iris points
    timestamp_ms: int = 0
