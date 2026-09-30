"""Webcam access."""

from __future__ import annotations

import sys

import cv2


def open_webcam(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Open a webcam and request a resolution (the camera may pick the nearest it supports)."""
    # DirectShow opens much faster than the default Media Foundation backend on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Keep only the newest frame: a deeper driver buffer hands the tracker stale
    # frames when it runs slower than the camera, adding hidden latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
