"""Webcam access."""

from __future__ import annotations

import sys

import cv2


def open_webcam(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Open a webcam and request a resolution (the camera may pick the nearest it supports).

    640x480 by default: the tracker downsizes to 640 wide anyway, so a 1280x720
    capture only adds decoding and a 10 ms resize per frame. Measured +1.5 fps
    on the demo laptop (8.3 -> 9.8), which matters because sign segmentation
    needs about 10 fps.
    """
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
