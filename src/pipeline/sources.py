"""Frame sources for the pipeline: the webcam, or a video replayed as if live."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.capture import open_webcam


class WebcamSource:
    def __init__(self, index: int = 0, size: tuple[int, int] | None = None):
        self._capture = open_webcam(index, *size) if size else open_webcam(index)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        self._capture.release()


class VideoSource:
    """Replays a video file at its real frame rate. Like a live camera, it hands
    over whatever frame is current when read() is called: frames the pipeline is
    too slow for are dropped, and a fast reader is made to wait."""

    def __init__(self, path: str | Path, size: tuple[int, int] | None = None, loop: bool = False):
        self._size = size  # resize frames to this (width, height), e.g. to match a webcam
        self._loop = loop  # start over at the end instead of stopping
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video {path}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._started: float | None = None
        self._next = 0  # index of the next frame in the file
        self.dropped = 0

    def read(self) -> np.ndarray | None:
        frame = self._read_once()
        if frame is None and self._loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._started, self._next = None, 0
            frame = self._read_once()
        return frame

    def _read_once(self) -> np.ndarray | None:
        if self._started is None:
            self._started = time.monotonic()
        due = (time.monotonic() - self._started) * self.fps  # frame a camera would show now
        if due < self._next:
            time.sleep((self._next - due) / self.fps)
        else:
            while self._next < int(due):
                if not self._capture.grab():
                    return None
                self._next += 1
                self.dropped += 1
        ok, frame = self._capture.read()
        self._next += 1
        if not ok:
            return None
        return cv2.resize(frame, self._size, interpolation=cv2.INTER_LINEAR) if self._size else frame

    def close(self) -> None:
        self._capture.release()


def open_source(spec: str, size: tuple[int, int] | None = None, loop: bool = False):
    """A webcam index ("0") or a path to a video file (optionally looping)."""
    return WebcamSource(int(spec), size) if spec.isdigit() else VideoSource(spec, size, loop)
