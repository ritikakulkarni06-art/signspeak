"""Frame sources for the pipeline: the webcam, or a video replayed as if live."""

from __future__ import annotations

import threading
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


class SwitchableSource:
    """A frame source the demo can switch while the pipeline runs: the live
    source (webcam, or a replay video) or a recorded demo clip.

    Only where frames come from changes. A clip is read through VideoSource, so
    it is paced in real time and frames the pipeline is too slow for are dropped,
    exactly as with a camera; every frame then goes through the same tracking,
    recognition, translation and speech as live input. Around the clip, its first
    frame is held for a moment (so the cut from the previous picture doesn't read
    as hand motion) and its last frame afterwards (a still signer, so the sign and
    the sentence end the way they do live). Between clips the last frame stays
    up, like a paused video.

    Every frame is fitted to one size (letterboxed if its shape differs): MediaPipe's
    tracker fails outright when the frame size changes mid-stream.

    use_live() and play_clip() may be called from any thread; the switch happens
    on the next read(), on the pipeline's thread. on_change(info) reports every
    switch, from the pipeline's thread.
    """

    LEAD_IN_S = 0.5
    HOLD_S = 2.0
    STILL_FPS = 30.0  # a held frame is handed out at a camera's pace, not as fast as asked

    def __init__(self, open_live, live_label: str = "Webcam", on_change=None, start_live: bool = True,
                 idle_frame: np.ndarray | None = None, size: tuple[int, int] = (640, 480)):  # fmt: skip
        self._open_live = open_live  # () -> a source with read() and close()
        self.live_label = live_label
        self._on_change = on_change or (lambda info: None)
        self._lock = threading.Lock()
        self._request: tuple | None = ("live", None, None) if start_live else None
        self._live = None
        self._clip: VideoSource | None = None
        self._phase = "still"  # live | lead_in | clip | hold | still
        self._phase_until = 0.0
        self.size = size  # (width, height) of every frame handed out; the webcam's default capture size
        self._still = fit_frame(idle_frame, size) if idle_frame is not None else np.zeros((size[1], size[0], 3), np.uint8)
        self._last_still = 0.0
        self.info = {"mode": "live" if start_live else "demo", "clip": None, "playing": False}

    # ------------------------------------------------------------ any thread

    def use_live(self) -> None:
        with self._lock:
            self._request = ("live", None, None)

    def play_clip(self, path: str | Path, clip: dict | None = None) -> None:
        """Play this recorded clip next. `clip` is what the UI shows about it."""
        with self._lock:
            self._request = ("clip", Path(path), clip or {"file": Path(path).name})

    # ------------------------------------------------------------ pipeline thread

    def read(self) -> np.ndarray | None:
        self._apply_request()
        while True:
            if self._phase == "live":
                frame = self._live.read()
                if frame is not None:
                    self._still = fit_frame(frame, self.size)
                    return self._still
                self._close_live()
                self._phase = "still"
                self._changed(mode="demo", clip=None, playing=False,
                              notice=f"The {self.live_label.lower()} stopped. Switched to Demo Clip mode.")  # fmt: skip
                continue
            if self._phase == "clip":
                frame = self._clip.read()
                if frame is not None:
                    self._still = fit_frame(frame, self.size)
                    return self._still
                self._clip.close()
                self._clip = None
                self._phase, self._phase_until = "hold", time.monotonic() + self.HOLD_S
                continue
            # a held frame: lead-in, hold after a clip, or still between clips
            now = time.monotonic()
            if self._phase == "lead_in" and now >= self._phase_until:
                self._phase = "clip"
                continue
            if self._phase == "hold" and now >= self._phase_until:
                self._phase = "still"
                self._changed(playing=False)
            time.sleep(max(0.0, self._last_still + 1.0 / self.STILL_FPS - now))
            self._last_still = time.monotonic()
            return self._still

    def close(self) -> None:
        self._close_live()
        if self._clip is not None:
            self._clip.close()
            self._clip = None

    def _apply_request(self) -> None:
        with self._lock:
            request, self._request = self._request, None
        if request is None:
            return
        kind, path, clip = request
        if kind == "live":
            if self._phase == "live":
                return
            try:
                live = self._open_live()
            except Exception as error:  # noqa: BLE001 - the demo carries on with clips
                self._changed(mode="demo", notice=f"Couldn't open the {self.live_label.lower()} ({error}). Use Demo Clip mode.")
                return
            self._stop_clip()
            self._live, self._phase = live, "live"
            self._changed(mode="live", clip=None, playing=False)
            return
        try:
            clip_source = VideoSource(path)
        except Exception as error:  # noqa: BLE001
            self._changed(notice=f"Couldn't open the demo clip: {error}")
            return
        self._close_live()
        self._stop_clip()
        first = first_frame(path)  # shown during the lead-in
        if first is not None:
            self._still = fit_frame(first, self.size)
        self._clip = clip_source
        self._phase, self._phase_until = "lead_in", time.monotonic() + self.LEAD_IN_S
        self._changed(mode="demo", clip=clip, playing=True)

    def _stop_clip(self) -> None:
        if self._clip is not None:
            self._clip.close()
            self._clip = None

    def _close_live(self) -> None:
        if self._live is not None:
            self._live.close()
            self._live = None

    def _changed(self, notice: str | None = None, **info) -> None:
        self.info = {**self.info, **info}
        self._on_change({**self.info, "notice": notice})


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The frame at exactly `size` (width, height): scaled, and letterboxed with
    black if its aspect ratio differs. Unchanged if it already fits."""
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    scale = min(width / frame.shape[1], height / frame.shape[0])
    w, h = round(frame.shape[1] * scale), round(frame.shape[0] * scale)
    canvas = np.zeros((height, width, 3), np.uint8)
    x, y = (width - w) // 2, (height - h) // 2
    canvas[y : y + h, x : x + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def first_frame(path: str | Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None
