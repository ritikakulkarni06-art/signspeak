"""Sliding-window buffer that cuts the per-frame feature stream into sign segments.

PLACEHOLDER SEGMENTATION: sign boundaries are found with a motion heuristic, i.e.
a sign ends when the hands go still (or drop out of view) for a short pause.
Real signing often flows from one sign into the next without a clear pause, and
holds are sometimes part of the sign itself, so this should later be replaced
by a learned segmentation model (e.g. a per-frame boundary classifier over the
same feature stream). SignSegmenter.push() is the seam to swap it out behind.

Motion is measured per hand as:
    wrist speed        in shoulder widths / second (body frame, from the features)
  + shape_weight * mean joint speed in hand sizes / second (fingers moving,
                       e.g. fingerspelling with a mostly still wrist)
and the fastest hand counts. Everything is per second, using frame timestamps,
so thresholds don't depend on the frame rate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .features import HAND_POINTS, LAYOUT

_HANDS = ("left_hand", "right_hand")


@dataclass(frozen=True)
class SegmenterConfig:
    window_size: int = 30  # frames kept in the rolling window (~1 s at 30 fps, ~2 s at 15 fps)
    # Thresholds are provisional (not yet calibrated on real signing); tune them
    # with the motion bar in demo/segment_signs.py.
    start_speed: float = 0.8  # motion above this starts a sign
    stop_speed: float = 0.4  # motion below this counts as a pause (lower than start: hysteresis)
    # Stillness needed to end a sign is pause_ms plus the 1-2 frames the smoothed
    # motion takes to fall below stop_speed.
    pause_ms: int = 200
    min_segment_ms: int = 300  # movement shorter than this (start threshold to pause) is dropped as a twitch
    max_segment_ms: int = 5000  # longer ones are cut, so continuous motion still produces output
    preroll_ms: int = 200  # frames from before the start threshold kept, so the onset isn't clipped
    shape_weight: float = 0.25  # hand sizes -> roughly shoulder widths
    smoothing: float = 0.6  # weight of the newest motion reading in the running average


@dataclass
class SignSegment:
    features: np.ndarray  # (num_frames, FEATURE_DIM)
    timestamps_ms: np.ndarray  # (num_frames,)
    end_reason: str  # "pause", "max_length" or "end_of_stream"
    # Last frame whose raw hand motion was above stop_speed: when the gesture
    # physically ended. The segment itself closes one pause later.
    gesture_end_ms: int = -1

    @property
    def num_frames(self) -> int:
        return len(self.features)

    @property
    def duration_ms(self) -> int:
        return int(self.timestamps_ms[-1] - self.timestamps_ms[0])


class SignSegmenter:
    """Feed frames with push(); it returns a SignSegment when one closes, else None."""

    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()
        self._window: deque[tuple[int, np.ndarray]] = deque(maxlen=self.config.window_size)
        self._segment: list[tuple[int, np.ndarray]] | None = None
        self._motion = 0.0
        self._still_since_ms: int | None = None
        self._active_start_ms = 0  # when motion crossed start_speed for the open segment
        self._last_motion_ms = -1  # last frame with raw motion above stop_speed
        self._frame_interval_ms = 0  # time between the last two frames
        self._last_segment_end_ms = -1

    @property
    def window(self) -> np.ndarray:
        """The last window_size feature vectors, oldest first: (<= window_size, FEATURE_DIM)."""
        return np.stack([features for _, features in self._window]) if self._window else np.empty((0, 0))

    @property
    def motion(self) -> float:
        """Smoothed hand motion, in the same units as the start/stop thresholds."""
        return self._motion

    @property
    def in_sign(self) -> bool:
        return self._segment is not None

    def push(self, features: np.ndarray, timestamp_ms: int) -> SignSegment | None:
        cfg = self.config
        previous = self._window[-1] if self._window else None
        if previous is not None:
            self._frame_interval_ms = timestamp_ms - previous[0]
        reading = _hand_motion(previous, (timestamp_ms, features), cfg.shape_weight)
        if reading is not None:  # None: can't measure this frame, keep the last value
            self._motion = cfg.smoothing * reading + (1 - cfg.smoothing) * self._motion
            if reading >= cfg.stop_speed:
                self._last_motion_ms = timestamp_ms
        self._window.append((timestamp_ms, features))

        if self._segment is None:
            if self._motion >= cfg.start_speed:
                earliest = timestamp_ms - cfg.preroll_ms
                self._segment = [
                    frame for frame in self._window if frame[0] >= earliest and frame[0] > self._last_segment_end_ms
                ]
                self._active_start_ms = timestamp_ms
                self._still_since_ms = None
            return None

        self._segment.append((timestamp_ms, features))
        if self._motion < cfg.stop_speed:
            if self._still_since_ms is None:
                self._still_since_ms = timestamp_ms
            if timestamp_ms - self._still_since_ms >= cfg.pause_ms:
                return self._close("pause")
        else:
            self._still_since_ms = None
        if timestamp_ms - self._segment[0][0] >= cfg.max_segment_ms:
            return self._close("max_length")
        return None

    def flush(self) -> SignSegment | None:
        """Close any open segment, e.g. when the stream ends."""
        return self._close("end_of_stream") if self._segment else None

    def _close(self, reason: str) -> SignSegment | None:
        frames, self._segment = self._segment, None
        # Judge length by the movement itself, not the preroll and closing-pause frames around it.
        active_end_ms = self._still_since_ms if self._still_since_ms is not None else frames[-1][0]
        gesture_end_ms = self._last_motion_ms if self._last_motion_ms >= frames[0][0] else active_end_ms
        self._still_since_ms = None
        self._last_segment_end_ms = frames[-1][0]
        # Each frame stands for the interval until the next one. Without counting it,
        # a real sign at a low frame rate (7.5 fps: 133 ms per frame) measured one
        # frame short and was discarded as a twitch: 8 of 26 test signs were lost.
        if active_end_ms - self._active_start_ms + self._frame_interval_ms < self.config.min_segment_ms:
            return None
        return SignSegment(
            features=np.stack([features for _, features in frames]),
            timestamps_ms=np.array([timestamp for timestamp, _ in frames], np.int64),
            end_reason=reason,
            gesture_end_ms=int(gesture_end_ms),
        )


def segment_signs(
    frames: Iterable[tuple[np.ndarray, int]], config: SegmenterConfig | None = None
) -> Iterator[SignSegment]:
    """Yield a SignSegment each time a sign completes in a stream of (features, timestamp_ms)."""
    segmenter = SignSegmenter(config)
    for features, timestamp_ms in frames:
        segment = segmenter.push(features, timestamp_ms)
        if segment is not None:
            yield segment
    segment = segmenter.flush()
    if segment is not None:
        yield segment


def _hand_motion(previous: tuple[int, np.ndarray] | None, current: tuple[int, np.ndarray], shape_weight: float):
    """Motion of the fastest hand between two frames: 0.0 if no hand is up (hands
    lowered = resting), None if it can't be measured this frame (no previous
    frame, a hand just appeared, or the body frame is missing)."""
    timestamp_ms, features = current
    raised = [hand for hand in _HANDS if features[LAYOUT[f"{hand}.present"]][0] > 0]
    if not raised:
        return 0.0
    if previous is None or timestamp_ms <= previous[0]:
        return None
    prev_features = previous[1]
    if not (features[LAYOUT["pose.present"]][0] > 0 and prev_features[LAYOUT["pose.present"]][0] > 0):
        return None

    dt = (timestamp_ms - previous[0]) / 1000
    speeds = []
    for hand in raised:
        if prev_features[LAYOUT[f"{hand}.present"]][0] == 0:
            continue
        location, shape = LAYOUT[f"{hand}.location"], LAYOUT[f"{hand}.shape"]
        wrist_speed = np.linalg.norm(features[location][:2] - prev_features[location][:2]) / dt
        joint_moves = (features[shape] - prev_features[shape]).reshape(HAND_POINTS, 3)
        shape_speed = np.linalg.norm(joint_moves, axis=1).mean() / dt
        speeds.append(wrist_speed + shape_weight * shape_speed)
    return max(speeds) if speeds else None
