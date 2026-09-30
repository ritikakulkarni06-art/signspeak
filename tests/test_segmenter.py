"""Segmenter tests on synthetic feature streams (no camera or model needed).

Run from the project root: python -m unittest discover -s tests
"""

import unittest

import numpy as np

from src.capture.features import FEATURE_DIM, LAYOUT
from src.capture.segmenter import SegmenterConfig, SignSegmenter, segment_signs

FRAME_MS = 66  # ~15 fps, what the Holistic tracker manages on CPU


def _frame(x: float, y: float = 0.5, hand: bool = True, pose: bool = True) -> np.ndarray:
    features = np.zeros(FEATURE_DIM, np.float32)
    features[LAYOUT["pose.present"]] = float(pose)
    if hand:
        features[LAYOUT["right_hand.present"]] = 1.0
        features[LAYOUT["right_hand.location"]] = (x, y, 0.25)
    return features


def _stream(motion_plan):
    """motion_plan: list of (seconds, speed in shoulder widths/s). Yields (features, timestamp_ms)."""
    timestamp, x = 0, 0.0
    for seconds, speed in motion_plan:
        for _ in range(round(seconds * 1000 / FRAME_MS)):
            x += speed * FRAME_MS / 1000
            yield _frame(x), timestamp
            timestamp += FRAME_MS


class SegmenterTests(unittest.TestCase):
    def test_two_signs_separated_by_pauses(self):
        plan = [(1.0, 0.0), (0.6, 2.0), (1.0, 0.0), (0.8, 2.0), (1.0, 0.0)]
        segments = list(segment_signs(_stream(plan)))
        self.assertEqual([s.end_reason for s in segments], ["pause", "pause"])
        for segment, active_s in zip(segments, (0.6, 0.8)):
            # Active motion plus a little preroll and the closing pause.
            self.assertGreater(segment.duration_ms, active_s * 1000)
            self.assertLess(segment.duration_ms, active_s * 1000 + 600)
            self.assertEqual(segment.features.shape, (segment.num_frames, FEATURE_DIM))

    def test_still_hands_and_jitter_produce_nothing(self):
        rng = np.random.default_rng(0)
        frames = [(_frame(rng.normal(0, 0.005), 0.5 + rng.normal(0, 0.005)), i * FRAME_MS) for i in range(100)]
        self.assertEqual(list(segment_signs(frames)), [])

    def test_twitch_shorter_than_min_segment_is_dropped(self):
        self.assertEqual(list(segment_signs(_stream([(1.0, 0.0), (0.07, 8.0), (1.0, 0.0)]))), [])

    def test_lowering_hands_ends_a_sign(self):
        frames = list(_stream([(0.5, 0.0), (0.6, 2.0)]))
        last_ts = frames[-1][1]
        frames += [(_frame(0, hand=False), last_ts + FRAME_MS * (i + 1)) for i in range(10)]
        segments = list(segment_signs(frames))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].end_reason, "pause")

    def test_pose_dropout_mid_sign_does_not_split_it(self):
        frames = list(_stream([(0.5, 0.0), (1.2, 2.0), (1.0, 0.0)]))
        for i in range(12, 15):  # three frames without pose in the middle of the motion
            features, ts = frames[i]
            features = features.copy()
            features[LAYOUT["pose.present"]] = 0.0
            frames[i] = (features, ts)
        self.assertEqual(len(list(segment_signs(frames))), 1)

    def test_continuous_motion_is_cut_at_max_length(self):
        config = SegmenterConfig(max_segment_ms=1000)
        segments = list(segment_signs(_stream([(3.5, 2.0)]), config))
        self.assertGreaterEqual(len(segments), 3)
        self.assertEqual(segments[0].end_reason, "max_length")

    def test_open_segment_is_flushed_at_end_of_stream(self):
        segments = list(segment_signs(_stream([(0.5, 0.0), (0.8, 2.0)])))
        self.assertEqual([s.end_reason for s in segments], ["end_of_stream"])

    def test_gesture_end_is_the_last_moving_frame_not_the_close(self):
        frames = list(_stream([(0.5, 0.0), (0.6, 2.0), (1.0, 0.0)]))
        last_moving = frames[round(1100 / FRAME_MS) - 1][1]
        segment = next(segment_signs(frames))
        self.assertAlmostEqual(segment.gesture_end_ms, last_moving, delta=FRAME_MS)
        self.assertGreater(segment.timestamps_ms[-1] - segment.gesture_end_ms, 250)  # closed after the pause

    def test_window_keeps_last_n_frames(self):
        segmenter = SignSegmenter(SegmenterConfig(window_size=5))
        for i in range(12):
            segmenter.push(_frame(float(i)), i * FRAME_MS)
        window = segmenter.window
        self.assertEqual(window.shape, (5, FEATURE_DIM))
        self.assertEqual(window[0][LAYOUT["right_hand.location"]][0], 7.0)

    def test_segments_do_not_share_frames(self):
        plan = [(0.5, 0.0), (0.6, 2.0), (0.5, 0.0), (0.6, -2.0), (1.0, 0.0)]
        segments = list(segment_signs(_stream(plan)))
        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0].timestamps_ms[-1], segments[1].timestamps_ms[0])


if __name__ == "__main__":
    unittest.main()
