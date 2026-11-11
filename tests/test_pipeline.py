"""Pipeline tests with stub stages (no camera, model, network, or speakers).

Run from the project root: python -m unittest discover -s tests
"""

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.pipeline import STAGES, BufferConfig, GlossBuffer, SignEvent, UtteranceTrace
from src.pipeline.live import SpeechStage, sign_duration_s
from src.pipeline.sources import SwitchableSource, VideoSource
from src.pipeline.trace import now_ms
from src.tts import Audio


def _sign(label="hello", confidence=0.9, t=0.0, intensity=0.0):
    return SignEvent(label, confidence, gesture_end_ms=t, detected_ms=t + 300, classified_ms=t + 305,
                     duration_s=1.2, intensity=intensity)  # fmt: skip


class GlossBufferTests(unittest.TestCase):
    def setUp(self):
        self.buffer = GlossBuffer(BufferConfig(gap_ms=700, hands_down_ms=350, max_signs=3, min_confidence=0.3))

    def test_pause_ends_the_utterance(self):
        self.buffer.add(_sign("hello", t=0))
        self.assertIsNone(self.buffer.poll(900, hands_visible=True, signing=False))
        signs, reason = self.buffer.poll(1010, hands_visible=True, signing=False)
        self.assertEqual(([s.label for s in signs], reason), (["hello"], "gap"))
        self.assertIsNone(self.buffer.poll(2000, hands_visible=True, signing=False))

    def test_no_flush_while_the_next_sign_is_under_way(self):
        self.buffer.add(_sign(t=0))
        self.assertIsNone(self.buffer.poll(5000, hands_visible=True, signing=True))

    def test_lowered_hands_end_the_utterance_early(self):
        self.buffer.add(_sign(t=0))
        self.assertIsNone(self.buffer.poll(310, hands_visible=False, signing=False))
        self.assertEqual(self.buffer.poll(670, hands_visible=False, signing=False)[1], "hands_down")

    def test_max_signs_and_unsure_signs(self):
        self.assertFalse(self.buffer.add(_sign("what", confidence=0.1)))
        for label in ("a", "b", "c"):
            self.buffer.add(_sign(label))
        self.assertEqual(self.buffer.poll(0, hands_visible=True, signing=True)[1], "max_signs")


class TraceTests(unittest.TestCase):
    def test_stages_add_up_to_end_to_end(self):
        trace = UtteranceTrace(1, [_sign(t=0), _sign(t=1000)], flushed_ms=2000, flush_reason="gap",
                               translate_start_ms=2001, translate_end_ms=2002, tts_end_ms=2400, audio_start_ms=2450)  # fmt: skip
        stages = trace.stages()
        self.assertAlmostEqual(sum(stages[name] for name in STAGES), stages["end_to_end"])
        self.assertEqual(stages["end_to_end"], 1450)  # measured from the *last* sign's gesture end


class SignDurationTests(unittest.TestCase):
    def test_duration_follows_the_real_frame_rate_not_the_training_rate(self):
        # the same 1.25 s sign seen by a 15 fps replay and by an 8 fps webcam
        self.assertAlmostEqual(sign_duration_s(np.arange(20) * 1000 / 15, 19), 19 / 15)
        self.assertAlmostEqual(sign_duration_s(np.arange(11) * 125.0, 10), 1.25)
        self.assertEqual(sign_duration_s(np.array([0.0]), 1), 0.0)


class StubTranslator:
    def translate(self, gloss):
        time.sleep(0.01)
        return "I think this idea is really important."


class StubEngine:
    def render(self, ssml):
        self.ssml = ssml
        time.sleep(0.02)
        tone = 0.3 * np.sin(np.arange(8000) * 2 * np.pi * 200 / 16000)  # audible, so trimming keeps it
        return Audio(tone.astype(np.float32), 16000, ssml)


class StubPlayback:
    def __init__(self, start_ms):
        self.start_ms = start_ms

    def wait(self, timeout=None):
        pass


class StubPlayer:
    def play(self, audio):
        return StubPlayback(now_ms() + 40)


class SpeechStageTests(unittest.TestCase):
    def test_stamps_each_boundary_in_order(self):
        signs = [_sign("idea", t=now_ms() - 1500), _sign("important", t=now_ms() - 1200, intensity=0.9)]
        trace = UtteranceTrace(1, signs, flushed_ms=now_ms(), flush_reason="gap")
        engine = StubEngine()
        trace, playback = SpeechStage(StubTranslator(), engine, StubPlayer()).speak(trace)
        order = [trace.flushed_ms, trace.translate_start_ms, trace.translate_end_ms, trace.tts_end_ms, trace.audio_start_ms]
        self.assertEqual(order, sorted(order))
        self.assertGreaterEqual(trace.stages()["translate"], 9)
        self.assertGreaterEqual(trace.stages()["tts"], 19)
        self.assertEqual(trace.emphasized, ["important."])  # the intense sign's word
        self.assertIn('pitch="+', engine.ssml)
        self.assertTrue(trace.audio_played)


class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "clip.avi"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
        for i in range(30):  # one second
            writer.write(np.full((48, 64, 3), i * 8, np.uint8))
        writer.release()

    def tearDown(self):
        self.tmp.cleanup()

    def test_fast_reader_is_held_to_real_time(self):
        source = VideoSource(self.path)
        started, frames = time.monotonic(), 0
        while source.read() is not None:
            frames += 1
        source.close()
        self.assertGreater(time.monotonic() - started, 0.85)
        self.assertEqual(frames + source.dropped, 30)

    def test_slow_reader_gets_frames_dropped_like_a_live_camera(self):
        source = VideoSource(self.path)
        frames = 0
        while source.read() is not None:
            frames += 1
            time.sleep(0.1)
        source.close()
        self.assertLess(frames, 15)
        self.assertGreater(source.dropped, 15)


class FakeCamera:
    def __init__(self, frames=None, value=200):
        self.frames, self.value, self.closed = frames, value, False

    def read(self):
        if self.frames is not None:
            if self.frames == 0:
                return None
            self.frames -= 1
        return np.full((48, 64, 3), self.value, np.uint8)

    def close(self):
        self.closed = True


class SwitchableSourceTests(unittest.TestCase):
    """Demo Clip mode: the clip replaces the camera as the frame source, and
    every one of its frames is still handed to the pipeline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self.tmp.name) / "clip.avi"
        writer = cv2.VideoWriter(str(self.clip), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
        for i in range(15):  # half a second, each frame a different shade
            writer.write(np.full((48, 64, 3), 10 + i * 8, np.uint8))
        writer.release()
        self.changes = []
        self.camera = FakeCamera()
        SwitchableSource.LEAD_IN_S, SwitchableSource.HOLD_S = 0.1, 0.1

    def tearDown(self):
        SwitchableSource.LEAD_IN_S, SwitchableSource.HOLD_S = 0.5, 2.0
        self.tmp.cleanup()

    def _source(self, **kwargs):
        return SwitchableSource(lambda: self.camera, on_change=self.changes.append, **kwargs)

    def test_clip_plays_in_real_time_then_holds_its_last_frame(self):
        source = self._source()
        self.assertEqual(source.read().mean(), 200)  # the camera
        source.play_clip(self.clip, {"file": "clip.avi", "label": "please"})
        started, shades = time.monotonic(), []
        while time.monotonic() - started < 1.2:
            shades.append(round(source.read().mean()))
        self.assertTrue(self.camera.closed)  # the camera is released while the clip plays
        clip_shades = [s for s in shades if s != shades[0]]
        self.assertEqual(clip_shades, sorted(clip_shades))  # in order, never repeated backwards
        # held on the last frame delivered afterwards (a late reader may miss the very last one, as live)
        self.assertEqual(shades[-1], clip_shades[-1])
        self.assertGreaterEqual(shades[-1], 10 + 12 * 8)
        self.assertEqual([(c["mode"], c["playing"]) for c in self.changes], [("live", False), ("demo", True), ("demo", False)])
        self.assertEqual(self.changes[1]["clip"]["label"], "please")
        source.close()

    def test_back_to_live(self):
        source = self._source()
        source.read()
        source.play_clip(self.clip)
        source.read()
        self.camera = FakeCamera(value=99)
        source.use_live()
        self.assertEqual(source.read().mean(), 99)
        self.assertEqual(self.changes[-1]["mode"], "live")
        source.close()

    def test_missing_camera_falls_back_to_demo_mode_instead_of_stopping(self):
        def no_camera():
            raise RuntimeError("no webcam")

        source = SwitchableSource(no_camera, on_change=self.changes.append)
        self.assertIsNotNone(source.read())  # a still frame, not the end of the stream
        self.assertEqual(self.changes[-1]["mode"], "demo")
        self.assertIn("no webcam", self.changes[-1]["notice"])

    def test_camera_that_stops_mid_session_falls_back_too(self):
        self.camera = FakeCamera(frames=2)
        source = self._source()
        frames = [source.read() for _ in range(4)]
        self.assertTrue(all(frame is not None for frame in frames))
        self.assertEqual(self.changes[-1]["mode"], "demo")

    def test_every_frame_has_one_size_whatever_the_source(self):
        # MediaPipe's tracker crashes if the frame size changes mid-stream.
        self.camera.read = lambda: np.full((720, 1280, 3), 50, np.uint8)
        source = self._source(size=(64, 48))
        self.assertEqual(source.read().shape, (48, 64, 3))  # 16:9 letterboxed into 4:3
        source.play_clip(self.clip)
        self.assertEqual(source.read().shape, (48, 64, 3))
        source.close()

    def test_start_in_demo_mode_without_opening_the_camera(self):
        source = self._source(start_live=False)
        self.assertEqual(source.read().shape, (480, 640, 3))
        self.assertFalse(self.changes)
        self.assertEqual(source.info["mode"], "demo")


if __name__ == "__main__":
    unittest.main()
