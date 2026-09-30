"""Demo UI server tests: routes, event stream, and video stream with a fake
pipeline feeding the state (no camera, models, or speakers).

Run from the project root: python -m unittest discover -s tests
"""

import json
import threading
import unittest

import cv2
import numpy as np

from demo.app import STREAM_WIDTH, DemoState, create_app
from src.capture.landmarks import FrameLandmarks
from src.pipeline import SignEvent, UtteranceTrace


def _sign(label="hello", confidence=0.8, t=1000.0):
    return SignEvent(label, confidence, t, t + 300, t + 305, 1.2, 0.0)


def _events(response):
    """Parsed SSE events from a streaming response, one per next()."""
    for chunk in response.response:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        if text.startswith("data: "):
            yield json.loads(text[len("data: ") :])


class DemoAppTests(unittest.TestCase):
    def setUp(self):
        self.state = DemoState("Webcam")
        self.client = create_app(self.state).test_client()

    def test_page_and_assets_are_served(self):
        with self.client.get("/") as page:
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"SignSpeak", page.data)
            self.assertNotIn(b'src="/video"', page.data)  # started by script after load, so the tab stops "loading"
        for asset in ("/static/app.js", "/static/app.css"):
            with self.client.get(asset) as response:
                self.assertEqual(response.status_code, 200, asset)

    def test_event_stream_starts_with_a_snapshot_then_live_events(self):
        response = self.client.get("/events")
        events = _events(response)
        snapshot = next(events)
        self.assertEqual((snapshot["type"], snapshot["status"]), ("snapshot", "starting"))
        self.state.on_ready({"translator": "offline rules", "voice": "David", "labels": ["yes", "no"], "warmup_ms": {}})
        self.state.on_sign(_sign("yes"), kept=True)
        self.assertEqual(next(events)["status"], "listening")
        sign = next(events)
        self.assertEqual((sign["type"], sign["label"], sign["kept"]), ("sign", "yes", True))
        response.close()

    def test_spoken_sentence_updates_history_and_summary(self):
        self.state.on_ready({"translator": "rules", "voice": "David", "labels": [], "warmup_ms": {}})
        trace = UtteranceTrace(1, [_sign("yes")], 1400, "gap", sentence="Yes.", translate_start_ms=1401,
                               translate_end_ms=1402, tts_end_ms=1450, audio_start_ms=1540)  # fmt: skip
        self.state.on_spoken(trace)
        state = self.state.state()
        self.assertEqual(state["status"], "speaking")
        self.assertEqual(state["history"][0]["sentence"], "Yes.")
        self.assertEqual(state["history"][0]["end_to_end"], 540)
        self.state.on_speech_done(trace)
        self.assertEqual(self.state.state()["status"], "listening")

    def test_errors_are_fatal_only_before_the_pipeline_is_ready(self):
        self.state.on_error("camera not found")
        self.assertEqual(self.state.state()["status"], "error")
        other = DemoState("Webcam")
        other.on_ready({"translator": "", "voice": "", "labels": [], "warmup_ms": {}})
        other.on_error("one sentence failed")
        self.assertEqual(other.state()["status"], "listening")

    def test_video_stream_serves_jpeg_frames(self):
        frame = np.full((720, 1280, 3), 90, np.uint8)
        self.state.on_frame(frame, FrameLandmarks((1280, 720)), signing=False)
        response = self.client.get("/video")
        chunk = next(iter(response.response))
        response.close()
        self.assertTrue(chunk.startswith(b"--frame"))
        jpeg = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (round(720 * STREAM_WIDTH / 1280), STREAM_WIDTH))  # scaled to the stream width

    def test_video_opened_before_the_first_frame_waits_instead_of_failing(self):
        # Regression: a tab opened during warm-up used to crash the stream (HTTP 500).
        frame = np.full((480, 640, 3), 60, np.uint8)
        threading.Timer(0.3, self.state.on_frame, (frame, FrameLandmarks((640, 480)), False)).start()
        response = self.client.get("/video")
        chunk = next(iter(response.response))
        response.close()
        self.assertTrue(chunk.startswith(b"--frame"))

    def test_clear_empties_history(self):
        self.state.on_sign(_sign(), kept=True)
        self.assertEqual(self.client.post("/api/clear").status_code, 200)
        self.assertEqual(self.state.state()["current"], [])


if __name__ == "__main__":
    unittest.main()
