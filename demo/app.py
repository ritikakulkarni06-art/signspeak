"""SignSpeak demo UI: the live pipeline in the browser.

Run from the project root:
    python -m demo.app                                                   # webcam
    python -m demo.app --source data/raw/replay/utterances.mp4 --loop    # backup: replayed real signing

It opens http://127.0.0.1:8000. Speech plays on this computer's speakers,
through the same low-latency path the latency numbers were measured on.

The server runs the pipeline on a background thread and serves:
    /            the page (demo/static; no external scripts or fonts, works offline)
    /video       webcam feed with the landmark overlay (MJPEG)
    /events      signs, sentences and latency as they happen (Server-Sent Events)
    /api/state   the current state as JSON (health check)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import queue
import statistics
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify

from src.capture import draw_landmarks
from src.pipeline.live import Pipeline, PipelineConfig, PipelineListener
from src.pipeline.trace import now_ms

STATIC = Path(__file__).resolve().parent / "static"
HISTORY = 8  # sentences kept on screen
# The browser decoding the stream shares this laptop's CPU with the tracker
# (measured: a full page in software-rendered Edge cut the capture loop from
# 19.7 to 14.5 fps), so stream no more pixels or frames than the eye needs.
STREAM_WIDTH = 800
STREAM_FPS = 15
TARGET_MS = 1500
log = logging.getLogger("signspeak.demo")


class DemoState(PipelineListener):
    """Receives the pipeline's callbacks, keeps the newest frame for the video
    stream, and fans events out to every open browser tab. A slow tab can never
    block the pipeline: each tab has a bounded queue and events are dropped for it
    rather than waited on."""

    def __init__(self, mode: str):
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._frame_ready = threading.Condition()
        self._frame = None
        self._landmarks = None
        self._frame_id = 0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_stats = 0.0
        self._signing = False
        self.snapshot = {
            "status": "starting",
            "message": "Warming up the models…",
            "mode": mode,
            "translator": "",
            "voice": "",
            "labels": [],
            "fps": 0.0,
            "current": [],
            "history": [],
            "latencies": [],
        }

    # ------------------------------------------------------------ browser side

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass

    def state(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.snapshot)

    def clear(self) -> None:
        with self._lock:
            self.snapshot.update(history=[], latencies=[], current=[])
        self.publish({"type": "cleared"})

    def fatal(self, message: str) -> None:
        with self._lock:
            self.snapshot.update(status="error", message=message)
        self.publish({"type": "status", "status": "error", "message": message})

    def stopped(self, message: str) -> None:
        with self._lock:
            self.snapshot.update(status="stopped", message=message)
        self.publish({"type": "status", "status": "stopped", "message": message})

    # ------------------------------------------------------------ pipeline callbacks

    def on_ready(self, info: dict) -> None:
        with self._lock:
            self.snapshot.update(status="listening", message="", translator=info["translator"],
                                 voice=info["voice"], labels=info["labels"])  # fmt: skip
        self.publish({"type": "status", "status": "listening", "message": "", **info})

    def on_error(self, message: str) -> None:
        with self._lock:
            starting = self.snapshot["status"] == "starting"
        if starting:  # nothing is running yet: this is fatal
            self.fatal(message)
        else:  # e.g. one sentence failed to speak; the session goes on
            self.publish({"type": "notice", "message": message})

    def on_frame(self, frame, landmarks, signing: bool) -> None:
        with self._frame_ready:
            self._frame, self._landmarks = frame, landmarks
            self._frame_id += 1
            self._frame_ready.notify_all()
        now = time.perf_counter()
        self._frame_times.append(now)
        if signing != self._signing:
            self._signing = signing
            self.publish({"type": "signing", "value": signing})
        if now - self._last_stats >= 1.0 and len(self._frame_times) > 1:
            self._last_stats = now
            fps = round((len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0]), 1)
            with self._lock:
                self.snapshot["fps"] = fps
                self.snapshot["frames"] = self._frame_id  # running count, for measuring fps over any window
            self.publish({"type": "stats", "fps": fps})

    def on_sign(self, sign, kept: bool) -> None:
        chip = {"label": sign.label, "confidence": round(sign.confidence, 2), "kept": kept}
        if kept:
            with self._lock:
                self.snapshot["current"].append(chip)
        self.publish({"type": "sign", **chip, "elapsed_ms": round(now_ms() - sign.gesture_end_ms)})

    def on_utterance(self, trace) -> None:
        self.publish({
            "type": "utterance",
            "index": trace.index,
            "glosses": [s.label for s in trace.signs],
            "elapsed_ms": round(now_ms() - trace.signs[-1].gesture_end_ms),
        })  # fmt: skip

    def on_spoken(self, trace) -> None:
        stages = {name: round(value) for name, value in trace.stages().items()}
        entry = {
            "index": trace.index,
            "sentence": trace.sentence,
            "glosses": [s.label for s in trace.signs],
            "stages": stages,
            "end_to_end": stages["end_to_end"],
            "time": time.strftime("%H:%M:%S"),
        }
        with self._lock:
            history = self.snapshot["history"]
            history.insert(0, entry)
            del history[HISTORY:]
            latencies = self.snapshot["latencies"]
            latencies.append(entry["end_to_end"])
            del latencies[:-100]
            self.snapshot.update(current=[], status="speaking")
            summary = {
                "median_ms": round(statistics.median(latencies)),
                "count": len(latencies),
                "under_target": sum(v < TARGET_MS for v in latencies),
            }
        self.publish({"type": "spoken", **entry, **summary})

    def on_speech_done(self, trace) -> None:
        with self._lock:
            if self.snapshot["status"] == "speaking":
                self.snapshot["status"] = "listening"
        self.publish({"type": "speech_done", "index": trace.index})

    # ------------------------------------------------------------ video

    def mjpeg(self, width: int = STREAM_WIDTH, max_fps: float = STREAM_FPS):
        """Multipart JPEG frames: mirrored like a mirror, with the landmark overlay.
        Rendered here, on the web thread, so the camera loop pays nothing for it."""
        last_id = 0  # frame ids start at 1: a tab opened during warm-up waits for the first frame
        while True:
            with self._frame_ready:
                if not self._frame_ready.wait_for(lambda: self._frame_id != last_id, timeout=2.0):
                    continue
                frame, landmarks, last_id = self._frame, self._landmarks, self._frame_id
            if frame is None:
                continue
            started = time.perf_counter()
            view = cv2.flip(frame, 1)
            if view.shape[1] > width:
                view = cv2.resize(view, (width, round(view.shape[0] * width / view.shape[1])), interpolation=cv2.INTER_AREA)
            draw_landmarks(view, landmarks, mirror=True)
            ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                data = jpeg.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(data), data)
            time.sleep(max(0.0, 1.0 / max_fps - (time.perf_counter() - started)))


def create_app(state: DemoState) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/video")
    def video():
        return Response(state.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})  # fmt: skip

    @app.get("/events")
    def events():
        subscriber = state.subscribe()

        def stream():
            try:
                yield _sse({"type": "snapshot", **state.state()})
                while True:
                    try:
                        yield _sse(subscriber.get(timeout=10))
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                state.unsubscribe(subscriber)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})  # fmt: skip

    @app.get("/api/state")
    def api_state():
        return jsonify(state.state())

    @app.post("/api/clear")
    def api_clear():
        state.clear()
        return jsonify(ok=True)

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam index, or a video file (backup demo)")
    parser.add_argument("--loop", action="store_true", help="loop a video source")
    parser.add_argument("--translator", choices=["auto", "claude", "rules"], default="auto")
    parser.add_argument("--voice", help="part of a voice name, e.g. Mark or Zira")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-9s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    state = DemoState("Webcam" if args.source.isdigit() else "Replay")
    pipeline = Pipeline(
        PipelineConfig(source=args.source, loop_source=args.loop, show_window=False,
                       translator=args.translator, voice=args.voice),  # fmt: skip
        state,
    )

    def run_pipeline() -> None:
        try:
            print(pipeline.run(), flush=True)
            state.stopped("The video source ended. Restart the demo to continue.")
        except Exception as error:  # noqa: BLE001 - shown on the page
            log.exception("pipeline stopped")
            if state.state()["status"] != "error":
                state.fatal(f"The pipeline stopped: {error}")

    worker = threading.Thread(target=run_pipeline, name="pipeline", daemon=True)
    worker.start()
    url = f"http://{args.host}:{args.port}"
    print(f"\n  SignSpeak demo: {url}   (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    try:
        create_app(state).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        pipeline.stop()
        worker.join(timeout=15)


if __name__ == "__main__":
    main()
