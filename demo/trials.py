"""Run the full demo flow on scripted sentences; log accuracy, latency and crashes.

Run from the project root:
    python -m demo.trials                 # all 10 trials
    python -m demo.trials --trials 3

Each trial is one sentence signed by real people: ASL Citizen test-split clips
(signers the model never trained on), stitched into a video once and cached in
data/raw/replay/trials/. Every trial uses a different set of signers. The trial
then runs the real demo server on it (`python -m demo.app --source <video>`) in
its own process and attaches what the page attaches: the event stream and the
video stream. If the optional `websocket-client` package is installed, it also
opens the page itself in headless Edge, collects JavaScript errors, and checks
that the page shows the sentence and latency that were spoken.

Recorded per trial: every sign the classifier emitted (including ones dropped as
too unsure), the sentences spoken with their latency, and any crash, traceback,
error state or JS exception. Results print as a table and are saved to
logs/trials_<time>.json.
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from data.vocabulary import ASL_CITIZEN

ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = ROOT / "data" / "raw" / "replay" / "trials"
PORT = 8766
TARGET_MS = 1500
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
TRIALS = [  # one sentence each, in vocabulary labels
    ("hello", "name", "what"),
    ("help", "need", "please"),
    ("thank you",),
    ("yes", "understand"),
    ("sorry", "again", "please"),
    ("why", "problem"),
    ("sign", "language", "learn", "want"),
    ("idea", "good", "think"),
    ("three", "example", "important"),
    ("understand", "no"),
]


@dataclass
class TrialResult:
    index: int
    expected: list[str]
    predicted: list[dict] = field(default_factory=list)  # every classified sign: label, confidence, kept
    spoken: list[dict] = field(default_factory=list)  # sentence, glosses, end_to_end, stages
    frames_streamed: int = 0
    capture_fps: float = 0.0  # pipeline frames per second while the video played
    errors: list[str] = field(default_factory=list)
    crashed: bool = False
    page_checked: bool = False
    seconds: float = 0.0

    @property
    def said_signs(self) -> list[str]:
        return [p["label"] for p in self.predicted if p["kept"]]

    def correct(self, labels: list[str]) -> int:
        """Signs recognized correctly, by aligning expected and recognized sequences."""
        matcher = difflib.SequenceMatcher(a=self.expected, b=labels, autojunk=False)
        return sum(block.size for block in matcher.get_matching_blocks())


# ---------------------------------------------------------------- clients


def _frames(url: str) -> int:
    return json.load(urllib.request.urlopen(url + "/api/state", timeout=5)).get("frames", 0)


def follow_events(url: str, result: TrialResult, done: threading.Event) -> None:
    ready = None  # (time, frames) when the pipeline started reading the video
    try:
        with urllib.request.urlopen(url + "/events", timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: ") :])
                kind = event["type"]
                if kind == "sign":
                    result.predicted.append({k: event[k] for k in ("label", "confidence", "kept")})
                elif kind == "spoken":
                    result.spoken.append({k: event[k] for k in ("sentence", "glosses", "end_to_end", "stages")})
                elif kind == "notice":
                    result.errors.append(f"notice: {event['message']}")
                elif kind in ("status", "snapshot") and event["status"] == "listening" and ready is None:
                    ready = (time.monotonic(), _frames(url))
                elif kind in ("status", "snapshot") and event["status"] in ("stopped", "error"):
                    if event["status"] == "error":
                        result.errors.append(f"error state: {event['message']}")
                    elif ready is not None:
                        result.capture_fps = round((_frames(url) - ready[1]) / (time.monotonic() - ready[0]), 1)
                    done.set()
                if done.is_set():
                    return
    except Exception as error:  # noqa: BLE001
        if not done.is_set():
            result.errors.append(f"event stream broke: {error!r}")


def follow_video(url: str, result: TrialResult, done: threading.Event) -> None:
    try:
        with urllib.request.urlopen(url + "/video", timeout=30) as response:
            while not done.is_set():
                chunk = response.read(65536)
                if not chunk:
                    break
                result.frames_streamed += chunk.count(b"--frame")
    except Exception as error:  # noqa: BLE001
        if not done.is_set():
            result.errors.append(f"video stream broke: {error!r}")


class EdgePage:
    """The demo page in headless Edge, driven over the DevTools protocol."""

    def __init__(self, url: str, port: int = 9341):
        import websocket  # optional dependency

        self._profile = tempfile.mkdtemp(prefix="signspeak-edge-")
        self._process = subprocess.Popen(
            [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run", f"--remote-debugging-port={port}",
             f"--user-data-dir={self._profile}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        target = None
        for _ in range(100):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1))
                target = next(t for t in targets if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Edge still starting
                time.sleep(0.2)
        if target is None:
            raise RuntimeError("headless Edge did not start")
        self._ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
        self._ids = iter(range(1, 1_000_000))
        self.js_errors: list[str] = []
        self._call("Runtime.enable")
        self._call("Page.enable")
        self._call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1, mobile=False)
        self._call("Page.navigate", url=url)

    def _call(self, method: str, **params):
        message_id = next(self._ids)
        self._ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(self._ws.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                details = message["params"]["exceptionDetails"]
                self.js_errors.append(details.get("exception", {}).get("description", details.get("text", "?")))
            elif message.get("method") == "Runtime.consoleAPICalled" and message["params"]["type"] == "error":
                self.js_errors.append(" ".join(str(a.get("value", a.get("description", ""))) for a in message["params"]["args"]))
            if message.get("id") == message_id:
                return message.get("result", {})

    def poll(self) -> None:
        self._call("Runtime.evaluate", expression="1")  # drains queued console/exception events

    def shown(self) -> dict:
        expression = """JSON.stringify({
          sentence: document.getElementById('sentence').textContent,
          latency: document.getElementById('latency-value').textContent,
          status: document.getElementById('status-text').textContent})"""
        return json.loads(self._call("Runtime.evaluate", expression=expression)["result"]["value"])

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001 - tearing down anyway
            pass
        # Edge runs a tree of processes; killing only the parent can leave it hanging.
        subprocess.run(["taskkill", "/PID", str(self._process.pid), "/T", "/F"], capture_output=True)
        try:
            self._process.wait(10)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(self._profile, ignore_errors=True)


# ---------------------------------------------------------------- one trial


def build_trial_video(index: int, labels: tuple[str, ...], archive) -> Path:
    from demo.build_replay_video import build_video

    path = TRIAL_DIR / f"trial_{index:02d}.mp4"
    if not path.exists():
        # clip index 1 + trial number: signers differ from each other and from utterances.mp4 (index 0)
        build_video([tuple(ASL_CITIZEN[label] for label in labels)], path, clip_index=1 + index, archive=archive)
    return path


def run_trial(index: int, labels: tuple[str, ...], video: Path, log_dir: Path, use_page: bool) -> TrialResult:
    result = TrialResult(index, list(labels))
    url = f"http://127.0.0.1:{PORT}"
    log_path = log_dir / f"trial_{index:02d}_server.log"
    started = time.monotonic()
    done = threading.Event()
    page = None
    with open(log_path, "w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "demo.app", "--source", str(video), "--no-browser", "--port", str(PORT)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip
    try:
        for _ in range(150):  # the server answers within a second; the pipeline warms up behind it
            if server.poll() is not None:
                break
            try:
                urllib.request.urlopen(url + "/api/state", timeout=1).read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        threads = [
            threading.Thread(target=follow_events, args=(url, result, done), daemon=True),
            threading.Thread(target=follow_video, args=(url, result, done), daemon=True),
        ]
        for thread in threads:
            thread.start()
        if use_page:
            try:
                page = EdgePage(url)
                result.page_checked = True
            except Exception as error:  # noqa: BLE001
                result.errors.append(f"page check unavailable: {error!r}")
        deadline = started + 150
        while not done.is_set() and time.monotonic() < deadline:
            if server.poll() is not None:
                result.crashed = True
                result.errors.append(f"server process exited with code {server.returncode}")
                break
            if page:
                page.poll()
            time.sleep(1)
        if not done.is_set() and not result.crashed:
            result.errors.append("timed out waiting for the run to finish")
        if page:
            time.sleep(0.5)
            page.poll()
            shown = page.shown()
            result.errors += [f"JS: {error}" for error in page.js_errors]
            if result.spoken:
                last = result.spoken[-1]
                if shown["sentence"] != last["sentence"] or shown["latency"].replace(",", "") != str(last["end_to_end"]):
                    result.errors.append(f"page shows {shown} but last spoken was {last['sentence']!r} {last['end_to_end']} ms")
    finally:
        done.set()
        if page:
            try:
                page.close()
            except Exception as error:  # noqa: BLE001 - never let cleanup end the whole run
                result.errors.append(f"harness: closing the page failed: {error!r}")
        server.terminate()
        try:
            server.wait(15)
        except subprocess.TimeoutExpired:
            server.kill()
    if "Traceback" in log_path.read_text(encoding="utf-8", errors="replace"):
        result.errors.append(f"traceback in {log_path.name}")
    result.seconds = round(time.monotonic() - started, 1)
    return result


# ---------------------------------------------------------------- report


def report(results: list[TrialResult]) -> str:
    lines = [f"{'#':>2}  {'signed':<34}{'recognized (said)':<36}{'spoken':<34}{'latency':>9}  problems"]
    for r in results:
        latencies = "/".join(str(s["end_to_end"]) for s in r.spoken) or "-"
        said = " ".join(r.said_signs) or "-"
        dropped = [p["label"] for p in r.predicted if not p["kept"]]
        if dropped:
            said += f" (+unsure: {' '.join(dropped)})"
        spoken = " | ".join(s["sentence"] for s in r.spoken) or "-"
        problems = "; ".join(r.errors) or ("CRASH" if r.crashed else "none")
        lines.append(f"{r.index:>2}  {' '.join(r.expected):<34}{said[:35]:<36}{spoken[:33]:<34}{latencies:>9}  {problems}")

    expected = sum(len(r.expected) for r in results)
    said_correct = sum(r.correct(r.said_signs) for r in results)
    raw_correct = sum(r.correct([p["label"] for p in r.predicted]) for r in results)
    exact = sum(r.said_signs == r.expected for r in results)
    one_sentence = sum(len(r.spoken) == 1 for r in results)
    latencies = [s["end_to_end"] for r in results for s in r.spoken]
    lines += [
        "",
        f"Signs recognized and said correctly: {said_correct}/{expected} ({said_correct / expected:.0%}); "
        f"top-1 incl. signs dropped as unsure: {raw_correct}/{expected} ({raw_correct / expected:.0%})",
        f"Whole sentence exactly right: {exact}/{len(results)}; spoken as one sentence (not split): {one_sentence}/{len(results)}",
    ]
    if latencies:
        ordered = sorted(latencies)
        lines.append(
            f"Latency over {len(latencies)} spoken sentences: median {statistics.median(latencies):.0f} ms, "
            f"90th percentile {ordered[int(0.9 * (len(ordered) - 1))]} ms, max {max(latencies)} ms; "
            f"under {TARGET_MS} ms: {sum(v < TARGET_MS for v in latencies)}/{len(latencies)}"
        )
    predicted = sum(len(r.predicted) for r in results)
    lines.append(f"Signs the segmenter found: {predicted}/{expected}; "
                 f"capture loop {statistics.mean(r.capture_fps for r in results):.1f} fps on average")  # fmt: skip
    crashes = sum(r.crashed for r in results)
    with_errors = sum(bool(r.errors) for r in results)
    lines.append(f"Crashes: {crashes}; trials with any error: {with_errors}/{len(results)}; "
                 f"page checked in {sum(r.page_checked for r in results)} trials")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=len(TRIALS))
    parser.add_argument("--no-page", action="store_true", help="skip the headless-Edge page check")
    args = parser.parse_args(argv)

    try:
        import websocket  # noqa: F401

        use_page = not args.no_page and EDGE.exists()
    except ImportError:
        use_page = False
        print("websocket-client not installed: skipping the page check (server and streams are still tested)")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = ROOT / "logs" / f"trials_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    trials = TRIALS[: args.trials]
    archive = None
    if not all((TRIAL_DIR / f"trial_{i:02d}.mp4").exists() for i in range(len(trials))):
        from data.prepare_dataset import ASL_CITIZEN_URL
        from data.remote_zip import RemoteZip

        archive = RemoteZip(ASL_CITIZEN_URL)
    results = []
    for index, labels in enumerate(trials):
        video = build_trial_video(index, labels, archive)
        result = run_trial(index, labels, video, log_dir, use_page)
        results.append(result)
        print(f"trial {index}: signed {' '.join(labels)!r} -> said {' '.join(result.said_signs)!r}, "
              f"{[s['end_to_end'] for s in result.spoken]} ms, {result.capture_fps} fps, "
              f"errors: {result.errors or 'none'}", flush=True)  # fmt: skip
    summary = report(results)
    print("\n" + summary)
    path = ROOT / "logs" / f"trials_{stamp}.json"
    path.write_text(json.dumps({"results": [asdict(r) for r in results], "summary": summary}, indent=2))
    print(f"\nSaved {path} (server logs in {log_dir})")


if __name__ == "__main__":
    main()
