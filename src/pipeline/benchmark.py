"""Measure what each latency optimization buys, where it acts.

Run from the project root:
    python -m src.pipeline.benchmark [--runs 4]

1. Per-frame cost: every frame of the replay video, upscaled to webcam size
   (1280x720), goes through tracking + features + segmentation + preview window
   in order, with no real-time pacing. Same frames for every configuration, so
   the comparison is deterministic. This is what the preview-window and
   tracking-resolution changes act on.
2. End-to-end: the replay through the whole pipeline in real time, --runs times
   per configuration, each configuration adding one change. Reported both as
   end-to-end latency and as processing latency (end-to-end minus the buffer
   and queue waits): those waits depend on where sentences break, not on how
   fast the pipeline runs, and would otherwise swamp the comparison.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import replace

import cv2
import numpy as np

from src.capture import HolisticTracker, draw_landmarks, extract_features
from src.capture.segmenter import SegmenterConfig, SignSegmenter

from .live import ROOT, Pipeline, PipelineConfig, _fit_width

REPLAY = ROOT / "data" / "raw" / "replay" / "utterances.mp4"
WEBCAM_SIZE = (1280, 720)
BASE = PipelineConfig(source=str(REPLAY), source_size=WEBCAM_SIZE)  # the Stage 7 settings
STEPS = [
    ("baseline (Stage 7)", {}),
    ("+ half-size preview window", {"preview_scale": 0.5}),
    ("+ PyTorch on 1 thread", {"torch_threads": 1}),
    ("+ faster sign-end detection", {"segmenter": SegmenterConfig(pause_ms=120, smoothing=0.8)}),
    ("+ track at 640x360", {"track_max_width": 640}),
]
FRAME_CONFIGS = [
    ("full-size preview (Stage 7)", 1.0, None),
    ("half-size preview", 0.5, None),
    ("half-size preview + track at 640x360", 0.5, 640),
    ("no preview window (reference)", None, None),
]


def load_frames(limit: int) -> list[np.ndarray]:
    capture, frames = cv2.VideoCapture(str(REPLAY)), []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, WEBCAM_SIZE))
    capture.release()
    return frames


def frame_costs(frames: list[np.ndarray]) -> None:
    print(f"\nPer-frame cost ({len(frames)} replay frames at {WEBCAM_SIZE[0]}x{WEBCAM_SIZE[1]}, same frames each time):")
    print(f"  {'configuration':<40}{'tracking':>10}{'whole frame':>13}{'max fps':>9}")
    for name, preview, track_width in FRAME_CONFIGS:
        segmenter, track, total = SignSegmenter(), [], []
        with HolisticTracker() as tracker:
            for i, frame in enumerate(frames):
                start = time.perf_counter()
                landmarks = tracker.process(_fit_width(frame, track_width), int(i * 1000 / 30))
                tracked = time.perf_counter()
                segmenter.push(extract_features(landmarks), landmarks.timestamp_ms)
                if preview is not None:
                    view = cv2.flip(cv2.resize(frame, None, fx=preview, fy=preview) if preview != 1 else frame, 1)
                    draw_landmarks(view, landmarks, mirror=True)
                    cv2.imshow("benchmark", view)
                    cv2.waitKey(1)
                track.append((tracked - start) * 1000)
                total.append((time.perf_counter() - start) * 1000)
        cv2.destroyAllWindows()
        print(f"  {name:<40}{np.median(track):8.1f}ms{np.median(total):11.1f}ms{1000 / np.median(total):9.1f}")


def end_to_end(runs: int, steps: int) -> None:
    rows, config = [], BASE
    for name, change in STEPS[:steps]:
        config = replace(config, **change)
        traces, fps, signs = [], [], 0
        for _ in range(runs):
            pipeline = Pipeline(config)
            report = pipeline.run()
            traces += [t for t in pipeline.traces if t.flush_reason != "end" and t.error is None]
            fps.append(float(report.split("Capture loop: ")[1].split(" ")[0]))
            signs += pipeline.signs_seen
        stages = [t.stages() for t in traces]
        median = lambda values: statistics.median(values) if values else float("nan")  # noqa: E731
        processing = [s["end_to_end"] - s["buffer"] - s["queue"] for s in stages]
        rows.append((name, len(traces), signs, np.mean(fps), median([s["detect"] for s in stages]),
                     median([s["classify"] for s in stages]), median(processing),
                     median([s["end_to_end"] for s in stages]), sum(s["end_to_end"] < 1500 for s in stages)))  # fmt: skip
        print(f"  done: {name}", flush=True)

    print(f"\nEnd-to-end ({runs} real-time replays per configuration, 8 signs each; each row adds one change):")
    print(f"  {'configuration':<30}{'utts':>5}{'signs':>6}{'fps':>6}{'detect':>8}{'classify':>9}{'processing':>12}{'end-to-end':>12}{'<1.5s':>8}")
    for name, n, signs, fps, detect, classify, processing, e2e, met in rows:
        print(f"  {name:<30}{n:5d}{signs:6d}{fps:6.1f}{detect:6.0f}ms{classify:7.1f}ms{processing:10.0f}ms{e2e:10.0f}ms{met:5d}/{n}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=4, help="real-time replays per configuration")
    parser.add_argument("--steps", type=int, default=len(STEPS), help="only the first N configurations")
    parser.add_argument("--frames", type=int, default=300, help="frames for the per-frame cost table")
    parser.add_argument("--skip-frame-costs", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if not args.skip_frame_costs:
        frame_costs(load_frames(args.frames))
    end_to_end(args.runs, args.steps)


if __name__ == "__main__":
    main()
