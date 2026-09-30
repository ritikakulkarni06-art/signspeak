"""Live sign segmentation check: sign in front of the webcam, and each time a
sign closes the console prints "sign segment captured" with its frame count.

Run from the project root:
    python -m demo.segment_signs [--camera 0] [--window 30] [--start 0.8] [--stop 0.4] [--pause-ms 250]

The window shows the landmarks, the segmenter state, and a live motion bar with
the start/stop thresholds marked, for tuning. Keys: q or Esc to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.capture import HolisticTracker, draw_landmarks, extract_features, open_webcam
from src.capture.segmenter import SegmenterConfig, SignSegment, SignSegmenter

WINDOW = "SignSpeak - sign segmentation"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, AMBER, GRAY = (255, 255, 255), (80, 220, 80), (0, 190, 255), (170, 170, 170)
FLASH_MS = 1200  # how long the "captured" banner stays up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = SegmenterConfig()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=defaults.window_size, help="rolling window size in frames")
    parser.add_argument("--start", type=float, default=defaults.start_speed, help="motion that starts a sign")
    parser.add_argument("--stop", type=float, default=defaults.stop_speed, help="motion that counts as a pause")
    parser.add_argument("--pause-ms", type=int, default=defaults.pause_ms, help="pause length that ends a sign")
    args = parser.parse_args()

    config = SegmenterConfig(
        window_size=args.window, start_speed=args.start, stop_speed=args.stop, pause_ms=args.pause_ms
    )
    segmenter = SignSegmenter(config)
    print(f"Segmenting with {config}")
    print("Sign in front of the camera; pause briefly or lower your hands between signs.\n")

    cap = open_webcam(args.camera)
    count, last_segment, last_segment_at = 0, None, 0.0
    try:
        with HolisticTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Webcam stopped returning frames.")
                    break
                landmarks = tracker.process(frame)
                segment = segmenter.push(extract_features(landmarks), landmarks.timestamp_ms)
                if segment is not None:
                    count += 1
                    _report(count, segment)
                    last_segment, last_segment_at = segment, time.monotonic()

                view = cv2.flip(frame, 1)
                draw_landmarks(view, landmarks, mirror=True)
                recent = last_segment if (time.monotonic() - last_segment_at) * 1000 < FLASH_MS else None
                _draw_status(view, segmenter, count, recent)
                cv2.imshow(WINDOW, view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        segment = segmenter.flush()
        if segment is not None:
            count += 1
            _report(count, segment)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"\n{count} segment(s) captured.")


def _report(index: int, segment: SignSegment) -> None:
    print(
        f"sign segment captured: {segment.num_frames} frames "
        f"({segment.duration_ms / 1000:.2f} s, #{index}, ended by {segment.end_reason})",
        flush=True,
    )


def _draw_status(image: np.ndarray, segmenter: SignSegmenter, count: int, recent: SignSegment | None) -> None:
    cfg = segmenter.config
    scale = image.shape[1] / 1280
    margin, thickness, font_scale = int(12 * scale), max(1, int(1.5 * scale)), 0.7 * scale
    panel = image[: int(96 * scale), :]
    panel[:] = (panel * 0.35).astype(np.uint8)

    state, color = ("SIGNING", AMBER) if segmenter.in_sign else ("waiting", GRAY)
    cv2.putText(image, f"{state}   segments: {count}", (margin, int(32 * scale)), FONT, font_scale, color, thickness)

    # Motion bar: full width = 2x the start threshold, with stop/start ticks.
    bar_x0, bar_y, bar_w, bar_h = margin, int(52 * scale), int(500 * scale), int(22 * scale)
    full = 2 * cfg.start_speed
    filled = int(bar_w * min(segmenter.motion / full, 1.0))
    cv2.rectangle(image, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + bar_h), GRAY, 1)
    cv2.rectangle(image, (bar_x0, bar_y), (bar_x0 + filled, bar_y + bar_h), color, -1)
    for threshold, label in ((cfg.stop_speed, "stop"), (cfg.start_speed, "start")):
        tick_x = bar_x0 + int(bar_w * threshold / full)
        cv2.line(image, (tick_x, bar_y - 4), (tick_x, bar_y + bar_h + 4), WHITE, thickness)
        cv2.putText(image, label, (tick_x + 4, bar_y + bar_h + int(16 * scale)), FONT, 0.45 * scale, WHITE, 1)
    cv2.putText(
        image,
        f"motion {segmenter.motion:.2f}",
        (bar_x0 + bar_w + margin, bar_y + bar_h - 4),
        FONT,
        font_scale,
        WHITE,
        thickness,
    )

    if recent is not None:
        text = f"sign segment captured: {recent.num_frames} frames"
        (w, h), _ = cv2.getTextSize(text, FONT, 1.0 * scale, 2)
        x, y = (image.shape[1] - w) // 2, image.shape[0] - int(40 * scale)
        cv2.rectangle(image, (x - margin, y - h - margin), (x + w + margin, y + margin), (0, 0, 0), -1)
        cv2.putText(image, text, (x, y), FONT, 1.0 * scale, GREEN, 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
