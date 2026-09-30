"""Live tracking check: webcam -> Holistic -> feature vector, with landmarks drawn
over the video so you can judge tracking quality.

Run from the project root:
    python -m demo.live_capture [--camera 0] [--no-mirror]

Keys: q or Esc to quit, m to toggle mirror view.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.capture import FEATURE_DIM, LAYOUT, NMM_NAMES, HolisticTracker, draw_landmarks, extract_features, open_webcam

WINDOW = "SignSpeak - capture check"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, RED, GRAY = (255, 255, 255), (80, 220, 80), (80, 80, 255), (170, 170, 170)
PARTS = (("L hand", "left_hand"), ("R hand", "right_hand"), ("pose", "pose"), ("face", "face"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--no-mirror", action="store_true", help="show the raw camera view instead of a mirror")
    args = parser.parse_args()

    mirror = not args.no_mirror
    cap = open_webcam(args.camera)
    fps, last_frame_time = 0.0, time.perf_counter()
    try:
        with HolisticTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Webcam stopped returning frames.")
                    break

                start = time.perf_counter()
                landmarks = tracker.process(frame)  # raw frame, so left/right stay correct
                features = extract_features(landmarks)
                latency_ms = (time.perf_counter() - start) * 1000

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
                fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
                last_frame_time = now

                view = cv2.flip(frame, 1) if mirror else frame.copy()
                draw_landmarks(view, landmarks, mirror=mirror)
                _draw_hud(view, features, fps, latency_ms, mirror)
                cv2.imshow(WINDOW, view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    mirror = not mirror
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _draw_hud(image: np.ndarray, features: np.ndarray, fps: float, latency_ms: float, mirror: bool) -> None:
    """Show stats read back from the feature vector itself, so the HUD checks the vector, not just the tracker."""
    scale = image.shape[1] / 1280
    line_height = int(28 * scale)
    font_scale = 0.6 * scale
    panel_rows = 4
    panel = image[: line_height * panel_rows + int(12 * scale), :]
    panel[:] = (panel * 0.35).astype(np.uint8)

    def put(text: str, x: int, row: int, color=WHITE) -> int:
        y = int(line_height * (row + 1))
        cv2.putText(image, text, (x, y), FONT, font_scale, color, max(1, int(scale * 1.5)), cv2.LINE_AA)
        return x + cv2.getTextSize(text, FONT, font_scale, 1)[0][0]

    margin = int(12 * scale)
    put(f"{fps:4.1f} fps | tracking + features {latency_ms:5.1f} ms | vector {FEATURE_DIM}-d", margin, 0)

    x = margin
    for label, part in PARTS:
        present = features[LAYOUT[f"{part}.present"]][0] > 0
        x = put(f"{label}: {'on' if present else 'off'}", x, 1, GREEN if present else RED) + int(24 * scale)

    if features[LAYOUT["face.present"]][0] > 0:
        roll, pitch, yaw = np.degrees(features[LAYOUT["face.head_pose"]])
        nmm = dict(zip(NMM_NAMES, features[LAYOUT["face.nmm"]]))
        put(f"head  roll {roll:+6.1f}  pitch {pitch:+6.1f}  yaw {yaw:+6.1f} deg", margin, 2)
        put(
            f"brows R {nmm['brow_raise_right']:.2f} L {nmm['brow_raise_left']:.2f}   "
            f"eyes R {nmm['eye_open_right']:.2f} L {nmm['eye_open_left']:.2f}   "
            f"mouth open {nmm['mouth_open']:.2f} wide {nmm['mouth_width']:.2f}",
            margin,
            3,
        )
    else:
        put("face not tracked - non-manual markers unavailable", margin, 2, GRAY)

    footer_y = image.shape[0] - margin
    cv2.putText(
        image,
        f"q quit | m mirror ({'on' if mirror else 'off'})",
        (margin, footer_y),
        FONT,
        font_scale,
        GRAY,
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
