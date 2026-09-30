"""Build a replay video for measuring pipeline latency with real signing but no
live signer: ASL Citizen test clips stitched into sentences.

Run from the project root:
    python -m demo.build_replay_video
    python -m src.pipeline.live --source data/raw/replay/utterances.mp4 --no-window

Each clip is trimmed to where a hand is visible, so the signs follow each other
the way continuous signing does, with a short hold between signs and a longer
rest (an empty frame, as if the signer stepped back) after each sentence.
The clips come from different signers. The output lands in data/raw
(gitignored): ASL Citizen may not be redistributed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from data.prepare_dataset import ASL_CITIZEN_URL
from data.remote_zip import RemoteZip
from data.vocabulary import ASL_CITIZEN
from src.capture import HolisticTracker

ROOT = Path(__file__).resolve().parents[1]
UTTERANCES = [("HELLO", "NAME", "WHAT1"), ("HELP", "NEED", "PLEASE"), ("UNDERSTAND", "NO")]
SIZE, FPS = (640, 480), 30.0
HOLD_BETWEEN_SIGNS_S = 0.25
REST_AFTER_SENTENCE_S = 3.0
LEAD_IN_S = 1.5
MARGIN_FRAMES = 3
OUT = ROOT / "data" / "raw" / "replay" / "utterances.mp4"


def pick_clip(gloss: str, index: int = 0) -> str:
    """The index-th test-split clip of this gloss (wrapping around). Test-split
    signers were never trained on; different indexes give different signers."""
    with open(ROOT / "data" / "raw" / "asl_citizen" / "splits" / "test.csv", encoding="utf-8") as file:
        clips = [row["Video file"] for row in csv.DictReader(file) if row["Gloss"] == gloss]
    if not clips:
        raise KeyError(gloss)
    return clips[index % len(clips)]


def active_frames(video: bytes, scratch: Path) -> list[np.ndarray]:
    """The clip's frames where a hand is visible, plus a few frames either side."""
    scratch.write_bytes(video)
    capture = cv2.VideoCapture(str(scratch))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, SIZE))
    capture.release()
    with HolisticTracker() as tracker:
        hands = [
            i
            for i, frame in enumerate(frames)
            if (lm := tracker.process(frame, int(i * 1000 / FPS))).left_hand is not None or lm.right_hand is not None
        ]
    if not hands:
        return frames
    return frames[max(0, hands[0] - MARGIN_FRAMES) : hands[-1] + 1 + MARGIN_FRAMES]


def build_video(utterances, out: Path, clip_index: int = 0, archive: RemoteZip | None = None) -> list[dict]:
    """Write a video of these sentences (tuples of ASL Citizen glosses) to `out`,
    using each gloss's clip_index-th test clip. Returns the manifest."""
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = archive or RemoteZip(ASL_CITIZEN_URL)
    label_of = {gloss: label for label, gloss in ASL_CITIZEN.items()}
    blank = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    scratch = out.with_name(f"_{out.stem}_clip.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    manifest, clock = [], LEAD_IN_S
    for _ in range(int(LEAD_IN_S * FPS)):
        writer.write(blank)
    for glosses in utterances:
        entry = {"start_s": round(clock, 2), "glosses": [label_of[g] for g in glosses], "clips": []}
        for k, gloss in enumerate(glosses):
            name = pick_clip(gloss, clip_index)
            frames = active_frames(archive.read(f"ASL_Citizen/videos/{name}"), scratch)
            if k < len(glosses) - 1:
                frames += [frames[-1]] * int(HOLD_BETWEEN_SIGNS_S * FPS)
            for frame in frames:
                writer.write(frame)
            clock += len(frames) / FPS
            entry["clips"].append(name)
            print(f"  {gloss:<11} {name}  {len(frames) / FPS:.2f} s")
        for _ in range(int(REST_AFTER_SENTENCE_S * FPS)):
            writer.write(blank)
        clock += REST_AFTER_SENTENCE_S
        manifest.append(entry)
    writer.release()
    scratch.unlink(missing_ok=True)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out} ({clock:.1f} s) and {out.with_suffix('.json').name}")
    return manifest


def main() -> None:
    build_video(UTTERANCES, OUT)


if __name__ == "__main__":
    main()
