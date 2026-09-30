"""Build a labeled sign dataset in the live pipeline's feature format.

Run from the project root:
    python -m data.prepare_dataset asl_citizen      # ASL: ~1,550 clips, ~0.8 GB streamed, ~1 hour
    python -m data.prepare_dataset include          # ISL: ~730 clips, ~11 GB streamed (slow server: hours)
    python -m data.prepare_dataset include --signs hello "thank you" --max-per-sign 2    # quick trial

For every clip of the vocabulary in data/vocabulary.py, it:
  1. fetches just that clip out of the dataset's zip over HTTP range requests
     (nothing else is downloaded), or out of a local copy of the zip (--zip-dir);
  2. runs it through Stage 1 (HolisticTracker + extract_features, the same code
     as the live pipeline), resampled to the live frame rate (--fps, default 15);
  3. trims leading and trailing frames with no hands, caches the result, and
     deletes the video, so disk use stays at a few clips at a time.
It then writes train/val/test tensors to data/processed/<dataset>/.

Splits are the datasets' official ones. ASL Citizen's are signer-independent
(no signer appears in two splits); INCLUDE's are its published
train/val/test lists.

Re-running is safe: clips already extracted are skipped, so an interrupted run
or a flaky server just means running the command again.

Licenses: INCLUDE is CC BY 4.0 (attribution required). ASL Citizen is under the
Microsoft Research License: non-commercial research only, and neither the data
nor data derived from it (including these tensors) may be redistributed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tempfile
import time
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch

from data.remote_zip import RemoteZip
from data.vocabulary import ASL_CITIZEN, INCLUDE
from src.capture import FEATURE_DIM, LAYOUT, HolisticTracker, extract_features
from src.capture.sequence import center_crop, to_right_dominant, trim_to_hands

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")

ASL_CITIZEN_URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
INCLUDE_URL = "https://zenodo.org/records/4010759/files/{name}?download=1"
INCLUDE_SPLITS_URL = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_{split}.txt"
INCLUDE_PARTS = {  # category -> number of zip parts on Zenodo
    "Adjectives": 8, "Animals": 2, "Clothes": 2, "Colours": 2, "Days_and_Time": 3,
    "Electronics": 2, "Greetings": 2, "Home": 4, "Jobs": 2, "Means_of_Transportation": 2,
    "People": 5, "Places": 4, "Pronouns": 2, "Seasons": 1, "Society": 3,
}  # fmt: skip


@dataclass(frozen=True)
class DatasetInfo:
    language: str
    license: str
    citation: str
    download_workers: int


DATASETS = {
    "asl_citizen": DatasetInfo(
        language="ASL (American Sign Language)",
        license="Microsoft Research License: non-commercial research only; "
        "the data and data derived from it may not be redistributed.",
        citation="Desai et al., ASL Citizen: A Community-Sourced Dataset for Advancing "
        "Isolated Sign Language Recognition, NeurIPS 2023 Datasets and Benchmarks.",
        download_workers=4,
    ),
    "include": DatasetInfo(
        language="ISL (Indian Sign Language)",
        license="CC BY 4.0 (attribution required).",
        citation="Sridhar et al., INCLUDE: A Large Scale Dataset for Indian Sign Language "
        "Recognition, ACM Multimedia 2020. doi:10.1145/3394171.3413528",
        download_workers=2,  # Zenodo is slow and returns 504s under parallel load
    ),
}


@dataclass(frozen=True)
class Clip:
    clip_id: str
    label: str
    split: str
    archive: str  # zip URL or path
    member: str  # file inside the zip


# ---------------------------------------------------------------- clip lists


def list_asl_citizen(vocab: dict[str, str], zip_dir: Path | None) -> tuple[list[Clip], dict[str, RemoteZip]]:
    location = str(zip_dir / "ASL_Citizen.zip") if zip_dir else ASL_CITIZEN_URL
    print(f"Reading the ASL Citizen archive directory ({location}) ...", flush=True)
    archive = RemoteZip(location)
    label_of = {gloss: label for label, gloss in vocab.items()}
    clips = []
    for split in SPLITS:
        rows = csv.DictReader(io.StringIO(archive.read(f"ASL_Citizen/splits/{split}.csv").decode("utf-8")))
        for row in rows:
            label = label_of.get(row["Gloss"])
            if label:
                video = row["Video file"]
                clips.append(Clip(Path(video).stem, label, split, location, f"ASL_Citizen/videos/{video}"))
    return clips, {location: archive}


def list_include(
    vocab: dict[str, tuple[str, str]], zip_dir: Path | None
) -> tuple[list[Clip], dict[str, RemoteZip]]:
    label_of = {pair: label for label, pair in vocab.items()}
    split_dir = ROOT / "data" / "raw" / "include" / "splits"
    listed = []
    for split in SPLITS:
        split_file = split_dir / f"include_{split}.txt"
        if not split_file.exists():
            split_dir.mkdir(parents=True, exist_ok=True)
            split_file.write_bytes(_download(INCLUDE_SPLITS_URL.format(split=split)))
        for line in split_file.read_text().splitlines():
            path = line.strip()
            label = label_of.get(_include_category_word(path)) if path else None
            if label:
                listed.append((path, label, split))

    archives, archive_of = {}, {}
    for category in sorted({category for category, _ in vocab.values()}):
        parts = INCLUDE_PARTS[category]
        for part in range(1, parts + 1):
            name = f"{category}_{part}of{parts}.zip"
            location = str(zip_dir / name) if zip_dir else INCLUDE_URL.format(name=name)
            print(f"Reading the directory of {name} ...", flush=True)
            archive = archives[location] = RemoteZip(location)
            archive_of.update(dict.fromkeys(archive.names, location))

    clips, missing = [], []
    for path, label, split in listed:
        if path in archive_of:
            clips.append(Clip(_include_clip_id(path), label, split, archive_of[path], path))
        else:
            missing.append(path)
    if missing:
        print(f"warning: {len(missing)} listed clips aren't in the archives, e.g. {missing[:3]}")
    return clips, archives


def _include_category_word(path: str) -> tuple[str, str]:
    category, folder = path.split("/")[:2]
    return category, re.sub(r"^\d+\.\s*", "", folder)  # "48. Hello" -> "Hello"


def _include_clip_id(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(path).with_suffix("").as_posix()).strip("_")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "signspeak-data-prep"})) as r:
        return r.read()


def cap_per_sign(clips: list[Clip], limit: int | None) -> list[Clip]:
    if not limit:
        return clips
    taken, kept = Counter(), []
    for clip in sorted(clips, key=lambda c: c.clip_id):
        if taken[clip.label, clip.split] < limit:
            taken[clip.label, clip.split] += 1
            kept.append(clip)
    return kept


# ---------------------------------------------------------------- Stage 1 extraction


class FramePicker:
    """Chooses which source frames to keep so a clip plays at target_fps (the
    frame nearest each tick). Never duplicates frames when the source is slower."""

    def __init__(self, source_fps: float, target_fps: float):
        self._step = max(source_fps / target_fps, 1.0)  # in source frames
        self._next = 0.0

    def keep(self, index: int) -> bool:
        if index + 0.5 < self._next:
            return False
        while self._next <= index + 0.5:
            self._next += self._step
        return True


def extract_clip(video_path: Path, target_fps: float) -> np.ndarray | None:
    """Stage 1 on a video file: (frames, FEATURE_DIM) at target_fps, trimmed to
    where hands are visible, or None if no hand is ever tracked."""
    capture = cv2.VideoCapture(str(video_path))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or not np.isfinite(source_fps):
        source_fps = 30.0
    picker = FramePicker(source_fps, target_fps)
    frames = []
    # A fresh tracker per clip: VIDEO mode carries tracking state between frames,
    # which must not leak from one clip into the next.
    with HolisticTracker() as tracker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if picker.keep(index):
                timestamp_ms = int(round(index * 1000 / source_fps))
                frames.append(extract_features(tracker.process(frame, timestamp_ms)))
            index += 1
    capture.release()
    return trim_to_hands(np.stack(frames)) if frames else None


def process_clips(
    clips: list[Clip], archives: dict[str, RemoteZip], cache_dir: Path, fps: float, workers: int
) -> dict[str, int]:
    """Extract every clip not already cached. Returns counts of what happened."""
    skipped_file = cache_dir / "no_hands.json"
    no_hands = set(json.loads(skipped_file.read_text())) if skipped_file.exists() else set()
    todo = [c for c in clips if c.clip_id not in no_hands and not (cache_dir / f"{c.clip_id}.npy").exists()]
    print(f"{len(clips)} clips in the vocabulary: {len(clips) - len(todo)} already done, {len(todo)} to process.")
    outcome = Counter()
    if not todo:
        return outcome

    with tempfile.TemporaryDirectory(prefix="signspeak-") as tmp:

        def fetch(clip: Clip) -> Path:
            path = Path(tmp) / f"{clip.clip_id}{Path(clip.member).suffix}"
            path.write_bytes(archives[clip.archive].read(clip.member))
            return path

        started = time.perf_counter()
        for done, (clip, video, error) in enumerate(_prefetched(fetch, todo, workers), 1):
            if error is not None:
                outcome["download_failed"] += 1
                status = f"download failed ({error!r:.80}), will retry on the next run"
            else:
                try:
                    sequence = extract_clip(video, fps)
                finally:
                    video.unlink(missing_ok=True)
                if sequence is None:
                    outcome["no_hands"] += 1
                    no_hands.add(clip.clip_id)
                    skipped_file.write_text(json.dumps(sorted(no_hands)))
                    status = "skipped: no hands tracked"
                else:
                    outcome["extracted"] += 1
                    np.save(cache_dir / f"{clip.clip_id}.npy", sequence.astype(np.float32))
                    status = f"{len(sequence)} frames"
            remaining_min = (time.perf_counter() - started) / done * (len(todo) - done) / 60
            print(f"[{done}/{len(todo)}] {clip.split:<5} {clip.label:<14} {status}  (~{remaining_min:.0f} min left)", flush=True)
    return outcome


def _prefetched(fn, items, workers: int):
    """Yield (item, result, error) in order, running fn on up to 2 * workers items ahead."""
    items = iter(items)
    with ThreadPoolExecutor(workers) as pool:
        pending = deque((item, pool.submit(fn, item)) for item in islice(items, 2 * workers))
        while pending:
            item, future = pending.popleft()
            upcoming = next(items, None)
            if upcoming is not None:
                pending.append((upcoming, pool.submit(fn, upcoming)))
            try:
                yield item, future.result(), None
            except Exception as error:  # noqa: BLE001 - reported per clip, run continues
                yield item, None, error


# ---------------------------------------------------------------- tensors


def build_splits(
    clips: list[Clip], cache_dir: Path, labels: list[str], max_frames: int, right_dominant: bool = True
) -> dict[str, dict]:
    """Pad each split's cached sequences into one tensor.

    features: (N, T, FEATURE_DIM) float32, zero-padded after each clip's length
    lengths:  (N,) valid frames per clip
    labels:   (N,) index into label_names
    mirrored: (N,) True where the clip was mirrored to make it right-hand dominant

    With right_dominant, clips where the left hand does most of the moving are
    mirrored, so the dominant hand is always in the right_hand slot whatever the
    signer's handedness or whether the recording was mirrored. Live segments get
    the same rule in src.recognition.model.SignRecognizer.
    """
    index = {label: i for i, label in enumerate(labels)}
    splits = {}
    for split in SPLITS:
        sequences, targets, ids, mirrored = [], [], [], []
        for clip in clips:
            path = cache_dir / f"{clip.clip_id}.npy"
            if clip.split != split or not path.exists():
                continue
            sequence = center_crop(np.load(path), max_frames)
            flip = False
            if right_dominant:
                sequence, flip = to_right_dominant(sequence)
            sequences.append(sequence)
            mirrored.append(flip)
            targets.append(index[clip.label])
            ids.append(clip.clip_id)
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        features = torch.zeros(len(sequences), int(lengths.max()) if sequences else 0, FEATURE_DIM)
        for i, sequence in enumerate(sequences):
            features[i, : len(sequence)] = torch.from_numpy(sequence)
        splits[split] = {
            "features": features,
            "lengths": lengths,
            "labels": torch.tensor(targets, dtype=torch.long),
            "mirrored": torch.tensor(mirrored, dtype=torch.bool),
            "clip_ids": ids,
        }
    return splits


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--signs", nargs="+", metavar="LABEL", help="only these labels (default: whole vocabulary)")
    parser.add_argument("--max-per-sign", type=int, help="at most this many clips per sign per split (quick trials)")
    parser.add_argument("--fps", type=float, default=15.0, help="frame rate to resample to; match the live tracker")
    parser.add_argument("--max-frames", type=int, default=90, help="longer clips are center-cropped to this")
    parser.add_argument(
        "--keep-handedness", action="store_true", help="don't mirror left-hand-dominant clips to right-hand dominant"
    )
    parser.add_argument("--workers", type=int, help="parallel clip downloads")
    parser.add_argument("--zip-dir", type=Path, help="folder with already-downloaded dataset zips (default: stream)")
    parser.add_argument("--out", type=Path, help="output folder (default: data/processed/<dataset>)")
    args = parser.parse_args(argv)

    info = DATASETS[args.dataset]
    vocab = ASL_CITIZEN if args.dataset == "asl_citizen" else INCLUDE
    labels = args.signs or list(vocab)
    unknown = [label for label in labels if label not in vocab]
    if unknown:
        parser.error(f"not in the {args.dataset} vocabulary: {unknown}. Choose from: {list(vocab)}")
    subset = {label: vocab[label] for label in labels}

    out_dir = args.out or ROOT / "data" / "processed" / args.dataset
    cache_dir = out_dir / f"clips_{args.fps:g}fps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {info.language}, {len(labels)} signs\nLicense: {info.license}\n")

    lister = list_asl_citizen if args.dataset == "asl_citizen" else list_include
    clips, archives = lister(subset, args.zip_dir)
    clips = cap_per_sign(clips, args.max_per_sign)
    absent = [label for label in labels if not any(c.label == label for c in clips)]
    if absent:
        print(f"warning: no clips found for {absent}")

    outcome = process_clips(clips, archives, cache_dir, args.fps, args.workers or info.download_workers)
    splits = build_splits(clips, cache_dir, labels, args.max_frames, right_dominant=not args.keep_handedness)

    meta = {
        "dataset": args.dataset,
        "language": info.language,
        "license": info.license,
        "citation": info.citation,
        "source": str(args.zip_dir) if args.zip_dir else "streamed from the official download",
        "fps": args.fps,
        "feature_dim": FEATURE_DIM,
        "feature_layout": {name: [s.start, s.stop] for name, s in LAYOUT.items()},
        "max_frames": args.max_frames,
        "label_names": labels,
        "dataset_labels": {label: subset[label] for label in labels},
        "clips_per_split": {split: len(data["labels"]) for split, data in splits.items()},
        "clips_per_label": {
            split: {labels[i]: n for i, n in sorted(Counter(data["labels"].tolist()).items())}
            for split, data in splits.items()
        },
        "last_run": dict(outcome),
        "right_hand_dominant": not args.keep_handedness,
        "mirrored_fraction": {
            split: round(float(data["mirrored"].float().mean()), 3) if len(data["mirrored"]) else 0.0
            for split, data in splits.items()
        },
    }
    for split, data in splits.items():
        torch.save({**data, "label_names": labels, "fps": args.fps, "feature_dim": FEATURE_DIM}, out_dir / f"{split}.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved to {out_dir}:")
    for split, data in splits.items():
        shape = tuple(data["features"].shape)
        print(f"  {split}.pt  {shape[0]:5d} clips  features {shape}  mirrored to right-hand dominant: {meta['mirrored_fraction'][split]:.0%}")
    thin = [label for label in labels if meta["clips_per_label"]["train"].get(label, 0) < 5]
    if thin:
        print(f"  fewer than 5 training clips: {thin}")
    if outcome.get("download_failed"):
        print(f"  {outcome['download_failed']} downloads failed; run the same command again to retry them.")


if __name__ == "__main__":
    main()
