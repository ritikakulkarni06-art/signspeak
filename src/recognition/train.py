"""Train the sign classifier on data/processed/<dataset>/ and keep the best checkpoint.

Run from the project root:
    python -m src.recognition.train asl_citizen
    python -m src.recognition.train asl_citizen --num-signs 10     # smaller vocabulary, for comparison

Each epoch logs train/val loss and accuracy to the console and to
models/<name>_train_log.csv. The checkpoint with the best validation accuracy
is saved to models/<name>.pt (default name: <dataset>_sign_classifier).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.capture import FEATURE_DIM, LAYOUT
from src.recognition.model import ModelConfig, SignClassifier, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
Clip = tuple[np.ndarray, int]  # (frames x FEATURE_DIM, label index)

_PRESENCE = np.zeros(FEATURE_DIM, bool)
for _name, _slice in LAYOUT.items():
    if _name.endswith(".present"):
        _PRESENCE[_slice] = True


def load_clips(data_dir: Path, split: str, keep_labels: list[str] | None = None) -> tuple[list[Clip], list[str]]:
    """Clips of one split as unpadded arrays. With keep_labels, only those signs,
    re-indexed in that order (matching is by name, so vocabularies can differ)."""
    data = torch.load(data_dir / f"{split}.pt")
    names = data["label_names"]
    clips = [(data["features"][i, :n].numpy(), int(data["labels"][i])) for i, n in enumerate(data["lengths"].tolist())]
    if keep_labels is not None:
        new_index = {names.index(label): k for k, label in enumerate(keep_labels) if label in names}
        clips = [(sequence, new_index[label]) for sequence, label in clips if label in new_index]
        names = list(keep_labels)
    return clips, names


def subsample_per_sign(clips: list[Clip], fraction: float, rng: np.random.Generator) -> list[Clip]:
    """Keep a random `fraction` of each sign's clips (at least one)."""
    by_sign: dict[int, list[Clip]] = {}
    for clip in clips:
        by_sign.setdefault(clip[1], []).append(clip)
    kept = []
    for group in by_sign.values():
        count = max(1, round(len(group) * fraction))
        kept += [group[i] for i in rng.permutation(len(group))[:count]]
    return kept


def feature_stats(clips: list[Clip]) -> tuple[torch.Tensor, torch.Tensor]:
    frames = np.concatenate([sequence for sequence, _ in clips])
    mean, std = frames.mean(0), frames.std(0)
    std[std < 1e-4] = 1.0  # constant features (e.g. always-present pose) pass through unscaled
    return torch.from_numpy(mean).float(), torch.from_numpy(std).float()


def augment(sequence: np.ndarray, rng: np.random.Generator, noise_std: np.ndarray) -> np.ndarray:
    """Variation a new signer or take would add: signing speed (0.8-1.25x),
    where the clip starts and ends, and small landmark jitter."""
    count = len(sequence)
    resampled = np.round(np.linspace(0, count - 1, max(3, round(count * rng.uniform(0.8, 1.25))))).astype(int)
    sequence = sequence[resampled]
    keep = max(3, round(len(sequence) * rng.uniform(0.8, 1.0)))
    start = rng.integers(0, len(sequence) - keep + 1)
    sequence = sequence[start : start + keep].copy()
    jitter = rng.normal(0.0, 1.0, sequence.shape).astype(np.float32) * noise_std
    sequence[:, ~_PRESENCE] += jitter[:, ~_PRESENCE]
    return sequence


def batches(clips: list[Clip], batch_size: int, rng: np.random.Generator | None = None, noise_std=None):
    """Padded (features, lengths, labels) batches; shuffled and augmented when rng is given."""
    order = rng.permutation(len(clips)) if rng is not None else np.arange(len(clips))
    for start in range(0, len(order), batch_size):
        chunk = [clips[i] for i in order[start : start + batch_size]]
        sequences = [augment(s, rng, noise_std) if rng is not None else s for s, _ in chunk]
        lengths = torch.tensor([len(s) for s in sequences])
        features = torch.zeros(len(sequences), int(lengths.max()), FEATURE_DIM)
        for i, sequence in enumerate(sequences):
            features[i, : len(sequence)] = torch.from_numpy(sequence)
        yield features, lengths, torch.tensor([label for _, label in chunk])


def run_epoch(model, clips, loss_fn, batch_size, optimizer=None, rng=None, noise_std=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for features, lengths, labels in batches(clips, batch_size, rng if training else None, noise_std):
            logits = model(features, lengths)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(clips), correct / len(clips)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="folder name under data/processed, e.g. asl_citizen")
    parser.add_argument("--data-dir", type=Path, help="default: data/processed/<dataset>")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--name", help="checkpoint name (default: <dataset>_sign_classifier)")
    parser.add_argument("--num-signs", type=int, help="train on only the first N signs of the vocabulary")
    parser.add_argument(
        "--train-fraction", type=float, default=1.0, help="use this share of each sign's training clips (learning curves)"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=40, help="stop after this many epochs without a better val accuracy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05, help="jitter, as a fraction of each feature's std")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data_dir = args.data_dir or ROOT / "data" / "processed" / args.dataset
    meta_file = data_dir / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    all_labels = torch.load(data_dir / "train.pt")["label_names"]
    labels = all_labels[: args.num_signs] if args.num_signs else all_labels
    name = args.name or (f"{args.dataset}_{len(labels)}signs_sign_classifier" if args.num_signs else f"{args.dataset}_sign_classifier")
    train, _ = load_clips(data_dir, "train", labels)
    if args.train_fraction < 1.0:
        train = subsample_per_sign(train, args.train_fraction, rng)
    val, _ = load_clips(data_dir, "val", labels)
    if not train or not val:
        raise SystemExit(f"need train and val clips in {data_dir}; got {len(train)} train, {len(val)} val")

    mean, std = feature_stats(train)
    config = ModelConfig(num_classes=len(labels), hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout)
    model = SignClassifier(config, mean, std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    noise_std = (args.noise * std).numpy()
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"{args.dataset}: {len(labels)} signs | {len(train)} train / {len(val)} val clips | "
        f"{parameters / 1e3:.0f}k parameters | chance = {1 / len(labels):.1%}"
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{name}.pt"
    log_path = args.models_dir / f"{name}_train_log.csv"
    best = (-1.0, float("inf"))  # (val accuracy, val loss)
    best_epoch = 0
    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss, train_acc = run_epoch(model, train, loss_fn, args.batch_size, optimizer, rng, noise_std)
            val_loss, val_acc = run_epoch(model, val, loss_fn, args.batch_size)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            seconds = time.perf_counter() - started
            improved = (val_acc, -val_loss) > (best[0], -best[1])
            if improved:
                best, best_epoch = (val_acc, val_loss), epoch
                save_checkpoint(
                    checkpoint_path,
                    model,
                    labels,
                    dataset=args.dataset,
                    fps=meta.get("fps", 15.0),
                    max_frames=meta.get("max_frames", 90),
                    right_dominant=meta.get("right_dominant", True),
                    epoch=epoch,
                    val_accuracy=val_acc,
                    train_args=vars(args) | {"data_dir": str(data_dir), "models_dir": str(args.models_dir)},
                )
            log.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{lr:.6f}", f"{seconds:.1f}"])
            log_file.flush()
            print(
                f"epoch {epoch:3d}  train loss {train_loss:.3f} acc {train_acc:6.1%}  |  "
                f"val loss {val_loss:.3f} acc {val_acc:6.1%}  {'* saved' if improved else ''}",
                flush=True,
            )
            if epoch - best_epoch >= args.patience:
                print(f"No val improvement for {args.patience} epochs; stopping.")
                break

    print(f"\nBest val accuracy {best[0]:.1%} at epoch {best_epoch} -> {checkpoint_path}")
    print(f"Per-epoch log: {log_path}")
    print(f"Next: python -m src.recognition.evaluate {args.dataset} --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
