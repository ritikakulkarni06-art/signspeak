"""Report test-set accuracy and the confusion matrix for a trained sign classifier.

Run from the project root:
    python -m src.recognition.evaluate asl_citizen
    python -m src.recognition.evaluate asl_citizen --checkpoint models/asl_citizen_10signs_sign_classifier.pt

Prints top-1/top-5 accuracy, per-sign accuracy and the most-confused pairs (plus
the full matrix for small vocabularies). Saves the confusion matrix as
models/<name>_confusion_<split>.csv and .png, and a summary .json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.recognition.model import load_checkpoint
from src.recognition.train import ROOT, batches, load_clips

_PRINT_MATRIX_UP_TO = 12  # signs; larger matrices only go to the .csv/.png


@torch.no_grad()
def predict(model, clips, batch_size: int = 64) -> np.ndarray:
    """Class probabilities, (clips, classes)."""
    model.eval()
    return np.concatenate([model(x, lengths).softmax(-1).numpy() for x, lengths, _ in batches(clips, batch_size)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="folder name under data/processed, e.g. asl_citizen")
    parser.add_argument("--checkpoint", type=Path, help="default: models/<dataset>_sign_classifier.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=Path, help="default: data/processed/<dataset>")
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint or ROOT / "models" / f"{args.dataset}_sign_classifier.pt"
    data_dir = args.data_dir or ROOT / "data" / "processed" / args.dataset
    model, checkpoint = load_checkpoint(checkpoint_path)
    labels = checkpoint["label_names"]
    clips, _ = load_clips(data_dir, args.split, labels)
    truth = np.array([label for _, label in clips])
    probs = predict(model, clips)
    predicted = probs.argmax(1)
    top5 = (np.argsort(-probs, axis=1)[:, :5] == truth[:, None]).any(1)

    classes = len(labels)
    confusion = np.zeros((classes, classes), int)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(1)
    per_sign = confusion.diagonal() / support.clip(min=1)
    accuracy = float((predicted == truth).mean())

    print(f"{checkpoint_path.name} on {args.dataset}/{args.split}: {len(clips)} clips, {classes} signs")
    print(f"  top-1 accuracy {accuracy:.1%}   top-5 {top5.mean():.1%}   chance {1 / classes:.1%}")
    print(f"  (best val accuracy during training: {checkpoint['val_accuracy']:.1%}, epoch {checkpoint['epoch']})\n")

    print("Per-sign accuracy (worst first):")
    for i in np.argsort(per_sign):
        if support[i] == 0:
            continue
        wrong = confusion[i].copy()
        wrong[i] = 0
        worst = f"  often -> {labels[wrong.argmax()]} ({wrong.max()})" if wrong.max() else ""
        print(f"  {labels[i]:<14} {per_sign[i]:6.1%}  ({confusion[i, i]}/{confusion[i].sum()}){worst}")
    unseen = [labels[i] for i in range(classes) if support[i] == 0]
    if unseen:
        print(f"  no {args.split} clips for: {', '.join(unseen)}")

    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = sorted(((off_diagonal[t, p], t, p) for t, p in zip(*np.nonzero(off_diagonal))), reverse=True)[:10]
    if pairs:
        print("\nMost confused (true -> predicted):")
        for count, t, p in pairs:
            print(f"  {labels[t]:<14} -> {labels[p]:<14} {count} of {confusion[t].sum()}")

    if classes <= _PRINT_MATRIX_UP_TO:
        width = max(len(label) for label in labels)
        print("\nConfusion matrix (rows = true, columns = predicted):")
        print(" " * (width + 1) + " ".join(f"{label[:4]:>4}" for label in labels))
        for i, label in enumerate(labels):
            print(f"{label:>{width}} " + " ".join(f"{n:4d}" for n in confusion[i]))

    stem = checkpoint_path.with_suffix("")
    with open(f"{stem}_confusion_{args.split}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true \\ predicted", *labels])
        writer.writerows([label, *row] for label, row in zip(labels, confusion.tolist()))
    _plot(confusion, labels, f"{stem}_confusion_{args.split}.png", f"{args.dataset} {args.split}: {accuracy:.1%} top-1")
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "clips": len(clips),
        "signs": classes,
        "top1_accuracy": accuracy,
        "top5_accuracy": float(top5.mean()),
        "per_sign_accuracy": {
            label: (round(float(rate), 4) if n else None) for label, rate, n in zip(labels, per_sign, support)
        },
    }
    Path(f"{stem}_eval_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {stem.name}_confusion_{args.split}.csv/.png and {stem.name}_eval_{args.split}.json in {stem.parent}")


def _plot(confusion: np.ndarray, labels: list[str], path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = confusion / confusion.sum(1, keepdims=True).clip(min=1)
    size = max(6.0, len(labels) * 0.26)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
