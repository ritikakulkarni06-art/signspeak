"""Order-free baseline for diagnosing the LSTM: each clip is summarized by the
mean, std, and max of its standardized features over time (frame order thrown
away), then a linear classifier is trained on that summary.

    python -m src.recognition.baseline asl_citizen [--num-signs N]

Reading the result against the LSTM:
  * LSTM clearly better  -> motion over time matters and the architecture uses it.
  * both similar and low -> the features or data are the limit, not the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.recognition.train import ROOT, feature_stats, load_clips


def summarize(clips, mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for sequence, _ in clips:
        z = (sequence - mean) / std
        rows.append(np.concatenate([z.mean(0), z.std(0), z.max(0)]))
    return torch.tensor(np.stack(rows), dtype=torch.float32), torch.tensor([label for _, label in clips])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--num-signs", type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    data_dir = args.data_dir or ROOT / "data" / "processed" / args.dataset
    labels = torch.load(data_dir / "train.pt")["label_names"]
    labels = labels[: args.num_signs] if args.num_signs else labels
    splits = {split: load_clips(data_dir, split, labels)[0] for split in ("train", "val", "test")}
    mean, std = (t.numpy() for t in feature_stats(splits["train"]))
    x_train, y_train = summarize(splits["train"], mean, std)

    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(x_train.shape[1], len(labels)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=args.weight_decay)
    best_val, best_state = -1.0, None
    x_val, y_val = summarize(splits["val"], mean, std)
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x_train), y_train, label_smoothing=0.1).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val).argmax(1) == y_val).float().mean().item()
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        accuracy = {split: (model(x).argmax(1) == y).float().mean().item() for split, (x, y) in
                    ((s, summarize(splits[s], mean, std)) for s in ("train", "val", "test"))}  # fmt: skip
    print(
        f"order-free linear baseline, {len(labels)} signs: "
        f"train {accuracy['train']:.1%}  val {accuracy['val']:.1%}  test {accuracy['test']:.1%}"
    )


if __name__ == "__main__":
    main()
