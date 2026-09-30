"""Sign classifier: a sequence of per-frame feature vectors -> one vocabulary sign.

Architecture, sized for ~15 training clips per sign:
    per frame:  standardize -> feature dropout -> Linear(356 -> hidden), LayerNorm, GELU
    sequence:   bidirectional LSTM over the valid frames (padding is packed away)
    pooling:    mean and max over time, concatenated
    head:       LayerNorm -> dropout -> Linear(-> number of signs)

An LSTM rather than a transformer: with this little data, recurrent models get
good accuracy with far less data and tuning, and an epoch takes seconds on a
CPU. The standardization statistics are stored in the model, so a checkpoint is
all you need to run it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.capture import FEATURE_DIM
from src.capture.sequence import center_crop, to_right_dominant, trim_to_hands


@dataclass
class ModelConfig:
    num_classes: int
    feature_dim: int = FEATURE_DIM
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.4
    feature_dropout: float = 0.1


class SignClassifier(nn.Module):
    def __init__(
        self, config: ModelConfig, feature_mean: torch.Tensor | None = None, feature_std: torch.Tensor | None = None
    ):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim) if feature_mean is None else feature_mean)
        self.register_buffer("feature_std", torch.ones(config.feature_dim) if feature_std is None else feature_std)
        self.feature_dropout = nn.Dropout(config.feature_dropout)
        self.frame_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Dropout(config.dropout), nn.Linear(4 * hidden, config.num_classes))

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """features: (batch, frames, feature_dim), zero-padded; lengths: (batch,)
        valid frames per clip (each >= 1). Returns logits (batch, num_classes)."""
        frames = features.shape[1]
        valid = (torch.arange(frames, device=features.device)[None, :] < lengths[:, None])[..., None]
        x = self.frame_encoder(self.feature_dropout((features - self.feature_mean) / self.feature_std))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=frames)
        mean = (out * valid).sum(1) / lengths[:, None].clamp(min=1)
        peak = out.masked_fill(~valid, float("-inf")).max(1).values
        return self.head(torch.cat([mean, peak], dim=1))


def save_checkpoint(path: Path, model: SignClassifier, label_names: list[str], **extra) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "label_names": label_names, **extra}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[SignClassifier, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = SignClassifier(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


class SignRecognizer:
    """Classifies a live segment (from SignSegmenter) the way training clips
    were prepared: trimmed to where hands are visible, mirrored to right-hand
    dominant, and center-cropped. Segments should come from the tracker at the
    checkpoint's frame rate (checkpoint["fps"])."""

    def __init__(self, checkpoint_path: Path):
        self.model, checkpoint = load_checkpoint(checkpoint_path)
        self.label_names: list[str] = checkpoint["label_names"]
        self.max_frames: int = checkpoint["max_frames"]
        self.right_dominant: bool = checkpoint["right_dominant"]
        self.fps: float = checkpoint["fps"]

    @torch.no_grad()
    def predict(self, sequence: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Top-k (label, probability) pairs, best first. Empty if no hands were seen."""
        prepared = trim_to_hands(sequence)
        if prepared is None:
            return []
        if self.right_dominant:
            prepared, _ = to_right_dominant(prepared)
        prepared = center_crop(prepared, self.max_frames)
        logits = self.model(torch.from_numpy(np.ascontiguousarray(prepared, np.float32))[None], torch.tensor([len(prepared)]))
        best = logits.softmax(-1)[0].topk(min(top_k, len(self.label_names)))
        return [(self.label_names[i], float(p)) for p, i in zip(best.values, best.indices)]
