"""Classifier tests on random data (no dataset, camera, or model file needed).

Run from the project root: python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.capture.features import FEATURE_DIM, LAYOUT
from src.recognition.model import ModelConfig, SignClassifier, SignRecognizer, save_checkpoint
from src.recognition.train import _PRESENCE, augment


def _clip(frames: int, seed: int = 0) -> np.ndarray:
    clip = np.random.default_rng(seed).normal(size=(frames, FEATURE_DIM)).astype(np.float32)
    clip[:, _PRESENCE] = 1.0
    return clip


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SignClassifier(ModelConfig(num_classes=5)).eval()

    def test_output_shape(self):
        logits = self.model(torch.randn(3, 20, FEATURE_DIM), torch.tensor([20, 12, 5]))
        self.assertEqual(tuple(logits.shape), (3, 5))

    def test_padding_does_not_change_the_prediction(self):
        short = torch.from_numpy(_clip(8))
        alone = self.model(short[None], torch.tensor([8]))
        padded = torch.zeros(2, 30, FEATURE_DIM)
        padded[0, :8] = short
        padded[1] = torch.from_numpy(_clip(30, seed=1))
        batched = self.model(padded, torch.tensor([8, 30]))
        torch.testing.assert_close(batched[0], alone[0], atol=1e-5, rtol=1e-5)

    def test_checkpoint_round_trip_and_recognizer(self):
        labels = ["yes", "no", "hello", "help", "thank you"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, self.model, labels, fps=15.0, max_frames=90, right_dominant=True, epoch=1, val_accuracy=0.5)
            recognizer = SignRecognizer(path)
        clip = _clip(20)
        with torch.no_grad():
            expected = self.model(torch.from_numpy(clip)[None], torch.tensor([20])).softmax(-1)[0]
        top = recognizer.predict(clip, top_k=3)
        self.assertEqual([label for label, _ in top], [labels[i] for i in expected.topk(3).indices])
        self.assertAlmostEqual(top[0][1], float(expected.max()), places=5)

    def test_recognizer_ignores_segments_without_hands(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, self.model, list("abcde"), fps=15.0, max_frames=90, right_dominant=True, epoch=1, val_accuracy=0.5)
            recognizer = SignRecognizer(path)
        self.assertEqual(recognizer.predict(np.zeros((10, FEATURE_DIM), np.float32)), [])


class AugmentTests(unittest.TestCase):
    def test_keeps_presence_flags_and_a_usable_length(self):
        rng = np.random.default_rng(0)
        clip = _clip(20)
        for _ in range(50):
            out = augment(clip, rng, np.full(FEATURE_DIM, 0.1, np.float32))
            self.assertGreaterEqual(len(out), 3)
            self.assertTrue((out[:, LAYOUT["right_hand.present"]] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
