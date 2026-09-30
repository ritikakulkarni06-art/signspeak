"""Dataset preparation tests (no network, camera, or model needed).

Run from the project root: python -m unittest discover -s tests
"""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from data.prepare_dataset import Clip, FramePicker, build_splits, cap_per_sign
from data.remote_zip import RemoteZip
from src.capture.features import FEATURE_DIM, LAYOUT
from src.capture.sequence import trim_to_hands


class RemoteZipTests(unittest.TestCase):
    def test_reads_stored_and_deflated_members(self):
        payload = bytes(range(256)) * 5000
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("clips/", "")
                archive.writestr("clips/stored.mov", payload, compress_type=zipfile.ZIP_STORED)
                archive.writestr("clips/deflated.mov", payload, compress_type=zipfile.ZIP_DEFLATED)
            remote = RemoteZip(path)
            self.assertEqual(sorted(remote.names), ["clips/deflated.mov", "clips/stored.mov"])
            self.assertEqual(remote.read("clips/stored.mov"), payload)
            self.assertEqual(remote.read("clips/deflated.mov"), payload)


class FramePickerTests(unittest.TestCase):
    @staticmethod
    def kept(source_fps, target_fps, frames=60):
        picker = FramePicker(source_fps, target_fps)
        return [i for i in range(frames) if picker.keep(i)]

    def test_halves_30_to_15(self):
        self.assertEqual(self.kept(30, 15), list(range(0, 60, 2)))

    def test_25_to_15_keeps_about_three_fifths(self):
        self.assertAlmostEqual(len(self.kept(25, 15)), 36, delta=1)

    def test_never_upsamples(self):
        self.assertEqual(self.kept(10, 15), list(range(60)))


def _sequence(frames, hand_frames=()):
    sequence = np.zeros((frames, FEATURE_DIM), np.float32)
    for i in hand_frames:
        sequence[i, LAYOUT["right_hand.present"]] = 1.0
    return sequence


class TrimTests(unittest.TestCase):
    def test_trims_to_hands_with_margin(self):
        trimmed = trim_to_hands(_sequence(20, range(5, 10)), margin=2)
        self.assertEqual(len(trimmed), 9)  # frames 3..11

    def test_no_hands_is_dropped(self):
        self.assertIsNone(trim_to_hands(_sequence(20)))


class BuildSplitsTests(unittest.TestCase):
    def test_pads_and_crops_per_split(self):
        clips = [
            Clip("a", "yes", "train", "", ""),
            Clip("b", "no", "train", "", ""),
            Clip("c", "yes", "test", "", ""),
            Clip("missing", "no", "val", "", ""),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            np.save(cache / "a.npy", np.ones((4, FEATURE_DIM), np.float32))
            np.save(cache / "b.npy", np.ones((12, FEATURE_DIM), np.float32))
            np.save(cache / "c.npy", np.ones((5, FEATURE_DIM), np.float32))
            splits = build_splits(clips, cache, ["yes", "no"], max_frames=8)

        train = splits["train"]
        self.assertEqual(tuple(train["features"].shape), (2, 8, FEATURE_DIM))
        self.assertEqual(train["lengths"].tolist(), [4, 8])
        self.assertEqual(train["labels"].tolist(), [0, 1])
        self.assertEqual(float(train["features"][0, 4:].abs().sum()), 0.0)  # padding
        self.assertEqual(len(splits["val"]["labels"]), 0)
        self.assertEqual(splits["test"]["clip_ids"], ["c"])

    def test_left_hand_dominant_clips_are_mirrored(self):
        left_signer = np.zeros((6, FEATURE_DIM), np.float32)
        left_signer[:, LAYOUT["pose.present"]] = 1.0
        left_signer[:, LAYOUT["left_hand.present"]] = 1.0
        left_signer[:, LAYOUT["left_hand.location"].start] = np.linspace(0.2, 0.8, 6)  # moving hand
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "a.npy", left_signer)
            clips = [Clip("a", "yes", "train", "", "")]
            as_recorded = build_splits(clips, Path(tmp), ["yes"], 90, right_dominant=False)["train"]
            canonical = build_splits(clips, Path(tmp), ["yes"], 90)["train"]

        self.assertFalse(as_recorded["mirrored"][0])
        self.assertTrue(canonical["mirrored"][0])
        clip = canonical["features"][0].numpy()
        self.assertTrue((clip[:, LAYOUT["right_hand.present"]] == 1).all())
        self.assertFalse(clip[:, LAYOUT["left_hand.present"]].any())
        np.testing.assert_allclose(clip[:, LAYOUT["right_hand.location"].start], -np.linspace(0.2, 0.8, 6))

    def test_cap_per_sign_is_per_split(self):
        clips = [Clip(f"{s}{i}", "yes", s, "", "") for s in ("train", "test") for i in range(5)]
        self.assertEqual(len(cap_per_sign(clips, 2)), 4)


if __name__ == "__main__":
    unittest.main()
