"""TTS tests. SSML and signal mapping are tested offline; rendering tests run
only where a Windows speech engine is available.

Run from the project root: python -m unittest discover -s tests
"""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.capture.features import FEATURE_DIM, LAYOUT
from src.tts import Audio, EmphasisRegion, build_ssml, create_engine, rate_percent, synthesize
from src.tts.prosody import word_strengths
from src.tts.signals import (
    NEUTRAL_NMM,
    TYPICAL_SIGN_S,
    emphasis_for_sentence,
    expression_intensity,
    pace_from_durations,
)

SSML_NS = "{http://www.w3.org/2001/10/synthesis}"


class ProsodyTests(unittest.TestCase):
    def test_rate_follows_pace_and_is_clamped(self):
        self.assertEqual(rate_percent(1.0), 0)
        self.assertEqual(rate_percent(1.3), 30)
        self.assertEqual(rate_percent(0.75), -25)
        self.assertEqual(rate_percent(5.0), 60)
        self.assertEqual(rate_percent(0.1), -40)

    def test_region_forms_overlap_and_bounds(self):
        strengths = word_strengths(5, [1, (2, 4, 0.5), EmphasisRegion(3, 9, 0.8), (-2, 0)])
        self.assertEqual(strengths, [0.0, 1.0, 0.5, 0.8, 0.8])
        with self.assertRaises(TypeError):
            word_strengths(3, ["loud"])

    def test_ssml_is_well_formed_and_escaped(self):
        ssml = build_ssml("Q&A <now> is fine", pace=1.2, emphasis_regions=[(0, 1)])
        root = ET.fromstring(ssml)  # raises if the markup is broken
        outer = root.find(f"{SSML_NS}prosody")
        self.assertEqual(outer.get("rate"), "+20%")
        self.assertIn("Q&A", "".join(root.itertext()))

    def test_emphasis_wraps_only_the_flagged_words(self):
        root = ET.fromstring(build_ssml("I think this idea is really important.", emphasis_regions=[(5, 7, 1.0)]))
        inner = root.find(f"{SSML_NS}prosody/{SSML_NS}prosody")
        self.assertEqual(inner.text, "really important.")
        self.assertEqual((inner.get("pitch"), inner.get("volume"), inner.get("rate")), ("+30%", "+40%", "-15%"))
        self.assertIsNotNone(root.find(f"{SSML_NS}prosody/{SSML_NS}break"))  # strong -> lead-in pause

    def test_weak_emphasis_is_milder_and_has_no_pause(self):
        root = ET.fromstring(build_ssml("one two three", emphasis_regions=[(1, 2, 0.2)]))
        inner = root.find(f"{SSML_NS}prosody/{SSML_NS}prosody")
        self.assertEqual(inner.get("pitch"), "+12%")
        self.assertIsNone(root.find(f"{SSML_NS}prosody/{SSML_NS}break"))


class TrimTests(unittest.TestCase):
    def test_trims_padding_but_keeps_the_speech(self):
        rate = 16000
        speech = 0.5 * np.sin(np.arange(rate // 2) * 2 * np.pi * 200 / rate).astype(np.float32)  # 500 ms
        padded = np.concatenate([np.zeros(rate // 10), speech, np.zeros(rate * 3 // 4)]).astype(np.float32)
        trimmed = Audio(padded, rate).trim_silence(keep_lead_ms=10, keep_tail_ms=80)
        self.assertAlmostEqual(trimmed.duration, 0.5 + 0.010 + 0.080, delta=0.02)
        self.assertEqual(len(Audio(np.zeros(100, np.float32), rate).trim_silence().samples), 0)


def _sequence(frames: int, nmm: np.ndarray | None) -> np.ndarray:
    sequence = np.zeros((frames, FEATURE_DIM), np.float32)
    if nmm is not None:
        sequence[:, LAYOUT["face.present"]] = 1.0
        sequence[:, LAYOUT["face.nmm"]] = nmm
    return sequence


class SignalTests(unittest.TestCase):
    def test_pace_from_sign_durations(self):
        self.assertAlmostEqual(pace_from_durations([TYPICAL_SIGN_S] * 3), 1.0)
        self.assertGreater(pace_from_durations([0.8, 0.9, 0.85]), 1.3)  # fast signer
        self.assertEqual(pace_from_durations([5.0]), 0.6)  # clamped
        self.assertEqual(pace_from_durations([]), 1.0)

    def test_expression_intensity(self):
        self.assertAlmostEqual(expression_intensity(_sequence(10, NEUTRAL_NMM)), 0.0, places=5)
        raised = NEUTRAL_NMM.copy()
        raised[:2] += 0.08  # both brows up by the full-scale amount
        self.assertAlmostEqual(expression_intensity(_sequence(10, raised)), 1.0, places=5)
        self.assertEqual(expression_intensity(_sequence(10, None)), 0.0)  # no face tracked

    def test_emphasis_follows_the_sign_into_the_sentence(self):
        sentence = "Thank you, I think this idea is really important."
        regions = emphasis_for_sentence(sentence, [("important", 0.9), ("idea", 0.2), ("thank you", 0.6), ("banana", 1.0)])
        self.assertEqual(regions, [EmphasisRegion(8, 9, 0.9), EmphasisRegion(0, 2, 0.6)])


def _engine_available() -> bool:
    try:
        create_engine()
        return True
    except RuntimeError:
        return False


@unittest.skipUnless(_engine_available(), "no Windows speech engine")
class SynthesisTests(unittest.TestCase):
    def test_slower_pace_gives_longer_audio(self):
        slow = synthesize("Could you say that again, please?", pace=0.7)
        fast = synthesize("Could you say that again, please?", pace=1.4)
        self.assertGreater(slow.duration, fast.duration * 1.4)

    def test_emphasis_renders_and_saves(self):
        audio = synthesize("I need more time.", pace=1.0, emphasis_regions=[(2, 4, 1.0)])
        self.assertGreater(audio.duration, 0.5)
        self.assertLessEqual(float(np.abs(audio.samples).max()), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            audio.save(path)
            reloaded = Audio.from_wav_bytes(path.read_bytes())
        self.assertEqual(reloaded.sample_rate, audio.sample_rate)
        self.assertEqual(len(reloaded.samples), len(audio.samples))

    def test_empty_text_is_silence(self):
        self.assertEqual(synthesize("   ").duration, 0.0)


if __name__ == "__main__":
    unittest.main()
