"""Hear the prosody controls: one sentence at two pace/emphasis settings, plus a
neutral reference, saved as WAV files.

Run from the project root:
    python -m demo.tts_demo [--play] [--voice Mark]

Writes demo/tts_samples/{0_neutral,1_calm_slow,2_fast_emphatic}.wav.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.tts import EmphasisRegion, build_ssml, create_engine

SENTENCE = "I think this idea is really important."
# Word indices: I(0) think(1) this(2) idea(3) is(4) really(5) important.(6)
SETTINGS = {
    "0_neutral": dict(pace=1.0, emphasis=[]),
    "1_calm_slow": dict(pace=0.75, emphasis=[]),  # slow signing, neutral face
    "2_fast_emphatic": dict(pace=1.35, emphasis=[EmphasisRegion(5, 7, 1.0)]),  # fast signing, raised brows
}
OUT_DIR = Path(__file__).resolve().parent / "tts_samples"


TAIL = 0.28  # "really important." is roughly the last 28% of the spoken sentence


def tail_vs_head(samples: np.ndarray, rate: int) -> tuple[float, float, float]:
    """Median pitch (Hz) of the start of the sentence and of its last words, and
    how much louder (dB) the last words are. The emphasized words sit at the end,
    so this is where emphasis should show up. Medians keep the pitch tracker's
    occasional octave errors from dominating."""
    level = np.abs(samples)
    voiced = np.flatnonzero(level > 0.02 * level.max())
    speech = samples[voiced[0] : voiced[-1] + 1]
    split = int(len(speech) * (1 - TAIL))
    head, tail = speech[:split], speech[split:]

    def median_pitch(segment: np.ndarray) -> float:
        frame, hop = int(0.04 * rate), int(0.01 * rate)
        low, high = int(rate / 300), int(rate / 60)
        pitches = []
        for start in range(0, len(segment) - frame, hop):
            chunk = segment[start : start + frame] * np.hanning(frame)
            if np.sqrt(np.mean(chunk**2)) < 0.05 * np.abs(segment).max():
                continue
            corr = np.correlate(chunk, chunk, "full")[frame - 1 :]
            lag = low + int(np.argmax(corr[low:high]))
            if corr[lag] > 0.4 * corr[0]:
                pitches.append(rate / lag)
        return float(np.median(pitches)) if pitches else float("nan")

    rms = lambda segment: float(np.sqrt(np.mean(segment**2)))  # noqa: E731
    return median_pitch(head), median_pitch(tail), 20 * np.log10(rms(tail) / rms(head))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--play", action="store_true", help="play each sample after writing it")
    parser.add_argument("--voice", help="part of a voice name, e.g. Mark or Zira")
    args = parser.parse_args()

    engine = create_engine(args.voice)
    OUT_DIR.mkdir(exist_ok=True)
    print(f'Engine: {engine.name}, voice: {engine.voice}\nSentence: "{SENTENCE}"\n')
    print(f"{'sample':<16}{'pace':<6}{'emphasis on':<20}{'length':>7}{'pitch start':>13}{'pitch end':>11}{'end louder by':>15}")
    for name, setting in SETTINGS.items():
        audio = engine.render(build_ssml(SENTENCE, setting["pace"], setting["emphasis"]))
        path = OUT_DIR / f"{name}.wav"
        audio.save(path)
        head_pitch, tail_pitch, louder = tail_vs_head(audio.samples, audio.sample_rate)
        words = SENTENCE.split()
        emphasized = ", ".join(" ".join(words[r.start : r.end]) for r in setting["emphasis"]) or "none"
        print(
            f"{name:<16}{setting['pace']:<6}{emphasized:<20}{audio.duration:6.2f}s"
            f"{head_pitch:10.0f} Hz{tail_pitch:8.0f} Hz{louder:12.1f} dB"
        )
        if args.play:
            audio.play()
    print(f"\nFiles in {OUT_DIR}")


if __name__ == "__main__":
    main()
