"""Rehearsal log: what you signed, what SignSpeak heard, and whether it worked.

Run it next to the live pipeline, in a second terminal:

    python -m src.pipeline.live          # terminal 1: sign in front of the camera
    python -m demo.rehearse              # terminal 2: log each attempt

For each attempt you type the sign you performed and say whether the audio
sounded right. Everything else - what was recognized, the confidence, the
sentence, whether audio reached the speakers, the latency - is read from the
pipeline's own per-utterance log, so the numbers are measured, not remembered.

Attempts where nothing was recognized are recorded too, as "not detected". They
are the ones a hand-written log quietly loses, and they are what separates "it
mostly worked" from a reliability number you can defend.

Rows are appended to rehearsals/rehearsal_<date>_<time>.csv.

At the end of a session (or any time):

    python -m demo.rehearse summary              # every rehearsal so far
    python -m demo.rehearse summary --file rehearsals/rehearsal_....csv

which prints per-sign success rates, latency, and the signs that failed more
than once.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
REHEARSAL_DIR = ROOT / "rehearsals"
COLUMNS = [
    "time", "intended", "recognized", "confidence", "sentence",
    "audio_ok", "latency_ms", "correct", "detected", "end_reason", "note",
]  # fmt: skip


# ---------------------------------------------------------------- the pipeline's own log


def newest_trace_log(after: float = 0.0) -> Path | None:
    """The most recently written logs/pipeline_*.jsonl, if any."""
    logs = [p for p in LOG_DIR.glob("pipeline_*.jsonl") if p.stat().st_mtime >= after]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


class TraceReader:
    """Reads utterances the pipeline appends to its log while we wait."""

    def __init__(self, path: Path | None = None, started: float | None = None):
        self._started = time.time() if started is None else started
        self._path = path
        self._offset = path.stat().st_size if path and path.exists() else 0

    def new_records(self) -> list[dict]:
        """Every utterance written since the last call (following log rotation)."""
        newest = newest_trace_log(self._started) if self._path is None else self._path
        if newest is None:
            return []
        if newest != self._path:  # the pipeline restarted, or started after us
            self._path, self._offset = newest, 0
        records = []
        with open(self._path, encoding="utf-8") as file:
            file.seek(self._offset)
            for line in file:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:  # half-written line; pick it up next time
                        return records
            self._offset = file.tell()
        return records


# ---------------------------------------------------------------- one row


@dataclass
class Attempt:
    intended: str
    record: dict | None
    audio_ok: bool | None = None
    note: str = ""

    @property
    def recognized(self) -> str:
        if not self.record:
            return ""
        return " ".join(sign["label"] for sign in self.record["signs"])

    @property
    def confidence(self) -> str:
        if not self.record:
            return ""
        return " ".join(f"{sign['confidence']:.2f}" for sign in self.record["signs"])

    @property
    def correct(self) -> bool:
        return bool(self.record) and self.recognized.split() == self.intended.split()

    def row(self) -> dict:
        record = self.record or {}
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "intended": self.intended,
            "recognized": self.recognized,
            "confidence": self.confidence,
            "sentence": record.get("sentence", ""),
            "audio_ok": "" if self.audio_ok is None else ("yes" if self.audio_ok else "no"),
            "latency_ms": round(record["stages_ms"]["end_to_end"]) if record.get("stages_ms") else "",
            "correct": "yes" if self.correct else "no",
            "detected": "yes" if self.record else "no",
            "end_reason": record.get("flush_reason", ""),
            "note": self.note,
        }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------- summary


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def per_sign(rows: list[dict]) -> dict[str, dict]:
    """Attempts and successes for each individual sign.

    A multi-sign attempt is aligned against what was recognized, so one wrong
    sign in a sentence doesn't mark the others wrong.
    """
    stats: dict[str, dict] = {}
    for row in rows:
        intended = row["intended"].split()
        recognized = row["recognized"].split()
        matcher = difflib.SequenceMatcher(a=intended, b=recognized, autojunk=False)
        matched = set()
        for block in matcher.get_matching_blocks():
            matched.update(range(block.a, block.a + block.size))
        for i, sign in enumerate(intended):
            entry = stats.setdefault(sign, {"attempts": 0, "correct": 0, "misses": 0, "wrong": []})
            entry["attempts"] += 1
            if i in matched:
                entry["correct"] += 1
            elif not recognized:
                entry["misses"] += 1
            else:
                entry["wrong"].append(recognized[min(i, len(recognized) - 1)])
    return stats


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "No rehearsal attempts logged yet. Run `python -m demo.rehearse` while you practise."
    lines = []
    latencies = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
    detected = sum(r["detected"] == "yes" for r in rows)
    correct = sum(r["correct"] == "yes" for r in rows)
    audio_bad = sum(r["audio_ok"] == "no" for r in rows)

    lines.append(f"{len(rows)} attempts logged")
    lines.append(f"  detected at all:       {detected}/{len(rows)} ({detected / len(rows):.0%})")
    lines.append(f"  recognized correctly:  {correct}/{len(rows)} ({correct / len(rows):.0%})")
    if audio_bad:
        lines.append(f"  audio sounded wrong:   {audio_bad}")
    if latencies:
        ordered = sorted(latencies)
        over = sum(v >= 1500 for v in latencies)
        lines.append(f"  latency: mean {statistics.mean(latencies):.0f} ms, median {statistics.median(latencies):.0f} ms, "
                     f"worst {max(latencies):.0f} ms; {over} of {len(latencies)} at or over 1.5 s")  # fmt: skip

    stats = per_sign(rows)
    lines.append("\nPer sign (worst first):")
    lines.append(f"  {'sign':<14}{'ok':>6}{'tried':>7}{'rate':>7}   usual problem")
    for sign, entry in sorted(stats.items(), key=lambda kv: (kv[1]["correct"] / kv[1]["attempts"], -kv[1]["attempts"])):
        rate = entry["correct"] / entry["attempts"]
        problem = ""
        if entry["misses"]:
            problem = f"not detected x{entry['misses']}"
        if entry["wrong"]:
            common = max(set(entry["wrong"]), key=entry["wrong"].count)
            problem = (problem + ", " if problem else "") + f"heard as {common}"
        lines.append(f"  {sign:<14}{entry['correct']:>6}{entry['attempts']:>7}{rate:>6.0%}   {problem}")

    unreliable = {s: e for s, e in stats.items() if e["attempts"] - e["correct"] > 1}
    if unreliable:
        lines.append("\nFailed more than once - avoid these on stage, or practise them more:")
        for sign, entry in sorted(unreliable.items(), key=lambda kv: kv[1]["correct"] - kv[1]["attempts"]):
            lines.append(f"  {sign:<14} failed {entry['attempts'] - entry['correct']} of {entry['attempts']} attempts")
    else:
        lines.append("\nNo sign failed more than once.")
    return "\n".join(lines)


# ---------------------------------------------------------------- interactive session


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "q"


def session(path: Path) -> None:
    print(__doc__.split("Rows are appended")[0].strip())
    print(f"\nLogging to {path.relative_to(ROOT)}")
    print("Sign, wait for the voice, then type what you signed. Enter alone repeats the last sign.")
    print("Type 'q' when you are done.\n")
    reader = TraceReader()
    attempts, last_sign = 0, ""
    while True:
        answer = ask(f"[{attempts + 1}] what did you sign? ")
        if answer.lower() in ("q", "quit", "exit"):
            break
        intended = answer or last_sign
        if not intended:
            print("    (type the sign you performed, e.g. 'hello' or 'help need please')")
            continue
        last_sign = intended

        records = reader.new_records()
        record = records[-1] if records else None
        if len(records) > 1:
            print(f"    note: {len(records)} sentences were spoken since the last attempt; logging the last one")
        if record is None:
            print("    NOT DETECTED - nothing was spoken for that attempt (logged as a miss)")
            attempt = Attempt(intended, None, note=ask("    note (optional): "))
        else:
            heard = " ".join(f"{s['label']} {s['confidence']:.0%}" for s in record["signs"])
            latency = record["stages_ms"]["end_to_end"]
            print(f"    heard: {heard}")
            print(f"    said:  {record['sentence']!r}  ({latency:.0f} ms, audio {'played' if record['audio_played'] else 'DID NOT PLAY'})")
            audio = ask("    did the audio sound right? [Y/n] ").lower()
            attempt = Attempt(intended, record, audio_ok=not audio.startswith("n"))
            if not attempt.correct:
                attempt.note = ask("    note (optional): ")
        append_row(path, attempt.row())
        attempts += 1
        print(f"    logged{'' if attempt.correct else '  (counted as a failure)'}\n")

    print(f"\n{attempts} attempts written to {path.relative_to(ROOT)}")
    if attempts:
        print()
        print(summarize(load_rows([path])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    summary = sub.add_parser("summary", help="summarize rehearsals instead of logging new ones")
    summary.add_argument("--file", type=Path, help="one CSV (default: every file in rehearsals/)")
    args = parser.parse_args(argv)

    if args.command == "summary":
        paths = [args.file] if args.file else sorted(REHEARSAL_DIR.glob("rehearsal_*.csv"))
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"No such file: {missing[0]}")
            return 2
        print(summarize(load_rows(paths)))
        return 0

    session(REHEARSAL_DIR / f"rehearsal_{datetime.now():%Y%m%d_%H%M%S}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
