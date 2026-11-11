"""Rehearsal log tests. No pipeline: trace records are written by hand.

Run from the project root: python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from demo import rehearse


def trace(labels_and_confidence, sentence="A sentence.", latency=800.0, audio=True, reason="gap"):
    return {
        "signs": [{"label": label, "confidence": c} for label, c in labels_and_confidence],
        "sentence": sentence,
        "audio_played": audio,
        "flush_reason": reason,
        "stages_ms": {"end_to_end": latency},
    }


class AttemptTests(unittest.TestCase):
    def test_row_takes_the_machine_numbers_from_the_trace(self):
        row = rehearse.Attempt("hello", trace([("hello", 0.91)], "Hello!", 812.4), audio_ok=True).row()
        self.assertEqual(row["recognized"], "hello")
        self.assertEqual(row["confidence"], "0.91")
        self.assertEqual(row["sentence"], "Hello!")
        self.assertEqual((row["latency_ms"], row["correct"], row["detected"], row["audio_ok"]), (812, "yes", "yes", "yes"))

    def test_a_miss_is_recorded_with_no_numbers(self):
        row = rehearse.Attempt("four", None, note="never picked it up").row()
        self.assertEqual((row["recognized"], row["latency_ms"], row["detected"], row["correct"]), ("", "", "no", "no"))
        self.assertEqual(row["note"], "never picked it up")

    def test_wrong_recognition_is_not_correct(self):
        self.assertFalse(rehearse.Attempt("easy", trace([("good", 0.4)])).correct)

    def test_multi_sign_attempt_must_match_in_order(self):
        self.assertTrue(rehearse.Attempt("help need", trace([("help", 0.8), ("need", 0.7)])).correct)
        self.assertFalse(rehearse.Attempt("help need", trace([("need", 0.7), ("help", 0.8)])).correct)


class TraceReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patch = mock.patch.object(rehearse, "LOG_DIR", self.dir)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, record):
        path = self.dir / name
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        return path

    def test_reads_only_records_written_after_it_started(self):
        self.write("pipeline_1.jsonl", trace([("old", 0.9)]))
        reader = rehearse.TraceReader(started=0)
        reader.new_records()  # drain what already existed
        self.write("pipeline_1.jsonl", trace([("new", 0.9)]))
        records = reader.new_records()
        self.assertEqual([s["label"] for r in records for s in r["signs"]], ["new"])

    def test_picks_up_a_log_created_after_it_started(self):
        reader = rehearse.TraceReader(started=0)
        self.assertEqual(reader.new_records(), [])
        self.write("pipeline_2.jsonl", trace([("hello", 0.9)]))
        self.assertEqual(len(reader.new_records()), 1)
        self.assertEqual(reader.new_records(), [])  # not a second time

    def test_half_written_line_is_left_for_next_time(self):
        path = self.write("pipeline_3.jsonl", trace([("hello", 0.9)]))
        with open(path, "a", encoding="utf-8") as file:
            file.write('{"signs": [{"label": "par')
        reader = rehearse.TraceReader(started=0, path=path)
        reader._offset = 0
        self.assertEqual(len(reader.new_records()), 1)


class SummaryTests(unittest.TestCase):
    def rows(self, *specs):
        out = []
        for intended, recognized, latency, correct in specs:
            out.append({"time": "t", "intended": intended, "recognized": recognized, "confidence": "",
                        "sentence": "", "audio_ok": "yes", "latency_ms": latency, "correct": correct,
                        "detected": "yes" if recognized else "no", "end_reason": "gap", "note": ""})
        return out

    def test_per_sign_counts_and_failure_reasons(self):
        stats = rehearse.per_sign(self.rows(
            ("hello", "hello", "700", "yes"),
            ("hello", "", "", "no"),
            ("easy", "good", "800", "no"),
            ("easy", "good", "820", "no"),
        ))
        self.assertEqual((stats["hello"]["attempts"], stats["hello"]["correct"], stats["hello"]["misses"]), (2, 1, 1))
        self.assertEqual((stats["easy"]["attempts"], stats["easy"]["correct"]), (2, 0))
        self.assertEqual(stats["easy"]["wrong"], ["good", "good"])

    def test_one_wrong_sign_does_not_fail_the_others_in_a_sentence(self):
        stats = rehearse.per_sign(self.rows(("help need please", "help want please", "900", "no")))
        self.assertEqual(stats["help"]["correct"], 1)
        self.assertEqual(stats["please"]["correct"], 1)
        self.assertEqual(stats["need"]["correct"], 0)

    def test_summary_lists_signs_that_failed_more_than_once(self):
        text = rehearse.summarize(self.rows(
            ("hello", "hello", "700", "yes"),
            ("easy", "good", "800", "no"),
            ("easy", "", "", "no"),
            ("easy", "good", "810", "no"),
        ))
        self.assertIn("Failed more than once", text)
        self.assertIn("easy", text)
        self.assertNotRegex(text, r"hello\s+failed")
        self.assertIn("mean 770 ms", text)

    def test_summary_without_repeated_failures_says_so(self):
        self.assertIn("No sign failed more than once", rehearse.summarize(self.rows(("hello", "hello", "700", "yes"))))

    def test_empty_log(self):
        self.assertIn("No rehearsal attempts", rehearse.summarize([]))


class RoundTripTests(unittest.TestCase):
    def test_rows_written_then_summarized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehearsal_test.csv"
            rehearse.append_row(path, rehearse.Attempt("hello", trace([("hello", 0.9)], latency=700)).row())
            rehearse.append_row(path, rehearse.Attempt("four", None).row())
            text = rehearse.summarize(rehearse.load_rows([path]))
        self.assertIn("2 attempts logged", text)
        self.assertIn("detected at all:       1/2", text)
        self.assertIn("recognized correctly:  1/2", text)


if __name__ == "__main__":
    unittest.main()
