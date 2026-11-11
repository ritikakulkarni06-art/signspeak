"""Check whether the Claude translation path really works, and what it costs.

Run from the project root:
    python -m demo.check_translation_api

The Stage 5 unit tests use a stub client, so they pass whether or not the API
works. This script talks to the real API, and it fails loudly instead of quietly
falling back to the offline rules the way the live pipeline deliberately does.

It reports, in order:
  1. where the key comes from and whether it is well formed (never printing it),
  2. whether the API accepts it, with the exact error if not,
  3. every Stage 5 gloss sequence translated by the model, marked API or FALLBACK,
  4. per-call latency: first call, warm calls, cached repeats,
  5. what that would add to the measured end-to-end latency.

Exit code 0 only if every sequence came from the API.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

from src.translation.examples import format_gloss  # noqa: F401  (imported for parity with the real prompt)
from src.translation.rules import RuleBasedTranslator
from src.translation.translator import ClaudeTranslator

# The hand-written sequences from the Stage 5 tests (tests/test_translation.py).
SEQUENCES = [
    ["hello", "name", "what"],
    ["help", "need", "please"],
    ["time", "more", "need"],
    ["next", "discuss", "what"],
    ["sign", "language", "learn", "easy", "no"],
    ["idea", "good", "think"],
    ["example", "three", "important"],
    ["finish", "thank you"],
    ["you", "deaf", "hearing", "which"],
    ["why", "you", "sad"],
    ["he", "understand", "no"],
    ["nine", "language", "bad", "six"],  # a recognition error: content must survive
]
PIPELINE_MEDIAN_MS = 750  # measured end-to-end with offline translation (translate ~0 ms)
TARGET_MS = 1500


class CountingClient:
    """Wraps the real client so we can tell an API answer from a silent fallback."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls += 1
            return self._outer._inner.messages.create(**kwargs)


def describe_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  ANTHROPIC_API_KEY is not set in this process.")
        print("  Set it for future terminals with:  setx ANTHROPIC_API_KEY sk-ant-...")
        return None
    problems = []
    if not key.startswith("sk-ant-"):
        problems.append("does not start with sk-ant-")
    if key.strip() != key:
        problems.append("has leading or trailing whitespace")
    if any(c in "\"'" for c in key):
        problems.append("contains quote characters")
    shape = "looks well formed" if not problems else "SUSPICIOUS: " + ", ".join(problems)
    print(f"  key found: {len(key)} chars, starts {key[:12]}..., ends ...{key[-4:]} ({shape})")
    return key


def check_auth(client) -> str | None:
    """None on success, else the error text."""
    try:
        client.messages.create(model="claude-opus-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
        return None
    except Exception as error:  # noqa: BLE001 - reported to the user verbatim
        return str(error)


def build_client():
    """The real API client. Replaced by the tests to exercise the success path."""
    import anthropic

    return anthropic.Anthropic(timeout=30, max_retries=0)


def main() -> int:
    print("1. Key")
    if describe_key() is None:
        return 2

    print("\n2. Authentication")
    started = time.perf_counter()
    failure = check_auth(build_client())
    elapsed = (time.perf_counter() - started) * 1000
    if failure is not None:
        print(f"  REJECTED after {elapsed:.0f} ms: {failure[:300]}")
        print("\n  The key itself is being refused, not the request format. Mint a new key at")
        print("  https://console.anthropic.com/settings/keys, then set ANTHROPIC_API_KEY and re-run.")
        print("  The demo is unaffected: it runs on the offline rules by default.")
        return 1
    print(f"  accepted ({elapsed:.0f} ms, including TLS setup)")

    counting = CountingClient(build_client())
    translator = ClaudeTranslator(client=counting, model=os.environ.get("SIGNSPEAK_TRANSLATION_MODEL", "claude-opus-5"))
    rules = RuleBasedTranslator()

    print("\n3. Stage 5 sequences through the model")
    latencies, from_api = [], 0
    for gloss in SEQUENCES:
        before = counting.calls
        started = time.perf_counter()
        sentence = translator.translate(list(gloss))
        took = (time.perf_counter() - started) * 1000
        called = counting.calls > before
        from_api += called
        latencies.append(took) if called else None
        mark = "API " if called else "FALLBACK"
        same = "  (same as rules)" if called and sentence == rules.translate(list(gloss)) else ""
        print(f"  {mark} {took:6.0f} ms  {' '.join(gloss):<38} -> {sentence!r}{same}")

    print("\n4. Latency")
    if latencies:
        first, rest = latencies[0], latencies[1:]
        print(f"  first call {first:.0f} ms" + (f"; warm calls median {statistics.median(rest):.0f} ms, "
              f"min {min(rest):.0f}, max {max(rest):.0f}" if rest else ""))  # fmt: skip
        started = time.perf_counter()
        translator.translate(list(SEQUENCES[0]))  # repeat: served from the in-process cache
        print(f"  repeated sequence (cached, no call): {(time.perf_counter() - started) * 1000:.1f} ms")
        warm = statistics.median(rest) if rest else first
        print(f"\n5. Effect on end-to-end latency")
        print(f"  measured pipeline median with offline rules: {PIPELINE_MEDIAN_MS} ms (translation ~0 ms)")
        print(f"  with the API instead: about {PIPELINE_MEDIAN_MS + warm:.0f} ms "
              f"({'still under' if PIPELINE_MEDIAN_MS + warm < TARGET_MS else 'OVER'} the {TARGET_MS} ms target)")

    if from_api != len(SEQUENCES):
        print(f"\n  {len(SEQUENCES) - from_api} of {len(SEQUENCES)} sequences fell back to the offline rules.")
        return 1
    print(f"\n  All {from_api} sequences came from the API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
