"""Collects recognized signs into an utterance and decides when it is complete.

An utterance is released when the signer pauses (no new sign within gap_ms of
the last one), lowers their hands out of view for hands_down_ms, or reaches
max_signs. These waits are latency by design: a sentence can't be translated
until it is known to be over. Lower them to speak sooner, at the risk of
splitting a sentence in two.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trace import SignEvent


@dataclass(frozen=True)
class BufferConfig:
    gap_ms: float = 700.0
    hands_down_ms: float = 350.0
    max_signs: int = 8
    min_confidence: float = 0.25  # below this a prediction is too unsure to say


class GlossBuffer:
    def __init__(self, config: BufferConfig | None = None):
        self.config = config or BufferConfig()
        self._signs: list[SignEvent] = []
        self._hands_gone_since: float | None = None

    @property
    def pending(self) -> list[str]:
        return [sign.label for sign in self._signs]

    def add(self, sign: SignEvent) -> bool:
        """Queue a recognized sign; returns False if it was too unsure to keep."""
        if sign.confidence < self.config.min_confidence:
            return False
        self._signs.append(sign)
        return True

    def poll(self, now_ms: float, hands_visible: bool, signing: bool) -> tuple[list[SignEvent], str] | None:
        """Call every frame. Returns (signs, reason) when the utterance is complete."""
        if hands_visible:
            self._hands_gone_since = None
        elif self._hands_gone_since is None:
            self._hands_gone_since = now_ms
        if not self._signs:
            return None
        if len(self._signs) >= self.config.max_signs:
            return self.flush("max_signs")
        if signing:  # the next sign is already under way
            return None
        if self._hands_gone_since is not None and now_ms - self._hands_gone_since >= self.config.hands_down_ms:
            return self.flush("hands_down")
        if now_ms - self._signs[-1].classified_ms >= self.config.gap_ms:
            return self.flush("gap")
        return None

    def flush(self, reason: str) -> tuple[list[SignEvent], str] | None:
        if not self._signs:
            return None
        signs, self._signs = self._signs, []
        return signs, reason
