"""Gloss sequence -> natural English sentence.

Sign language grammar is not English grammar, so the recognizer's output needs
rewriting before it is spoken: ["hello", "name", "what"] means "Hello! What's
your name?", not "hello name what".

This stage is deliberately swappable. `GlossTranslator` is the interface, and
the prototype ships two implementations:

  * ClaudeTranslator - prompts a pretrained Claude model with a fixed set of
    gloss -> sentence examples (few-shot). Nothing is trained.
  * RuleBasedTranslator (rules.py) - an offline role-based grammar. It is used
    when there is no API key and whenever an API call fails, so the live
    pipeline always says something.

A trained seq2seq model can replace either one: implement `translate()`, then
pass it as `translate_gloss(..., translator=...)` or install it with
`set_default_translator()`.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from .examples import EXAMPLES, SYSTEM_PROMPT, format_gloss
from .rules import RuleBasedTranslator, normalize

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 256  # a sentence; deliberately short
# Models that accept output_config.effort (older/smaller ones return 400).
_EFFORT_MODELS = ("claude-opus-", "claude-sonnet-5", "claude-fable-")


@runtime_checkable
class GlossTranslator(Protocol):
    """Anything that turns gloss tokens into a sentence."""

    def translate(self, gloss: list[str]) -> str: ...


class ClaudeTranslator:
    """Prompts a pretrained Claude model with few-shot gloss -> sentence examples.

    Falls back to `fallback` (rule-based by default) when the API is unreachable,
    refuses, or returns nothing, so a live session never stalls on the network.
    After a failure it stops calling the API for a while (for the whole session
    if the key is invalid or out of credit), so each sentence doesn't pay for a
    doomed request. Repeated gloss sequences are answered from a small cache.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        client=None,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 1,
        effort: str = "low",
        fallback: GlossTranslator | None = None,
        cache_size: int = 256,
        retry_pause: float = 30.0,
    ):
        self.model = model or os.environ.get("SIGNSPEAK_TRANSLATION_MODEL") or DEFAULT_MODEL
        self.effort = effort
        self.fallback = RuleBasedTranslator() if fallback is None else fallback
        self.retry_pause = retry_pause
        self._paused_until = 0.0
        self._cache: OrderedDict[tuple[str, ...], str] = OrderedDict()
        self._cache_size = cache_size
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported here so the offline path doesn't need the SDK

            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    @property
    def api_disabled(self) -> bool:
        """True once an auth or billing failure switched this session to the offline fallback."""
        return self._paused_until == float("inf")

    def build_messages(self, gloss: list[str]) -> list[dict]:
        """Few-shot examples as alternating turns, then the real gloss."""
        messages: list[dict] = []
        for example_gloss, sentence in EXAMPLES:
            messages.append({"role": "user", "content": format_gloss(example_gloss)})
            messages.append({"role": "assistant", "content": sentence})
        messages.append({"role": "user", "content": format_gloss(gloss)})
        return messages

    def translate(self, gloss: list[str]) -> str:
        tokens = normalize(gloss)
        if not tokens:
            return ""
        key = tuple(tokens)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if time.monotonic() < self._paused_until:
            return self.fallback.translate(tokens)

        request = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            # The system prompt and examples are identical on every call, so they cache.
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": self.build_messages(tokens),
        }
        if self.model.startswith(_EFFORT_MODELS):
            request["output_config"] = {"effort": self.effort}

        try:
            response = self._client.messages.create(**request)
        except Exception as error:  # noqa: BLE001 - network/auth/billing: speak something anyway
            status = getattr(error, "status_code", None)
            lasting = status in (401, 403) or (status == 400 and "credit" in str(error).lower())
            self._paused_until = float("inf") if lasting else time.monotonic() + self.retry_pause
            log.warning(
                "translation API call failed (%s); using the offline translator %s",
                error,
                "for the rest of this session" if lasting else f"for {self.retry_pause:.0f} s",
            )
            return self.fallback.translate(tokens)

        if getattr(response, "stop_reason", None) == "refusal":
            log.warning("translation refused by the model; using the offline fallback")
            return self.fallback.translate(tokens)
        sentence = "".join(block.text for block in response.content if block.type == "text").strip().strip('"')
        if not sentence:
            return self.fallback.translate(tokens)

        self._cache[key] = sentence
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return sentence


_default: GlossTranslator | None = None


def get_default_translator() -> GlossTranslator:
    """Claude when an API key is available, the offline rules otherwise."""
    global _default
    if _default is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                _default = ClaudeTranslator()
            except Exception as error:  # noqa: BLE001 - SDK missing or client config bad
                log.warning("could not start the Claude translator (%s); using the offline rules", error)
                _default = RuleBasedTranslator()
        else:
            log.info("ANTHROPIC_API_KEY is not set; using the offline rule-based translator")
            _default = RuleBasedTranslator()
    return _default


def set_default_translator(translator: GlossTranslator | None) -> None:
    """Install a translator (e.g. a trained seq2seq model). None resets to the default."""
    global _default
    _default = translator


def translate_gloss(gloss_sequence: list[str], translator: GlossTranslator | None = None) -> str:
    """Turn recognized gloss tokens into one natural English sentence.

    >>> translate_gloss(["hello", "name", "what"])
    "Hello! What's your name?"

    Returns "" for an empty sequence. Never raises for API problems: it falls
    back to the offline translator instead.
    """
    return (translator or get_default_translator()).translate(list(gloss_sequence))
