"""Gloss sequences -> natural sentences."""

from .examples import EXAMPLES, SYSTEM_PROMPT
from .rules import RuleBasedTranslator
from .translator import (
    ClaudeTranslator,
    GlossTranslator,
    get_default_translator,
    set_default_translator,
    translate_gloss,
)

__all__ = [
    "EXAMPLES",
    "SYSTEM_PROMPT",
    "ClaudeTranslator",
    "GlossTranslator",
    "RuleBasedTranslator",
    "get_default_translator",
    "set_default_translator",
    "translate_gloss",
]
