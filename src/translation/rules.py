"""Offline gloss -> English rules: the translator used when no model is reachable.

A small role-based grammar over the prototype's vocabularies (ASL and ISL, see
data/vocabulary.py). Each token is tagged with a role (question word, verb,
adjective, noun, number, negation, ...), greetings and thanks are peeled off
the ends, and the rest fills one of a handful of sentence templates:

    hello name what              -> Hello! What's your name?
    sign language learn easy no  -> Learning sign language isn't easy.
    idea good think              -> I think the idea is good.

It never drops or invents content: a sequence that doesn't fit a template
comes back as its tokens joined into a sentence ("Nine language bad six."),
so a recognition error stays visible instead of being papered over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

QUESTION_SIGNS = {"what", "why", "how", "who", "where", "when", "which"}
VERBS = {
    "help", "understand", "know", "think", "want", "need", "learn", "ask", "answer",
    "discuss", "finish", "repeat", "explain", "agree", "dream", "paint",
}  # fmt: skip
ADJECTIVES = {
    "good", "bad", "wrong", "easy", "hard", "important", "deaf", "hearing", "new", "old",
    "happy", "sad", "slow", "fast", "loud", "quiet", "big", "small", "long", "short",
}  # fmt: skip
NOUNS = {
    "name", "time", "idea", "problem", "example", "language", "sign", "sign language",
    "question", "computer", "teacher", "student", "school", "university", "science",
    "technology", "minute", "hour", "day", "week", "money", "team",
}  # fmt: skip
MASS_NOUNS = {"sign language", "language", "time", "help", "science", "technology", "money"}  # no "the"
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
TRAILING_ADVERBS = {"again", "next", "now", "today", "tomorrow", "yesterday"}
NEGATIONS = {"no", "not", "never"}
PRONOUNS = {"i", "you", "we", "he", "she", "it", "they", "you (plural)"}
OPENERS = {
    "hello": "Hello!", "welcome": "Welcome!", "yes": "Yes.", "alright": "Alright.",
    "good morning": "Good morning!", "good afternoon": "Good afternoon!", "good evening": "Good evening!",
    "how are you": "How are you?",
}  # fmt: skip
CLOSERS = {"thank you": "Thank you.", "sorry": "Sorry.", "good night": "Good night.", "pleased": "Pleased to meet you."}
COMPOUNDS = {("sign", "language"): "sign language"}
# Whole utterances that templates can't express well.
PHRASES = {
    ("yes",): "Yes.",
    ("no",): "No.",
    ("please",): "Please.",
    ("understand", "no"): "I don't understand.",
    ("no", "understand"): "I don't understand.",
    ("again", "please"): "Could you say that again, please?",
    ("please", "again"): "Could you say that again, please?",
    ("idea", "good"): "That's a good idea.",
    ("good", "idea"): "That's a good idea.",
}
_QUESTION_SUBJECT = {"discuss": "we"}  # "What do we discuss next?", not "...do you..."
# Words a template may render differently; used to check nothing signed was dropped.
_SURFACE_FORMS = {"finish": ("finish", "done"), "which": ("which", "or")}
_BE = {"i": "am", "you": "are", "we": "are", "they": "are", "you (plural)": "are"}  # others: "is"


def normalize(gloss: list[str]) -> list[str]:
    """Trim, lowercase, drop empties, and collapse a sign repeated back-to-back
    (the segmenter can emit the same sign twice while a hold is held)."""
    tokens: list[str] = []
    for token in gloss:
        token = " ".join(str(token).split()).lower()
        if token and (not tokens or token != tokens[-1]):
            tokens.append(token)
    return tokens


class RuleBasedTranslator:
    """Offline translator; see the module docstring."""

    def translate(self, gloss: list[str]) -> str:
        tokens = normalize(gloss)
        if not tokens:
            return ""
        return PHRASES.get(tuple(tokens)) or compose(tokens)


def compose(tokens: list[str]) -> str:
    tokens = _merge_compounds(tokens)
    before, after = [], []
    while tokens and tokens[0] in OPENERS:
        before.append(OPENERS[tokens.pop(0)])
    while tokens and tokens[-1] in CLOSERS:
        after.insert(0, CLOSERS[tokens.pop()])
    core = (PHRASES.get(tuple(tokens)) or _clause(tokens)) if tokens else ""
    return " ".join(part for part in (*before, core, *after) if part)


@dataclass
class _Roles:
    question: list[str] = field(default_factory=list)
    verbs: list[str] = field(default_factory=list)
    adjectives: list[str] = field(default_factory=list)
    nouns: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    adverbs: list[str] = field(default_factory=list)
    pronouns: list[str] = field(default_factory=list)
    more: bool = False
    negated: bool = False
    polite: bool = False
    unknown: list[str] = field(default_factory=list)


def _merge_compounds(tokens: list[str]) -> list[str]:
    merged, i = [], 0
    while i < len(tokens):
        pair = tuple(tokens[i : i + 2])
        if pair in COMPOUNDS:
            merged.append(COMPOUNDS[pair])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def _roles(tokens: list[str]) -> _Roles:
    roles = _Roles()
    for token in tokens:
        if token in QUESTION_SIGNS:
            roles.question.append(token)
        elif token in VERBS:
            roles.verbs.append(token)
        elif token in ADJECTIVES:
            roles.adjectives.append(token)
        elif token in NOUNS:
            roles.nouns.append(token)
        elif token in NUMBERS:
            roles.numbers.append(token)
        elif token in TRAILING_ADVERBS:
            roles.adverbs.append(token)
        elif token in PRONOUNS:
            roles.pronouns.append("you" if token == "you (plural)" else token)
        elif token == "more":
            roles.more = True
        elif token in NEGATIONS:
            roles.negated = True
        elif token == "please":
            roles.polite = True
        else:
            roles.unknown.append(token)
    return roles


def _clause(tokens: list[str]) -> str:
    roles = _roles(tokens)
    sentence = None
    if not roles.unknown:
        sentence = _question(roles) if roles.question else _statement(roles)
    if sentence is not None and not _keeps_content(sentence, roles):
        sentence = None
    if sentence is None:  # doesn't fit a template: say exactly what was signed
        text = " ".join(tokens)
        return text[0].upper() + text[1:] + ("?" if roles.question else ".")
    if roles.polite:
        sentence = sentence[:-1] + ", please" + sentence[-1]
    return sentence[0].upper() + sentence[1:]


def _keeps_content(sentence: str, roles: _Roles) -> bool:
    """True if every signed content word made it into the sentence. Templates
    are allowed to add words, never to lose them."""
    text = sentence.lower()
    signed = roles.question + roles.verbs + roles.adjectives + roles.nouns + roles.numbers + roles.adverbs + roles.pronouns
    for word in signed:
        if not any(re.search(r"\b" + re.escape(form), text) for form in _SURFACE_FORMS.get(word, (word,))):
            return False
    if roles.more and not re.search(r"\bmore\b", text):
        return False
    return not roles.negated or bool(re.search(r"n't\b|\bnot\b", text))


def _noun_phrase(roles: _Roles, article: str = "the") -> tuple[str, bool] | None:
    """(text, is_plural) for the clause's noun, e.g. "three examples", "more time"."""
    if len(roles.nouns) > 1 or len(roles.numbers) > 1:
        return None
    if not roles.nouns:
        if roles.numbers:
            return roles.numbers[0], False
        return ("more", False) if roles.more else None
    noun = roles.nouns[0]
    count = NUMBERS[roles.numbers[0]] if roles.numbers else None
    if count and count > 1:
        noun = noun + ("es" if noun.endswith("s") else "s")
    if roles.numbers:
        determiner = roles.numbers[0]
    elif roles.more:
        determiner = "more"
    else:
        determiner = "" if roles.nouns[0] in MASS_NOUNS else article
    return " ".join(word for word in (determiner, noun) if word), bool(count and count > 1)


def _be(subject: str, plural: bool = False, negated: bool = False) -> str:
    verb = "are" if plural else _BE.get(subject, "is")
    if negated:
        return {"am": "am not", "are": "aren't", "is": "isn't"}[verb]
    return verb


def _question(roles: _Roles) -> str | None:
    if len(roles.question) > 1:
        return None
    word = roles.question[0]
    subject = roles.pronouns[0] if roles.pronouns else None

    options = roles.adjectives + roles.nouns
    if word == "which" and len(options) >= 2 and not roles.verbs:
        subject = subject or "you"
        return f"{_be(subject)} {subject} {', '.join(options[:-1])} or {options[-1]}?"

    if roles.verbs:
        if len(roles.verbs) > 1 or roles.adjectives:
            return None
        verb = roles.verbs[0]
        subject = subject or _QUESTION_SUBJECT.get(verb, "you")
        auxiliary = ("doesn't" if subject in {"he", "she", "it"} else "don't") if roles.negated else (
            "does" if subject in {"he", "she", "it"} else "do"
        )
        obj = _noun_phrase(roles)
        words = [word, auxiliary, subject, verb, obj[0] if obj else "", *roles.adverbs]
        return " ".join(w for w in words if w) + "?"

    if roles.nouns == ["name"] and not roles.numbers and not roles.adjectives and word == "what":
        return "What's your name?"
    if roles.nouns == ["time"] and not roles.numbers and not roles.adjectives and word == "what":
        return "What time is it?"

    if len(roles.adjectives) == 1:
        adjective = roles.adjectives[0]
        if roles.nouns:
            phrase = _noun_phrase(roles)
            if phrase is None:
                return None
            return f"{word} {_be('', phrase[1], roles.negated)} {phrase[0]} {adjective}?"
        subject = subject or "you"
        return f"{word} {_be(subject, negated=roles.negated)} {subject} {adjective}?"

    if roles.adjectives:
        return None
    phrase = _noun_phrase(roles)
    if phrase:
        return f"{word} {_be('', phrase[1])} {phrase[0]}?"
    return f"{word}?" if not roles.pronouns else None


def _statement(roles: _Roles) -> str | None:
    subject = roles.pronouns[0] if roles.pronouns else "i"
    third_person = subject in {"he", "she", "it"}

    def say(verb: str, complement: str = "") -> str:
        if roles.negated:
            words = [subject, "doesn't" if third_person else "don't", verb]
        else:
            words = [subject, verb + (("es" if verb.endswith("s") else "s") if third_person else "")]
        return " ".join([*words, complement, *roles.adverbs]).replace("  ", " ").strip() + "."

    verbs = roles.verbs
    wanting = next((verb for verb in verbs if verb in {"want", "need"}), None)
    if wanting:
        others = [verb for verb in verbs if verb != wanting]
        if len(others) > 1 or roles.adjectives:
            return None
        phrase = _noun_phrase(roles)
        if others == ["help"] and phrase is None:
            complement = "help"  # "I need help", not "I need to help"
        elif others:
            complement = f"to {others[0]}" + (f" {phrase[0]}" if phrase else "")
        elif phrase:
            complement = phrase[0]
        else:
            return None
        return say(wanting, complement)

    if len(verbs) == 1:
        verb = verbs[0]
        if roles.adjectives:
            phrase = _noun_phrase(roles, article="")
            if len(roles.adjectives) > 1 or phrase is None:
                return None
            adjective = roles.adjectives[0]
            if verb == "think":  # "idea good think" -> I think the idea is good.
                full = _noun_phrase(roles)
                return f"{subject} think {full[0]} {_be('', full[1], roles.negated)} {adjective}."
            # "sign language learn easy no" -> Learning sign language isn't easy.
            return f"{_gerund(verb)} {phrase[0]} {_be('', False, roles.negated)} {adjective}."
        if verb == "finish" and not roles.nouns and subject == "i":
            return "I'm not done." if roles.negated else "I'm done."
        phrase = _noun_phrase(roles)
        return say(verb, phrase[0] if phrase else "")
    if verbs:
        return None

    phrase = _noun_phrase(roles)
    if len(roles.adjectives) == 1:
        adjective = roles.adjectives[0]
        if phrase:
            return f"{phrase[0]} {_be('', phrase[1], roles.negated)} {adjective}."
        if roles.pronouns:
            return f"{subject} {_be(subject, negated=roles.negated)} {adjective}."
        return f"it's {'not ' if roles.negated else ''}{adjective}."
    if not roles.adjectives and phrase and not roles.negated:
        return " ".join([phrase[0], *roles.adverbs]) + "."
    return None


def _gerund(verb: str) -> str:
    return (verb[:-1] if verb.endswith("e") and not verb.endswith("ee") else verb) + "ing"
