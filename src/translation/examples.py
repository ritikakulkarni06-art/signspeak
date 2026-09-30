"""The prompt: what the model is told, and the gloss -> sentence examples it is shown.

These examples are the whole "training" of this stage, so they are the first
thing to edit when output reads wrong. They use the prototype's own vocabulary
(see data/vocabulary.py) and a presentation / Q&A register.
"""

SYSTEM_PROMPT = """You turn sign-language gloss into natural English for someone presenting and taking questions.

A gloss sequence lists the signs a person produced, in signing order. Sign language grammar is not English grammar: there is no copula, question signs often come last, time markers come first, and articles, plurals and tenses are dropped. Recover the English sentence the signer meant.

Rules:
- Reply with the sentence only. No quotes, no gloss, no explanation, no alternatives.
- One sentence, unless the gloss clearly holds two.
- Add the small words English needs (is, are, the, do, can) but never add new content or invent details the gloss does not contain.
- If the gloss has a question sign (what, why, how, who, where, when), write a question.
- Use "I" when the gloss names no subject and the meaning is clearly about the signer; use "you" when the signer is addressing someone.
- Keep it brief and natural, the way a person speaks in a meeting.
- If the gloss is too garbled to read, reply with the glosses joined by spaces, as a sentence."""

# (gloss tokens, natural sentence)
EXAMPLES: tuple[tuple[list[str], str], ...] = (
    (["hello", "name", "what"], "Hello! What's your name?"),
    (["understand", "no"], "I don't understand."),
    (["again", "please"], "Could you say that again, please?"),
    (["help", "need"], "I need help."),
    (["idea", "good"], "That's a good idea."),
    (["sign", "language", "learn", "want"], "I want to learn sign language."),
    (["discuss", "next", "what"], "What should we discuss next?"),
    (["question", "answer", "time", "more", "need"], "I need more time to answer the question."),
    (["example", "three", "want"], "I'd like three examples."),
    (["problem", "hard", "but", "important"], "The problem is hard, but it's important."),
    (["you", "deaf", "hearing", "which"], "Are you deaf or hearing?"),
    (["thank you", "finish"], "Thank you, I'm done."),
)


def format_gloss(gloss: list[str]) -> str:
    """How a gloss sequence is written in the prompt."""
    return " ".join(token.upper() for token in gloss)
