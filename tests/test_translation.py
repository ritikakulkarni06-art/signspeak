"""Translation tests. No network: the API client is stubbed.

Run from the project root: python -m unittest discover -s tests
"""

import unittest
from types import SimpleNamespace

from src.translation import ClaudeTranslator, RuleBasedTranslator, set_default_translator, translate_gloss
from src.translation.examples import EXAMPLES, SYSTEM_PROMPT
from src.translation.translator import normalize


class StubClient:
    """Stands in for anthropic.Anthropic."""

    def __init__(self, text="What's your name?", error=None, stop_reason="end_turn"):
        self.text, self.error, self.stop_reason = text, error, stop_reason
        self.requests = []

    @property
    def messages(self):
        return self

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        blocks = [SimpleNamespace(type="text", text=self.text)] if self.text is not None else []
        return SimpleNamespace(content=blocks, stop_reason=self.stop_reason)


class NormalizeTests(unittest.TestCase):
    def test_lowercases_trims_and_collapses_repeats(self):
        self.assertEqual(normalize([" Hello ", "hello", "NAME", "what"]), ["hello", "name", "what"])
        self.assertEqual(normalize(["thank  you"]), ["thank you"])
        self.assertEqual(normalize([" ", ""]), [])


class RuleBasedTests(unittest.TestCase):
    def setUp(self):
        self.translator = RuleBasedTranslator()

    def test_phrase_table(self):
        self.assertEqual(self.translator.translate(["understand", "no"]), "I don't understand.")
        self.assertEqual(self.translator.translate(["again", "please"]), "Could you say that again, please?")
        self.assertEqual(self.translator.translate(["thank you"]), "Thank you.")

    def test_grammar_on_hand_written_sequences(self):
        cases = {
            ("hello", "name", "what"): "Hello! What's your name?",
            ("you", "name", "what"): "What's your name?",
            ("help", "need", "please"): "I need help, please.",
            ("time", "more", "need"): "I need more time.",
            ("next", "discuss", "what"): "What do we discuss next?",
            ("sign", "language", "learn", "easy", "no"): "Learning sign language isn't easy.",
            ("idea", "good", "think"): "I think the idea is good.",
            ("example", "three", "important"): "Three examples are important.",
            ("finish", "thank you"): "I'm done. Thank you.",
            ("you", "deaf", "hearing", "which"): "Are you deaf or hearing?",
            ("problem", "what"): "What is the problem?",
            ("why", "you", "sad"): "Why are you sad?",
            ("sign", "language", "learn", "want"): "I want to learn sign language.",
            ("he", "understand", "no"): "He doesn't understand.",
            ("good morning", "teacher", "where"): "Good morning! Where is the teacher?",
        }
        for gloss, expected in cases.items():
            self.assertEqual(self.translator.translate(list(gloss)), expected, gloss)

    def test_unparseable_sequences_are_said_as_signed(self):
        # Content is never dropped or invented: a recognition error stays visible.
        self.assertEqual(self.translator.translate(["nine", "language", "bad", "six"]), "Nine language bad six.")
        self.assertEqual(self.translator.translate(["banana", "what"]), "Banana what?")

    def test_empty_sequence(self):
        self.assertEqual(self.translator.translate([]), "")


class ClaudeTranslatorTests(unittest.TestCase):
    def test_prompt_carries_system_and_few_shot_examples(self):
        client = StubClient()
        translator = ClaudeTranslator(client=client, model="claude-opus-5")
        self.assertEqual(translator.translate(["hello", "name", "what"]), "What's your name?")

        request = client.requests[0]
        self.assertEqual(request["model"], "claude-opus-5")
        self.assertEqual(request["system"][0]["text"], SYSTEM_PROMPT)
        self.assertEqual(request["output_config"], {"effort": "low"})
        self.assertNotIn("temperature", request)  # rejected by current models
        self.assertEqual(len(request["messages"]), 2 * len(EXAMPLES) + 1)
        self.assertEqual(request["messages"][0]["content"], "HELLO NAME WHAT")
        self.assertEqual(request["messages"][1]["content"], EXAMPLES[0][1])
        self.assertEqual(request["messages"][-1], {"role": "user", "content": "HELLO NAME WHAT"})

    def test_effort_is_omitted_for_models_that_reject_it(self):
        client = StubClient()
        ClaudeTranslator(client=client, model="claude-haiku-4-5").translate(["yes"])
        self.assertNotIn("output_config", client.requests[0])

    def test_repeated_sequences_are_cached(self):
        client = StubClient()
        translator = ClaudeTranslator(client=client, model="claude-opus-5")
        translator.translate(["yes"])
        translator.translate(["yes"])
        self.assertEqual(len(client.requests), 1)

    def test_api_failure_falls_back_offline(self):
        translator = ClaudeTranslator(client=StubClient(error=ConnectionError("no network")), model="claude-opus-5")
        self.assertEqual(translator.translate(["understand", "no"]), "I don't understand.")

    def test_bad_key_stops_api_calls_for_the_session(self):
        class AuthError(Exception):
            status_code = 401

        client = StubClient(error=AuthError("API key is invalid."))
        translator = ClaudeTranslator(client=client, model="claude-opus-5")
        self.assertEqual(translator.translate(["understand", "no"]), "I don't understand.")
        self.assertEqual(translator.translate(["help", "need"]), "I need help.")
        self.assertEqual(len(client.requests), 1)

    def test_transient_failure_pauses_then_retries(self):
        client = StubClient(error=ConnectionError("timeout"))
        translator = ClaudeTranslator(client=client, model="claude-opus-5", retry_pause=0.0)
        translator.translate(["yes"])
        client.error = None
        self.assertEqual(translator.translate(["hello", "name", "what"]), "What's your name?")
        self.assertEqual(len(client.requests), 2)

    def test_refusal_and_empty_reply_fall_back_offline(self):
        refused = ClaudeTranslator(client=StubClient(stop_reason="refusal"), model="claude-opus-5")
        self.assertEqual(refused.translate(["thank you"]), "Thank you.")
        empty = ClaudeTranslator(client=StubClient(text="  "), model="claude-opus-5")
        self.assertEqual(empty.translate(["thank you"]), "Thank you.")

    def test_quotes_are_stripped(self):
        translator = ClaudeTranslator(client=StubClient(text='"Thank you."'), model="claude-opus-5")
        self.assertEqual(translator.translate(["thank you"]), "Thank you.")


class TranslateGlossTests(unittest.TestCase):
    def tearDown(self):
        set_default_translator(None)

    def test_uses_the_installed_default_translator(self):
        set_default_translator(RuleBasedTranslator())
        self.assertEqual(translate_gloss(["understand", "no"]), "I don't understand.")

    def test_explicit_translator_wins(self):
        set_default_translator(RuleBasedTranslator())
        stub = ClaudeTranslator(client=StubClient(text="Hello there."), model="claude-opus-5")
        self.assertEqual(translate_gloss(["hello"], translator=stub), "Hello there.")

    def test_empty_input(self):
        set_default_translator(RuleBasedTranslator())
        self.assertEqual(translate_gloss([]), "")


if __name__ == "__main__":
    unittest.main()
