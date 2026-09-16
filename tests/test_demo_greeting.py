"""
Offline tests for the demo's bare-greeting courtesy reply (30_demo.py).

No API calls, like the rest of tests/. The property that matters is the
negative one: a greeting that also carries a request must NOT be short-circuited
into the canned reply, because that message is a real support issue and has to
reach the drafter.

  ./.venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import importlib.util
import os
import sys
import unittest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_spec = importlib.util.spec_from_file_location("demo_mod", os.path.join(_SCRIPTS, "30_demo.py"))
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

import reply_guard  # noqa: E402  (needs _SCRIPTS on sys.path first)


class BareGreetingsAreRecognised(unittest.TestCase):
    def test_plain_greetings(self):
        for text in ["hi", "Hi", "HI", "hello", "Hello!", "hey", "yo", "hiya",
                     "howdy", "sup", "Hii", "greetings"]:
            self.assertTrue(demo.is_bare_greeting(text), text)

    def test_greeting_with_filler(self):
        for text in ["hi there", "Hello there!", "hey guys", "hi all",
                     "good morning", "Good Evening", "hey team", "hi dropbox"]:
            self.assertTrue(demo.is_bare_greeting(text), text)

    def test_handles_and_punctuation_are_ignored(self):
        for text in ["@DropboxSupport hi", "hi!!!", "  hello  ", "@115712 hey there"]:
            self.assertTrue(demo.is_bare_greeting(text), text)


class RealMessagesAreNotGreetings(unittest.TestCase):
    """The failure that would actually matter: swallowing a support request."""

    def test_greeting_plus_request_is_not_bare(self):
        for text in [
            "hi, my files won't sync",
            "hello I was charged twice this month",
            "hey can you help me recover a deleted folder",
            "hi there, how do I share a folder?",
            "good morning, my account is locked",
        ]:
            self.assertFalse(demo.is_bare_greeting(text), text)

    def test_other_no_action_messages_are_not_greetings(self):
        """Praise and spam share the no_action_needed label but are not greetings."""
        for text in [
            "thanks guys, you're the best!",
            "you're the worst support team ever",
            "check out my crypto giveaway at example.com",
            "we are hiring senior engineers, apply now",
        ]:
            self.assertFalse(demo.is_bare_greeting(text), text)

    def test_empty_and_junk(self):
        for text in ["", "   ", "???", "@DropboxSupport", "12345"]:
            self.assertFalse(demo.is_bare_greeting(text), text)

    def test_long_message_is_never_a_greeting(self):
        self.assertFalse(demo.is_bare_greeting("hi hi hi hi hi hi hi hi"))


class CourtesyReplyIsSafe(unittest.TestCase):
    def test_it_passes_the_placeholder_guard(self):
        """The canned string must satisfy the same guard drafted replies do."""
        ok, hits = reply_guard.validate(demo.COURTESY_REPLY)
        self.assertTrue(ok, f"courtesy reply tripped the guard: {hits}")

    def test_it_fits_a_tweet_and_names_no_one(self):
        self.assertLessEqual(len(demo.COURTESY_REPLY), 280)
        self.assertNotIn("@", demo.COURTESY_REPLY)


class PolicyIsUnchanged(unittest.TestCase):
    """The courtesy reply must not leak into anything the evaluation scores."""

    def test_escalation_table_still_routes_greetings_to_no_action(self):
        import escalation
        action, _ = escalation.INTENT_DEFAULTS["no_action_needed"]
        self.assertEqual(action, escalation.NO_ACTION)

    def test_decide_is_not_involved_in_the_courtesy_path(self):
        """decide() has no knowledge of greetings -- 12_evaluate.py is unaffected."""
        import escalation
        result = {"intent": "no_action_needed", "confidence": 0.95, "turn_type": "first_contact"}
        self.assertEqual(escalation.decide(result).action, escalation.NO_ACTION)


if __name__ == "__main__":
    unittest.main()
