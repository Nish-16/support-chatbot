"""
Non-API checks that the v2.1 boundary rules are executable and consistent.

Run from anywhere:
    ./.venv/Scripts/python.exe -m unittest discover -s tests -v

None of these call the LLM, so none of them show the classifier FOLLOWS the
rules. They show that the classifier and the labeler are shown the same rules,
that every example is a real, correctly labelled dev tweet, that the
adjudicated dev labels agree with each rule's stated winner, and that the
production prompt did not change. The holdout split is read only to assert
that no example comes from it.
"""
import contextlib
import hashlib
import importlib.util
import io
import os
import re
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)  # groq_lib reads intents.json relative to the working directory

import groq_lib  # noqa: E402
import taxonomy_rules as tr  # noqa: E402

# sha256 of the p1 system prompt, taken from groq_lib BEFORE p3 existed.
from eval_integrity import P1_PROMPT_SHA256 as P1_SHA256  # noqa: E402

THIRTEEN_INTENTS = [
    "account_access", "security_account_compromise", "phishing_abuse_report",
    "billing_subscription", "storage_quota_plan_limits", "sync_app_bug",
    "service_outage", "sharing_permissions", "data_loss_recovery",
    "how_to_usage", "feature_request", "complaint_dissatisfaction",
    "no_action_needed",
]

# Rules whose winner is fixed, and rules that only rule an intent out. R1a,
# R1d, R2 and R1e route to "the topic", which no single intent captures.
RULE_WINNER = {
    "R1b": {"feature_request"},
    "R1c": {"complaint_dissatisfaction"},
    "R3a": {"sharing_permissions"},
    "R3b": {"storage_quota_plan_limits", "sync_app_bug"},
    "R4": {"storage_quota_plan_limits"},
    "R6a": {"billing_subscription"},
    "R6b": {"storage_quota_plan_limits"},
    "R7a": {"account_access"},
    "R7b": {"how_to_usage"},
    "R8a": {"sync_app_bug"},
    "R8b": {"service_outage"},
    "R9a": {"no_action_needed"},
}
RULE_EXCLUDES = {
    "R1a": {"complaint_dissatisfaction"},
    "R1d": {"complaint_dissatisfaction"},
    "R5": {"storage_quota_plan_limits"},
}
# Adjudicated rows that contradict their recorded rule. Listed, not relabeled:
# this change does not touch the golden set. Review at the next label pass.
KNOWN_EXCEPTIONS = {
    "1000504": "v3_rule R7a on a security_account_compromise row; R7 separates account_access "
               "from how_to_usage and says nothing about compromise -- the code looks misapplied",
    "1057223": "same as 1000504",
}
LABELER_ONLY_CODES = {"R9b"}


def _norm(text: str) -> str:
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("&amp;", "&")):
        text = text.replace(a, b)
    return " ".join(text.split()).lower()


def _golden() -> pd.DataFrame:
    if not hasattr(_golden, "cache"):
        gold = pd.read_csv("data/golden_labels_v3.csv", dtype={"customer_tweet_id": str})
        split = pd.read_csv("data/golden_split.csv", dtype={"customer_tweet_id": str})
        _golden.cache = gold.merge(split[["customer_tweet_id", "split"]], on="customer_tweet_id")
    return _golden.cache


def _load_labeler():
    spec = importlib.util.spec_from_file_location("label_mod", "scripts/08_label_golden_set.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntentSetUnchanged(unittest.TestCase):
    def test_exactly_the_thirteen_v2_intents(self):
        self.assertEqual(groq_lib.INTENT_NAMES, THIRTEEN_INTENTS)

    def test_turn_type_and_flags_are_not_intents(self):
        self.assertFalse(set(groq_lib.TURN_TYPE_NAMES) & set(groq_lib.INTENT_NAMES))
        self.assertFalse(set(groq_lib.FLAG_SPEC) & set(groq_lib.INTENT_NAMES))


class ProductionPromptUnchanged(unittest.TestCase):
    def test_default_prompt_version_is_still_p1(self):
        self.assertEqual(groq_lib.PROMPT_VERSION, "p1")

    def test_p1_prompt_is_byte_identical_to_before(self):
        for prompt in (groq_lib.build_system_prompt("p1"), groq_lib.build_system_prompt()):
            self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), P1_SHA256)

    def test_p1_carries_no_rules(self):
        self.assertNotIn(tr.RULES_TEXT, groq_lib.build_system_prompt("p1"))

    def test_unknown_prompt_version_is_refused(self):
        with self.assertRaises(ValueError):
            groq_lib.build_system_prompt("p2")

    def test_cache_namespace_keeps_p3_apart(self):
        self.assertIn(":p3:", groq_lib.cache_namespace(prompt_version="p3"))
        self.assertNotEqual(groq_lib.cache_namespace(prompt_version="p3"), groq_lib.cache_namespace())


class ClassifierAndLabelerSeeTheSameRules(unittest.TestCase):
    def test_p3_is_p1_plus_the_rules_block(self):
        p1, p3 = groq_lib.build_system_prompt("p1"), groq_lib.build_system_prompt("p3")
        self.assertIn(tr.PROMPT_BLOCK, p3)
        self.assertEqual(p3.replace(tr.PROMPT_BLOCK + "\n\n", "", 1), p1)
        self.assertEqual(p3.count("Respond with ONLY this JSON shape"), 1)

    def test_labeler_prints_the_same_rules_and_examples(self):
        labeler = _load_labeler()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            labeler.print_precedence_rules()
        shown = out.getvalue()
        self.assertIn(tr.RULES_TEXT, shown)
        for ex in tr.EXAMPLES:
            self.assertIn(f'- "{ex.text}" -> {ex.intent}, not {ex.not_intent}', shown)

    def test_every_example_line_the_model_sees_the_labeler_sees(self):
        model_lines = [l for l in tr.PROMPT_BLOCK.splitlines() if l.startswith('- "')]
        labeler_lines = set(tr.LABELER_BLOCK.splitlines())
        self.assertTrue(model_lines)
        self.assertEqual([l for l in model_lines if l not in labeler_lines], [])

    def test_labeler_no_longer_shows_the_brand_reply(self):
        with open("scripts/08_label_golden_set.py", encoding="utf-8") as f:
            self.assertNotIn("row.brand_text", f.read())


class RulesMatchTheTaxonomy(unittest.TestCase):
    def setUp(self):
        with open("taxonomy.md", encoding="utf-8") as f:
            text = f.read()
        self.part11 = text[text.index("# Part 11"):]

    def test_every_rule_code_exists_in_part_11(self):
        codes = {c for r in tr.RULES for c in r.codes} | {e.rule for e in tr.EXAMPLES}
        missing = [c for c in sorted(codes) if not re.search(rf"\b{c}\b", self.part11)]
        self.assertEqual(missing, [])

    def test_rules_name_only_real_intents_and_fields(self):
        allowed = set(groq_lib.INTENT_NAMES) | {"secondary_intent", "turn_type"}
        tokens = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", tr.RULES_TEXT))
        self.assertEqual(tokens - allowed, set())

    def test_every_rule_code_used_in_dev_labels_is_covered(self):
        dev = _golden()[_golden().split == "dev"]
        used = {c for v in dev.v3_rule.dropna() for c in str(v).split("/")}
        covered = {c for r in tr.RULES for c in r.codes} | LABELER_ONLY_CODES
        self.assertEqual(used - covered, set())


class BoundaryExamples(unittest.TestCase):
    def test_every_required_boundary_has_examples_on_both_sides(self):
        for key, (_, side_a, side_b) in tr.BOUNDARIES.items():
            winners = {e.intent for e in tr.EXAMPLES if e.boundary == key}
            self.assertIn(side_a, winners, f"boundary {key} has no {side_a} example")
            self.assertIn(side_b, winners, f"boundary {key} has no {side_b} example")

    def test_example_intents_are_real_and_distinct(self):
        for ex in tr.EXAMPLES:
            self.assertIn(ex.intent, groq_lib.INTENT_NAMES)
            self.assertIn(ex.not_intent, groq_lib.INTENT_NAMES)
            self.assertNotEqual(ex.intent, ex.not_intent)
            self.assertIn(ex.boundary, tr.BOUNDARIES)

    def test_examples_are_dev_tweets_quoted_verbatim_with_their_v3_label(self):
        gold = _golden().set_index("customer_tweet_id")
        for ex in (e for e in tr.EXAMPLES if e.tweet_id):
            with self.subTest(tweet=ex.tweet_id):
                self.assertIn(ex.tweet_id, gold.index)
                row = gold.loc[ex.tweet_id]
                self.assertEqual(row.split, "dev", "examples must never come from the holdout")
                self.assertEqual(row.v3_status, "resolved")
                self.assertEqual(row.human_intent_v3, ex.intent)
                self.assertIn(_norm(ex.text), _norm(row.customer_text))

    def test_illustrative_examples_stay_the_exception(self):
        self.assertLessEqual(sum(1 for e in tr.EXAMPLES if e.tweet_id is None), 2)


class AdjudicatedDevLabelsFollowTheRules(unittest.TestCase):
    def _dev_rows(self):
        dev = _golden()
        return dev[(dev.split == "dev") & (dev.v3_status == "resolved") & dev.v3_rule.notna()]

    def test_labels_agree_with_each_rules_winner(self):
        violations = []
        for r in self._dev_rows().itertuples():
            code = str(r.v3_rule)
            if r.customer_tweet_id in KNOWN_EXCEPTIONS:
                continue
            if code in RULE_WINNER and r.human_intent_v3 not in RULE_WINNER[code]:
                violations.append((r.customer_tweet_id, code, r.human_intent_v3))
            if code in RULE_EXCLUDES and r.human_intent_v3 in RULE_EXCLUDES[code]:
                violations.append((r.customer_tweet_id, code, r.human_intent_v3))
        self.assertEqual(violations, [])

    def test_known_exceptions_are_still_real(self):
        rows = self._dev_rows().set_index("customer_tweet_id")
        for tweet_id in KNOWN_EXCEPTIONS:
            with self.subTest(tweet=tweet_id):
                self.assertIn(tweet_id, rows.index, "exception no longer in the data -- remove it")
                row = rows.loc[tweet_id]
                self.assertNotIn(row.human_intent_v3, RULE_WINNER.get(str(row.v3_rule), set()),
                                 "exception now agrees with its rule -- remove it")


if __name__ == "__main__":
    unittest.main()
