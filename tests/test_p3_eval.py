"""
Checks on the p1-vs-p3 experiment infrastructure (scripts/28_p3_eval.py). No
API calls, no labels needed.

Run from anywhere:
    ./.venv/Scripts/python.exe -m unittest discover -s tests -v

What is covered: the existing evaluation state has not moved, the frozen set
is intact and clean of every tweet the project has already touched, both
prompts are the ones the set was frozen against, the labelling path cannot see
predictions, the two R7a exceptions are still exceptions, and the statistics
the verdict rests on compute what they claim.
"""
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import eval_integrity as integrity  # noqa: E402
import groq_lib  # noqa: E402
import taxonomy_rules  # noqa: E402

spec = importlib.util.spec_from_file_location("p3_eval", "scripts/28_p3_eval.py")
p3_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3_eval)


class ExistingEvaluationUnchanged(unittest.TestCase):
    def test_frozen_files_and_cache_prefix_unchanged(self):
        self.assertEqual(integrity.frozen_state_problems(), [])

    def test_p1_is_byte_identical_to_production(self):
        self.assertEqual(groq_lib.PROMPT_VERSION, "p1")
        self.assertEqual(p3_eval.prompt_sha256("p1"), integrity.P1_PROMPT_SHA256)

    def test_p3_carries_the_shared_rules(self):
        self.assertIn(taxonomy_rules.PROMPT_BLOCK, groq_lib.build_system_prompt("p3"))

    def test_r7a_exceptions_are_still_exceptions_and_documented(self):
        gold = pd.read_csv("data/golden_labels_v3.csv", dtype={"customer_tweet_id": str}).set_index("customer_tweet_id")
        with open("taxonomy.md", encoding="utf-8") as f:
            part12 = f.read().split("# Part 12", 1)[-1]
        for tid, label in integrity.KNOWN_R7A_EXCEPTIONS.items():
            self.assertEqual(gold.at[tid, "human_intent_v3"], label)
            self.assertEqual(gold.at[tid, "v3_rule"], "R7a")
            self.assertIn(tid, part12)
        from test_taxonomy_rules import KNOWN_EXCEPTIONS
        self.assertEqual(set(KNOWN_EXCEPTIONS), set(integrity.KNOWN_R7A_EXCEPTIONS))


@unittest.skipUnless(os.path.exists(p3_eval.MANIFEST_PATH), "eval set not built yet")
class FrozenEvalSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(p3_eval.MANIFEST_PATH, encoding="utf-8") as f:
            cls.manifest = json.load(f)
        cls.eval_set = p3_eval.load_set()

    def test_set_matches_its_frozen_hash_and_size(self):
        self.assertEqual(integrity.file_sha256(p3_eval.SET_PATH), self.manifest["set_sha256"])
        self.assertEqual(len(self.eval_set), p3_eval.EVAL_N)
        self.assertEqual(list(self.eval_set.eval_index), list(range(1, p3_eval.EVAL_N + 1)))

    def test_prompts_are_the_ones_the_set_was_frozen_against(self):
        for version in p3_eval.PROMPTS:
            self.assertEqual(p3_eval.prompt_sha256(version), self.manifest["prompt_sha256"][version])

    def test_no_tweet_was_already_touched_by_the_project(self):
        used = integrity.used_tweet_ids(self.manifest["cache_lines_at_freeze"])
        for source, ids in used.items():
            with self.subTest(source=source):
                self.assertEqual(set(self.eval_set.customer_tweet_id) & ids, set())

    def test_no_tweet_is_in_the_golden_split(self):
        split = pd.read_csv("data/golden_split.csv", dtype={"customer_tweet_id": str})
        self.assertEqual(set(self.eval_set.customer_tweet_id) & set(split.customer_tweet_id), set())

    def test_no_tweet_is_quoted_in_docs_or_rule_examples(self):
        reference = integrity.reference_shingles()
        quoted = [r.customer_tweet_id for r in self.eval_set.itertuples()
                  if integrity.shingles(r.customer_text) & reference]
        self.assertEqual(quoted, [])

    def test_no_duplicate_tweets_or_texts(self):
        self.assertFalse(self.eval_set.customer_tweet_id.duplicated().any())
        self.assertFalse(self.eval_set.customer_text.map(integrity.normalize_for_match).duplicated().any())

    def test_labels_if_any_belong_to_the_set(self):
        labels = p3_eval.load_labels()
        if labels.empty:
            self.skipTest("no labels yet")
        self.assertTrue(set(labels.customer_tweet_id) <= set(p3_eval.scored_set().customer_tweet_id))
        self.assertTrue((pd.to_numeric(labels.eval_index) <= p3_eval.SCORED_N).all())
        self.assertTrue(set(labels.human_intent) <= set(groq_lib.INTENT_NAMES))
        self.assertEqual(set(labels.set_sha256), {self.manifest["set_sha256"]})


@unittest.skipUnless(os.path.exists(p3_eval.MANIFEST_PATH), "eval set not built yet")
class ScoringAmendment(unittest.TestCase):
    """The first 250 frozen tweets are scored: 250 per the assignment, 150
    after ~200 labels were lost (amendment 2), 250 again after the n=150
    report was read (amendment 3)."""

    def test_scored_prefix_is_250_of_the_frozen_300(self):
        self.assertEqual(p3_eval.SCORED_N, 250)
        self.assertLessEqual(p3_eval.SCORED_N, p3_eval.EVAL_N)
        scored = p3_eval.scored_set()
        self.assertEqual(list(scored.eval_index), list(range(1, p3_eval.SCORED_N + 1)))

    def test_amendment_is_recorded_against_the_frozen_set(self):
        with open(p3_eval.AMENDMENT_PATH, encoding="utf-8") as f:
            amendment = json.load(f)
        with open(p3_eval.MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(amendment["scored_n"], p3_eval.SCORED_N)
        self.assertEqual(amendment["set_sha256"], manifest["set_sha256"])
        self.assertEqual(amendment["state_at_amendment"]["labels_beyond_scored_n"], 0)
        # An amendment made after a report was read must keep that result.
        if amendment["state_at_amendment"]["predictions_viewed"]:
            kept = amendment["result_before_amendment"]["outputs_preserved_as"]
            self.assertTrue(kept)
            for path in kept:
                self.assertTrue(os.path.exists(path), path)

    def test_label_run_status_report_use_only_the_scored_set(self):
        for fn in (p3_eval.label, p3_eval.run, p3_eval.status, p3_eval.report):
            source = inspect.getsource(fn)
            self.assertIn("scored_set()", source, fn.__name__)
            self.assertNotIn("load_set()", source, fn.__name__)


class LabelsFileHandling(unittest.TestCase):
    """A labelling session creates the labels file before its first label is
    saved. --status must read that as 'no labels', and a restarted session
    must still write the header."""

    def setUp(self):
        import tempfile
        self._original = p3_eval.LABELS_PATH
        handle, self.path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        p3_eval.LABELS_PATH = self.path

    def tearDown(self):
        p3_eval.LABELS_PATH = self._original
        os.remove(self.path)

    def test_empty_labels_file_reads_as_no_labels(self):
        self.assertEqual(os.path.getsize(self.path), 0)
        labels = p3_eval.load_labels()
        self.assertTrue(labels.empty)
        self.assertEqual(list(labels.columns), p3_eval.LABEL_FIELDS)

    def test_label_writes_a_header_into_an_empty_file(self):
        source = inspect.getsource(p3_eval.label)
        self.assertIn("os.path.getsize(LABELS_PATH) == 0", source)


class LabellingStaysBlind(unittest.TestCase):
    def test_label_path_never_reads_predictions(self):
        source = inspect.getsource(p3_eval.label)
        for forbidden in ("ClassificationCache", "predictions(", "classify", "_intent\"]"):
            self.assertNotIn(forbidden, source)

    def test_run_and_status_print_counts_not_intents(self):
        for fn in (p3_eval.run, p3_eval.status):
            source = inspect.getsource(fn)
            self.assertNotIn("['intent']", source)
            self.assertNotIn('["intent"]', source)


class Statistics(unittest.TestCase):
    def test_mcnemar_exact_known_values(self):
        self.assertAlmostEqual(p3_eval.mcnemar_exact(0, 6), 0.03125)
        self.assertAlmostEqual(p3_eval.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(p3_eval.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(p3_eval.mcnemar_exact(1, 9), 2 * (1 + 10) / 2 ** 10)

    def test_paired_difference_counts_and_interval(self):
        p1 = [True] * 50 + [False] * 50
        p3 = [True] * 50 + [True] * 10 + [False] * 40
        stats = p3_eval.paired_difference(p1, p3, n_boot=2000)
        self.assertEqual((stats["fixes"], stats["breaks"]), (10, 0))
        self.assertAlmostEqual(stats["diff"], 0.10)
        self.assertLess(stats["ci"][0], 0.10)
        self.assertGreater(stats["ci"][1], 0.10)
        self.assertGreater(stats["ci"][0], 0)

    def test_identical_predictions_give_zero_width_interval(self):
        stats = p3_eval.paired_difference([True, False] * 20, [True, False] * 20, n_boot=500)
        self.assertEqual(stats["ci"], (0.0, 0.0))
        self.assertEqual(stats["mcnemar_p"], 1.0)

    def test_verdict_rules(self):
        better = p3_eval.verdict({"ci": (0.01, 0.08)}, missed_p1=5, missed_p3=5)
        self.assertTrue(better[0].startswith("P3 BETTER"))
        worse_escalation = p3_eval.verdict({"ci": (0.01, 0.08)}, missed_p1=5, missed_p3=6)
        self.assertTrue(worse_escalation[0].startswith("INCONCLUSIVE"))
        not_better = p3_eval.verdict({"ci": (-0.06, 0.02)}, missed_p1=5, missed_p3=5)
        self.assertTrue(not_better[0].startswith("P3 NOT BETTER"))
        unclear = p3_eval.verdict({"ci": (-0.02, 0.05)}, missed_p1=5, missed_p3=5)
        self.assertTrue(unclear[0].startswith("INCONCLUSIVE"))

    def test_escalation_metrics(self):
        esc = pd.Series([True, True, False, False])
        human = pd.Series([True, False, True, False])
        m = p3_eval.escalation_metrics(esc, human)
        self.assertEqual((m["missed"], m["over"], m["escalated"]), (1, 1, 2))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)


class BoundaryDefinitions(unittest.TestCase):
    def test_boundaries_mirror_taxonomy_rules(self):
        self.assertEqual(set(p3_eval.BOUNDARY_SIDES), set(taxonomy_rules.BOUNDARIES))
        for key, (_, side_a, side_b) in taxonomy_rules.BOUNDARIES.items():
            wide_a, wide_b = p3_eval.BOUNDARY_SIDES[key]
            self.assertIn(side_a, wide_a)
            self.assertIn(side_b, wide_b)
            self.assertFalse(wide_a & wide_b)
            self.assertTrue((wide_a | wide_b) <= set(groq_lib.INTENT_NAMES))

    def test_boundary_table_counts_cross_errors(self):
        df = pd.DataFrame({
            "gold":      ["storage_quota_plan_limits", "billing_subscription", "storage_quota_plan_limits"],
            "p1_intent": ["billing_subscription", "billing_subscription", "storage_quota_plan_limits"],
            "p3_intent": ["storage_quota_plan_limits", "storage_quota_plan_limits", "storage_quota_plan_limits"],
        })
        d = next(b for b in p3_eval.boundary_table(df) if b["key"] == "D")
        self.assertEqual((d["n"], d["p1_cross"], d["p3_cross"], d["cross_fixed"], d["cross_new"]), (3, 1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
