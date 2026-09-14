"""
Checks on prompt p4 (scripts/prompt_p4.py). No API calls.

p4 must be p3 plus one block and nothing else, must leave p1 (production) and
p3 (frozen in the eval manifest) untouched, and must not quote any tweet from
the frozen evaluation set it is scored on.
"""
import hashlib
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
import prompt_p4  # noqa: E402
import taxonomy_rules  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class P4Prompt(unittest.TestCase):
    def test_production_is_still_p1(self):
        self.assertEqual(groq_lib.PROMPT_VERSION, "p1")
        self.assertEqual(_sha(groq_lib.build_system_prompt()), integrity.P1_PROMPT_SHA256)

    def test_p3_still_matches_the_frozen_manifest(self):
        with open("data/p3_eval_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(_sha(groq_lib.build_system_prompt("p3")), manifest["prompt_sha256"]["p3"])

    def test_p4_is_p3_plus_one_block(self):
        p3, p4 = groq_lib.build_system_prompt("p3"), groq_lib.build_system_prompt("p4")
        self.assertEqual(p4.replace(prompt_p4.P4_BLOCK + "\n\n", "", 1), p3)
        self.assertIn(taxonomy_rules.PROMPT_BLOCK, p4)
        self.assertEqual(p4.count("Respond with ONLY this JSON shape"), 1)

    def test_p4_has_its_own_cache_namespace(self):
        namespaces = {groq_lib.cache_namespace(prompt_version=v) for v in ("p1", "p3", "p4")}
        self.assertEqual(len(namespaces), 3)

    def test_p4_block_quotes_no_eval_tweet(self):
        eval_set = pd.read_csv("data/p3_eval_set.csv", dtype={"customer_tweet_id": str})
        block = integrity.shingles(prompt_p4.P4_BLOCK)
        quoted = [r.eval_index for r in eval_set.itertuples() if integrity.shingles(r.customer_text) & block]
        self.assertEqual(quoted, [])

    def test_p4_block_uses_only_real_intent_names(self):
        import re
        named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", prompt_p4.P4_BLOCK)) - {"secondary_intent"}
        self.assertTrue(named <= set(groq_lib.INTENT_NAMES), named - set(groq_lib.INTENT_NAMES))


if __name__ == "__main__":
    unittest.main()
