"""
Trivial + simple baselines, per README.md's "Baselines (defined before
building the LLM agent)" section -- so the eventual results table isn't
retrofitted to make the LLM agent look good, and shows what you get for
free vs. what the LLM actually adds.

Independent of hand-labeling completion for MOST of what's here:
  - trivial classify/draft/escalate need no labels at all.
  - simple draft (1-NN retrieval) and simple escalate (keyword regex)
    need no labels either.
  - simple classify (TF-IDF + LogisticRegression) is the one piece that
    DOES need labels -- it fits on the golden set.

Labels: reads data/golden_labels_v3.csv and scores against the v3 label,
skipping the rows marked insufficient_context, so this baseline is measured
on exactly the same 174 rows as the LLM in 12_evaluate.py. Scoring a
baseline on one label version and the LLM on another makes the comparison
table meaningless; that mistake is documented in WHAT_WENT_WRONG.md.

Usage:
  ./.venv/Scripts/python.exe scripts/10_baselines.py
"""
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from retrieval import ReplyRetriever
from groq_lib import INTENT_NAMES

GOLDEN_LABELS_PATH = "data/golden_labels_v3.csv"


def read_labels(path: str = GOLDEN_LABELS_PATH) -> pd.DataFrame:
    """Golden set with the *reported* label in `human_intent`.

    v3 re-adjudicated every row and marked 15 of them insufficient_context
    (unroutable without the prior turn); those are dropped here for the same
    reason 12_evaluate.py drops them. Falls back to whatever `human_intent`
    holds if handed a pre-v3 file, so the old numbers stay reproducible.
    """
    labels = pd.read_csv(path)
    if "v3_status" in labels.columns:
        labels = labels[labels["v3_status"] == "resolved"].copy()
    if "human_intent_v3" in labels.columns:
        labels["human_intent"] = labels["human_intent_v3"]
    return labels.dropna(subset=["customer_text", "human_intent"])


# --------------------------------------------------------------------
# Trivial baseline
# --------------------------------------------------------------------

class TrivialBaseline:
    """Majority-class classification, one fixed canned reply, escalate
    100% of tickets. The floor every other approach has to beat."""

    CANNED_REPLY = ("Hi, thanks for reaching out! We're looking into this and "
                     "will follow up shortly. In the meantime, DM us your account "
                     "email so we can look into it.")

    def __init__(self, golden_labels_path: str = GOLDEN_LABELS_PATH):
        self.majority_intent = self._compute_majority_intent(golden_labels_path)

    @staticmethod
    def _compute_majority_intent(path: str) -> str:
        try:
            labels = read_labels(path)
            if len(labels):
                return labels["human_intent"].value_counts().idxmax()
        except FileNotFoundError:
            pass
        # No labels yet (or file missing) -- fall back to the intent
        # with the most candidates in the pool, as the least-bad guess
        # available before any hand-labeling exists. Not a substitute
        # for the real golden-set majority once labeling completes.
        candidates = pd.read_csv("data/golden_candidates.csv")
        return candidates["suggested_intent"].value_counts().idxmax()

    def classify(self, text: str) -> str:
        return self.majority_intent

    def draft(self, text: str) -> str:
        return self.CANNED_REPLY

    def should_escalate(self, text: str) -> bool:
        return True  # escalate 100% of tickets, per README


# --------------------------------------------------------------------
# Simple baseline
# --------------------------------------------------------------------

# Keyword regex escalation, per README.md's "Simple baseline" section.
# Deliberately crude (word-boundary substring match, no NLP) -- it's
# meant to be a floor for the LLM's flag-based escalation logic
# (escalation.py) to beat, not a competitor to it.
ESCALATE_KEYWORDS = [
    r"\brefund\b", r"\blost\b", r"\bstolen\b", r"\bsue\b", r"\blawyer\b",
    r"\blegal\b", r"\bhacked\b", r"\bcompromised\b", r"\bunauthorized\b",
    r"\bcancel(l)?ing?\b", r"\bcompetitor\b",
]
_ESCALATE_RE = re.compile("|".join(ESCALATE_KEYWORDS), re.IGNORECASE)


class SimpleBaseline:
    def __init__(self, golden_labels_path: str = GOLDEN_LABELS_PATH):
        self.retriever = ReplyRetriever()
        self._clf = None
        self._vec = None
        self._n_train = 0
        self._fit_classifier(golden_labels_path)

    def _fit_classifier(self, path: str):
        try:
            labels = read_labels(path)
        except FileNotFoundError:
            return
        if len(labels) < 10:
            return  # not enough rows yet for a meaningful fit -- classify() falls back to None
        self._vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2), stop_words="english")
        X = self._vec.fit_transform(labels["customer_text"])
        y = labels["human_intent"]
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(X, y)
        self._n_train = len(labels)

    def classify(self, text: str) -> str | None:
        """Returns None if there aren't enough golden labels yet to fit
        on -- caller should treat that as "baseline not ready", not as
        a real prediction of an actual class."""
        if self._clf is None:
            return None
        return self._clf.predict(self._vec.transform([text]))[0]

    def draft(self, text: str, exclude_tweet_id=None) -> str:
        """1-nearest-neighbor retrieval: return the raw historical brand
        reply text for the single most similar past customer tweet, no
        LLM rewriting."""
        neighbors = self.retriever.top_k(text, k=1, exclude_tweet_id=exclude_tweet_id)
        if neighbors.empty:
            return TrivialBaseline.CANNED_REPLY  # no similar-enough past example -- fall back rather than return nothing
        return neighbors.iloc[0]["brand_text"]

    def should_escalate(self, text: str) -> bool:
        return bool(_ESCALATE_RE.search(text))


def fit_and_report():
    trivial = TrivialBaseline()
    simple = SimpleBaseline()

    print("=== Trivial baseline ===")
    print(f"Majority-class intent: {trivial.majority_intent}")
    print(f"Canned reply: {trivial.CANNED_REPLY}")
    print("Escalation: always")

    print("\n=== Simple baseline ===")
    if simple._clf is None:
        print(f"Classifier: not fit -- need >=10 rows in {GOLDEN_LABELS_PATH}, "
              f"have {0 if not __import__('os').path.exists(GOLDEN_LABELS_PATH) else len(pd.read_csv(GOLDEN_LABELS_PATH))}. "
              f"Rerun after more labeling.")
    else:
        golden = read_labels()
        X = simple._vec.transform(golden["customer_text"])
        y = golden["human_intent"]
        # 5-fold CV accuracy, not train accuracy -- LogisticRegression on
        # TF-IDF features will happily overfit ~85 examples across 13
        # classes and report a misleadingly high train-set number.
        n_folds = min(5, y.value_counts().min())  # can't do more folds than the smallest class has examples
        if n_folds >= 2:
            scores = cross_val_score(simple._clf, X, y, cv=n_folds)
            print(f"Classifier trained on {simple._n_train} labeled examples, {n_folds}-fold CV accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
        else:
            print(f"Classifier trained on {simple._n_train} labeled examples -- too few per-class examples for CV yet.")
    print("Drafting: 1-NN retrieval (raw historical reply, no rewriting)")
    print(f"Escalation keywords: {ESCALATE_KEYWORDS}")

    return trivial, simple


if __name__ == "__main__":
    fit_and_report()
