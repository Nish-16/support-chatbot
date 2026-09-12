"""
TF-IDF vs. embedding retrieval, measured rather than assumed.

Adding a vector store is easy to justify with a story ("embeddings capture
meaning") and hard to justify with a number. This script produces the number.

The metric
----------
Retrieval quality needs relevance labels, and the golden set has them: 174
tweets, each with a hand-adjudicated intent. So the evaluation restricts the
corpus to the golden set itself and asks, for each tweet, whether the
neighbours it retrieves share its intent.

  precision@k -- of the k retrieved neighbours, what fraction share the
                 query's gold intent
  MRR         -- 1/rank of the first neighbour sharing the gold intent

Both retrievers index exactly the same 174 documents and answer exactly the
same 174 queries, each excluding itself.

What this does and does not prove
---------------------------------
It measures *topical* retrieval on a 174-document corpus. That is a proxy: a
good grounding example shares the customer's problem, and same-intent is the
best available stand-in for that. It is not the production setting (5,938
rows), and same-intent is not identical to same-problem -- two
`sync_app_bug` tweets can need different answers. Reported as a directional
comparison between two retrievers on one axis, not as retrieval accuracy.

Also reported: the case that motivated the whole experiment. Two golden
tweets about the missing green-check icon are near-identical in meaning and
share almost no vocabulary. TF-IDF scores them 0.153. Whether embeddings fix
that specific pair is a concrete, checkable claim.

Usage: ./.venv/Scripts/python.exe scripts/24_retrieval_ab.py
       (needs the Chroma index -- scripts/vector_retrieval.py --build)
"""
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval import _clean

LABELS_PATH = "data/golden_labels_v3.csv"
K = 3

# The pair from the consistency audit: same meaning, disjoint vocabulary.
# Loaded by id from the real data rather than retyped -- a paraphrase would
# quietly measure a different pair than the one the audit reported on.
GREEN_CHECK_IDS = (1863374, 615215)


def load_golden() -> pd.DataFrame:
    df = pd.read_csv(LABELS_PATH)
    df = df[df["v3_status"] == "resolved"]
    return df[["customer_tweet_id", "customer_text", "human_intent_v3"]].reset_index(drop=True)


class TfidfIndex:
    name = "TF-IDF (incumbent)"

    def __init__(self, texts: list[str]):
        self.vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vec.fit_transform([_clean(t) for t in texts])

    def rank(self, query: str) -> np.ndarray:
        """Document indices, most similar first."""
        sims = cosine_similarity(self.vec.transform([_clean(query)]), self.matrix)[0]
        return sims.argsort()[::-1]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.vec.transform([_clean(a)]), self.vec.transform([_clean(b)])
        return float(cosine_similarity(va, vb)[0][0])


class VectorIndex:
    name = "Embeddings (Chroma + MiniLM)"

    def __init__(self, texts: list[str]):
        import chromadb
        # A throwaway collection in a temp dir: the A/B must index exactly the
        # 174 evaluation documents, not the 5,938-row production collection,
        # or the two retrievers would be answering from different corpora.
        self._dir = tempfile.mkdtemp(prefix="chroma_ab_")
        client = chromadb.PersistentClient(path=self._dir)
        self.collection = client.get_or_create_collection(
            name="retrieval_ab", metadata={"hnsw:space": "cosine"})
        self.collection.add(
            ids=[str(i) for i in range(len(texts))],
            documents=[_clean(t) for t in texts],
        )
        self.n = len(texts)

    def rank(self, query: str) -> np.ndarray:
        res = self.collection.query(query_texts=[_clean(query)], n_results=self.n)
        return np.array([int(i) for i in res["ids"][0]])

    def similarity(self, a: str, b: str) -> float:
        ef = self.collection._embedding_function
        va, vb = ef([_clean(a)])[0], ef([_clean(b)])[0]
        va, vb = np.array(va), np.array(vb)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def cleanup(self):
        shutil.rmtree(self._dir, ignore_errors=True)


def evaluate(index, golden: pd.DataFrame, k: int = K) -> dict:
    intents = golden["human_intent_v3"].values
    precisions, rr = [], []
    t0 = time.time()
    for i, query in enumerate(golden["customer_text"]):
        order = [j for j in index.rank(query) if j != i]  # never retrieve yourself
        top = order[:k]
        hits = [intents[j] == intents[i] for j in top]
        precisions.append(np.mean(hits) if hits else 0.0)
        first = next((r for r, j in enumerate(order, 1) if intents[j] == intents[i]), None)
        rr.append(1.0 / first if first else 0.0)
    elapsed = time.time() - t0
    return {
        "p_at_1": float(np.mean([p for p in
                                 [1.0 if intents[[j for j in index.rank(q) if j != i][0]] == intents[i] else 0.0
                                  for i, q in enumerate(golden["customer_text"])]])),
        "p_at_k": float(np.mean(precisions)),
        "mrr": float(np.mean(rr)),
        "query_ms": elapsed / len(golden) * 1000,
    }


def green_check(golden: pd.DataFrame):
    """The motivating case, scored on the PRODUCTION indexes (5,938 rows), not
    the 174-document A/B corpus -- that is the setting the 0.153 figure in the
    consistency audit came from, and the only one comparable to it."""
    by_id = golden.set_index("customer_tweet_id")["customer_text"]
    try:
        a, b = (by_id.loc[i] for i in GREEN_CHECK_IDS)
    except KeyError:
        print("\n  (green-check pair not present in the label file; skipped)")
        return

    print("\n  The motivating case -- same meaning, disjoint vocabulary,")
    print("  scored on the full 5,938-row production indexes:")
    print(f"    A: {a}")
    print(f"    B: {b}")

    from retrieval import ReplyRetriever
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    tf = ReplyRetriever()
    va, vb = tf.vectorizer.transform([_clean(a)]), tf.vectorizer.transform([_clean(b)])
    print(f"\n    TF-IDF (incumbent)               similarity {float(_cos(va, vb)[0][0]):.3f}")

    try:
        from vector_retrieval import VectorReplyRetriever
        vr = VectorReplyRetriever()
        ef = vr.collection._embedding_function
        ea, eb = np.array(ef([_clean(a)])[0]), np.array(ef([_clean(b)])[0])
        sim = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        print(f"    Embeddings (Chroma + MiniLM)     similarity {sim:.3f}")
    except Exception as exc:
        print(f"    Embeddings: unavailable ({exc}) -- run vector_retrieval.py --build")


def main():
    golden = load_golden()
    texts = golden["customer_text"].tolist()
    print(f"Corpus and query set: {len(golden)} golden-labelled tweets, "
          f"{golden['human_intent_v3'].nunique()} intents.")
    print(f"Chance precision@{K} (random neighbour sharing the query's intent): "
          f"{(golden['human_intent_v3'].value_counts(normalize=True) ** 2).sum():.3f}\n")

    results = {}
    for cls in (TfidfIndex, VectorIndex):
        t0 = time.time()
        index = cls(texts)
        build = time.time() - t0
        stats = evaluate(index, golden)
        stats["build_s"] = build
        results[cls.name] = stats
        if hasattr(index, "cleanup"):
            index.cleanup()

    print(f"{'retriever':32s} {'P@1':>7s} {f'P@{K}':>7s} {'MRR':>7s} "
          f"{'build':>8s} {'query':>9s}")
    for name, s in results.items():
        print(f"{name:32s} {s['p_at_1']:7.3f} {s['p_at_k']:7.3f} {s['mrr']:7.3f} "
              f"{s['build_s']:7.2f}s {s['query_ms']:8.2f}ms")

    a, b = list(results.values())
    delta = b["p_at_k"] - a["p_at_k"]
    print(f"\n  Embeddings over TF-IDF on P@{K}: {delta:+.3f} "
          f"({delta / max(a['p_at_k'], 1e-9) * 100:+.1f}% relative)")
    print(f"  Cost of that: {b['build_s'] / max(a['build_s'], 1e-9):.0f}x build time, "
          f"{b['query_ms'] / max(a['query_ms'], 1e-9):.0f}x query latency, "
          f"and an 83MB model download.")

    green_check(golden)

    print("\n  CAVEAT: this measures topical retrieval on a 174-document corpus.")
    print("  Same-intent is a proxy for same-problem, and the production corpus")
    print("  is 5,938 rows. Directional, not an accuracy figure.")


if __name__ == "__main__":
    main()
