"""
Embedding retrieval over dropbox_paired.csv, backed by Chroma.

Drop-in alternative to retrieval.ReplyRetriever: same `top_k(query_text,
k, exclude_tweet_id)` signature, same returned columns, so 09_draft_reply
and 13_reply_eval can switch between them with a flag and the comparison
is apples to apples.

Why a vector store at all, given TF-IDF answers in ~2ms
-------------------------------------------------------
TF-IDF matches *wording*. The golden set contains a documented case it
structurally cannot handle: "why did you remove the green check..." and
"I'm missing the green check, now see only the white box" are about the
same thing at a TF-IDF cosine similarity of **0.23** -- near-identical
meaning, almost no shared vocabulary. Embeddings score the same pair
0.53. Retrieval that treats those as unrelated hands the drafter the
wrong grounding examples.

Embeddings are the fix for that class. The vector *store* is not the
interesting part at 4,504 documents -- a numpy array would do -- but
Chroma gives persistence, metadata filtering and a stable query API for
free, and keeps the door open to a corpus that doesn't fit in memory.

Measured, and not a free win: 24_retrieval_ab.py shows +72% relative
P@3 over TF-IDF, but that has not yet translated into better replies --
see the retrieval section of README.md.

Cost, stated honestly: this is the only part of the project that needs a
model download (~80MB ONNX MiniLM, cached after the first run) and it
takes ~1 minute to embed the corpus once. The index is persisted to
data/chroma/, so subsequent runs are instant and offline. The TF-IDF path
remains the default precisely so `12_evaluate.py` and the headline stay
reproducible with nothing but pandas and scikit-learn.

Build the index (once):  ./.venv/Scripts/python.exe scripts/vector_retrieval.py --build
"""
import argparse
import os

import pandas as pd

from retrieval import PAIRED_PATH, RETRIEVAL_VERSION, _clean, load_pairs

CHROMA_PATH = "data/chroma"
COLLECTION = "dropbox_pairs"


class VectorReplyRetriever:
    """Same contract as ReplyRetriever, backed by Chroma + MiniLM embeddings."""

    def __init__(self, paired_path: str = PAIRED_PATH, persist_path: str = CHROMA_PATH,
                 build: bool = False):
        import chromadb  # imported lazily: the TF-IDF path must not require chromadb

        # Same collapsed corpus as the TF-IDF path -- one row per customer
        # tweet, replies stitched -- so the two retrievers index identical
        # documents and the A/B compares retrieval, not corpus shape.
        self.df = load_pairs(paired_path)

        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            # Cosine, to match what the TF-IDF path reports. Chroma defaults to
            # squared L2, which would make the two retrievers' `similarity`
            # columns incomparable -- the whole point of this class is that
            # they ARE comparable.
            metadata={"hnsw:space": "cosine"},
        )
        # Document ids are positions in self.df, so an index built against a
        # different corpus shape would map every hit to the wrong row and
        # return confidently wrong neighbours. Count mismatch is the cheap
        # detector; rebuild rather than ask the caller to remember.
        if build or self.collection.count() != len(self.df):
            if self.collection.count() not in (0, len(self.df)):
                print(f"Index holds {self.collection.count()} docs but the corpus has "
                      f"{len(self.df)} -- stale, rebuilding.")
            self._build()

    def _build(self):
        """Embed and index every pair. Idempotent -- ids are row indices, so a
        rerun upserts rather than duplicating."""
        texts = self.df["customer_text"].map(_clean).tolist()
        print(f"Embedding {len(texts)} rows into {CHROMA_PATH} "
              f"(first run downloads ~80MB MiniLM, then cached)...")
        # Batched: Chroma caps a single add, and batching keeps peak memory flat.
        batch = 500
        for start in range(0, len(texts), batch):
            stop = min(start + batch, len(texts))
            chunk = self.df.iloc[start:stop]
            self.collection.upsert(
                ids=[str(i) for i in range(start, stop)],
                documents=texts[start:stop],
                metadatas=[{"customer_tweet_id": int(r.customer_tweet_id)}
                           for r in chunk.itertuples()],
            )
            print(f"  {stop}/{len(texts)}")
        print(f"Indexed {self.collection.count()} rows.")

    def top_k(self, query_text: str, k: int = 3, exclude_tweet_id=None) -> pd.DataFrame:
        """The k most semantically similar past (customer_text, brand_text)
        pairs, with a `similarity` column on the same 0-1 cosine scale the
        TF-IDF retriever reports.

        Over-fetches so that dropping the query's own record (when drafting for
        a tweet that is itself in the corpus) still returns k neighbours,
        rather than silently returning k-1."""
        result = self.collection.query(
            query_texts=[_clean(query_text)],
            n_results=min(k + 5, self.collection.count()),
        )
        rows = []
        for doc_id, meta, distance in zip(result["ids"][0],
                                          result["metadatas"][0],
                                          result["distances"][0]):
            if len(rows) >= k:
                break
            row = self.df.iloc[int(doc_id)]
            if exclude_tweet_id is not None and row["customer_tweet_id"] == exclude_tweet_id:
                continue
            # Chroma returns cosine DISTANCE; 1 - d puts it on the same scale
            # as cosine_similarity in the TF-IDF path.
            similarity = 1.0 - float(distance)
            if similarity <= 0:
                break  # same rule as TF-IDF: fewer examples beats irrelevant ones
            rows.append({
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "brand_text": row["brand_text"],
                "similarity": similarity,
            })
        return pd.DataFrame(rows)


def get_retriever(kind: str = "tfidf"):
    """One place that maps a --retriever flag to an implementation, so callers
    don't each grow their own if/else."""
    if kind == "tfidf":
        from retrieval import ReplyRetriever
        return ReplyRetriever()
    if kind == "vector":
        return VectorReplyRetriever()
    raise ValueError(f"unknown retriever {kind!r} -- use 'tfidf' or 'vector'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the index")
    args = ap.parse_args()

    retriever = VectorReplyRetriever(build=args.build)
    demo = "I can't log in to my account, it keeps saying wrong password even after I reset it"
    neighbors = retriever.top_k(demo, k=3)
    print(f"\nQuery: {demo}\n")
    for _, row in neighbors.iterrows():
        print(f"  sim={row['similarity']:.3f} tweet_id={row['customer_tweet_id']}")
        print(f"    customer: {row['customer_text']}")
        print(f"    brand:    {row['brand_text']}\n")
