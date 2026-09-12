"""
Shared TF-IDF retrieval over dropbox_paired.csv, used for grounded reply
drafting (09_draft_reply.py) and the simple-baseline 1-NN drafter
(10_baselines.py) -- pulled out to one place so both use the same index
instead of two slightly different TF-IDF fits.

TF-IDF, not embeddings: no vector DB / embedding API in this project
yet, and 5,938 rows fits comfortably in memory as a sparse matrix --
cosine similarity over it is milliseconds. Revisit if retrieval quality
turns out to be the bottleneck once the eval harness can measure it.

One customer tweet, one grounding example
-----------------------------------------
The file has 5,938 rows but only 4,504 unique customer tweets: a brand reply
split across several tweets appears as several rows sharing one customer
tweet. Ranked retrieval put those rows next to each other, so a k=3 lookup
routinely spent two of its three slots on the *same customer situation*.
Measured before this was fixed: 7 of 12 embedding queries returned a
duplicate, and distinct neighbours averaged 2.17 of 3.

So the corpus is collapsed to one row per customer tweet at load time, with
that tweet's brand replies stitched back together in `brand_created_at`
order. Ordering is taken from the data, not assumed from row order -- which
is what makes stitching safe to do here rather than leaving the fragments
apart. `k` now means k distinct past situations.

Remaining caveat: some resolutions happen entirely off-thread ("we've
replied to your DM!"), so a stitched reply can still contain no answer.
That is a property of the dataset, and the deflection policy in
09_draft_reply.py exists because of it.
"""
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAIRED_PATH = "data/dropbox_paired.csv"

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")

# Bumped whenever a change alters what top_k returns. Stored alongside every
# reply-eval row, so results produced under different retrieval behaviour are
# never silently averaged together -- the same reason rubric_version exists.
RETRIEVAL_VERSION = "dedup1"


def _clean(text: str) -> str:
    """Strip @mentions and URLs before vectorizing -- otherwise every
    tweet's shared "@DropboxSupport" mention and boilerplate short-link
    domain dominate the vocabulary and swamp the actual topic signal."""
    text = _URL_RE.sub(" ", str(text))
    text = _MENTION_RE.sub(" ", text)
    return text


def load_pairs(paired_path: str = PAIRED_PATH) -> pd.DataFrame:
    """One row per customer tweet, with its brand replies stitched in
    chronological order. Shared by both retrievers so they index exactly the
    same corpus and stay comparable."""
    df = pd.read_csv(paired_path)
    df = df.dropna(subset=["customer_text", "brand_text"])

    # Twitter's "Mon Nov 20 19:21:31 +0000 2017" format. errors="coerce" keeps
    # an unparseable timestamp from dropping the row; those sort last, which
    # is no worse than the arbitrary order they had before.
    order = pd.to_datetime(df["brand_created_at"], format="%a %b %d %H:%M:%S %z %Y",
                           errors="coerce")
    df = df.assign(_order=order).sort_values(["customer_tweet_id", "_order"],
                                             na_position="last")

    stitched = (df.groupby("customer_tweet_id", sort=False)
                  .agg(customer_text=("customer_text", "first"),
                       brand_text=("brand_text", lambda s: " ".join(s.astype(str))),
                       n_parts=("brand_text", "size"))
                  .reset_index())
    return stitched


class ReplyRetriever:
    def __init__(self, paired_path: str = PAIRED_PATH):
        self.df = load_pairs(paired_path)
        cleaned = self.df["customer_text"].map(_clean)
        self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(cleaned)

    def top_k(self, query_text: str, k: int = 3, exclude_tweet_id=None) -> pd.DataFrame:
        """Returns the k most textually-similar past (customer_text,
        brand_text) pairs, ranked by cosine similarity, with a
        `similarity` column. exclude_tweet_id lets a caller drafting a
        reply for a tweet that's ALSO in dropbox_paired.csv (e.g. a
        golden-set example) exclude its own exact record from its own
        neighbor list."""
        query_vec = self.vectorizer.transform([_clean(query_text)])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        order = sims.argsort()[::-1]
        rows = []
        for idx in order:
            if len(rows) >= k:
                break
            if sims[idx] <= 0:
                break  # no more relevant matches -- padding with zero-similarity rows is worse than fewer examples
            row = self.df.iloc[idx]
            if exclude_tweet_id is not None and row["customer_tweet_id"] == exclude_tweet_id:
                continue
            rows.append({
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "brand_text": row["brand_text"],
                "similarity": float(sims[idx]),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    retriever = ReplyRetriever()
    demo = "I can't log in to my account, it keeps saying wrong password even after I reset it"
    neighbors = retriever.top_k(demo, k=3)
    print(f"Query: {demo}\n")
    for _, row in neighbors.iterrows():
        print(f"  sim={row['similarity']:.3f} tweet_id={row['customer_tweet_id']}")
        print(f"    customer: {row['customer_text']}")
        print(f"    brand:    {row['brand_text']}\n")
