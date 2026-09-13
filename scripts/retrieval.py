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

Generic replies (optional filter)
---------------------------------
Some resolutions happen entirely off-thread ("we've replied to your DM!"), so
a stitched reply can still contain no answer. Retrieval ranks on the
*customer* tweet, so a perfect match on the problem can still hand the drafter
a reply with nothing in it -- the live green-check query got two "we'll add
your voice to the feedback" neighbours from the vector retriever, and a vaguer
draft for it. `filter_generic=True` skips those neighbours; see
is_generic_reply for the definition. Off by default, and stamped as
GENERIC_FILTER_VERSION on eval rows when on, so it is measured before it is
trusted.

Measured 2026-09-13 (README section 3): no reply-quality benefit on either
retriever -- every difference was within the same-configuration rerun noise.
It stays OFF. Kept for provenance, not as a recommended setting.
"""
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAIRED_PATH = "data/dropbox_paired.csv"

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")

# Bumped whenever a change alters what top_k returns. Stored alongside every
# reply-eval row, so results produced under different retrieval behaviour are
# never silently averaged together -- the same reason rubric_version exists.
RETRIEVAL_VERSION = "dedup1"

# Bumped whenever is_generic_reply's definition changes. Only stamped on eval
# rows produced with the filter ON, so unfiltered labels stay as they were.
GENERIC_FILTER_VERSION = "filt1"

# Phrases that carry nothing another customer could use. Removed as SPANS, not
# whole sentences: "thanks for the feedback" opens plenty of replies that go on
# to give a real answer (144 of 665 pattern matches on the corpus were over 180
# characters, most of them substantive), and dropping the sentence would take
# the answer with it.
_FILLER_RES = [re.compile(p) for p in (
    # the resolution already happened in DMs
    r"(replied|replying|responded|responding|reached (back )?out|sent over|following up)"
    r"[^.!?]{0,40}\b(dms?|message|inbox)\b",
    r"(check|look at) your (dms?|inbox)",
    r"bear with us",
    # feedback forwarded to some team: acknowledges, answers nothing
    r"add your voice[^.!?]*",
    r"\b(pass|passed|forward|forwarding|share|shared)\b[^.!?]{0,30}\b(along|over|feedback|team)\b",
    r"thanks? (you )?for (the|your|sharing|writing|reaching|checking)\w*( (feedback|this|that|in|out|us))?",
    r"we appreciate[^.!?]*",
    # dated incident status: true once, an invented fact when reused
    r"(disruption|temporary issue|experiencing issues?|this problem)[^.!?]{0,60}\b(resolved|fixed)",
    r"(up and running|fully restored)",
    r"still (having|experiencing) (any )?(issues|this)",
    # pleasantries
    r"(apologies|sorry) for the (inconvenience|trouble|delay)",
    r"hope (this|that|it) (helps|clarifies)",
    r"^\s*(hi|hey|hello|hola|bonjour)\s+\w+",
)]
_NON_CONTENT = ENGLISH_STOP_WORDS | {
    "dropbox", "thanks", "thank", "thx", "cheers", "please", "sorry", "apologies",
    "inconvenience", "sure", "hear", "happy", "day", "regards", "kind", "great",
    "amp", "let", "know", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "weekend",
}
_WORD_RE = re.compile(r"[a-z][a-z']+")
# Audited on 4,504 replies (2026-09-13). At 5, short real answers were flagged
# ("generally, clicking anywhere else will close it"; "do you have a ticket
# number we can investigate?"). At 4 they survive. Misses are the cheaper error:
# a filler neighbour the filter lets through is what happened before it existed,
# while a substantive one it drops is grounding the drafter never sees.
MIN_CONTENT_WORDS = 4


def is_generic_reply(text: str) -> bool:
    """True when a historical brand reply has nothing reusable in it: no link,
    and fewer than MIN_CONTENT_WORDS content words once filler phrases,
    handles, greetings and stop words are removed.

    Not a quality judgment on the original reply -- "we've replied to your DM!"
    was a fine thing to tweet at that customer. It is a judgment on whether it
    helps draft a reply to a DIFFERENT customer, which it does not. Replies that
    ask for something specific ("do you have any ticket IDs as reference to your
    support interactions?") keep their content words and are not flagged."""
    t = str(text).lower().replace("’", "'")
    if _URL_RE.search(t):
        return False  # a link is a pointer to the answer, which is what the drafter needs
    t = _MENTION_RE.sub(" ", t)
    for pattern in _FILLER_RES:
        t = pattern.sub(" ", t)
    words = [w.split("'")[0] for w in _WORD_RE.findall(t)]
    return sum(w not in _NON_CONTENT for w in words) < MIN_CONTENT_WORDS


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
    # Judged on the stitched reply: a "1/2" fragment that says nothing can be
    # followed by a "2/2" that carries the answer.
    stitched["is_generic"] = stitched["brand_text"].map(is_generic_reply)
    return stitched


class ReplyRetriever:
    def __init__(self, paired_path: str = PAIRED_PATH, filter_generic: bool = False):
        self.df = load_pairs(paired_path)
        self.filter_generic = filter_generic
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
            if self.filter_generic and row["is_generic"]:
                continue
            rows.append({
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "brand_text": row["brand_text"],
                "similarity": float(sims[idx]),
                "is_generic": bool(row["is_generic"]),
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
