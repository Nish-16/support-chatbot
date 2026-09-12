"""
Step 2b: for a shortlist of mid-sized brands, compute the REAL filtered
dataset size -- not just the brand's own replies, but every tweet that
belongs to a thread involving that brand (the customer's tweets too).

Logic: find every tweet_id the brand ever replied to
(in_response_to_tweet_id on the brand's own rows), then count:
  - the brand's own reply tweets
  - the customer tweets those replies were responding to
This double-counts a bit if a customer tweet got multiple replies, so
we dedupe by tweet_id.
"""
import pandas as pd

df = pd.read_csv("data/twcs/twcs.csv")

shortlist = [
    "DropboxSupport",
    "DellCares",
    "HPSupport",
    "GloCare",
    "AskCiti",
    "VirginAtlantic",
    "TacoBellTeam",
    "nationalrailenq",
]

for brand in shortlist:
    brand_replies = df[df["author_id"] == brand]
    replied_to_ids = brand_replies["in_response_to_tweet_id"].dropna().unique()
    customer_tweets = df[df["tweet_id"].isin(replied_to_ids)]

    total_ids = set(brand_replies["tweet_id"]) | set(customer_tweets["tweet_id"])
    print(f"{brand:20s} brand replies={len(brand_replies):6d}  "
          f"customer tweets={len(customer_tweets):6d}  "
          f"total filtered tweets={len(total_ids):6d}")
