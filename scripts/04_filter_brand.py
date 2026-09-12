"""
Step 3: filter the full dataset down to DropboxSupport conversations
and pair each customer tweet with the brand's reply to it.

Why pairs, not raw rows: a customer's tweet on its own is what we'll
classify by intent, and the brand's reply next to it is what we'll use
later to ground our own generated replies ("here's how Dropbox actually
handled this kind of issue before"). Having them side by side now also
makes the data much easier to read through.

Note: some conversations are multi-turn (back-and-forth), so a single
customer tweet_id can appear more than once if Dropbox replied, the
customer replied again, and Dropbox replied again. For now we keep it
simple: one row per (customer tweet -> its direct brand reply).
"""
import pandas as pd

BRAND = "DropboxSupport"

df = pd.read_csv("data/twcs/twcs.csv")

brand_replies = df[df["author_id"] == BRAND].copy()
brand_replies = brand_replies.rename(columns={
    "tweet_id": "brand_tweet_id",
    "text": "brand_text",
    "created_at": "brand_created_at",
})

customer_tweets = df[df["inbound"] == True].copy()
customer_tweets = customer_tweets.rename(columns={
    "tweet_id": "customer_tweet_id",
    "text": "customer_text",
    "created_at": "customer_created_at",
    "author_id": "customer_id",
})

# join: brand reply's in_response_to_tweet_id == customer's tweet_id
paired = brand_replies.merge(
    customer_tweets[["customer_tweet_id", "customer_text", "customer_created_at", "customer_id"]],
    left_on="in_response_to_tweet_id",
    right_on="customer_tweet_id",
    how="inner",
)

paired = paired[[
    "customer_tweet_id", "customer_id", "customer_created_at", "customer_text",
    "brand_tweet_id", "brand_created_at", "brand_text",
]]

print(f"paired customer<->brand tweets for {BRAND}: {len(paired)}")
paired.to_csv("data/dropbox_paired.csv", index=False)
print("saved to data/dropbox_paired.csv")
