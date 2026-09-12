"""
Step 4: pull a random sample of paired conversations to read through
and start defining intent categories by hand.

random_state=42 makes this reproducible -- rerunning gives the same
sample, so if we come back to this later we see the same examples.
"""
import pandas as pd

df = pd.read_csv("data/dropbox_paired.csv")
sample = df.sample(n=40, random_state=42)

with open("data/reading_sample.txt", "w", encoding="utf-8") as f:
    for i, row in enumerate(sample.itertuples(), 1):
        f.write(f"--- example {i} (customer_tweet_id={row.customer_tweet_id}) ---\n")
        f.write(f"CUSTOMER: {row.customer_text}\n")
        f.write(f"DROPBOX:  {row.brand_text}\n\n")

print("wrote data/reading_sample.txt")
