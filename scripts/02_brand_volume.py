"""
Step 2: figure out which brand to focus on.

Brand accounts are rows where inbound == False (the company replying)
and author_id is a real handle (not an anonymized customer number).
We count how many *support replies* each brand sent -- that's a decent
proxy for how much data we'll have to work with after filtering the
whole dataset down to just that brand's conversations.
"""
import pandas as pd

df = pd.read_csv("data/twcs/twcs.csv")

brand_replies = df[df["inbound"] == False]
volume = brand_replies["author_id"].value_counts()

print("top 30 brands by number of support replies sent:")
print(volume.count())
print(volume.head(54))

n = len(volume)
middle = n // 2
print()
print("middle 10 brands:")
print(volume.iloc[middle-5:middle+5])
