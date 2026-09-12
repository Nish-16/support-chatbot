"""
Step 1: load the raw Kaggle CSV and print its shape/structure.

The dataset is one flat table of tweets (not pre-split into threads).
Each row is a single tweet, and rows link to each other via tweet IDs
to reconstruct a conversation. We're just looking at the shape of that
table for now.
"""
import pandas as pd

df = pd.read_csv("data/twcs/twcs.csv")

print("shape (rows, columns):", df.shape)
print()
print("columns and dtypes:")
print(df.dtypes)
print()
print("first 5 rows:")
print(df.head())
print()
print("null counts per column:")
print(df.isnull().sum())
