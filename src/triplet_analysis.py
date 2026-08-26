import pandas as pd
from itertools import combinations
from collections import Counter
import numpy as np


# ==============================
# Configuration
# ==============================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]


# ==============================
# Load data
# ==============================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:")
print(df.shape)


# ==============================
# Generate triplets
# ==============================

triplet_counter = Counter()

for _, row in df.iterrows():

    numbers = sorted(
        row[number_columns].astype(int).tolist()
    )

    triplets = combinations(numbers, 3)

    for triplet in triplets:
        triplet_counter[triplet] += 1


# ==============================
# Convert to DataFrame
# ==============================

triplet_df = pd.DataFrame(
    triplet_counter.items(),
    columns=["triplet", "frequency"]
)

triplet_df = triplet_df.sort_values(
    "frequency",
    ascending=False
).reset_index(drop=True)


# ==============================
# Statistics
# ==============================

print("\nTotal unique triplets:")
print(len(triplet_df))


print("\nTop 20 most frequent triplets:")
print(
    triplet_df.head(20).to_string(index=False)
)


print("\nTriplet frequency statistics:")
print("--------------------------------")

print("Mean:")
print(triplet_df["frequency"].mean())

print("\nStandard deviation:")
print(triplet_df["frequency"].std())

print("\nMinimum:")
print(triplet_df["frequency"].min())

print("\nMaximum:")
print(triplet_df["frequency"].max())

print("\nMedian:")
print(triplet_df["frequency"].median())

print("\n95th percentile:")
print(
    np.percentile(
        triplet_df["frequency"],
        95
    )
)

print("\n99th percentile:")
print(
    np.percentile(
        triplet_df["frequency"],
        99
    )
)