import pandas as pd
from itertools import combinations
from collections import Counter

# Load cleaned dataset
df = pd.read_csv("data/processed/lotto_6aus49_clean.csv")

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

pair_counter = Counter()

# Go through every draw
for _, row in df[number_columns].iterrows():

    numbers = row.tolist()

    # Generate all pairs from the 6 numbers
    pairs = combinations(numbers, 2)

    # Count each pair
    pair_counter.update(pairs)

# Convert to DataFrame
pair_frequency = pd.DataFrame(
    pair_counter.items(),
    columns=["pair", "frequency"]
)

# Sort by frequency
pair_frequency = pair_frequency.sort_values(
    "frequency",
    ascending=False
)

print("Total unique pairs:", len(pair_frequency))

print("\nTop 20 most frequent pairs:")
print(pair_frequency.head(20).to_string(index=False))
print("\nPair frequency statistics:")

print("Mean:")
print(pair_frequency["frequency"].mean())

print("\nStandard deviation:")
print(pair_frequency["frequency"].std())

print("\nMinimum:")
print(pair_frequency["frequency"].min())

print("\nMaximum:")
print(pair_frequency["frequency"].max())

print("\nMedian:")
print(pair_frequency["frequency"].median())

print("\n95th percentile:")
print(pair_frequency["frequency"].quantile(0.95))

print("\n99th percentile:")
print(pair_frequency["frequency"].quantile(0.99))