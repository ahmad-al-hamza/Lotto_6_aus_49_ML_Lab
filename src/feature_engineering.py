import pandas as pd
import numpy as np


# =========================
# Load data
# =========================

df = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

number_columns = [
    "n1", "n2", "n3",
    "n4", "n5", "n6"
]


# =========================
# Basic draw features
# =========================

df["sum"] = df[number_columns].sum(axis=1)

df["mean"] = df[number_columns].mean(axis=1)

df["std"] = df[number_columns].std(axis=1)

df["min"] = df[number_columns].min(axis=1)

df["max"] = df[number_columns].max(axis=1)

df["range"] = (
    df["max"] - df["min"]
)


# =========================
# Odd / Even
# =========================

df["odd_count"] = (
    df[number_columns]
    .apply(lambda row: (row % 2 != 0).sum(), axis=1)
)

df["even_count"] = 6 - df["odd_count"]


# =========================
# Consecutive numbers
# =========================

def consecutive_features(numbers):

    numbers = sorted(numbers)

    consecutive_pairs = 0
    max_streak = 1
    current_streak = 1

    for i in range(1, len(numbers)):

        if numbers[i] == numbers[i - 1] + 1:

            consecutive_pairs += 1
            current_streak += 1

            max_streak = max(
                max_streak,
                current_streak
            )

        else:

            current_streak = 1

    return consecutive_pairs, max_streak


consecutive_results = df[number_columns].apply(
    lambda row: consecutive_features(row.tolist()),
    axis=1
)

df["consecutive_pairs"] = (
    consecutive_results.apply(lambda x: x[0])
)

df["consecutive_max"] = (
    consecutive_results.apply(lambda x: x[1])
)


# =========================
# Number frequency
# =========================

number_frequency = {}

for number in range(1, 50):

    number_frequency[number] = (
        df[number_columns]
        .eq(number)
        .sum()
    )


df["frequency_sum"] = (
    df[number_columns]
    .map(lambda x: number_frequency[x])
    .sum(axis=1)
)

df["frequency_mean"] = (
    df["frequency_sum"] / 6
)


# =========================
# Display results
# =========================

feature_columns = [
    "date",
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "range",
    "odd_count",
    "even_count",
    "consecutive_pairs",
    "consecutive_max",
    "frequency_sum",
    "frequency_mean"
]


print("\nFeature Dataset:")
print(
    df[feature_columns]
    .head(10)
    .to_string(index=False)
)


print("\nDataset shape:")
print(df.shape)


print("\nFeature columns:")
print(feature_columns)