import pandas as pd
import numpy as np


# =========================
# Load data
# =========================

df = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

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

df["range"] = df["max"] - df["min"]


# =========================
# Odd / Even
# =========================

df["odd_count"] = (
    df[number_columns]
    .apply(
        lambda row: (row % 2 != 0).sum(),
        axis=1
    )
)

df["even_count"] = 6 - df["odd_count"]


# =========================
# Consecutive numbers
# =========================

def consecutive_features(numbers):

    numbers = sorted(numbers)

    pairs = 0
    max_streak = 1
    current_streak = 1

    for i in range(1, len(numbers)):

        if numbers[i] == numbers[i - 1] + 1:

            pairs += 1
            current_streak += 1

            max_streak = max(
                max_streak,
                current_streak
            )

        else:

            current_streak = 1

    return pairs, max_streak


results = df[number_columns].apply(
    lambda row:
    consecutive_features(row.tolist()),
    axis=1
)

df["consecutive_pairs"] = results.apply(
    lambda x: x[0]
)

df["consecutive_max"] = results.apply(
    lambda x: x[1]
)


# =========================
# Historical number frequency
# =========================
#
# IMPORTANT:
# Only use previous draws.
#
# shift(1) prevents the current draw
# from influencing its own features.
# =========================

frequency_history = np.zeros(49)

frequency_means = []

for _, row in df.iterrows():

    current_numbers = row[number_columns].values

    total_previous = frequency_history.sum()

    if total_previous == 0:

        frequency_means.append(np.nan)

    else:

        current_frequencies = [
            frequency_history[int(number) - 1]
            for number in current_numbers
        ]

        frequency_means.append(
            np.mean(current_frequencies)
        )

    # Update history AFTER calculating feature

    for number in current_numbers:

        frequency_history[int(number) - 1] += 1


df["historical_frequency_mean"] = (
    frequency_means
)


# =========================
# Remove rows without history
# =========================

df = df.dropna(
    subset=["historical_frequency_mean"]
)


# =========================
# Select features
# =========================

feature_columns = [
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
    "historical_frequency_mean"
]


print("\nML Feature Dataset:")

print(
    df[
        ["date"] + feature_columns
    ]
    .head(10)
    .to_string(index=False)
)


print("\nDataset shape:")
print(df.shape)


print("\nFeature columns:")
print(feature_columns)


print("\nMissing values:")
print(
    df[feature_columns]
    .isna()
    .sum()
)