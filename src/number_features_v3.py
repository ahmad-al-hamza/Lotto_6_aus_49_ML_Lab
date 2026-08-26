import pandas as pd
import numpy as np

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]

WINDOWS = [5, 10, 20, 50, 100, 200]


# ==============================
# Load data
# ==============================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ==============================
# Basic draw features
# ==============================

df["draw_sum"] = df[NUMBER_COLUMNS].sum(axis=1)
df["draw_mean"] = df[NUMBER_COLUMNS].mean(axis=1)
df["draw_std"] = df[NUMBER_COLUMNS].std(axis=1)

df["draw_min"] = df[NUMBER_COLUMNS].min(axis=1)
df["draw_max"] = df[NUMBER_COLUMNS].max(axis=1)

df["draw_range"] = (
    df["draw_max"] - df["draw_min"]
)

df["odd_count"] = (
    df[NUMBER_COLUMNS] % 2
).sum(axis=1)

df["even_count"] = (
    df[NUMBER_COLUMNS] % 2 == 0
).sum(axis=1)


# ==============================
# Consecutive features
# ==============================

def consecutive_features(row):

    numbers = sorted(
        row[NUMBER_COLUMNS].astype(int).tolist()
    )

    consecutive_pairs = sum(
        numbers[i + 1] - numbers[i] == 1
        for i in range(len(numbers) - 1)
    )

    max_streak = 1
    current_streak = 1

    for i in range(1, len(numbers)):

        if numbers[i] == numbers[i - 1] + 1:
            current_streak += 1
            max_streak = max(
                max_streak,
                current_streak
            )
        else:
            current_streak = 1

    return consecutive_pairs, max_streak


consecutive = df.apply(
    consecutive_features,
    axis=1
)

df["consecutive_pairs"] = consecutive.apply(
    lambda x: x[0]
)

df["consecutive_max"] = consecutive.apply(
    lambda x: x[1]
)


# ==============================
# Previous draw features
# ==============================

previous_features = [
    "draw_sum",
    "draw_mean",
    "draw_std",
    "draw_range",
    "odd_count",
    "even_count",
    "consecutive_pairs",
    "consecutive_max"
]

for column in previous_features:

    df[f"previous_{column}"] = (
        df[column].shift(1)
    )


# ==============================
# Number-level dataset
# ==============================

rows = []

# Frequency history for each number
history = {
    number: []
    for number in range(1, 50)
}

last_seen = {
    number: None
    for number in range(1, 50)
}


for index, row in df.iterrows():

    current_date = row["date"]

    current_numbers = set(
        row[NUMBER_COLUMNS].astype(int)
    )

    for number in range(1, 50):

        # --------------------------
        # Historical frequency
        # --------------------------

        features = {}

        past = history[number]

        for window in WINDOWS:

            recent = past[-window:]

            features[f"freq_{window}"] = sum(
                recent
            )

        # --------------------------
        # Gap
        # --------------------------

        if last_seen[number] is None:
            gap = np.nan
        else:
            gap = index - last_seen[number]

        features["gap"] = gap

        # --------------------------
        # Recent vs long-term
        # --------------------------

        freq_20 = features["freq_20"]
        freq_200 = features["freq_200"]

        if freq_200 > 0:
            recent_vs_long = (
                freq_20 / (freq_200 / 10)
            )
        else:
            recent_vs_long = 0

        features["recent_vs_long"] = (
            recent_vs_long
        )

        # --------------------------
        # Previous draw information
        # --------------------------

        for column in previous_features:

            features[
                f"previous_{column}"
            ] = row[
                f"previous_{column}"
            ]

        # --------------------------
        # Time information
        # --------------------------

        features["draw_index"] = index

        features["year"] = (
            current_date.year
        )

        features["month"] = (
            current_date.month
        )

        features["day_of_week"] = (
            current_date.dayofweek
        )

        # --------------------------
        # Target
        # --------------------------

        target = int(
            number in current_numbers
        )

        rows.append({
            "date": current_date,
            "number": number,
            **features,
            "target": target
        })

    # ==========================
    # Update history AFTER
    # creating current features
    # ==========================

    for number in current_numbers:

        history[number].append(1)
        last_seen[number] = index

    for number in range(1, 50):

        if number not in current_numbers:
            history[number].append(0)


# ==============================
# Create dataset
# ==============================

features_df = pd.DataFrame(rows)

features_df = features_df.dropna()

features_df = features_df.reset_index(drop=True)


# ==============================
# Output
# ==============================

print("\nNumber Feature Dataset V3:")
print(
    features_df.head(20).to_string(
        index=False
    )
)

print("\nDataset shape:")
print(features_df.shape)

print("\nFeatures:")
print(
    features_df.columns.tolist()
)

print("\nTarget distribution:")
print(
    features_df["target"].value_counts()
)

print("\nTarget percentage:")
print(
    features_df["target"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nMissing values:")
print(
    features_df.isnull().sum()
)


# ==============================
# Save
# ==============================

features_df.to_csv(
    "data/processed/number_features_v3.csv",
    index=False
)

print(
    "\nSaved to: "
    "data/processed/number_features_v3.csv"
)