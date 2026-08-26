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
# Create binary matrix
# =========================

# Each row = draw
# Each column = number 1-49
#
# 1 = number appeared
# 0 = number did not appear

draw_matrix = np.zeros(
    (len(df), 49),
    dtype=np.int8
)

for i, row in df[number_columns].iterrows():

    for number in row:

        draw_matrix[i, int(number) - 1] = 1


# =========================
# Create samples
# =========================

samples = []

windows = [10, 50, 100]

for i in range(100, len(df) - 1):

    # -------------------------
    # Historical data only
    # -------------------------

    history = draw_matrix[:i]

    next_draw = draw_matrix[i + 1]

    # -------------------------
    # Features for numbers 1-49
    # -------------------------

    for number_index in range(49):

        number = number_index + 1

        features = {
            "date": df.loc[i, "date"],
            "number": number
        }

        # -------------------------
        # Frequency windows
        # -------------------------

        for window in windows:

            recent = history[-window:, number_index]

            features[f"freq_{window}"] = (
                recent.sum()
            )

        # -------------------------
        # Gap
        # -------------------------

        occurrences = np.where(
            history[:, number_index] == 1
        )[0]

        if len(occurrences) == 0:

            gap = i + 1

        else:

            gap = i - occurrences[-1]

        features["gap"] = gap

        # -------------------------
        # Target
        # -------------------------

        features["target"] = int(
            next_draw[number_index]
        )

        samples.append(features)


# =========================
# Create DataFrame
# =========================

features_df = pd.DataFrame(samples)


# =========================
# Results
# =========================

print("\nNumber Feature Dataset:")

print(
    features_df.head(20)
    .to_string(index=False)
)

print("\nDataset shape:")

print(features_df.shape)

print("\nTarget distribution:")

print(
    features_df["target"]
    .value_counts()
)

print("\nTarget percentage:")

print(
    features_df["target"]
    .value_counts(normalize=True)
    * 100
)

print("\nMissing values:")

print(
    features_df.isna().sum()
)
features_df.to_csv(
    "data/processed/number_features.csv",
    index=False
)

print(
    "\nSaved to:"
    " data/processed/number_features.csv"
)