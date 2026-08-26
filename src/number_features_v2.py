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
# Binary draw matrix
# =========================

draw_matrix = np.zeros(
    (len(df), 49),
    dtype=np.int8
)

for i, row in df[number_columns].iterrows():

    for number in row:

        draw_matrix[
            i,
            int(number) - 1
        ] = 1


# =========================
# Create samples
# =========================

samples = []

windows = [
    5,
    10,
    20,
    50,
    100,
    200
]


# Need at least 200 historical draws
for i in range(200, len(df) - 1):

    history = draw_matrix[:i]

    next_draw = draw_matrix[i + 1]

    for number_index in range(49):

        number = number_index + 1

        features = {
            "date": df.loc[i, "date"],
            "number": number
        }

        # =========================
        # Frequency features
        # =========================

        frequencies = {}

        for window in windows:

            recent = history[-window:, number_index]

            frequency = recent.sum()

            frequencies[window] = frequency

            features[
                f"freq_{window}"
            ] = frequency


        # =========================
        # Gap
        # =========================

        occurrences = np.where(
            history[:, number_index] == 1
        )[0]

        if len(occurrences) == 0:

            gap = i + 1

        else:

            gap = i - occurrences[-1]

        features["gap"] = gap


        # =========================
        # Recent vs long-term
        # =========================

        recent_rate = (
            frequencies[20] / 20
        )

        long_rate = (
            frequencies[200] / 200
        )

        if long_rate > 0:

            recent_vs_long = (
                recent_rate / long_rate
            )

        else:

            recent_vs_long = 1.0

        features[
            "recent_vs_long"
        ] = recent_vs_long


        # =========================
        # Target
        # =========================

        features["target"] = int(
            next_draw[number_index]
        )

        samples.append(features)


# =========================
# DataFrame
# =========================

features_df = pd.DataFrame(
    samples
)


# =========================
# Save
# =========================

features_df.to_csv(
    "data/processed/number_features_v2.csv",
    index=False
)


# =========================
# Results
# =========================

print("\nNumber Feature Dataset V2:")

print(
    features_df.head(20)
    .to_string(index=False)
)


print("\nDataset shape:")

print(
    features_df.shape
)


print("\nFeatures:")

print(
    features_df.columns.tolist()
)


print("\nTarget distribution:")

print(
    features_df["target"]
    .value_counts()
)


print("\nTarget percentage:")

print(
    features_df["target"]
    .value_counts(
        normalize=True
    ) * 100
)


print("\nMissing values:")

print(
    features_df.isna().sum()
)