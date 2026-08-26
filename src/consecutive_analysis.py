import pandas as pd


# -------------------------
# Load data
# -------------------------

df = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

number_columns = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6"
]


# -------------------------
# Calculate consecutive numbers
# -------------------------

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


# -------------------------
# Apply to every draw
# -------------------------

results = []

for _, row in df.iterrows():

    numbers = row[number_columns].tolist()

    pairs, max_streak = consecutive_features(numbers)

    results.append(
        {
            "consecutive_pairs": pairs,
            "consecutive_max": max_streak
        }
    )


features = pd.DataFrame(results)

df = pd.concat(
    [df, features],
    axis=1
)


# -------------------------
# Results
# -------------------------

print("Consecutive pairs distribution:")
print(
    df["consecutive_pairs"]
    .value_counts()
    .sort_index()
)

print("\nMaximum consecutive streak distribution:")
print(
    df["consecutive_max"]
    .value_counts()
    .sort_index()
)

print("\nAverage consecutive pairs:")
print(
    df["consecutive_pairs"].mean()
)

print("\nMaximum streak observed:")
print(
    df["consecutive_max"].max()
)