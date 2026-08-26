import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

N_FOLDS = 5
TOP_K = 6

RANDOM_SEED = 42
N_RANDOM_SIMULATIONS = 10000

np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TOP-6 WALK-FORWARD TEST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")


# ============================================================
# DETECT NUMBER COLUMNS
# ============================================================

number_columns = [
    col for col in df.columns
    if col.lower() in [
        "n1", "n2", "n3", "n4", "n5", "n6"
    ]
]

if len(number_columns) != 6:
    raise ValueError(
        f"Could not find exactly 6 number columns. Found: {number_columns}"
    )

print("\nNumber columns:")
print(number_columns)


# ============================================================
# CREATE DRAW TABLE
# ============================================================

draws = df[["date"] + number_columns].copy()

draws = draws.drop_duplicates(subset=["date"])

draws = draws.sort_values("date").reset_index(drop=True)

print(f"\nTotal draws: {len(draws)}")


# ============================================================
# FIND NUMBER RANGE
# ============================================================

all_numbers = draws[number_columns].values.flatten()

all_numbers = all_numbers[~pd.isna(all_numbers)]

min_number = int(np.min(all_numbers))
max_number = int(np.max(all_numbers))

print(f"Number range: {min_number} -> {max_number}")


# ============================================================
# HELPER: GET NUMBERS FROM DRAW
# ============================================================

def get_draw_numbers(row):
    return set(
        int(x)
        for x in row[number_columns].values
        if not pd.isna(x)
    )


# ============================================================
# BUILD WALK-FORWARD FOLDS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING WALK-FORWARD FOLDS")
print("=" * 70)

fold_sizes = np.array_split(
    np.arange(len(draws)),
    N_FOLDS
)

results = []


# ============================================================
# WALK FORWARD
# ============================================================

for fold_number in range(N_FOLDS):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold_number + 1}")
    print("=" * 70)

    test_indices = fold_sizes[fold_number]

    train_end = test_indices[0]

    train = draws.iloc[:train_end].copy()
    test = draws.iloc[test_indices].copy()

    if len(train) == 0:
        print("Skipping first fold because there is no training data.")
        continue

    print(
        f"Training: {train['date'].min()} -> "
        f"{train['date'].max()}"
    )

    print(
        f"Testing:  {test['date'].min()} -> "
        f"{test['date'].max()}"
    )

    print(f"Training draws: {len(train)}")
    print(f"Testing draws:  {len(test)}")


    # ========================================================
    # CALCULATE HISTORICAL FREQUENCY
    # ========================================================

    frequency = {}

    for number in range(min_number, max_number + 1):
        frequency[number] = 0

    for _, row in train.iterrows():

        draw_numbers = get_draw_numbers(row)

        for number in draw_numbers:
            frequency[number] += 1


    # ========================================================
    # SELECT TOP 6
    # ========================================================

    frequency_series = pd.Series(frequency)

    top6 = (
        frequency_series
        .sort_values(ascending=False)
        .head(TOP_K)
        .index
        .tolist()
    )

    top6 = sorted(top6)

    print("\nTop-6 selected from training:")
    print(top6)

    print("\nFrequencies:")

    for number in top6:
        print(
            f"Number {number:2d}: "
            f"{frequency[number]} appearances"
        )


    # ========================================================
    # TEST TOP-6
    # ========================================================

    hits = []

    for _, row in test.iterrows():

        draw_numbers = get_draw_numbers(row)

        hit_count = len(
            draw_numbers.intersection(top6)
        )

        hits.append(hit_count)

    hits = np.array(hits)

    mean_hits = hits.mean()

    max_hits = hits.max()

    total_hits = hits.sum()


    # ========================================================
    # RANDOM EXPECTATION
    # ========================================================

    draw_size = len(number_columns)

    population_size = max_number - min_number + 1

    random_expected_hits = (
        draw_size * TOP_K / population_size
    )

    difference = mean_hits - random_expected_hits

    difference_percent = (
        difference / random_expected_hits * 100
    )


    # ========================================================
    # RANDOM TOP-6 SIMULATION
    # ========================================================

    print("\nRunning random Top-6 simulation...")

    random_means = []

    test_draw_sets = [
        get_draw_numbers(row)
        for _, row in test.iterrows()
    ]

    for _ in range(N_RANDOM_SIMULATIONS):

        random_top6 = np.random.choice(
            np.arange(min_number, max_number + 1),
            size=TOP_K,
            replace=False
        )

        random_top6 = set(random_top6)

        random_hits = [
            len(draw_set.intersection(random_top6))
            for draw_set in test_draw_sets
        ]

        random_means.append(
            np.mean(random_hits)
        )

    random_means = np.array(random_means)

    random_sim_mean = random_means.mean()

    random_p95_low = np.percentile(
        random_means, 2.5
    )

    random_p95_high = np.percentile(
        random_means, 97.5
    )

    empirical_p = (
        np.sum(random_means >= mean_hits) + 1
    ) / (
        len(random_means) + 1
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nResults")
    print("-" * 40)

    print(f"Average hits:       {mean_hits:.6f}")
    print(f"Total hits:         {total_hits}")
    print(f"Maximum hits:       {max_hits}")

    print(
        f"Random expected:    "
        f"{random_expected_hits:.6f}"
    )

    print(
        f"Difference:         "
        f"{difference:+.6f}"
    )

    print(
        f"Difference %:       "
        f"{difference_percent:+.3f}%"
    )

    print(
        f"Random simulation:  "
        f"{random_sim_mean:.6f}"
    )

    print(
        f"Random 95% range:   "
        f"[{random_p95_low:.6f}, "
        f"{random_p95_high:.6f}]"
    )

    print(
        f"Empirical p-value:  "
        f"{empirical_p:.6f}"
    )


    # ========================================================
    # SAVE FOLD RESULT
    # ========================================================

    results.append({
        "fold": fold_number + 1,
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "test_start": test["date"].min(),
        "test_end": test["date"].max(),
        "train_draws": len(train),
        "test_draws": len(test),

        "top6": ",".join(
            str(x) for x in top6
        ),

        "average_hits": mean_hits,
        "total_hits": total_hits,
        "maximum_hits": max_hits,

        "random_expected": random_expected_hits,
        "difference": difference,
        "difference_percent": difference_percent,

        "random_sim_mean": random_sim_mean,
        "random_sim_low": random_p95_low,
        "random_sim_high": random_p95_high,

        "empirical_p": empirical_p
    })


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df[
        [
            "fold",
            "test_draws",
            "average_hits",
            "random_expected",
            "difference",
            "difference_percent",
            "empirical_p"
        ]
    ].to_string(index=False)
)


# ============================================================
# OVERALL STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL")
print("=" * 70)

mean_model = results_df["average_hits"].mean()

mean_random = results_df["random_expected"].mean()

mean_difference = results_df["difference"].mean()

mean_difference_percent = (
    mean_difference / mean_random * 100
)

mean_p = results_df["empirical_p"].mean()

print(
    f"Mean model hits:       "
    f"{mean_model:.6f}"
)

print(
    f"Mean random expected:  "
    f"{mean_random:.6f}"
)

print(
    f"Mean difference:       "
    f"{mean_difference:+.6f}"
)

print(
    f"Mean difference %:     "
    f"{mean_difference_percent:+.3f}%"
)

print(
    f"Mean empirical p:      "
    f"{mean_p:.6f}"
)


# ============================================================
# COUNT FOLDS
# ============================================================

above_random = (
    results_df["difference"] > 0
).sum()

below_random = (
    results_df["difference"] < 0
).sum()

print("\nFold consistency:")

print(
    f"Above random: {above_random}"
)

print(
    f"Below random: {below_random}"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

if (
    mean_difference > 0
    and above_random >= 4
    and mean_p < 0.05
):

    conclusion = (
        "Potential predictive signal detected."
    )

elif (
    abs(mean_difference_percent) < 1
    and above_random <= 3
):

    conclusion = (
        "No meaningful advantage over random selection."
    )

else:

    conclusion = (
        "Weak or unstable effect; "
        "insufficient evidence for a predictive advantage."
    )

print(f"\n{conclusion}")


# ============================================================
# SAVE
# ============================================================

output_path = Path(
    "data/processed/top6_walk_forward_results.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)