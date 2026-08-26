import os
import numpy as np
import pandas as pd
from itertools import combinations


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

OUTPUT_PATH = "data/processed/pairwise_interaction_results.csv"

N_NUMBERS = 49
PICK_SIZE = 6

N_FOLDS = 5
RANDOM_SIMULATIONS = 5000

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PAIRWISE INTERACTION TOP-6 WALK-FORWARD TEST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# DATE
# ============================================================

date_candidates = [
    "date",
    "Date",
    "datum",
    "Datum"
]

date_column = None

for col in date_candidates:
    if col in df.columns:
        date_column = col
        break

if date_column is None:
    raise ValueError(
        f"Could not find date column. Available columns: {list(df.columns)}"
    )

df[date_column] = pd.to_datetime(df[date_column])

df = df.sort_values(date_column).reset_index(drop=True)

print(
    f"Date range: "
    f"{df[date_column].min()} -> {df[date_column].max()}"
)


# ============================================================
# NUMBER COLUMNS
# ============================================================

number_columns = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6"
]

missing = [
    col for col in number_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing number columns: {missing}"
    )

print("\nNumber columns:")
print(number_columns)

print(f"\nTotal draws: {len(df)}")

# Validate range

all_numbers = df[number_columns].to_numpy()

if all_numbers.min() < 1 or all_numbers.max() > 49:
    raise ValueError("Numbers outside 1-49 detected.")


# ============================================================
# CONVERT DRAWS TO SETS
# ============================================================

draw_sets = [
    set(row)
    for row in all_numbers
]


# ============================================================
# PAIR INDEX
# ============================================================

pairs = list(
    combinations(range(1, N_NUMBERS + 1), 2)
)

pair_to_index = {
    pair: i
    for i, pair in enumerate(pairs)
}

N_PAIRS = len(pairs)

print(f"\nTotal possible pairs: {N_PAIRS}")


# ============================================================
# PAIR FREQUENCY
# ============================================================

def calculate_pair_frequency(draws):

    counts = np.zeros(N_PAIRS, dtype=np.int32)

    for draw in draws:

        for pair in combinations(sorted(draw), 2):

            idx = pair_to_index[pair]

            counts[idx] += 1

    return counts


# ============================================================
# NUMBER FREQUENCY
# ============================================================

def calculate_number_frequency(draws):

    counts = np.zeros(N_NUMBERS + 1, dtype=np.int32)

    for draw in draws:

        for number in draw:

            counts[number] += 1

    return counts


# ============================================================
# PAIR SCORE
# ============================================================

def calculate_pair_scores(pair_counts, number_counts):

    scores = np.zeros(N_PAIRS, dtype=np.float64)

    total_draws = max(
        1,
        number_counts.sum() // PICK_SIZE
    )

    for i, (a, b) in enumerate(pairs):

        observed = pair_counts[i]

        freq_a = number_counts[a]
        freq_b = number_counts[b]

        if freq_a == 0 or freq_b == 0:
            scores[i] = 0.0
            continue

        # Expected pair frequency under independence
        expected = (
            freq_a * freq_b
        ) / total_draws

        # Smoothed lift
        scores[i] = (
            observed + 1.0
        ) / (
            expected + 1.0
        )

    return scores


# ============================================================
# SELECT TOP-6
# ============================================================

def select_top6(
    pair_scores,
    number_counts
):

    # Start with individual frequency as a small
    # stabilizing component.

    number_score = (
        number_counts[1:] /
        max(1, number_counts[1:].max())
    )

    selected = []

    # Greedy selection.
    #
    # First number:
    # strongest individual frequency.
    #
    # Next numbers:
    # maximize interaction with already
    # selected numbers.

    first = int(
        np.argmax(number_score)
    ) + 1

    selected.append(first)

    remaining = set(
        range(1, N_NUMBERS + 1)
    )

    remaining.remove(first)

    while len(selected) < PICK_SIZE:

        best_number = None
        best_score = -np.inf

        for candidate in remaining:

            interaction_scores = []

            for selected_number in selected:

                pair = tuple(
                    sorted(
                        (candidate, selected_number)
                    )
                )

                idx = pair_to_index[pair]

                interaction_scores.append(
                    pair_scores[idx]
                )

            if interaction_scores:

                interaction = np.mean(
                    interaction_scores
                )

            else:
                interaction = 0.0

            individual = number_score[
                candidate - 1
            ]

            # Combination of individual strength
            # and pair interaction.
            score = (
                0.30 * individual +
                0.70 * interaction
            )

            if score > best_score:

                best_score = score
                best_number = candidate

        selected.append(best_number)

        remaining.remove(best_number)

    return sorted(selected)


# ============================================================
# RANDOM SIMULATION
# ============================================================

def random_top6_test(
    test_draws,
    simulations=5000,
    seed=42
):

    rng = np.random.default_rng(seed)

    n_test = len(test_draws)

    random_hits = np.zeros(
        simulations,
        dtype=np.float64
    )

    for s in range(simulations):

        selected = set(
            rng.choice(
                np.arange(1, 50),
                size=6,
                replace=False
            )
        )

        hits = 0

        for draw in test_draws:

            hits += len(
                selected.intersection(draw)
            )

        random_hits[s] = (
            hits / n_test
        )

    return random_hits


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

print("\n" + "=" * 70)
print("CREATING WALK-FORWARD FOLDS")
print("=" * 70)

n_total = len(df)

fold_size = n_total // (N_FOLDS + 1)

results = []


# ============================================================
# FOLDS
# ============================================================

for fold in range(N_FOLDS):

    print("\n" + "=" * 70)
    print(f"FOLD {fold + 1}")
    print("=" * 70)

    train_end = (
        fold + 1
    ) * fold_size

    test_start = train_end

    test_end = min(
        test_start + fold_size,
        n_total
    )

    if test_start >= n_total:
        break

    train_df = df.iloc[:train_end]

    test_df = df.iloc[
        test_start:test_end
    ]

    if len(train_df) == 0:
        print("Skipping fold because there is no training data.")
        continue

    print(
        f"Training: "
        f"{train_df[date_column].min()} -> "
        f"{train_df[date_column].max()}"
    )

    print(
        f"Testing:  "
        f"{test_df[date_column].min()} -> "
        f"{test_df[date_column].max()}"
    )

    print(
        f"Training draws: {len(train_df)}"
    )

    print(
        f"Testing draws:  {len(test_df)}"
    )


    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    train_draws = [
        set(row)
        for row in train_df[
            number_columns
        ].to_numpy()
    ]

    test_draws = [
        set(row)
        for row in test_df[
            number_columns
        ].to_numpy()
    ]


    # --------------------------------------------------------
    # CALCULATE PAIR STATISTICS
    # --------------------------------------------------------

    print("\nCalculating pair frequencies...")

    pair_counts = calculate_pair_frequency(
        train_draws
    )

    number_counts = calculate_number_frequency(
        train_draws
    )

    pair_scores = calculate_pair_scores(
        pair_counts,
        number_counts
    )


    # --------------------------------------------------------
    # TOP-6
    # --------------------------------------------------------

    selected = select_top6(
        pair_scores,
        number_counts
    )

    print("\nPairwise Top-6 selected:")

    print(selected)


    # --------------------------------------------------------
    # SHOW NUMBER FREQUENCY
    # --------------------------------------------------------

    print("\nNumber frequencies:")

    for number in selected:

        print(
            f"Number {number:2d}: "
            f"{number_counts[number]}"
        )


    # --------------------------------------------------------
    # SHOW PAIR SCORES
    # --------------------------------------------------------

    print("\nSelected pair scores:")

    for a, b in combinations(
        selected,
        2
    ):

        pair = tuple(
            sorted((a, b))
        )

        idx = pair_to_index[pair]

        print(
            f"({a:2d}, {b:2d}) "
            f"{pair_scores[idx]:.4f}"
        )


    # --------------------------------------------------------
    # MODEL TEST
    # --------------------------------------------------------

    total_hits = 0
    hit_distribution = []

    selected_set = set(selected)

    for draw in test_draws:

        hits = len(
            selected_set.intersection(draw)
        )

        total_hits += hits

        hit_distribution.append(hits)


    average_hits = (
        total_hits /
        len(test_draws)
    )


    # --------------------------------------------------------
    # RANDOM BASELINE
    # --------------------------------------------------------

    random_expected = (
        PICK_SIZE * PICK_SIZE / N_NUMBERS
    )

    print(
        "\nRunning random Top-6 simulation..."
    )

    random_results = random_top6_test(
        test_draws,
        simulations=RANDOM_SIMULATIONS,
        seed=RANDOM_SEED + fold
    )

    random_mean = random_results.mean()

    random_lower = np.percentile(
        random_results,
        2.5
    )

    random_upper = np.percentile(
        random_results,
        97.5
    )


    # --------------------------------------------------------
    # EMPIRICAL P-VALUE
    # --------------------------------------------------------

    empirical_p = (
        np.sum(
            random_results >= average_hits
        ) + 1
    ) / (
        len(random_results) + 1
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    difference = (
        average_hits -
        random_expected
    )

    difference_percent = (
        difference /
        random_expected
    ) * 100


    print("\nResults")
    print("-" * 40)

    print(
        f"Average hits:       "
        f"{average_hits:.6f}"
    )

    print(
        f"Total hits:         "
        f"{total_hits}"
    )

    print(
        f"Maximum hits:       "
        f"{max(hit_distribution)}"
    )

    print(
        f"Random expected:    "
        f"{random_expected:.6f}"
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
        f"{random_mean:.6f}"
    )

    print(
        f"Random 95% range:   "
        f"[{random_lower:.6f}, "
        f"{random_upper:.6f}]"
    )

    print(
        f"Empirical p-value:  "
        f"{empirical_p:.6f}"
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append({

        "fold": fold + 1,

        "train_draws": len(train_draws),

        "test_draws": len(test_draws),

        "average_hits": average_hits,

        "random_expected": random_expected,

        "difference": difference,

        "difference_percent":
            difference_percent,

        "random_mean":
            random_mean,

        "random_lower":
            random_lower,

        "random_upper":
            random_upper,

        "empirical_p":
            empirical_p,

        "selected_numbers":
            ",".join(
                map(str, selected)
            )
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

if len(results_df) > 0:

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
        ].to_string(
            index=False
        )
    )


# ============================================================
# OVERALL
# ============================================================

print("\n" + "=" * 70)
print("OVERALL")
print("=" * 70)

if len(results_df) > 0:

    mean_model = (
        results_df[
            "average_hits"
        ].mean()
    )

    mean_random = (
        results_df[
            "random_expected"
        ].mean()
    )

    mean_difference = (
        mean_model -
        mean_random
    )

    mean_difference_percent = (
        mean_difference /
        mean_random
    ) * 100

    mean_p = (
        results_df[
            "empirical_p"
        ].mean()
    )

    above = np.sum(
        results_df[
            "difference"
        ] > 0
    )

    below = np.sum(
        results_df[
            "difference"
        ] < 0
    )

    print(
        f"Mean model hits:       "
        f"{mean_model:.6f}"
    )

    print(
        f"Mean random expected:  "
        f"{mean_random:.6f}"
    )

    print(
        f"Mean difference:        "
        f"{mean_difference:+.6f}"
    )

    print(
        f"Mean difference %:      "
        f"{mean_difference_percent:+.3f}%"
    )

    print(
        f"Mean empirical p:       "
        f"{mean_p:.6f}"
    )

    print("\nFold consistency:")

    print(
        f"Above random: {above}"
    )

    print(
        f"Below random: {below}"
    )


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

if len(results_df) == 0:

    print(
        "No valid folds were produced."
    )

else:

    if (
        mean_difference > 0
        and
        mean_p < 0.05
        and
        above >= 3
    ):

        print(
            "Potential pairwise interaction advantage detected."
        )

    else:

        print(
            "No meaningful pairwise interaction "
            "advantage over random selection."
        )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)