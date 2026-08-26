import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
RESULT_PATH = "data/processed/stability_top6_walk_forward_results.csv"

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

N_NUMBERS = 49
TOP_K = 6

TEST_SIZE = 1008
STEP_SIZE = 1008

WINDOWS = [20, 50, 100, 200, 500]

RANDOM_SIMULATIONS = 5000

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    print("\nNumber columns:")
    print(NUMBER_COLS)

    print(f"\nTotal draws: {len(df)}")

    return df


# ============================================================
# NUMBER MATRIX
# ============================================================

def create_presence_matrix(df):

    matrix = np.zeros(
        (len(df), N_NUMBERS),
        dtype=np.int8
    )

    values = df[NUMBER_COLS].values

    for i, row in enumerate(values):

        for number in row:

            number = int(number)

            if 1 <= number <= N_NUMBERS:

                matrix[i, number - 1] = 1

    return matrix


# ============================================================
# RANDOM EXPECTED HITS
# ============================================================

def random_expected_hits():

    return TOP_K * 6 / N_NUMBERS


# ============================================================
# STABILITY SCORE
# ============================================================

def calculate_stability_scores(
    presence,
    train_end
):

    available = presence[:train_end]

    rows = []

    expected_rate = 6 / N_NUMBERS

    for number in range(1, N_NUMBERS + 1):

        idx = number - 1

        window_rates = []

        window_counts = []

        for window in WINDOWS:

            if len(available) < window:
                continue

            segment = available[-window:, idx]

            count = int(segment.sum())

            rate = count / window

            window_counts.append(count)
            window_rates.append(rate)

        if not window_rates:

            continue

        rates = np.array(window_rates, dtype=float)

        counts = np.array(window_counts, dtype=float)

        # ----------------------------------------------------
        # Mean recent rate
        # ----------------------------------------------------

        mean_rate = rates.mean()

        # ----------------------------------------------------
        # Stability
        #
        # Lower variation = more stable
        # ----------------------------------------------------

        std_rate = rates.std()

        stability = 1.0 / (1.0 + std_rate)

        # ----------------------------------------------------
        # Strength relative to random expectation
        # ----------------------------------------------------

        strength = mean_rate / expected_rate

        # ----------------------------------------------------
        # Long-term frequency
        # ----------------------------------------------------

        total_frequency = available[:, idx].sum()

        total_rate = total_frequency / len(available)

        # ----------------------------------------------------
        # Recent vs long-term
        # ----------------------------------------------------

        recent_rate = rates[0]

        recency_strength = recent_rate / expected_rate

        rows.append(
            {
                "number": number,
                "frequency": total_frequency,
                "total_rate": total_rate,
                "mean_window_rate": mean_rate,
                "std_window_rate": std_rate,
                "stability": stability,
                "strength": strength,
                "recent_rate": recent_rate,
                "recency_strength": recency_strength,
            }
        )

    result = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Normalize components
    # --------------------------------------------------------

    def minmax(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:

            return pd.Series(
                np.ones(len(series)),
                index=series.index
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    result["strength_norm"] = minmax(
        result["strength"]
    )

    result["stability_norm"] = minmax(
        result["stability"]
    )

    result["recency_norm"] = minmax(
        result["recency_strength"]
    )

    # --------------------------------------------------------
    # Final stability score
    # --------------------------------------------------------

    result["stability_score"] = (
        0.45 * result["strength_norm"]
        + 0.40 * result["stability_norm"]
        + 0.15 * result["recency_norm"]
    )

    result = result.sort_values(
        "stability_score",
        ascending=False
    ).reset_index(drop=True)

    return result


# ============================================================
# EVALUATE TOP-6
# ============================================================

def evaluate_top6(
    presence,
    test_start,
    test_end,
    selected
):

    selected_idx = np.array(
        selected,
        dtype=int
    ) - 1

    test = presence[test_start:test_end]

    hits = test[:, selected_idx].sum(axis=1)

    average_hits = hits.mean()

    total_hits = hits.sum()

    maximum_hits = hits.max()

    return (
        average_hits,
        total_hits,
        maximum_hits,
        hits
    )


# ============================================================
# RANDOM SIMULATION
# ============================================================

def random_simulation(
    presence,
    test_start,
    test_end
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    test = presence[test_start:test_end]

    n_tests = len(test)

    results = np.zeros(
        RANDOM_SIMULATIONS,
        dtype=float
    )

    numbers = np.arange(
        1,
        N_NUMBERS + 1
    )

    for simulation in range(
        RANDOM_SIMULATIONS
    ):

        selected = rng.choice(
            numbers,
            size=TOP_K,
            replace=False
        )

        selected_idx = selected - 1

        hits = test[
            :,
            selected_idx
        ].sum(axis=1)

        results[simulation] = hits.mean()

    mean_random = results.mean()

    lower = np.percentile(
        results,
        2.5
    )

    upper = np.percentile(
        results,
        97.5
    )

    return (
        mean_random,
        lower,
        upper,
        results
    )


# ============================================================
# EMPIRICAL P-VALUE
# ============================================================

def empirical_p_value(
    model_score,
    random_scores
):

    return np.mean(
        random_scores >= model_score
    )


# ============================================================
# WALK FORWARD FOLDS
# ============================================================

def create_folds(n_draws):

    folds = []

    train_end = TEST_SIZE

    while train_end + TEST_SIZE <= n_draws:

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            n_draws
        )

        folds.append(
            (
                train_end,
                test_start,
                test_end
            )
        )

        train_end += STEP_SIZE

    # Final partial fold
    if train_end < n_draws:

        test_start = train_end
        test_end = n_draws

        if test_end > test_start:

            folds.append(
                (
                    train_end,
                    test_start,
                    test_end
                )
            )

    return folds


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STABILITY TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    df = load_data()

    presence = create_presence_matrix(df)

    print(
        f"Number range: "
        f"{presence.shape[1]}"
    )

    print("\n")
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    folds = create_folds(
        len(df)
    )

    results = []

    random_expected = random_expected_hits()

    for fold_number, (
        train_end,
        test_start,
        test_end
    ) in enumerate(
        folds,
        start=1
    ):

        train_df = df.iloc[
            :train_end
        ]

        test_df = df.iloc[
            test_start:test_end
        ]

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold_number}")
        print("=" * 70)

        print(
            f"Training: "
            f"{train_df['date'].iloc[0]} -> "
            f"{train_df['date'].iloc[-1]}"
        )

        print(
            f"Testing:  "
            f"{test_df['date'].iloc[0]} -> "
            f"{test_df['date'].iloc[-1]}"
        )

        print(
            f"Training draws: {len(train_df)}"
        )

        print(
            f"Testing draws:  {len(test_df)}"
        )

        # ----------------------------------------------------
        # Calculate scores
        # ----------------------------------------------------

        print("\nCalculating stability scores...")

        score_table = calculate_stability_scores(
            presence,
            train_end
        )

        selected = (
            score_table
            .head(TOP_K)["number"]
            .astype(int)
            .tolist()
        )

        print("\nStability Top-6 selected:")
        print(selected)

        print("\nTop candidates:")

        display_columns = [
            "number",
            "frequency",
            "mean_window_rate",
            "std_window_rate",
            "stability",
            "strength",
            "recent_rate",
            "stability_score"
        ]

        print(
            score_table[
                display_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        print("\nEvaluating Stability Top-6...")

        (
            average_hits,
            total_hits,
            maximum_hits,
            hits
        ) = evaluate_top6(
            presence,
            test_start,
            test_end,
            selected
        )

        # ----------------------------------------------------
        # Random simulation
        # ----------------------------------------------------

        print(
            "\nRunning random Top-6 simulation..."
        )

        (
            random_mean,
            random_lower,
            random_upper,
            random_scores
        ) = random_simulation(
            presence,
            test_start,
            test_end
        )

        difference = (
            average_hits
            - random_expected
        )

        difference_percent = (
            difference
            / random_expected
            * 100
        )

        p_value = empirical_p_value(
            average_hits,
            random_scores
        )

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
            f"{maximum_hits}"
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
            f"{p_value:.6f}"
        )

        results.append(
            {
                "fold": fold_number,
                "test_draws": len(test_df),
                "average_hits": average_hits,
                "random_expected": random_expected,
                "difference": difference,
                "difference_percent": difference_percent,
                "empirical_p": p_value,
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # OVERALL
    # ========================================================

    mean_model = (
        results_df["average_hits"]
        .mean()
    )

    mean_random = (
        results_df["random_expected"]
        .mean()
    )

    mean_difference = (
        results_df["difference"]
        .mean()
    )

    mean_difference_percent = (
        results_df["difference_percent"]
        .mean()
    )

    mean_p = (
        results_df["empirical_p"]
        .mean()
    )

    above_random = (
        results_df["difference"] > 0
    ).sum()

    below_random = (
        results_df["difference"] < 0
    ).sum()

    print("\n")
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Mean stability hits:   "
        f"{mean_model:.6f}"
    )

    print(
        f"Mean random expected:   "
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
        f"Above random: {above_random}"
    )

    print(
        f"Below random: {below_random}"
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        mean_difference > 0
        and above_random > below_random
        and mean_p < 0.05
    ):

        print(
            "\nPossible stability signal detected."
        )

    else:

        print(
            "\nNo meaningful stability advantage "
            "over random selection."
        )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(RESULT_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\n")
    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()