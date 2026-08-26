import os
import random
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
RESULT_PATH = "data/processed/regime_top6_walk_forward_results.csv"

NUMBER_MIN = 1
NUMBER_MAX = 49
TOP_K = 6

TEST_SIZE = 1008
STEP_SIZE = 1008

RECENT_20 = 20
RECENT_50 = 50
BASELINE_WINDOW = 200

RANDOM_SIMULATIONS = 5000

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def find_number_columns(df):
    candidates = ["n1", "n2", "n3", "n4", "n5", "n6"]

    if all(c in df.columns for c in candidates):
        return candidates

    number_columns = [
        c for c in df.columns
        if str(c).lower().startswith("n")
        and str(c)[1:].isdigit()
    ]

    number_columns = sorted(
        number_columns,
        key=lambda x: int(str(x)[1:])
    )

    if len(number_columns) != 6:
        raise ValueError(
            f"Could not find exactly 6 number columns. "
            f"Found: {number_columns}"
        )

    return number_columns


def normalize(values):
    values = np.asarray(values, dtype=float)

    min_v = np.min(values)
    max_v = np.max(values)

    if max_v - min_v < 1e-12:
        return np.full_like(values, 0.5)

    return (values - min_v) / (max_v - min_v)


def calculate_frequency(draws, number_min=1, number_max=49):
    """
    Count appearances of each number.
    """

    counts = np.zeros(number_max + 1, dtype=float)

    for row in draws:
        for number in row:
            number = int(number)

            if number_min <= number <= number_max:
                counts[number] += 1

    return counts


def calculate_gap(draws, number_min=1, number_max=49):
    """
    Number of draws since each number last appeared.

    Larger gap = number has been absent longer.
    """

    gaps = np.full(number_max + 1, len(draws), dtype=float)

    for i in range(len(draws) - 1, -1, -1):

        for number in draws[i]:

            number = int(number)

            if number_min <= number <= number_max:

                if gaps[number] == len(draws):
                    gaps[number] = len(draws) - i

    return gaps


def calculate_regime_scores(train_draws):
    """
    Calculate regime-based score for every number.

    The idea:

    1. Long-term frequency
    2. Recent frequency
    3. Medium-term frequency
    4. Change from baseline
    5. Gap
    6. Recent-vs-baseline z-score

    Everything is calculated ONLY from training data.
    """

    n_numbers = NUMBER_MAX - NUMBER_MIN + 1

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    recent_20_draws = train_draws[-RECENT_20:]
    recent_50_draws = train_draws[-RECENT_50:]

    baseline_size = min(
        BASELINE_WINDOW,
        len(train_draws)
    )

    baseline_draws = train_draws[-baseline_size:]

    # Older data before recent window
    if len(train_draws) > RECENT_50:

        historical_draws = train_draws[:-RECENT_50]

        if len(historical_draws) > BASELINE_WINDOW:
            historical_draws = historical_draws[-BASELINE_WINDOW:]

    else:
        historical_draws = train_draws

    # --------------------------------------------------------
    # Frequencies
    # --------------------------------------------------------

    total_frequency = calculate_frequency(train_draws)

    freq_20 = calculate_frequency(recent_20_draws)

    freq_50 = calculate_frequency(recent_50_draws)

    baseline_frequency = calculate_frequency(
        baseline_draws
    )

    historical_frequency = calculate_frequency(
        historical_draws
    )

    # --------------------------------------------------------
    # Normalize frequency
    # --------------------------------------------------------

    long_term_score = normalize(
        total_frequency[NUMBER_MIN:NUMBER_MAX + 1]
    )

    recent20_score = normalize(
        freq_20[NUMBER_MIN:NUMBER_MAX + 1]
    )

    recent50_score = normalize(
        freq_50[NUMBER_MIN:NUMBER_MAX + 1]
    )

    # --------------------------------------------------------
    # Recent change
    # --------------------------------------------------------

    recent_rate = (
        freq_50[NUMBER_MIN:NUMBER_MAX + 1]
        / max(len(recent_50_draws), 1)
    )

    historical_rate = (
        historical_frequency[NUMBER_MIN:NUMBER_MAX + 1]
        / max(len(historical_draws), 1)
    )

    rate_change = recent_rate - historical_rate

    change_score = normalize(rate_change)

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    expected_per_draw = 6 / 49

    expected_recent = (
        len(recent_50_draws)
        * expected_per_draw
    )

    variance = (
        len(recent_50_draws)
        * expected_per_draw
        * (1 - expected_per_draw)
    )

    std = np.sqrt(max(variance, 1e-12))

    z_scores = (
        freq_50[NUMBER_MIN:NUMBER_MAX + 1]
        - expected_recent
    ) / std

    z_score_normalized = normalize(z_scores)

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    gaps = calculate_gap(train_draws)

    gap_values = gaps[NUMBER_MIN:NUMBER_MAX + 1]

    gap_score = normalize(gap_values)

    # --------------------------------------------------------
    # Regime strength
    # --------------------------------------------------------

    # Positive if recent behavior differs
    # from historical behavior.

    regime_strength = np.abs(rate_change)

    regime_strength_score = normalize(
        regime_strength
    )

    # --------------------------------------------------------
    # Combined regime score
    # --------------------------------------------------------

    score = (
        0.20 * long_term_score
        + 0.25 * recent20_score
        + 0.20 * recent50_score
        + 0.20 * change_score
        + 0.10 * z_score_normalized
        + 0.05 * regime_strength_score
    )

    numbers = np.arange(
        NUMBER_MIN,
        NUMBER_MAX + 1
    )

    result = pd.DataFrame({
        "number": numbers,
        "frequency": total_frequency[
            NUMBER_MIN:NUMBER_MAX + 1
        ],
        "freq_20": freq_20[
            NUMBER_MIN:NUMBER_MAX + 1
        ],
        "freq_50": freq_50[
            NUMBER_MIN:NUMBER_MAX + 1
        ],
        "historical_freq": historical_frequency[
            NUMBER_MIN:NUMBER_MAX + 1
        ],
        "rate_change": rate_change,
        "z_score": z_scores,
        "gap": gap_values,
        "regime_strength": regime_strength,
        "regime_score": score
    })

    result = result.sort_values(
        "regime_score",
        ascending=False
    ).reset_index(drop=True)

    return result


def calculate_hits(selected_numbers, test_draws):
    """
    Calculate average number of hits per draw.
    """

    selected = set(selected_numbers)

    hits = []

    for draw in test_draws:

        draw_set = set(int(x) for x in draw)

        hits.append(
            len(selected.intersection(draw_set))
        )

    hits = np.asarray(hits, dtype=float)

    return {
        "average_hits": float(np.mean(hits)),
        "total_hits": int(np.sum(hits)),
        "maximum_hits": int(np.max(hits)),
        "hits": hits
    }


def random_top6_simulation(test_draws, simulations=5000):
    """
    Monte Carlo random Top-6 baseline.

    Each simulation chooses 6 random numbers
    and evaluates them over the test period.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    averages = []

    test_sets = [
        set(int(x) for x in draw)
        for draw in test_draws
    ]

    for _ in range(simulations):

        selected = rng.choice(
            np.arange(NUMBER_MIN, NUMBER_MAX + 1),
            size=TOP_K,
            replace=False
        )

        selected = set(int(x) for x in selected)

        total_hits = 0

        for draw_set in test_sets:

            total_hits += len(
                selected.intersection(draw_set)
            )

        averages.append(
            total_hits / len(test_sets)
        )

    averages = np.asarray(averages)

    return {
        "mean": float(np.mean(averages)),
        "lower": float(np.percentile(averages, 2.5)),
        "upper": float(np.percentile(averages, 97.5))
    }


def empirical_p_value(
    model_average,
    test_draws,
    simulations=5000
):
    """
    Estimate probability that a random Top-6
    achieves >= model performance.
    """

    rng = np.random.default_rng(
        RANDOM_SEED + 123
    )

    test_sets = [
        set(int(x) for x in draw)
        for draw in test_draws
    ]

    random_averages = []

    numbers = np.arange(
        NUMBER_MIN,
        NUMBER_MAX + 1
    )

    for _ in range(simulations):

        selected = set(
            rng.choice(
                numbers,
                size=TOP_K,
                replace=False
            )
        )

        total_hits = 0

        for draw_set in test_sets:

            total_hits += len(
                selected.intersection(draw_set)
            )

        random_averages.append(
            total_hits / len(test_sets)
        )

    random_averages = np.asarray(
        random_averages
    )

    return float(
        np.mean(
            random_averages >= model_average
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REGIME TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_column = None

    for column in df.columns:

        if str(column).lower() in [
            "date",
            "draw_date",
            "datum"
        ]:
            date_column = column
            break

    if date_column is None:
        raise ValueError(
            "Could not find date column."
        )

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    df = df.sort_values(
        date_column
    ).reset_index(drop=True)

    print(
        f"Date range: "
        f"{df[date_column].min()} -> "
        f"{df[date_column].max()}"
    )

    # --------------------------------------------------------
    # Number columns
    # --------------------------------------------------------

    number_columns = find_number_columns(df)

    print("\nNumber columns:")
    print(number_columns)

    draws = (
        df[number_columns]
        .astype(int)
        .values
    )

    print(
        f"\nTotal draws: {len(draws)}"
    )

    print(
        f"Number range: "
        f"{draws.min()} -> {draws.max()}"
    )

    # --------------------------------------------------------
    # Random baseline expected
    # --------------------------------------------------------

    random_expected = (
        TOP_K * TOP_K / NUMBER_MAX
    )

    # --------------------------------------------------------
    # Walk-forward folds
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    results = []

    total_draws = len(draws)

    fold = 1

    train_end = TEST_SIZE

    while train_end < total_draws:

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            total_draws
        )

        train_draws = draws[:train_end]

        test_draws = draws[
            test_start:test_end
        ]

        if len(test_draws) == 0:
            break

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        print(
            f"Training: "
            f"{df.iloc[0][date_column]} -> "
            f"{df.iloc[train_end - 1][date_column]}"
        )

        print(
            f"Testing:  "
            f"{df.iloc[test_start][date_column]} -> "
            f"{df.iloc[test_end - 1][date_column]}"
        )

        print(
            f"Training draws: {len(train_draws)}"
        )

        print(
            f"Testing draws:  {len(test_draws)}"
        )

        # ----------------------------------------------------
        # Calculate regime
        # ----------------------------------------------------

        print(
            "\nCalculating regime scores..."
        )

        score_table = calculate_regime_scores(
            train_draws
        )

        selected = (
            score_table
            .head(TOP_K)["number"]
            .astype(int)
            .tolist()
        )

        print(
            "\nRegime Top-6 selected:"
        )

        print(selected)

        # ----------------------------------------------------
        # Display candidates
        # ----------------------------------------------------

        print("\nTop candidates:")

        display_columns = [
            "number",
            "frequency",
            "freq_20",
            "freq_50",
            "rate_change",
            "z_score",
            "gap",
            "regime_strength",
            "regime_score"
        ]

        print(
            score_table[
                display_columns
            ].head(10).to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        print(
            "\nEvaluating regime Top-6..."
        )

        evaluation = calculate_hits(
            selected,
            test_draws
        )

        average_hits = evaluation[
            "average_hits"
        ]

        # ----------------------------------------------------
        # Random simulation
        # ----------------------------------------------------

        print(
            "\nRunning random Top-6 simulation..."
        )

        random_result = random_top6_simulation(
            test_draws,
            RANDOM_SIMULATIONS
        )

        p_value = empirical_p_value(
            average_hits,
            test_draws,
            RANDOM_SIMULATIONS
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

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print("\nResults")
        print("-" * 40)

        print(
            f"Average hits:       "
            f"{average_hits:.6f}"
        )

        print(
            f"Total hits:         "
            f"{evaluation['total_hits']}"
        )

        print(
            f"Maximum hits:       "
            f"{evaluation['maximum_hits']}"
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
            f"{random_result['mean']:.6f}"
        )

        print(
            f"Random 95% range:   "
            f"[{random_result['lower']:.6f}, "
            f"{random_result['upper']:.6f}]"
        )

        print(
            f"Empirical p-value:  "
            f"{p_value:.6f}"
        )

        results.append({
            "fold": fold,
            "train_draws": len(train_draws),
            "test_draws": len(test_draws),
            "average_hits": average_hits,
            "random_expected": random_expected,
            "difference": difference,
            "difference_percent": difference_percent,
            "empirical_p": p_value,
            "selected_numbers": ",".join(
                map(str, selected)
            )
        })

        # ----------------------------------------------------
        # Next fold
        # ----------------------------------------------------

        fold += 1

        train_end += STEP_SIZE

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
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

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

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
            mean_model
            - mean_random
        )

        mean_difference_percent = (
            mean_difference
            / mean_random
            * 100
        )

        mean_p = (
            results_df[
                "empirical_p"
            ].mean()
        )

        above_random = int(
            (
                results_df[
                    "difference"
                ] > 0
            ).sum()
        )

        below_random = int(
            (
                results_df[
                    "difference"
                ] < 0
            ).sum()
        )

        print("\n")
        print("=" * 70)
        print("OVERALL")
        print("=" * 70)

        print(
            f"Mean regime hits:      "
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

        print("\nFold consistency:")

        print(
            f"Above random: {above_random}"
        )

        print(
            f"Below random: {below_random}"
        )

        # ----------------------------------------------------
        # Conclusion
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL CONCLUSION")
        print("=" * 70)

        if (
            mean_difference > 0
            and mean_p < 0.05
            and above_random > below_random
        ):

            print(
                "Potential regime advantage detected."
            )

        else:

            print(
                "No meaningful regime advantage "
                "over random selection."
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    print(RESULT_PATH)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()