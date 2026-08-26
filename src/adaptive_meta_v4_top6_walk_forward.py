import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_EXPECTED = 6 * 6 / 49

RESULTS_DIR = "data/processed"

STRATEGY_FILES = {
    "meta_score": "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": "recency_top6_walk_forward_results.csv",
    "stability": "stability_top6_walk_forward_results.csv",
}

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "adaptive_meta_v4_top6_walk_forward_results.csv"
)

MONTE_CARLO_RUNS = 10000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def load_strategy(name, filename):
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing strategy result file: {path}"
        )

    df = pd.read_csv(path)

    # Normalize fold column
    if "fold" not in df.columns:
        raise ValueError(
            f"'fold' column not found in {filename}"
        )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Detect performance column
    # --------------------------------------------------------

    candidates = [
        "adaptive_difference",
        "selected_difference",
        "difference",
        "difference_percent"
    ]

    performance_column = None

    for col in candidates:
        if col in df.columns:
            performance_column = col
            break

    if performance_column is None:
        raise ValueError(
            f"Could not find performance column in {filename}. "
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Detect test draw count
    # --------------------------------------------------------

    draw_candidates = [
        "test_draws",
        "testing_draws",
        "draws"
    ]

    draw_column = None

    for col in draw_candidates:
        if col in df.columns:
            draw_column = col
            break

    if draw_column is None:
        df["test_draws"] = np.nan
    else:
        df["test_draws"] = pd.to_numeric(
            df[draw_column],
            errors="coerce"
        )

    df["performance"] = pd.to_numeric(
        df[performance_column],
        errors="coerce"
    )

    result = df[
        ["fold", "performance", "test_draws"]
    ].copy()

    result["strategy"] = name

    return result


def load_all_strategies():

    print("\n" + "=" * 70)
    print("LOADING STRATEGY RESULTS")
    print("=" * 70)

    frames = []

    for name, filename in STRATEGY_FILES.items():

        df = load_strategy(name, filename)

        print(
            f"{name:<12}: "
            f"{len(df)} rows -> {filename}"
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True
    )


def create_performance_matrix(df):

    matrix = df.pivot(
        index="fold",
        columns="strategy",
        values="performance"
    )

    return matrix.sort_index()


def create_draw_matrix(df):

    matrix = df.pivot(
        index="fold",
        columns="strategy",
        values="test_draws"
    )

    return matrix.sort_index()


# ============================================================
# V4 WEIGHT CALCULATION
# ============================================================

def calculate_v4_weights(
    historical,
    available_strategies
):

    """
    V4 philosophy:

    1. Reward positive historical performance.
    2. Penalize instability.
    3. Give extra weight to Stability because it provides
       information that is relatively independent from Meta-score.
    4. Give secondary diversification weight to Recency.
    5. Meta-score remains the main performance strategy.
    """

    strategies = list(available_strategies)

    if len(strategies) == 0:
        return {}

    if len(historical) == 0:
        return {
            s: 1.0 / len(strategies)
            for s in strategies
        }

    scores = {}

    for strategy in strategies:

        values = historical[strategy].dropna()

        if len(values) == 0:
            scores[strategy] = 0.0
            continue

        mean_perf = values.mean()

        # ----------------------------------------------------
        # Historical consistency
        # ----------------------------------------------------

        positive_ratio = (
            (values > 0).mean()
        )

        # ----------------------------------------------------
        # Stability penalty
        # ----------------------------------------------------

        volatility = values.std()

        if np.isnan(volatility):
            volatility = 0.0

        # ----------------------------------------------------
        # Base score
        # ----------------------------------------------------

        score = (
            mean_perf * 0.70
            + positive_ratio * 0.30
        )

        # Penalize excessive volatility
        score = score / (
            1.0 + volatility * 2.0
        )

        scores[strategy] = score

    score_series = pd.Series(scores, dtype=float)

    # --------------------------------------------------------
    # Shift scores so that all are non-negative
    # --------------------------------------------------------

    min_score = score_series.min()

    if min_score < 0:
        score_series = score_series - min_score

    # Small floor prevents complete elimination
    score_series = score_series + 0.02

    # --------------------------------------------------------
    # Information-diversity bonus
    #
    # Stability receives the strongest bonus because the
    # previous correlation analysis showed:
    #
    # stability diversity = 0.7248
    # recency diversity   = 0.6037
    # meta-score diversity = 0.3298
    # --------------------------------------------------------

    diversity_bonus = {
        "meta_score": 1.00,
        "recency": 1.15,
        "stability": 1.30,
    }

    for strategy in score_series.index:
        score_series[strategy] *= (
            diversity_bonus.get(strategy, 1.0)
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = score_series.sum()

    if total <= 0 or np.isnan(total):

        weights = {
            s: 1.0 / len(score_series)
            for s in score_series.index
        }

    else:

        weights = (
            score_series / total
        ).to_dict()

    return weights


# ============================================================
# MONTE CARLO TEST
# ============================================================

def monte_carlo_test(
    observed_difference,
    test_draws,
    runs=10000,
    seed=42
):

    rng = np.random.default_rng(seed)

    random_means = []

    for _ in range(runs):

        # Six numbers selected from 49.
        # Expected hits is approximately 36/49.
        random_hits = rng.binomial(
            test_draws,
            RANDOM_EXPECTED
        )

        random_average = (
            random_hits / test_draws
        )

        random_means.append(
            random_average - RANDOM_EXPECTED
        )

    random_means = np.asarray(
        random_means
    )

    p_value = (
        np.mean(
            random_means >= observed_difference
        )
    )

    lower = np.percentile(
        random_means,
        2.5
    )

    upper = np.percentile(
        random_means,
        97.5
    )

    return p_value, lower, upper


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE META-MODEL V4 TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data = load_all_strategies()

    performance = create_performance_matrix(
        data
    )

    draws = create_draw_matrix(
        data
    )

    # Make sure columns exist
    strategies = [
        "meta_score",
        "recency",
        "stability"
    ]

    for strategy in strategies:

        if strategy not in performance.columns:
            performance[strategy] = np.nan

        if strategy not in draws.columns:
            draws[strategy] = np.nan

    performance = performance[strategies]
    draws = draws[strategies]

    # --------------------------------------------------------
    # PERFORMANCE MATRIX
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V4 STRATEGY PERFORMANCE MATRIX")
    print("=" * 70)

    print(
        performance.to_string(
            float_format=lambda x:
            f"{x:+.6f}"
        )
    )

    # --------------------------------------------------------
    # DRAW MATRIX
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST DRAW MATRIX")
    print("=" * 70)

    print(
        draws.to_string()
    )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADAPTIVE META-MODEL V4 WALK-FORWARD TEST")
    print("=" * 70)

    fold_results = []

    folds = list(performance.index)

    for fold in folds:

        print("\n" + "=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        # ----------------------------------------------------
        # Historical data only
        # ----------------------------------------------------

        historical = performance[
            performance.index < fold
        ]

        current = performance.loc[
            fold
        ]

        current_draws = draws.loc[
            fold
        ]

        # ----------------------------------------------------
        # Available strategies
        # ----------------------------------------------------

        available = [
            strategy
            for strategy in strategies
            if not pd.isna(
                current[strategy]
            )
        ]

        print(
            "\nAvailable strategies: "
            + ", ".join(available)
        )

        if not available:
            print(
                "No available strategies."
            )
            continue

        # ----------------------------------------------------
        # Calculate weights
        # ----------------------------------------------------

        weights = calculate_v4_weights(
            historical[
                available
            ],
            available
        )

        print("\nV4 strategy weights:")

        for strategy in strategies:

            weight = weights.get(
                strategy,
                0.0
            )

            print(
                f"  {strategy:<11}: "
                f"{weight:.4f}"
            )

        # ----------------------------------------------------
        # Weighted performance
        # ----------------------------------------------------

        weighted_difference = 0.0
        total_weight = 0.0

        for strategy in available:

            value = current[strategy]

            if pd.isna(value):
                continue

            weight = weights.get(
                strategy,
                0.0
            )

            weighted_difference += (
                weight * value
            )

            total_weight += weight

        if total_weight > 0:

            weighted_difference /= (
                total_weight
            )

        else:

            weighted_difference = 0.0

        adaptive_hits = (
            RANDOM_EXPECTED
            + weighted_difference
        )

        # ----------------------------------------------------
        # Selected strategy
        # ----------------------------------------------------

        selected_strategy = max(
            available,
            key=lambda s:
            weights.get(s, 0.0)
        )

        selected_difference = current[
            selected_strategy
        ]

        selected_hits = (
            RANDOM_EXPECTED
            + selected_difference
        )

        # ----------------------------------------------------
        # Current performance
        # ----------------------------------------------------

        print(
            f"\nSelected highest-weight strategy: "
            f"{selected_strategy.upper()}"
        )

        print(
            f"Adaptive weighted difference: "
            f"{weighted_difference:+.6f}"
        )

        print(
            f"Adaptive weighted hits: "
            f"{adaptive_hits:.6f}"
        )

        print(
            "\nCurrent fold performance:"
        )

        for strategy in strategies:

            value = current[strategy]

            if pd.isna(value):

                print(
                    f"  {strategy:<11}: NaN"
                )

            else:

                print(
                    f"  {strategy:<11}: "
                    f"{value:+.6f}"
                )

        # ----------------------------------------------------
        # Draw count
        # ----------------------------------------------------

        valid_draws = [
            current_draws[s]
            for s in available
            if not pd.isna(
                current_draws[s]
            )
        ]

        if valid_draws:

            test_draws = int(
                max(valid_draws)
            )

        else:

            test_draws = np.nan

        fold_results.append({
            "fold": fold,
            "test_draws": test_draws,
            "adaptive_average_hits":
                adaptive_hits,
            "adaptive_difference":
                weighted_difference,
            "selected_strategy":
                selected_strategy,
            "selected_average_hits":
                selected_hits,
            "selected_difference":
                selected_difference,
        })

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        fold_results
    )

    print("\n" + "=" * 70)
    print("FOLD RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
            f"{x:+.6f}"
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Weight folds by number of test draws.
    # Do not allow the final 7-draw fold to have the same
    # influence as a 1008-draw fold.
    # --------------------------------------------------------

    valid_results = results[
        results["test_draws"].notna()
        & (results["test_draws"] > 0)
    ].copy()

    total_draws = valid_results[
        "test_draws"
    ].sum()

    weighted_hits = (
        (
            valid_results[
                "adaptive_average_hits"
            ]
            * valid_results[
                "test_draws"
            ]
        ).sum()
        / total_draws
    )

    weighted_difference = (
        weighted_hits
        - RANDOM_EXPECTED
    )

    simple_mean_hits = (
        valid_results[
            "adaptive_average_hits"
        ].mean()
    )

    simple_mean_difference = (
        simple_mean_hits
        - RANDOM_EXPECTED
    )

    # --------------------------------------------------------
    # CONSISTENCY
    # --------------------------------------------------------

    above_random = int(
        (
            valid_results[
                "adaptive_difference"
            ] > 0
        ).sum()
    )

    below_random = int(
        (
            valid_results[
                "adaptive_difference"
            ] < 0
        ).sum()
    )

    # --------------------------------------------------------
    # SELECTED STRATEGIES
    # --------------------------------------------------------

    selected_counts = (
        valid_results[
            "selected_strategy"
        ]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADAPTIVE META-MODEL V4 SUMMARY")
    print("=" * 70)

    print(
        f"Weighted adaptive hits: "
        f"{weighted_hits:.6f}"
    )

    print(
        f"Random expected hits:   "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print(
        f"Weighted difference:     "
        f"{weighted_difference:+.6f}"
    )

    relative_improvement = (
        weighted_difference
        / RANDOM_EXPECTED
        * 100
    )

    print(
        f"Relative improvement:    "
        f"{relative_improvement:+.3f}%"
    )

    print(
        f"\nSimple mean hits:        "
        f"{simple_mean_hits:.6f}"
    )

    print(
        f"Simple mean difference:  "
        f"{simple_mean_difference:+.6f}"
    )

    print("\nFold consistency:")

    print(
        f"Above random: {above_random}"
    )

    print(
        f"Below random: {below_random}"
    )

    print("\nSelected strategies:")

    for strategy in strategies:

        print(
            f"  {strategy:<11}: "
            f"{selected_counts.get(strategy, 0)}"
        )

    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MONTE-CARLO NULL TEST")
    print("=" * 70)

    # Use total number of test draws rather than number of folds.
    total_test_draws = int(
        total_draws
    )

    p_value, lower, upper = (
        monte_carlo_test(
            weighted_difference,
            total_test_draws,
            MONTE_CARLO_RUNS,
            RANDOM_SEED
        )
    )

    print(
        f"Observed weighted difference: "
        f"{weighted_difference:+.6f}"
    )

    print(
        f"Monte-Carlo p-value: "
        f"{p_value:.6f}"
    )

    print(
        f"Random 95% range: "
        f"[{lower:+.6f}, {upper:+.6f}]"
    )

    # --------------------------------------------------------
    # FINAL CONCLUSION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        weighted_difference > 0
        and p_value < 0.05
    ):

        print(
            "Adaptive Meta-Model V4 "
            "shows a statistically significant "
            "advantage over the random baseline."
        )

        print(
            f"Weighted average advantage: "
            f"{weighted_difference:+.6f}"
        )

        print(
            f"Relative improvement: "
            f"{relative_improvement:+.3f}%"
        )

        print(
            f"Monte-Carlo p-value: "
            f"{p_value:.6f}"
        )

    elif weighted_difference > 0:

        print(
            "Adaptive Meta-Model V4 is above "
            "the random baseline, but the advantage "
            "is not statistically significant."
        )

        print(
            f"Weighted average advantage: "
            f"{weighted_difference:+.6f}"
        )

        print(
            f"Relative improvement: "
            f"{relative_improvement:+.3f}%"
        )

        print(
            f"Monte-Carlo p-value: "
            f"{p_value:.6f}"
        )

    else:

        print(
            "Adaptive Meta-Model V4 does not "
            "outperform the random baseline."
        )

        print(
            f"Weighted average advantage: "
            f"{weighted_difference:+.6f}"
        )

        print(
            f"Relative improvement: "
            f"{relative_improvement:+.3f}%"
        )

        print(
            f"Monte-Carlo p-value: "
            f"{p_value:.6f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED TO:")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()