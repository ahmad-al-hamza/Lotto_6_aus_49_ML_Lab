import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_EXPECTED = 6 * 6 / 49

MAX_WEIGHT = 0.40
MIN_WEIGHT = 0.05

# Number of historical folds used to estimate current performance
ROLLING_FOLDS = 3

# Shrink historical performance toward zero
SHRINKAGE = 0.50

# Minimum number of historical draws required before trusting
# a strategy strongly
MIN_HISTORICAL_DRAWS = 1000

MONTE_CARLO_RUNS = 10000

DATA_DIR = "data/processed"

STRATEGY_FILES = {
    "meta_score": "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": "recency_top6_walk_forward_results.csv",
    "stability": "stability_top6_walk_forward_results.csv",
    "diversity": "diversity_top6_walk_forward_results.csv",
    "ensemble": "ensemble_top6_walk_forward_results.csv",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_weights(scores):
    """
    Convert strategy scores into stable positive weights.

    Positive historical performance gets more weight.
    Negative performance does not automatically get zero weight;
    a minimum floor is retained to avoid overfitting.
    """

    scores = pd.Series(scores, dtype=float)

    valid = scores.dropna()

    if len(valid) == 0:
        return pd.Series(
            1.0 / len(scores),
            index=scores.index
        )

    # Center around zero
    centered = valid - valid.mean()

    # Softmax-like transformation
    temperature = max(centered.std(), 0.01)

    exp_scores = np.exp(
        np.clip(centered / temperature, -5, 5)
    )

    weights = exp_scores / exp_scores.sum()

    result = pd.Series(0.0, index=scores.index)

    for strategy in valid.index:
        result.loc[strategy] = weights.loc[strategy]

    # Apply maximum weight
    result = result.clip(
        lower=MIN_WEIGHT,
        upper=MAX_WEIGHT
    )

    # Only available strategies
    result[scores.isna()] = 0.0

    total = result.sum()

    if total <= 0:
        available = scores.notna()

        result.loc[available] = (
            1.0 / available.sum()
        )
    else:
        result = result / total

    return result


def load_strategy_results():

    print("=" * 70)
    print("LOADING STRATEGY RESULTS")
    print("=" * 70)

    matrices = {}
    draw_matrices = {}

    for strategy, filename in STRATEGY_FILES.items():

        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            print(
                f"{strategy:12s}: FILE NOT FOUND -> {filename}"
            )
            continue

        try:
            df = pd.read_csv(path)

            if "fold" not in df.columns:
                print(
                    f"{strategy:12s}: missing fold column"
                )
                continue

            # ------------------------------------------------
            # Identify performance column
            # ------------------------------------------------

            performance_column = None

            candidates = [
                "difference",
                "adaptive_difference",
                "selected_difference",
                "mean_difference",
            ]

            for col in candidates:
                if col in df.columns:
                    performance_column = col
                    break

            if performance_column is None:
                print(
                    f"{strategy:12s}: no performance column"
                )
                continue

            performance = (
                df.set_index("fold")[performance_column]
                .astype(float)
            )

            matrices[strategy] = performance

            # ------------------------------------------------
            # Test draw column
            # ------------------------------------------------

            draw_column = None

            for col in [
                "test_draws",
                "test_draw_count",
                "n_test",
            ]:
                if col in df.columns:
                    draw_column = col
                    break

            if draw_column is not None:
                draws = (
                    df.set_index("fold")[draw_column]
                    .astype(float)
                )
            else:
                # Most strategy files use 1008 observations
                draws = pd.Series(
                    1008.0,
                    index=performance.index
                )

            draw_matrices[strategy] = draws

            print(
                f"{strategy:12s}: "
                f"{len(performance)} rows -> {filename}"
            )

        except Exception as e:

            print(
                f"{strategy:12s}: ERROR -> {e}"
            )

    return matrices, draw_matrices


# ============================================================
# BUILD MATRIX
# ============================================================

def build_matrix(matrices):

    all_folds = sorted(
        set(
            fold
            for series in matrices.values()
            for fold in series.index
        )
    )

    strategies = list(STRATEGY_FILES.keys())

    performance = pd.DataFrame(
        index=all_folds,
        columns=strategies,
        dtype=float
    )

    for strategy in strategies:

        if strategy in matrices:

            for fold, value in matrices[strategy].items():

                performance.loc[
                    fold,
                    strategy
                ] = value

    return performance


def build_draw_matrix(draw_matrices, folds):

    strategies = list(STRATEGY_FILES.keys())

    draws = pd.DataFrame(
        index=folds,
        columns=strategies,
        dtype=float
    )

    for strategy in strategies:

        if strategy in draw_matrices:

            for fold, value in draw_matrices[strategy].items():

                draws.loc[
                    fold,
                    strategy
                ] = value

    return draws


# ============================================================
# HISTORICAL SCORE
# ============================================================

def historical_score(
    performance,
    draws,
    current_fold,
    strategy
):

    previous_folds = [
        f for f in performance.index
        if f < current_fold
    ]

    if not previous_folds:
        return 0.0

    previous_folds = previous_folds[
        -ROLLING_FOLDS:
    ]

    values = []

    weights = []

    for fold in previous_folds:

        value = performance.loc[
            fold,
            strategy
        ]

        n_draws = draws.loc[
            fold,
            strategy
        ]

        if pd.isna(value):
            continue

        if pd.isna(n_draws):
            n_draws = 1008

        values.append(value)
        weights.append(n_draws)

    if not values:
        return 0.0

    values = np.asarray(values)
    weights = np.asarray(weights)

    weighted_mean = np.average(
        values,
        weights=weights
    )

    # Shrink toward zero
    effective_draws = weights.sum()

    shrink_factor = (
        effective_draws /
        (effective_draws + SHRINKAGE * 1000)
    )

    score = (
        weighted_mean *
        shrink_factor
    )

    return score


# ============================================================
# V6 WEIGHTS
# ============================================================

def calculate_v6_weights(
    performance,
    draws,
    current_fold
):

    strategies = performance.columns

    scores = {}

    for strategy in strategies:

        # Current strategy must exist
        if pd.isna(
            performance.loc[
                current_fold,
                strategy
            ]
        ):
            scores[strategy] = np.nan
            continue

        score = historical_score(
            performance,
            draws,
            current_fold,
            strategy
        )

        scores[strategy] = score

    scores = pd.Series(scores)

    available = scores.notna()

    if not available.any():

        return pd.Series(
            1.0 / len(strategies),
            index=strategies
        )

    # --------------------------------------------------------
    # Convert historical scores to positive values
    # --------------------------------------------------------

    valid = scores[available]

    # Center at zero
    centered = valid - valid.mean()

    # Scale conservatively
    scale = max(
        centered.abs().mean(),
        0.005
    )

    transformed = np.exp(
        np.clip(
            centered / scale,
            -2,
            2
        )
    )

    raw = transformed / transformed.sum()

    weights = pd.Series(
        0.0,
        index=strategies
    )

    weights.loc[available] = raw

    # --------------------------------------------------------
    # Maximum weight
    # --------------------------------------------------------

    weights = weights.clip(
        upper=MAX_WEIGHT
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = weights.sum()

    if total <= 0:

        weights.loc[available] = (
            1.0 /
            available.sum()
        )

    else:

        weights /= total

    return weights


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_test(
    observed_difference,
    test_draws,
    runs=10000
):

    if test_draws <= 0:
        return np.nan, np.nan, np.nan

    # Simulate random Top-6 selections.
    #
    # Each draw has expected:
    #
    # 6 * 6 / 49 = 0.734694 hits

    p = RANDOM_EXPECTED

    simulated_hits = np.random.binomial(
        n=6,
        p=6 / 49,
        size=(runs, test_draws)
    )

    simulated_average = (
        simulated_hits.mean(axis=1)
    )

    simulated_difference = (
        simulated_average - RANDOM_EXPECTED
    )

    p_value = np.mean(
        np.abs(simulated_difference)
        >= abs(observed_difference)
    )

    lower = np.percentile(
        simulated_difference,
        2.5
    )

    upper = np.percentile(
        simulated_difference,
        97.5
    )

    return p_value, lower, upper


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE META-MODEL V6 TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print(
        f"Maximum strategy weight: "
        f"{MAX_WEIGHT:.2f}"
    )

    print(
        f"Minimum historical draws: "
        f"{MIN_HISTORICAL_DRAWS}"
    )

    print(
        f"Rolling folds: "
        f"{ROLLING_FOLDS}"
    )

    print(
        f"Shrinkage: "
        f"{SHRINKAGE}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    matrices, draw_matrices = (
        load_strategy_results()
    )

    if not matrices:

        print(
            "ERROR: No strategy results found."
        )

        return

    performance = build_matrix(
        matrices
    )

    draws = build_draw_matrix(
        draw_matrices,
        performance.index
    )

    print()
    print("=" * 70)
    print("PERFORMANCE DIFFERENCE MATRIX")
    print("=" * 70)

    print(
        performance.to_string(
            float_format=lambda x:
            f"{x:+.6f}"
        )
    )

    print()
    print("=" * 70)
    print("TEST DRAW MATRIX")
    print("=" * 70)

    print(
        draws.to_string()
    )

    # --------------------------------------------------------
    # Walk forward
    # --------------------------------------------------------

    results = []
    weights_history = []

    cumulative_hits = 0.0
    cumulative_draws = 0.0

    print()
    print("=" * 70)
    print("ADAPTIVE META-MODEL V6 WALK-FORWARD TEST")
    print("=" * 70)

    for fold in performance.index:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        available = performance.loc[
            fold
        ].dropna().index.tolist()

        if not available:

            print(
                "No available strategies."
            )

            continue

        print()
        print(
            "Available strategies: "
            + ", ".join(available)
        )

        # ----------------------------------------------------
        # Calculate weights
        # ----------------------------------------------------

        weights = calculate_v6_weights(
            performance,
            draws,
            fold
        )

        # Remove unavailable strategies
        for strategy in weights.index:

            if strategy not in available:

                weights.loc[strategy] = 0.0

        total = weights.sum()

        if total > 0:

            weights /= total

        print()
        print("V6 strategy weights:")

        for strategy in weights.index:

            print(
                f"  {strategy:12s}: "
                f"{weights.loc[strategy]:.4f}"
            )

        # ----------------------------------------------------
        # Selected strategy
        # ----------------------------------------------------

        selected = (
            weights[
                weights > 0
            ].idxmax()
        )

        print()
        print(
            "Selected highest-weight strategy: "
            + selected.upper()
        )

        # ----------------------------------------------------
        # Current fold performance
        # ----------------------------------------------------

        current = performance.loc[
            fold
        ]

        current = current[
            available
        ]

        adaptive_difference = 0.0

        for strategy in available:

            adaptive_difference += (
                weights.loc[strategy]
                *
                current.loc[strategy]
            )

        adaptive_hits = (
            RANDOM_EXPECTED
            +
            adaptive_difference
        )

        print(
            f"Adaptive weighted difference: "
            f"{adaptive_difference:+.6f}"
        )

        print(
            f"Adaptive weighted hits: "
            f"{adaptive_hits:.6f}"
        )

        print()
        print("Current fold performance:")

        for strategy in available:

            print(
                f"  {strategy:12s}: "
                f"{current.loc[strategy]:+.6f}"
            )

        # ----------------------------------------------------
        # Test draws
        # ----------------------------------------------------

        available_draws = draws.loc[
            fold,
            available
        ].dropna()

        if len(available_draws) > 0:

            test_draws = int(
                available_draws.max()
            )

        else:

            test_draws = 0

        # ----------------------------------------------------
        # Selected strategy performance
        # ----------------------------------------------------

        selected_difference = (
            current.loc[selected]
        )

        selected_hits = (
            RANDOM_EXPECTED
            +
            selected_difference
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append({
            "fold": fold,
            "test_draws": test_draws,
            "adaptive_average_hits":
                adaptive_hits,
            "adaptive_difference":
                adaptive_difference,
            "selected_strategy":
                selected,
            "selected_average_hits":
                selected_hits,
            "selected_difference":
                selected_difference,
        })

        weight_record = {
            "fold": fold
        }

        for strategy in weights.index:

            weight_record[strategy] = (
                weights.loc[strategy]
            )

        weights_history.append(
            weight_record
        )

        # ----------------------------------------------------
        # Cumulative weighted evaluation
        # ----------------------------------------------------

        if test_draws > 0:

            cumulative_hits += (
                adaptive_hits *
                test_draws
            )

            cumulative_draws += (
                test_draws
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    weights_df = pd.DataFrame(
        weights_history
    )

    if results_df.empty:

        print(
            "ERROR: No fold results."
        )

        return

    # --------------------------------------------------------
    # Weighted overall performance
    # --------------------------------------------------------

    weighted_hits = (
        cumulative_hits /
        cumulative_draws
    )

    weighted_difference = (
        weighted_hits -
        RANDOM_EXPECTED
    )

    relative_improvement = (
        weighted_difference /
        RANDOM_EXPECTED
        * 100
    )

    # --------------------------------------------------------
    # Simple fold mean
    # --------------------------------------------------------

    simple_mean_hits = (
        results_df[
            "adaptive_average_hits"
        ].mean()
    )

    simple_mean_difference = (
        simple_mean_hits -
        RANDOM_EXPECTED
    )

    # --------------------------------------------------------
    # Fold consistency
    # --------------------------------------------------------

    above_random = (
        results_df[
            "adaptive_difference"
        ] > 0
    ).sum()

    below_random = (
        results_df[
            "adaptive_difference"
        ] < 0
    ).sum()

    # --------------------------------------------------------
    # Selected strategies
    # --------------------------------------------------------

    selected_counts = (
        results_df[
            "selected_strategy"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Print fold results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FOLD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:+.6f}"
        )
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V6 FINAL EVALUATION")
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

    print(
        f"Relative improvement:    "
        f"{relative_improvement:+.3f}%"
    )

    print()
    print(
        f"Simple mean hits:        "
        f"{simple_mean_hits:.6f}"
    )

    print(
        f"Simple mean difference:  "
        f"{simple_mean_difference:+.6f}"
    )

    print()
    print("Fold consistency:")

    print(
        f"Above random: {above_random}"
    )

    print(
        f"Below random: {below_random}"
    )

    print()
    print("Selected strategies:")

    for strategy in STRATEGY_FILES:

        count = selected_counts.get(
            strategy,
            0
        )

        print(
            f"  {strategy:12s}: {count}"
        )

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MONTE-CARLO NULL TEST")
    print("=" * 70)

    p_value, lower, upper = (
        monte_carlo_test(
            weighted_difference,
            int(cumulative_draws),
            MONTE_CARLO_RUNS
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
    # Conclusion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        weighted_difference > 0
        and p_value < 0.05
    ):

        print(
            "V6 shows statistically significant "
            "improvement over the random baseline."
        )

    elif weighted_difference > 0:

        print(
            "V6 is above the random baseline, "
            "but the advantage is not statistically significant."
        )

    else:

        print(
            "V6 does not outperform the "
            "random baseline."
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

    # ========================================================
    # SAVE
    # ========================================================

    result_path = os.path.join(
        DATA_DIR,
        "adaptive_meta_v6_top6_walk_forward_results.csv"
    )

    weights_path = os.path.join(
        DATA_DIR,
        "adaptive_meta_v6_weights.csv"
    )

    results_df.to_csv(
        result_path,
        index=False
    )

    weights_df.to_csv(
        weights_path,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(result_path)
    print(weights_path)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()