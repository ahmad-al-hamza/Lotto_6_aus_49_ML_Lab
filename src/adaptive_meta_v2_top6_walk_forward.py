"""
ADAPTIVE META-MODEL V2 TOP-6 WALK-FORWARD TEST

Improvements over V1:
1. Handles NaN strategy results correctly.
2. Does not treat tiny test folds as equal to large folds.
3. Uses test-draw weighted overall evaluation.
4. Selects strategies only from historically available strategies.
5. Uses recency-weighted historical performance.
6. Includes a Monte-Carlo permutation test for the adaptive strategy.
7. Compares against the random Top-6 baseline.

Expected input files:
    data/processed/ensemble_top6_walk_forward_results.csv
    data/processed/recency_top6_walk_forward_results.csv
    data/processed/stability_top6_walk_forward_results.csv
    data/processed/diversity_top6_walk_forward_results.csv
    data/processed/meta_score_top6_walk_forward_results.csv

Output:
    data/processed/adaptive_meta_v2_top6_walk_forward_results.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

RANDOM_EXPECTED = 6 * 6 / 49

STRATEGIES = [
    "meta_score",
    "ensemble",
    "recency",
    "stability",
    "diversity",
]

RESULT_FILES = {
    "meta_score": DATA_DIR / "meta_score_top6_walk_forward_results.csv",
    "ensemble": DATA_DIR / "ensemble_top6_walk_forward_results.csv",
    "recency": DATA_DIR / "recency_top6_walk_forward_results.csv",
    "stability": DATA_DIR / "stability_top6_walk_forward_results.csv",
    "diversity": DATA_DIR / "diversity_top6_walk_forward_results.csv",
}

OUTPUT_FILE = DATA_DIR / "adaptive_meta_v2_top6_walk_forward_results.csv"

# Historical weighting.
# Larger value = more importance to recent folds.
RECENCY_DECAY = 0.70

# Minimum historical observations required before
# a strategy can receive a full adaptive weight.
MIN_HISTORY = 1

# Monte-Carlo simulations.
N_SIMULATIONS = 10000

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def normalize_weights(values):
    """
    Convert positive values into weights summing to 1.
    """
    values = np.asarray(values, dtype=float)

    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, 0.0)

    total = values.sum()

    if total <= 0:
        return np.ones(len(values)) / len(values)

    return values / total


def load_strategy_results():
    """
    Load all strategy result CSVs.

    The function expects each CSV to contain:
        fold
        average_hits
        test_draws
        difference

    It also supports:
        adaptive_average_hits
        adaptive_difference
        meta_difference
        meta_score
    depending on the source file.
    """

    frames = {}

    for strategy, path in RESULT_FILES.items():

        if not path.exists():
            print(f"WARNING: Missing file: {path}")
            continue

        df = pd.read_csv(path)

        print(
            f"Loaded {strategy:<10}: "
            f"{len(df)} fold results"
        )

        frames[strategy] = df

    return frames


def extract_performance(df, strategy):
    """
    Extract fold-level performance difference and test draws.
    """

    result = pd.DataFrame()

    if "fold" in df.columns:
        result["fold"] = df["fold"]

    else:
        result["fold"] = np.arange(1, len(df) + 1)

    # --------------------------------------------------------
    # Identify performance column
    # --------------------------------------------------------

    performance_columns = [
        "difference",
        "average_difference",
        "adaptive_difference",
        "meta_difference",
        "meta_score_difference",
    ]

    performance_col = None

    for col in performance_columns:
        if col in df.columns:
            performance_col = col
            break

    if performance_col is None:

        # Try to reconstruct from average hits.
        if "average_hits" in df.columns:
            result["performance"] = (
                df["average_hits"] - RANDOM_EXPECTED
            )

        elif "adaptive_average_hits" in df.columns:
            result["performance"] = (
                df["adaptive_average_hits"] - RANDOM_EXPECTED
            )

        elif "meta_score_average_hits" in df.columns:
            result["performance"] = (
                df["meta_score_average_hits"] - RANDOM_EXPECTED
            )

        else:
            raise ValueError(
                f"Could not find performance column for {strategy}"
            )

    else:
        result["performance"] = pd.to_numeric(
            df[performance_col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Test draw count
    # --------------------------------------------------------

    if "test_draws" in df.columns:
        result["test_draws"] = pd.to_numeric(
            df["test_draws"],
            errors="coerce"
        )

    else:
        # Fallback: assume equal-sized folds.
        result["test_draws"] = 1

    result["strategy"] = strategy

    return result


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_performance_matrix(frames):

    all_rows = []

    for strategy, df in frames.items():

        extracted = extract_performance(
            df,
            strategy
        )

        all_rows.append(extracted)

    if not all_rows:
        raise RuntimeError(
            "No strategy result files were loaded."
        )

    combined = pd.concat(
        all_rows,
        ignore_index=True
    )

    performance = combined.pivot_table(
        index="fold",
        columns="strategy",
        values="performance",
        aggfunc="first"
    )

    draws = combined.pivot_table(
        index="fold",
        columns="strategy",
        values="test_draws",
        aggfunc="first"
    )

    # Ensure all expected strategy columns exist.
    for strategy in STRATEGIES:

        if strategy not in performance.columns:
            performance[strategy] = np.nan

        if strategy not in draws.columns:
            draws[strategy] = np.nan

    performance = performance[STRATEGIES]
    draws = draws[STRATEGIES]

    return performance.sort_index(), draws.sort_index()


# ============================================================
# HISTORICAL WEIGHT CALCULATION
# ============================================================

def calculate_historical_weights(
    performance_matrix,
    current_fold
):
    """
    Calculate strategy weights using ONLY folds before
    the current fold.

    This prevents look-ahead leakage.

    Performance is converted into positive evidence.

    More recent historical folds receive greater weight.
    """

    history = performance_matrix[
        performance_matrix.index < current_fold
    ]

    if history.empty:
        return {
            strategy: 1.0 / len(STRATEGIES)
            for strategy in STRATEGIES
        }

    scores = {}

    for strategy in STRATEGIES:

        values = history[strategy].dropna()

        if len(values) < MIN_HISTORY:
            scores[strategy] = 0.0
            continue

        # ----------------------------------------------------
        # Recency weights
        # ----------------------------------------------------

        # oldest -> smallest weight
        # newest -> largest weight

        n = len(values)

        recency_weights = np.array([
            RECENCY_DECAY ** (n - 1 - i)
            for i in range(n)
        ])

        recency_weights /= recency_weights.sum()

        weighted_performance = np.sum(
            values.values * recency_weights
        )

        # ----------------------------------------------------
        # Positive evidence transform
        # ----------------------------------------------------

        # Shift negative performance toward zero.
        #
        # This means a strategy with negative historical
        # performance cannot dominate merely because of
        # numerical scaling.

        score = max(
            weighted_performance,
            0.0
        )

        scores[strategy] = score

    # --------------------------------------------------------
    # If every strategy has non-positive performance,
    # fall back to equal weights among available strategies.
    # --------------------------------------------------------

    positive_total = sum(scores.values())

    if positive_total <= 0:

        available = []

        for strategy in STRATEGIES:

            if performance_matrix.loc[
                performance_matrix.index < current_fold,
                strategy
            ].notna().any():

                available.append(strategy)

        if not available:
            available = STRATEGIES.copy()

        weight = 1.0 / len(available)

        return {
            strategy: weight if strategy in available else 0.0
            for strategy in STRATEGIES
        }

    weights = {
        strategy: scores[strategy] / positive_total
        for strategy in STRATEGIES
    }

    return weights


# ============================================================
# ADAPTIVE SELECTION
# ============================================================

def select_strategy(
    weights,
    current_performance
):
    """
    Select the strategy with the highest adaptive score.

    For the current fold, NaN strategies are unavailable and
    therefore cannot be selected.
    """

    candidates = []

    for strategy in STRATEGIES:

        current_value = current_performance.get(
            strategy,
            np.nan
        )

        if not np.isfinite(current_value):
            continue

        candidates.append(strategy)

    if not candidates:
        return None

    # Current expected contribution.
    #
    # Historical weight × current available performance
    # is NOT used for selection because current performance
    # is unknown at prediction time.
    #
    # Instead select by historical weight.

    selected = max(
        candidates,
        key=lambda s: weights.get(s, 0.0)
    )

    return selected


# ============================================================
# ADAPTIVE COMBINATION
# ============================================================

def calculate_adaptive_prediction(
    weights,
    current_performance
):
    """
    Calculate the weighted adaptive performance.

    IMPORTANT:
    NaN strategies are removed and remaining weights are
    renormalized.
    """

    available = []

    for strategy in STRATEGIES:

        value = current_performance.get(
            strategy,
            np.nan
        )

        if np.isfinite(value):
            available.append(strategy)

    if not available:
        return np.nan

    raw_weights = np.array([
        weights.get(strategy, 0.0)
        for strategy in available
    ])

    raw_weights = normalize_weights(
        raw_weights
    )

    values = np.array([
        current_performance[strategy]
        for strategy in available
    ])

    adaptive_difference = np.sum(
        raw_weights * values
    )

    return adaptive_difference


# ============================================================
# MONTE CARLO TEST
# ============================================================

def monte_carlo_test(
    adaptive_differences,
    test_draws,
    n_simulations=N_SIMULATIONS,
    seed=RANDOM_SEED
):
    """
    Monte-Carlo test.

    Null hypothesis:
        Adaptive strategy has no real advantage.

    We generate random Top-6 performance for each fold
    using the theoretical Binomial(6, 6/49) distribution.

    The test is weighted by actual test-draw counts.
    """

    rng = np.random.default_rng(seed)

    adaptive_differences = np.asarray(
        adaptive_differences,
        dtype=float
    )

    test_draws = np.asarray(
        test_draws,
        dtype=float
    )

    valid = (
        np.isfinite(adaptive_differences)
        &
        np.isfinite(test_draws)
        &
        (test_draws > 0)
    )

    adaptive_differences = adaptive_differences[valid]
    test_draws = test_draws[valid]

    if len(adaptive_differences) == 0:
        return np.nan, np.nan, np.nan

    actual_weighted_difference = np.average(
        adaptive_differences,
        weights=test_draws
    )

    simulated = np.zeros(
        n_simulations,
        dtype=float
    )

    p = 6 / 49

    for i in range(n_simulations):

        total_hits = 0.0
        total_draws = 0.0

        for draws in test_draws:

            draws = int(draws)

            hits = rng.binomial(
                n=6,
                p=p,
                size=draws
            )

            total_hits += hits.sum()
            total_draws += draws

        simulated_average_hits = (
            total_hits / total_draws
        )

        simulated[i] = (
            simulated_average_hits
            - RANDOM_EXPECTED
        )

    p_value = np.mean(
        simulated >= actual_weighted_difference
    )

    lower = np.percentile(
        simulated,
        2.5
    )

    upper = np.percentile(
        simulated,
        97.5
    )

    return (
        actual_weighted_difference,
        p_value,
        (lower, upper)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE META-MODEL V2 TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print()

    print("Loading strategy results...")

    frames = load_strategy_results()

    if not frames:
        raise RuntimeError(
            "No strategy result files available."
        )

    print()

    performance_matrix, draws_matrix = (
        build_performance_matrix(frames)
    )

    print("=" * 70)
    print("STRATEGY PERFORMANCE MATRIX")
    print("=" * 70)

    print(
        performance_matrix.to_string(
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    print()

    print("=" * 70)
    print("TEST DRAW MATRIX")
    print("=" * 70)

    print(
        draws_matrix.to_string(
            float_format=lambda x: f"{x:.0f}"
        )
    )

    print()

    print("=" * 70)
    print("ADAPTIVE META-MODEL V2 WALK-FORWARD TEST")
    print("=" * 70)

    results = []

    folds = sorted(
        performance_matrix.index
    )

    for fold in folds:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        current_performance = (
            performance_matrix.loc[fold]
        )

        current_draws = draws_matrix.loc[fold]

        # ----------------------------------------------------
        # Historical weights
        # ----------------------------------------------------

        weights = calculate_historical_weights(
            performance_matrix,
            fold
        )

        print()
        print("Historical strategy weights:")

        for strategy in STRATEGIES:

            print(
                f"  {strategy:<10}: "
                f"{weights[strategy]:.4f}"
            )

        # ----------------------------------------------------
        # Available strategies
        # ----------------------------------------------------

        available = [
            strategy
            for strategy in STRATEGIES
            if np.isfinite(
                current_performance.get(
                    strategy,
                    np.nan
                )
            )
        ]

        print()
        print(
            "Available strategies: "
            + ", ".join(available)
        )

        # ----------------------------------------------------
        # Select strategy
        # ----------------------------------------------------

        selected = select_strategy(
            weights,
            current_performance
        )

        if selected is None:

            print(
                "Selected strategy: NONE"
            )

        else:

            print(
                f"Selected strategy: "
                f"{selected.upper()}"
            )

        # ----------------------------------------------------
        # Adaptive weighted performance
        # ----------------------------------------------------

        adaptive_difference = (
            calculate_adaptive_prediction(
                weights,
                current_performance
            )
        )

        if np.isfinite(adaptive_difference):

            adaptive_hits = (
                RANDOM_EXPECTED
                + adaptive_difference
            )

            print(
                f"Adaptive weighted difference: "
                f"{adaptive_difference:+.6f}"
            )

            print(
                f"Adaptive weighted hits: "
                f"{adaptive_hits:.6f}"
            )

        else:

            adaptive_hits = np.nan

            print(
                "Adaptive weighted difference: NaN"
            )

        # ----------------------------------------------------
        # Selected strategy performance
        # ----------------------------------------------------

        if selected is not None:

            selected_difference = (
                current_performance[selected]
            )

            selected_hits = (
                RANDOM_EXPECTED
                + selected_difference
            )

            selected_draws = (
                current_draws[selected]
            )

        else:

            selected_difference = np.nan
            selected_hits = np.nan
            selected_draws = np.nan

        print()
        print("Current fold performance:")

        for strategy in STRATEGIES:

            value = current_performance.get(
                strategy,
                np.nan
            )

            if np.isfinite(value):

                print(
                    f"  {strategy:<10}: "
                    f"{value:+.6f}"
                )

            else:

                print(
                    f"  {strategy:<10}: NaN"
                )

        # ----------------------------------------------------
        # Determine fold draws
        # ----------------------------------------------------

        valid_draws = current_draws[
            current_draws.notna()
        ]

        if len(valid_draws) > 0:

            fold_draws = int(
                valid_draws.max()
            )

        else:

            fold_draws = 0

        results.append({
            "fold": fold,
            "test_draws": fold_draws,
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

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("FOLD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    # ========================================================
    # WEIGHTED SUMMARY
    # ========================================================

    valid = results_df[
        results_df["adaptive_difference"].notna()
        &
        results_df["test_draws"].notna()
        &
        (results_df["test_draws"] > 0)
    ].copy()

    if len(valid) == 0:

        print(
            "\nNo valid adaptive results."
        )

        return

    weighted_difference = np.average(
        valid["adaptive_difference"],
        weights=valid["test_draws"]
    )

    weighted_hits = (
        RANDOM_EXPECTED
        + weighted_difference
    )

    # Fold-level simple average for comparison.
    simple_difference = (
        valid["adaptive_difference"].mean()
    )

    simple_hits = (
        RANDOM_EXPECTED
        + simple_difference
    )

    relative_improvement = (
        weighted_difference
        / RANDOM_EXPECTED
        * 100
    )

    above_random = int(
        (valid["adaptive_difference"] > 0).sum()
    )

    below_random = int(
        (valid["adaptive_difference"] < 0).sum()
    )

    print()
    print("=" * 70)
    print("ADAPTIVE META-MODEL V2 SUMMARY")
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
        f"{simple_hits:.6f}"
    )

    print(
        f"Simple mean difference:  "
        f"{simple_difference:+.6f}"
    )

    print()

    print("Fold consistency:")

    print(
        f"Above random: {above_random}"
    )

    print(
        f"Below random: {below_random}"
    )

    # ========================================================
    # SELECTED STRATEGIES
    # ========================================================

    print()
    print("Selected strategies:")

    selected_counts = (
        results_df[
            results_df["selected_strategy"].notna()
        ]["selected_strategy"]
        .value_counts()
    )

    for strategy in STRATEGIES:

        print(
            f"  {strategy:<10}: "
            f"{selected_counts.get(strategy, 0)}"
        )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    print()
    print("=" * 70)
    print("MONTE-CARLO NULL TEST")
    print("=" * 70)

    actual_difference, p_value, ci = (
        monte_carlo_test(
            valid["adaptive_difference"].values,
            valid["test_draws"].values,
            N_SIMULATIONS,
            RANDOM_SEED
        )
    )

    print(
        f"Observed weighted difference: "
        f"{actual_difference:+.6f}"
    )

    print(
        f"Monte-Carlo p-value: "
        f"{p_value:.6f}"
    )

    if ci is not None:

        print(
            "Random 95% range: "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if weighted_difference > 0:

        if p_value < 0.05:

            print(
                "Adaptive Meta-Model V2 is above random "
                "with statistically significant evidence."
            )

        else:

            print(
                "Adaptive Meta-Model V2 is above random "
                "but the advantage is NOT statistically significant."
            )

    else:

        print(
            "Adaptive Meta-Model V2 does not outperform "
            "the random baseline."
        )

    print()
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

    # Add summary metadata columns.
    results_df["weighted_summary_difference"] = (
        weighted_difference
    )

    results_df["weighted_summary_hits"] = (
        weighted_hits
    )

    results_df["relative_improvement_percent"] = (
        relative_improvement
    )

    results_df["monte_carlo_p_value"] = (
        p_value
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED TO:")
    print("=" * 70)

    print(OUTPUT_FILE)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()