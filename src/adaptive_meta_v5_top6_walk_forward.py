"""
ADAPTIVE META-MODEL V5 TOP-6 WALK-FORWARD TEST

V5 improvements:
1. Sample-size weighted evaluation.
2. Shrinkage toward the random baseline.
3. Information-diversity adjustment.
4. Maximum strategy weight cap.
5. Minimum historical sample requirement.
6. No current-fold leakage.
7. Weighted final evaluation by number of test draws.
8. Monte-Carlo null test.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_EXPECTED = 6 / 49

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

STRATEGY_FILES = {
    "meta_score": DATA_DIR / "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": DATA_DIR / "recency_top6_walk_forward_results.csv",
    "stability": DATA_DIR / "stability_top6_walk_forward_results.csv",
    "diversity": DATA_DIR / "diversity_top6_walk_forward_results.csv",
    "ensemble": DATA_DIR / "ensemble_top6_walk_forward_results.csv",
}

OUTPUT_FILE = (
    DATA_DIR / "adaptive_meta_v5_top6_walk_forward_results.csv"
)

WEIGHTS_OUTPUT_FILE = (
    DATA_DIR / "adaptive_meta_v5_weights.csv"
)

MC_SIMULATIONS = 10000

# Do not allow one strategy to dominate the ensemble.
MAX_WEIGHT = 0.60

# Minimum number of historical test draws required
# before a strategy can strongly influence V5.
MIN_HISTORY_DRAWS = 500

# Shrinkage strength.
# Larger value = more conservative.
SHRINKAGE_STRENGTH = 1000

# How strongly information diversity affects weights.
DIVERSITY_POWER = 0.50


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_strategy_results():

    print_header("LOADING STRATEGY RESULTS")

    loaded = {}

    for strategy, path in STRATEGY_FILES.items():

        if not path.exists():
            print(f"{strategy:<12}: FILE NOT FOUND")
            continue

        try:
            df = pd.read_csv(path)

            # Try common fold column names
            if "fold" not in df.columns:
                print(f"{strategy:<12}: no fold column")
                continue

            loaded[strategy] = df.copy()

            print(
                f"{strategy:<12}: "
                f"{len(df)} rows -> {path.name}"
            )

        except Exception as exc:
            print(
                f"{strategy:<12}: ERROR -> {exc}"
            )

    return loaded


def find_difference_column(df):

    candidates = [
        "adaptive_difference",
        "difference",
        "mean_difference",
        "selected_difference",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def find_hits_column(df):

    candidates = [
        "adaptive_average_hits",
        "average_hits",
        "selected_average_hits",
        "mean_hits",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def find_draw_column(df):

    candidates = [
        "test_draws",
        "testing_draws",
        "draws",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_matrices(loaded):

    print_header("BUILDING PERFORMANCE MATRIX")

    strategy_names = list(loaded.keys())

    all_folds = sorted(
        set(
            fold
            for df in loaded.values()
            for fold in df["fold"].dropna().astype(int)
        )
    )

    performance = pd.DataFrame(
        index=all_folds,
        columns=strategy_names,
        dtype=float,
    )

    draws = pd.DataFrame(
        index=all_folds,
        columns=strategy_names,
        dtype=float,
    )

    hits = pd.DataFrame(
        index=all_folds,
        columns=strategy_names,
        dtype=float,
    )

    for strategy, df in loaded.items():

        difference_col = find_difference_column(df)
        hits_col = find_hits_column(df)
        draw_col = find_draw_column(df)

        if difference_col is None:
            print(
                f"WARNING: {strategy} "
                f"has no difference column"
            )
            continue

        for _, row in df.iterrows():

            fold = int(row["fold"])

            performance.loc[fold, strategy] = (
                pd.to_numeric(
                    row[difference_col],
                    errors="coerce",
                )
            )

            if hits_col is not None:
                hits.loc[fold, strategy] = (
                    pd.to_numeric(
                        row[hits_col],
                        errors="coerce",
                    )
                )

            if draw_col is not None:
                draws.loc[fold, strategy] = (
                    pd.to_numeric(
                        row[draw_col],
                        errors="coerce",
                    )
                )

    print()
    print("PERFORMANCE DIFFERENCE MATRIX")
    print(performance.to_string(float_format=lambda x: f"{x:+.6f}"))

    print()
    print("TEST DRAW MATRIX")
    print(draws.to_string())

    return performance, draws, hits


# ============================================================
# INFORMATION DIVERSITY
# ============================================================

def calculate_information_diversity(performance):

    print_header("CALCULATING INFORMATION DIVERSITY")

    correlation = performance.corr(
        method="pearson",
        min_periods=2,
    )

    diversity = {}

    for strategy in performance.columns:

        others = [
            s for s in performance.columns
            if s != strategy
        ]

        values = []

        for other in others:

            pair = performance[
                [strategy, other]
            ].dropna()

            if len(pair) >= 3:

                corr = pair[strategy].corr(
                    pair[other]
                )

                if pd.notna(corr):
                    values.append(abs(corr))

        if values:
            mean_abs_corr = np.mean(values)
            diversity[strategy] = 1.0 - mean_abs_corr
        else:
            diversity[strategy] = 0.5

    diversity_series = pd.Series(diversity)

    print()
    print(
        "Information diversity:"
    )

    for strategy, value in diversity_series.sort_values(
        ascending=False
    ).items():

        print(
            f"  {strategy:<12}: {value:.4f}"
        )

    return diversity_series, correlation


# ============================================================
# HISTORICAL STATISTICS
# ============================================================

def historical_statistics(
    performance,
    draws,
    current_fold,
):
    """
    Use only folds BEFORE current_fold.
    """

    history = performance[
        performance.index < current_fold
    ]

    history_draws = draws[
        draws.index < current_fold
    ]

    stats = {}

    for strategy in performance.columns:

        values = history[strategy].dropna()

        if len(values) == 0:
            stats[strategy] = {
                "mean": RANDOM_EXPECTED * 0,
                "sample_draws": 0,
                "folds": 0,
            }
            continue

        strategy_draws = (
            history_draws[strategy]
            .loc[values.index]
            .fillna(0)
        )

        total_draws = strategy_draws.sum()

        if total_draws <= 0:
            weighted_mean = values.mean()
        else:
            weighted_mean = np.average(
                values.values,
                weights=strategy_draws.values,
            )

        stats[strategy] = {
            "mean": weighted_mean,
            "sample_draws": float(total_draws),
            "folds": int(len(values)),
        }

    return stats


# ============================================================
# V5 WEIGHT CALCULATION
# ============================================================

def calculate_v5_weights(
    stats,
    diversity,
):

    raw_scores = {}

    for strategy, info in stats.items():

        historical_mean = info["mean"]
        sample_draws = info["sample_draws"]

        # ----------------------------------------------------
        # 1. Sample reliability
        # ----------------------------------------------------

        reliability = (
            sample_draws /
            (sample_draws + SHRINKAGE_STRENGTH)
        )

        # ----------------------------------------------------
        # 2. Shrink performance toward zero
        # ----------------------------------------------------

        shrunk_performance = (
            historical_mean * reliability
        )

        # ----------------------------------------------------
        # 3. Positive performance component
        #
        # Negative historical strategies are not forbidden,
        # but their contribution is heavily reduced.
        # ----------------------------------------------------

        positive_score = max(
            0.0,
            shrunk_performance
        )

        # ----------------------------------------------------
        # 4. Information diversity
        # ----------------------------------------------------

        diversity_value = diversity.get(
            strategy,
            0.5
        )

        diversity_factor = (
            max(diversity_value, 0.05)
            ** DIVERSITY_POWER
        )

        # ----------------------------------------------------
        # 5. Combined score
        # ----------------------------------------------------

        raw_score = (
            positive_score *
            diversity_factor
        )

        # If there is not enough historical data,
        # give only a very small base score.
        if sample_draws < MIN_HISTORY_DRAWS:
            raw_score *= 0.25

        raw_scores[strategy] = raw_score

    raw = pd.Series(raw_scores, dtype=float)

    # --------------------------------------------------------
    # If all scores are zero:
    # equal weights.
    # --------------------------------------------------------

    if raw.sum() <= 0:

        weights = pd.Series(
            1.0 / len(raw),
            index=raw.index,
        )

    else:

        weights = raw / raw.sum()

    # --------------------------------------------------------
    # Apply MAX_WEIGHT cap iteratively.
    # --------------------------------------------------------

    weights = apply_weight_cap(
        weights,
        MAX_WEIGHT,
    )

    return weights, raw


def apply_weight_cap(weights, max_weight):

    weights = weights.copy()

    if len(weights) == 0:
        return weights

    # Repeated redistribution.
    for _ in range(100):

        over = weights > max_weight

        if not over.any():
            break

        excess = (
            weights[over] - max_weight
        ).sum()

        weights[over] = max_weight

        under = ~over

        if under.sum() == 0:
            break

        under_total = weights[under].sum()

        if under_total <= 0:
            weights[under] += (
                excess / under.sum()
            )
        else:
            weights[under] += (
                weights[under] /
                under_total *
                excess
            )

    # Final normalization
    total = weights.sum()

    if total > 0:
        weights /= total

    return weights


# ============================================================
# V5 WALK FORWARD
# ============================================================

def run_walk_forward(
    performance,
    draws,
    diversity,
):

    print_header(
        "ADAPTIVE META-MODEL V5 WALK-FORWARD TEST"
    )

    folds = sorted(performance.index)

    fold_results = []
    weight_records = []

    for fold in folds:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        current = performance.loc[fold]

        available = current.dropna().index.tolist()

        if not available:
            print("No available strategies.")
            continue

        print()
        print(
            "Available strategies: "
            + ", ".join(available)
        )

        # ----------------------------------------------------
        # Historical information ONLY
        # ----------------------------------------------------

        stats_all = historical_statistics(
            performance,
            draws,
            fold,
        )

        stats = {
            strategy: stats_all[strategy]
            for strategy in available
        }

        # ----------------------------------------------------
        # Calculate weights
        # ----------------------------------------------------

        available_diversity = diversity.reindex(
            available
        ).fillna(0.5)

        weights, raw_scores = calculate_v5_weights(
            stats,
            available_diversity,
        )

        print()
        print("V5 strategy weights:")

        for strategy in available:
            print(
                f"  {strategy:<12}: "
                f"{weights[strategy]:.4f}"
            )

        selected = weights.idxmax()

        # ----------------------------------------------------
        # Current fold performance
        # ----------------------------------------------------

        current_perf = current[
            available
        ]

        # Weighted difference
        weighted_difference = float(
            np.sum(
                weights.values *
                current_perf.values
            )
        )

        adaptive_hits = (
            RANDOM_EXPECTED +
            weighted_difference
        )

        # ----------------------------------------------------
        # Selected strategy
        # ----------------------------------------------------

        selected_difference = float(
            current_perf[selected]
        )

        selected_hits = (
            RANDOM_EXPECTED +
            selected_difference
        )

        # ----------------------------------------------------
        # Draw count
        # ----------------------------------------------------

        current_draws = draws.loc[
            fold,
            available
        ].dropna()

        if len(current_draws) > 0:
            # Use max available draw count for fold
            test_draws = int(
                current_draws.max()
            )
        else:
            test_draws = 0

        print()
        print(
            f"Selected highest-weight strategy: "
            f"{selected.upper()}"
        )

        print(
            f"Adaptive weighted difference: "
            f"{weighted_difference:+.6f}"
        )

        print(
            f"Adaptive weighted hits: "
            f"{adaptive_hits:.6f}"
        )

        print()
        print("Current fold performance:")

        for strategy in available:

            print(
                f"  {strategy:<12}: "
                f"{current_perf[strategy]:+.6f}"
            )

        # ----------------------------------------------------
        # Save fold result
        # ----------------------------------------------------

        fold_results.append({
            "fold": fold,
            "test_draws": test_draws,
            "adaptive_average_hits": adaptive_hits,
            "adaptive_difference": weighted_difference,
            "selected_strategy": selected,
            "selected_average_hits": selected_hits,
            "selected_difference": selected_difference,
        })

        # ----------------------------------------------------
        # Save weights
        # ----------------------------------------------------

        record = {
            "fold": fold,
            "selected_strategy": selected,
        }

        for strategy in performance.columns:
            record[
                f"weight_{strategy}"
            ] = weights.get(
                strategy,
                0.0
            )

            if strategy in stats:
                record[
                    f"historical_mean_{strategy}"
                ] = stats[strategy]["mean"]

                record[
                    f"historical_draws_{strategy}"
                ] = stats[strategy]["sample_draws"]

        weight_records.append(record)

    return (
        pd.DataFrame(fold_results),
        pd.DataFrame(weight_records),
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def final_evaluation(
    fold_results,
):

    print_header("V5 FINAL EVALUATION")

    if fold_results.empty:
        print("No fold results.")
        return

    # --------------------------------------------------------
    # Simple mean
    # --------------------------------------------------------

    simple_mean_hits = (
        fold_results[
            "adaptive_average_hits"
        ].mean()
    )

    simple_difference = (
        simple_mean_hits -
        RANDOM_EXPECTED
    )

    # --------------------------------------------------------
    # Draw-weighted mean
    # --------------------------------------------------------

    valid = fold_results[
        fold_results["test_draws"] > 0
    ].copy()

    if not valid.empty:

        weighted_hits = np.average(
            valid["adaptive_average_hits"],
            weights=valid["test_draws"],
        )

    else:
        weighted_hits = simple_mean_hits

    weighted_difference = (
        weighted_hits -
        RANDOM_EXPECTED
    )

    relative_improvement = (
        weighted_difference /
        RANDOM_EXPECTED *
        100
    )

    # --------------------------------------------------------
    # Fold consistency
    # --------------------------------------------------------

    above = (
        fold_results[
            "adaptive_difference"
        ] > 0
    ).sum()

    below = (
        fold_results[
            "adaptive_difference"
        ] < 0
    ).sum()

    print()
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
        f"{simple_difference:+.6f}"
    )

    print()
    print("Fold consistency:")
    print(f"Above random: {above}")
    print(f"Below random: {below}")

    print()
    print("Selected strategies:")

    counts = (
        fold_results[
            "selected_strategy"
        ].value_counts()
    )

    for strategy in STRATEGY_FILES.keys():

        print(
            f"  {strategy:<12}: "
            f"{counts.get(strategy, 0)}"
        )

    return {
        "weighted_hits": weighted_hits,
        "weighted_difference": weighted_difference,
        "relative_improvement": relative_improvement,
    }


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_test(
    fold_results,
):

    print_header("MONTE-CARLO NULL TEST")

    valid = fold_results[
        fold_results["test_draws"] > 0
    ].copy()

    if valid.empty:
        print("No valid folds.")
        return np.nan

    observed = (
        np.average(
            valid["adaptive_average_hits"],
            weights=valid["test_draws"],
        )
        - RANDOM_EXPECTED
    )

    rng = np.random.default_rng(42)

    simulated = np.empty(
        MC_SIMULATIONS
    )

    weights = valid[
        "test_draws"
    ].values.astype(float)

    weights /= weights.sum()

    for i in range(MC_SIMULATIONS):

        # Null model:
        # each simulated fold is random around
        # the theoretical expectation.
        simulated_hits = rng.binomial(
            valid["test_draws"].values,
            RANDOM_EXPECTED,
        ) / valid["test_draws"].values

        simulated_mean = np.sum(
            simulated_hits * weights
        )

        simulated[i] = (
            simulated_mean -
            RANDOM_EXPECTED
        )

    p_value = np.mean(
        simulated <= observed
    )

    low, high = np.percentile(
        simulated,
        [2.5, 97.5]
    )

    print(
        f"Observed weighted difference: "
        f"{observed:+.6f}"
    )

    print(
        f"Monte-Carlo p-value: "
        f"{p_value:.6f}"
    )

    print(
        f"Random 95% range: "
        f"[{low:+.6f}, {high:+.6f}]"
    )

    return p_value


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ADAPTIVE META-MODEL V5 TOP-6 "
        "WALK-FORWARD TEST"
    )
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
        f"{MIN_HISTORY_DRAWS}"
    )

    print(
        f"Shrinkage strength: "
        f"{SHRINKAGE_STRENGTH}"
    )

    loaded = load_strategy_results()

    if len(loaded) < 2:
        print()
        print(
            "ERROR: Need at least two "
            "strategy result files."
        )
        return

    performance, draws, hits = (
        build_matrices(loaded)
    )

    diversity, correlation = (
        calculate_information_diversity(
            performance
        )
    )

    print_header("STRATEGY CORRELATION")

    print(
        correlation.to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    fold_results, weight_results = (
        run_walk_forward(
            performance,
            draws,
            diversity,
        )
    )

    if fold_results.empty:
        print("No results.")
        return

    print_header("FOLD RESULTS")

    print(
        fold_results.to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    summary = final_evaluation(
        fold_results
    )

    p_value = monte_carlo_test(
        fold_results
    )

    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    print_header("FINAL CONCLUSION")

    if summary is None:
        print("Unable to evaluate.")
        return

    difference = summary[
        "weighted_difference"
    ]

    improvement = summary[
        "relative_improvement"
    ]

    print(
        f"Weighted average advantage: "
        f"{difference:+.6f}"
    )

    print(
        f"Relative improvement: "
        f"{improvement:+.3f}%"
    )

    print(
        f"Monte-Carlo p-value: "
        f"{p_value:.6f}"
    )

    if (
        difference > 0
        and p_value < 0.05
    ):

        print()
        print(
            "V5 shows evidence of outperforming "
            "the random baseline."
        )

    elif difference > 0:

        print()
        print(
            "V5 is above the random baseline, "
            "but the advantage is not statistically "
            "significant."
        )

    else:

        print()
        print(
            "V5 does not outperform "
            "the random baseline."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = fold_results.copy()

    output[
        "random_expected"
    ] = RANDOM_EXPECTED

    output[
        "monte_carlo_p_value"
    ] = p_value

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    weight_results.to_csv(
        WEIGHTS_OUTPUT_FILE,
        index=False,
    )

    print()
    print_header("RESULTS SAVED")

    print(
        OUTPUT_FILE
    )

    print(
        WEIGHTS_OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
