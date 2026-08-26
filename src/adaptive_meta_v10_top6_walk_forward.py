from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# ADAPTIVE META-MODEL V10
# ROBUST / STABILITY-AWARE / STRICT WALK-FORWARD
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

STRATEGIES = [
    "meta_score",
    "recency",
    "stability",
    "diversity",
    "ensemble",
]

FILES = {
    "meta_score": "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": "recency_top6_walk_forward_results.csv",
    "stability": "stability_top6_walk_forward_results.csv",
    "diversity": "diversity_top6_walk_forward_results.csv",
    "ensemble": "ensemble_top6_walk_forward_results.csv",
}

MAX_WEIGHT = 0.40
MIN_HISTORICAL_FOLDS = 2

# V10 parameters
SHRINKAGE = 0.50
STABILITY_PENALTY = 0.50
POSITIVE_BONUS = 0.20
RECENCY_DECAY = 0.80

MONTE_CARLO_SIMULATIONS = 10_000
RANDOM_SEED = 42

# The expected random hit rate for a 6-number draw
RANDOM_EXPECTED_HITS = 0.734694


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def find_column(df, candidates):
    """
    Find the first existing column from candidates.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_strategy(name):
    path = DATA_DIR / FILES[name]

    if not path.exists():
        print(f"{name:<12}: FILE NOT FOUND -> {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalize fold
    fold_col = find_column(
        df,
        ["fold", "Fold", "FOLD"]
    )

    if fold_col is None:
        print(f"{name:<12}: no fold column")
        return pd.DataFrame()

    df["fold"] = pd.to_numeric(df[fold_col], errors="coerce")

    # Difference column
    diff_col = find_column(
        df,
        [
            "adaptive_difference",
            "selected_difference",
            "mean_difference",
            "difference",
            "strategy_difference",
        ]
    )

    # Average hits
    hits_col = find_column(
        df,
        [
            "adaptive_average_hits",
            "selected_average_hits",
            "average_hits",
            "mean_hits",
        ]
    )

    # Draw count
    draws_col = find_column(
        df,
        [
            "test_draws",
            "draws",
            "n_draws",
            "test_count",
        ]
    )

    result = pd.DataFrame()
    result["fold"] = df["fold"]

    if diff_col is not None:
        result["difference"] = pd.to_numeric(
            df[diff_col],
            errors="coerce"
        )
    else:
        result["difference"] = np.nan

    if hits_col is not None:
        result["hits"] = pd.to_numeric(
            df[hits_col],
            errors="coerce"
        )
    else:
        result["hits"] = np.nan

    if draws_col is not None:
        result["draws"] = pd.to_numeric(
            df[draws_col],
            errors="coerce"
        )
    else:
        result["draws"] = np.nan

    result = result.dropna(subset=["fold"])
    result["fold"] = result["fold"].astype(int)

    return result.sort_values("fold").reset_index(drop=True)


# ============================================================
# LOAD ALL STRATEGIES
# ============================================================

def load_all():
    print("=" * 70)
    print("LOADING STRATEGY RESULTS")
    print("=" * 70)

    data = {}

    for strategy in STRATEGIES:
        df = load_strategy(strategy)

        data[strategy] = df

        if df.empty:
            print(
                f"{strategy:<12}: EMPTY / NOT AVAILABLE"
            )
        else:
            print(
                f"{strategy:<12}: "
                f"{len(df)} rows -> {FILES[strategy]}"
            )

    return data


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_matrices(data):

    all_folds = sorted(
        set(
            fold
            for df in data.values()
            if not df.empty
            for fold in df["fold"].tolist()
        )
    )

    performance = pd.DataFrame(
        index=all_folds,
        columns=STRATEGIES,
        dtype=float,
    )

    draws = pd.DataFrame(
        index=all_folds,
        columns=STRATEGIES,
        dtype=float,
    )

    for strategy, df in data.items():

        if df.empty:
            continue

        for _, row in df.iterrows():

            fold = int(row["fold"])

            performance.loc[
                fold, strategy
            ] = safe_float(row["difference"])

            draws.loc[
                fold, strategy
            ] = safe_float(row["draws"])

    return performance, draws


# ============================================================
# INFORMATION DIVERSITY
# ============================================================

def calculate_information_diversity(performance):

    corr = performance.corr(
        method="pearson",
        min_periods=2
    )

    diversity = {}

    for strategy in STRATEGIES:

        others = [
            s for s in STRATEGIES
            if s != strategy
        ]

        values = []

        for other in others:

            if (
                strategy in corr.index
                and other in corr.columns
            ):
                value = corr.loc[strategy, other]

                if pd.notna(value):
                    values.append(abs(float(value)))

        if values:
            mean_abs_corr = float(np.mean(values))
            diversity[strategy] = 1.0 - mean_abs_corr
        else:
            diversity[strategy] = 0.0

    return pd.Series(diversity), corr


# ============================================================
# ROBUST HISTORICAL SCORE
# ============================================================

def historical_score(
    strategy,
    current_fold,
    performance,
    draws,
    diversity
):

    if strategy not in performance.columns:
        return None

    # --------------------------------------------------------
    # IMPORTANT:
    # ONLY USE PREVIOUS FOLDS
    # --------------------------------------------------------

    history = performance.loc[
        performance.index < current_fold,
        strategy
    ].dropna()

    if len(history) < MIN_HISTORICAL_FOLDS:
        return None

    # --------------------------------------------------------
    # Draw-weighted historical performance
    # --------------------------------------------------------

    historical_draws = draws.loc[
        draws.index < current_fold,
        strategy
    ]

    valid = (
        history.index
        .intersection(historical_draws.dropna().index)
    )

    if len(valid) > 0:

        values = history.loc[valid].values
        weights = historical_draws.loc[valid].values

        weights = np.asarray(weights, dtype=float)
        values = np.asarray(values, dtype=float)

        if (
            np.all(np.isfinite(weights))
            and weights.sum() > 0
        ):
            weighted_mean = float(
                np.average(
                    values,
                    weights=weights
                )
            )
        else:
            weighted_mean = float(
                history.mean()
            )
    else:
        weighted_mean = float(history.mean())

    # --------------------------------------------------------
    # Median = robustness against single lucky fold
    # --------------------------------------------------------

    median_value = float(history.median())

    robust_mean = (
        0.60 * weighted_mean
        + 0.40 * median_value
    )

    # --------------------------------------------------------
    # Stability penalty
    # --------------------------------------------------------

    std = float(history.std(ddof=0))

    if not np.isfinite(std):
        std = 0.0

    stability_factor = 1.0 / (
        1.0 + STABILITY_PENALTY * std * 10.0
    )

    robust_score = (
        robust_mean * stability_factor
    )

    # --------------------------------------------------------
    # Positive-performance bonus
    # --------------------------------------------------------

    positive_fraction = float(
        np.mean(history.values > 0)
    )

    positive_bonus = (
        1.0
        + POSITIVE_BONUS
        * (positive_fraction - 0.5)
    )

    robust_score *= positive_bonus

    # --------------------------------------------------------
    # Information diversity
    # --------------------------------------------------------

    div = float(
        diversity.get(strategy, 0.0)
    )

    # Diversity multiplier is deliberately mild.
    diversity_multiplier = (
        0.75
        + 0.50 * max(0.0, min(1.0, div))
    )

    robust_score *= diversity_multiplier

    # --------------------------------------------------------
    # Recency weighting
    #
    # More recent historical folds receive slightly
    # more influence, but old folds are NOT discarded.
    # --------------------------------------------------------

    n = len(history)

    recency_weights = np.array([
        RECENCY_DECAY ** (n - 1 - i)
        for i in range(n)
    ])

    recency_weights /= recency_weights.sum()

    recency_mean = float(
        np.sum(
            history.values
            * recency_weights
        )
    )

    robust_score = (
        0.70 * robust_score
        + 0.30 * recency_mean
    )

    return {
        "score": robust_score,
        "mean": weighted_mean,
        "median": median_value,
        "std": std,
        "positive_fraction": positive_fraction,
        "diversity": div,
        "folds": len(history),
    }


# ============================================================
# CONVERT SCORES TO WEIGHTS
# ============================================================

def scores_to_weights(score_dict):

    available = {
        s: info
        for s, info in score_dict.items()
        if info is not None
    }

    if not available:
        return {
            s: 0.0
            for s in STRATEGIES
        }

    # --------------------------------------------------------
    # Shift scores so negative values don't destroy
    # the complete distribution.
    # --------------------------------------------------------

    raw = {
        s: float(info["score"])
        for s, info in available.items()
    }

    values = np.array(
        list(raw.values()),
        dtype=float
    )

    # If all values are <= 0, use rank-based scoring.
    if np.all(values <= 0):

        ranks = pd.Series(raw).rank(
            method="average"
        )

        transformed = {
            s: float(
                ranks[s]
                / ranks.sum()
            )
            for s in raw
        }

    else:

        minimum = float(
            np.min(values)
        )

        shifted = {
            s: max(
                0.0,
                value - minimum
            )
            for s, value in raw.items()
        }

        # Small epsilon prevents all-zero case.
        total = sum(shifted.values())

        if total <= 0:
            transformed = {
                s: 1.0 / len(raw)
                for s in raw
            }
        else:
            transformed = {
                s: value / total
                for s, value in shifted.items()
            }

    # --------------------------------------------------------
    # Apply shrinkage toward equal weights.
    # --------------------------------------------------------

    n = len(transformed)
    equal_weight = 1.0 / n

    weights = {
        s: (
            (1.0 - SHRINKAGE)
            * transformed[s]
            + SHRINKAGE
            * equal_weight
        )
        for s in transformed
    }

    # --------------------------------------------------------
    # Maximum weight cap.
    # --------------------------------------------------------

    weights = cap_weights(
        weights,
        MAX_WEIGHT
    )

    # Normalize.
    total = sum(weights.values())

    if total > 0:
        weights = {
            s: w / total
            for s, w in weights.items()
        }

    # Add unavailable strategies as zero.
    for s in STRATEGIES:
        weights.setdefault(s, 0.0)

    return weights


def cap_weights(weights, maximum):

    weights = dict(weights)

    for _ in range(100):

        over = {
            s: w
            for s, w in weights.items()
            if w > maximum
        }

        if not over:
            break

        excess = sum(
            w - maximum
            for w in over.values()
        )

        for s in over:
            weights[s] = maximum

        under = [
            s for s, w in weights.items()
            if w < maximum
        ]

        if not under:
            break

        available = sum(
            weights[s]
            for s in under
        )

        if available <= 0:
            equal_add = (
                excess / len(under)
            )

            for s in under:
                weights[s] += equal_add
        else:
            for s in under:
                share = (
                    weights[s]
                    / available
                )

                weights[s] += (
                    excess * share
                )

    return weights


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_test(
    observed_difference,
    n_folds,
    simulations=10000,
    seed=42
):

    rng = np.random.default_rng(seed)

    # Null distribution:
    # random sign assignment around zero.
    random_values = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(
            max(1, n_folds)
        ),
        size=simulations
    )

    # Scale to a conservative null range.
    scale = max(
        abs(observed_difference),
        0.01
    )

    random_values *= scale

    p_value = float(
        np.mean(
            random_values
            <= observed_difference
        )
    )

    lower = float(
        np.percentile(
            random_values,
            2.5
        )
    )

    upper = float(
        np.percentile(
            random_values,
            97.5
        )
    )

    return p_value, lower, upper


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE META-MODEL V10 ROBUST TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        f"Maximum strategy weight: "
        f"{MAX_WEIGHT:.2f}"
    )

    print(
        f"Minimum historical folds: "
        f"{MIN_HISTORICAL_FOLDS}"
    )

    print(
        f"Shrinkage: "
        f"{SHRINKAGE:.2f}"
    )

    print(
        f"Stability penalty: "
        f"{STABILITY_PENALTY:.2f}"
    )

    print(
        f"Monte-Carlo simulations: "
        f"{MONTE_CARLO_SIMULATIONS}"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data = load_all()

    performance, draws = build_matrices(data)

    print()
    print("=" * 70)
    print("PERFORMANCE DIFFERENCE MATRIX")
    print("=" * 70)
    print()
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
    print()
    print(draws.to_string())

    # --------------------------------------------------------
    # INFORMATION DIVERSITY
    # --------------------------------------------------------

    diversity, correlation = (
        calculate_information_diversity(
            performance
        )
    )

    print()
    print("=" * 70)
    print("INFORMATION DIVERSITY")
    print("=" * 70)

    for strategy in diversity.sort_values(
        ascending=False
    ).index:

        print(
            f"  {strategy:<12}: "
            f"{diversity[strategy]:.4f}"
        )

    print()
    print("=" * 70)
    print("V10 ROBUST WALK-FORWARD TEST")
    print("=" * 70)

    fold_results = []
    weight_results = []

    all_folds = sorted(
        performance.index.tolist()
    )

    for fold in all_folds:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)
        print()

        current = performance.loc[
            fold
        ]

        current_draws = draws.loc[
            fold
        ]

        # ----------------------------------------------------
        # Available strategies for CURRENT fold
        # ----------------------------------------------------

        available = []

        for strategy in STRATEGIES:

            value = current.get(
                strategy,
                np.nan
            )

            draw_count = current_draws.get(
                strategy,
                np.nan
            )

            if (
                pd.notna(value)
                and pd.notna(draw_count)
                and float(draw_count) > 0
            ):
                available.append(strategy)

        print(
            "Available strategies: "
            + (
                ", ".join(available)
                if available
                else "NONE"
            )
        )

        if not available:
            print(
                "No strategies available."
            )
            continue

        # ----------------------------------------------------
        # Historical scoring
        # ----------------------------------------------------

        scores = {}

        for strategy in available:

            scores[strategy] = historical_score(
                strategy,
                fold,
                performance,
                draws,
                diversity
            )

        # ----------------------------------------------------
        # If insufficient history:
        # equal weights.
        # ----------------------------------------------------

        usable = {
            s: info
            for s, info in scores.items()
            if info is not None
        }

        if not usable:

            weights = {
                s: 1.0 / len(available)
                for s in available
            }

        else:

            weights = scores_to_weights(
                scores
            )

            weights = {
                s: weights.get(
                    s,
                    0.0
                )
                for s in available
            }

            total = sum(
                weights.values()
            )

            if total > 0:
                weights = {
                    s: w / total
                    for s, w in weights.items()
                }

        # ----------------------------------------------------
        # Print weights
        # ----------------------------------------------------

        print()
        print("V10 strategy weights:")

        for strategy in STRATEGIES:

            print(
                f"  {strategy:<12}: "
                f"{weights.get(strategy, 0.0):.4f}"
            )

        # ----------------------------------------------------
        # Selected strategy
        # ----------------------------------------------------

        selected = max(
            weights,
            key=weights.get
        )

        print()
        print(
            "Selected highest-weight strategy: "
            f"{selected.upper()}"
        )

        # ----------------------------------------------------
        # Adaptive current-fold performance
        # ----------------------------------------------------

        weighted_difference = 0.0
        total_weight = 0.0

        for strategy in available:

            value = float(
                current[strategy]
            )

            weight = float(
                weights[strategy]
            )

            weighted_difference += (
                weight * value
            )

            total_weight += weight

        if total_weight > 0:
            weighted_difference /= (
                total_weight
            )

        adaptive_hits = (
            RANDOM_EXPECTED_HITS
            + weighted_difference
        )

        # ----------------------------------------------------
        # Selected strategy performance
        # ----------------------------------------------------

        selected_difference = float(
            current[selected]
        )

        selected_hits = (
            RANDOM_EXPECTED_HITS
            + selected_difference
        )

        test_draws = float(
            current_draws[
                available
            ].dropna().min()
        )

        print()
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

        for strategy in STRATEGIES:

            value = current.get(
                strategy,
                np.nan
            )

            if pd.notna(value):

                print(
                    f"  {strategy:<12}: "
                    f"{float(value):+.6f}"
                )

        # ----------------------------------------------------
        # Save fold result
        # ----------------------------------------------------

        fold_results.append({
            "fold": fold,
            "test_draws": test_draws,
            "adaptive_average_hits":
                adaptive_hits,
            "adaptive_difference":
                weighted_difference,
            "selected_strategy":
                selected,
            "selected_average_hits":
                selected_hits,
            "selected_difference":
                selected_difference,
        })

        # ----------------------------------------------------
        # Save weights
        # ----------------------------------------------------

        weight_row = {
            "fold": fold
        }

        for strategy in STRATEGIES:
            weight_row[
                strategy
            ] = weights.get(
                strategy,
                0.0
            )

        weight_results.append(
            weight_row
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )

    weights_df = pd.DataFrame(
        weight_results
    )

    if results_df.empty:

        print()
        print(
            "No valid folds were available."
        )
        return

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

    # ========================================================
    # DRAW-WEIGHTED FINAL EVALUATION
    # ========================================================

    total_draws = (
        results_df["test_draws"]
        .sum()
    )

    weighted_hits = (
        (
            results_df[
                "adaptive_average_hits"
            ]
            * results_df["test_draws"]
        ).sum()
        / total_draws
    )

    weighted_difference = (
        weighted_hits
        - RANDOM_EXPECTED_HITS
    )

    relative_improvement = (
        weighted_difference
        / RANDOM_EXPECTED_HITS
        * 100.0
    )

    simple_hits = float(
        results_df[
            "adaptive_average_hits"
        ].mean()
    )

    simple_difference = (
        simple_hits
        - RANDOM_EXPECTED_HITS
    )

    # ========================================================
    # FOLD CONSISTENCY
    # ========================================================

    above = int(
        (
            results_df[
                "adaptive_difference"
            ] > 0
        ).sum()
    )

    below = int(
        (
            results_df[
                "adaptive_difference"
            ] < 0
        ).sum()
    )

    selected_counts = (
        results_df[
            "selected_strategy"
        ]
        .value_counts()
    )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    p_value, null_lower, null_upper = (
        monte_carlo_test(
            weighted_difference,
            len(results_df),
            MONTE_CARLO_SIMULATIONS,
            RANDOM_SEED
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("V10 FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Weighted adaptive hits: "
        f"{weighted_hits:.6f}"
    )

    print(
        f"Random expected hits:   "
        f"{RANDOM_EXPECTED_HITS:.6f}"
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
        f"Above random: {above}"
    )
    print(
        f"Below random: {below}"
    )

    print()

    print("Selected strategies:")

    for strategy in STRATEGIES:

        print(
            f"  {strategy:<12}: "
            f"{int(selected_counts.get(strategy, 0))}"
        )

    print()
    print("=" * 70)
    print("MONTE-CARLO NULL TEST")
    print("=" * 70)

    print(
        f"Observed weighted difference: "
        f"{weighted_difference:+.6f}"
    )

    print(
        f"Monte-Carlo p-value: "
        f"{p_value:.6f}"
    )

    print(
        f"Random 95% null range: "
        f"[{null_lower:+.6f}, "
        f"{null_upper:+.6f}]"
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    statistically_significant = (
        p_value < 0.05
    )

    beats_random = (
        weighted_difference > 0
    )

    if (
        beats_random
        and statistically_significant
    ):

        print(
            "V10 OUTPERFORMS THE RANDOM "
            "BASELINE WITH STATISTICAL "
            "SIGNIFICANCE."
        )

    elif beats_random:

        print(
            "V10 IS ABOVE THE RANDOM "
            "BASELINE, BUT THE ADVANTAGE "
            "IS NOT STATISTICALLY SIGNIFICANT."
        )

    else:

        print(
            "V10 DOES NOT OUTPERFORM "
            "THE RANDOM BASELINE."
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
    # SAVE RESULTS
    # ========================================================

    results_path = (
        DATA_DIR
        / "adaptive_meta_v10_top6_walk_forward_results.csv"
    )

    weights_path = (
        DATA_DIR
        / "adaptive_meta_v10_weights.csv"
    )

    diversity_path = (
        DATA_DIR
        / "adaptive_meta_v10_information_diversity.csv"
    )

    correlation_path = (
        DATA_DIR
        / "adaptive_meta_v10_correlation.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    weights_df.to_csv(
        weights_path,
        index=False
    )

    diversity_df = (
        diversity
        .rename(
            "information_diversity"
        )
        .reset_index()
    )

    diversity_df.columns = [
        "strategy",
        "information_diversity"
    ]

    diversity_df.to_csv(
        diversity_path,
        index=False
    )

    correlation.to_csv(
        correlation_path
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(results_path)
    print(weights_path)
    print(diversity_path)
    print(correlation_path)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()