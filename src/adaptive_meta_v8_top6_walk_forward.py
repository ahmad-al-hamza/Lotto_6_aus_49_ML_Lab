from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_EXPECTED_HITS = 6 / 49

MAX_STRATEGY_WEIGHT = 0.40
MIN_HISTORICAL_FOLDS = 2
MIN_HISTORICAL_DRAWS = 1500
MIN_CURRENT_DRAWS = 500
ROLLING_FOLDS = 3
STABILITY_PENALTY = 0.50
MONTE_CARLO_SIMULATIONS = 10000

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"


STRATEGY_FILES = {
    "meta_score": DATA_DIR / "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": DATA_DIR / "recency_top6_walk_forward_results.csv",
    "stability": DATA_DIR / "stability_top6_walk_forward_results.csv",
    "diversity": DATA_DIR / "diversity_top6_walk_forward_results.csv",
    "ensemble": DATA_DIR / "ensemble_top6_walk_forward_results.csv",
}


OUTPUT_RESULTS = (
    DATA_DIR / "adaptive_meta_v8_top6_walk_forward_results.csv"
)

OUTPUT_WEIGHTS = (
    DATA_DIR / "adaptive_meta_v8_weights.csv"
)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find the first existing column from candidates.
    """
    for column in candidates:
        if column in df.columns:
            return column

    return None


def load_strategy_file(name, path):
    """
    Load one strategy result file and normalize the columns.
    """

    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return None

    df = pd.read_csv(path)

    if df.empty:
        print(f"WARNING: Empty file: {path}")
        return None

    # --------------------------------------------------------
    # Fold column
    # --------------------------------------------------------

    fold_col = find_column(
        df,
        [
            "fold",
            "Fold",
            "FOLD",
        ],
    )

    if fold_col is None:
        raise ValueError(
            f"{name}: Could not find fold column."
        )

    # --------------------------------------------------------
    # Performance / difference column
    # --------------------------------------------------------

    diff_col = find_column(
        df,
        [
            "selected_difference",
            "adaptive_difference",
            "difference",
            "mean_difference",
            "performance_difference",
        ],
    )

    if diff_col is None:
        raise ValueError(
            f"{name}: Could not find performance difference column.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Test draws
    # --------------------------------------------------------

    draws_col = find_column(
        df,
        [
            "test_draws",
            "current_draws",
            "draws",
            "n_test",
            "test_size",
        ],
    )

    # --------------------------------------------------------
    # Hit rate
    # --------------------------------------------------------

    hits_col = find_column(
        df,
        [
            "selected_average_hits",
            "adaptive_average_hits",
            "average_hits",
            "hits",
        ],
    )

    result = pd.DataFrame()

    result["fold"] = pd.to_numeric(
        df[fold_col],
        errors="coerce",
    )

    result["difference"] = pd.to_numeric(
        df[diff_col],
        errors="coerce",
    )

    if draws_col is not None:
        result["test_draws"] = pd.to_numeric(
            df[draws_col],
            errors="coerce",
        )
    else:
        result["test_draws"] = np.nan

    if hits_col is not None:
        result["hits"] = pd.to_numeric(
            df[hits_col],
            errors="coerce",
        )
    else:
        result["hits"] = (
            RANDOM_EXPECTED_HITS + result["difference"]
        )

    result = result.dropna(
        subset=["fold", "difference"]
    )

    result["fold"] = result["fold"].astype(int)

    # --------------------------------------------------------
    # Remove duplicate folds
    # --------------------------------------------------------

    result = (
        result
        .sort_values("fold")
        .drop_duplicates(
            subset=["fold"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# LOAD ALL STRATEGIES
# ============================================================

def load_all_strategies():

    print("=" * 70)
    print("LOADING STRATEGY RESULTS")
    print("=" * 70)

    strategies = {}

    for name, path in STRATEGY_FILES.items():

        df = load_strategy_file(
            name,
            path,
        )

        if df is None:
            continue

        strategies[name] = df

        print(
            f"{name:<12}: "
            f"{len(df)} rows -> {path.name}"
        )

    if not strategies:
        raise RuntimeError(
            "No strategy result files were loaded."
        )

    return strategies


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_matrices(strategies):

    all_folds = sorted(
        set(
            fold
            for df in strategies.values()
            for fold in df["fold"].tolist()
        )
    )

    strategy_names = list(strategies.keys())

    difference_matrix = pd.DataFrame(
        index=all_folds,
        columns=strategy_names,
        dtype=float,
    )

    draws_matrix = pd.DataFrame(
        index=all_folds,
        columns=strategy_names,
        dtype=float,
    )

    hits_matrix = pd.DataFrame(
        index=all_folds,
        columns=strategy_names,
        dtype=float,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Construct each column exactly once.
    # This prevents duplicate-column bugs.
    # --------------------------------------------------------

    for strategy in strategy_names:

        df = strategies[strategy]

        for _, row in df.iterrows():

            fold = int(row["fold"])

            difference_matrix.loc[
                fold,
                strategy,
            ] = row["difference"]

            draws_matrix.loc[
                fold,
                strategy,
            ] = row["test_draws"]

            hits_matrix.loc[
                fold,
                strategy,
            ] = row["hits"]

    # --------------------------------------------------------
    # Force unique column names
    # --------------------------------------------------------

    difference_matrix = (
        difference_matrix.loc[
            :,
            ~difference_matrix.columns.duplicated(),
        ]
    )

    draws_matrix = (
        draws_matrix.loc[
            :,
            ~draws_matrix.columns.duplicated(),
        ]
    )

    hits_matrix = (
        hits_matrix.loc[
            :,
            ~hits_matrix.columns.duplicated(),
        ]
    )

    return (
        difference_matrix,
        draws_matrix,
        hits_matrix,
    )


# ============================================================
# INFORMATION DIVERSITY
# ============================================================

def calculate_information_diversity(
    performance_matrix,
):

    correlation = performance_matrix.corr(
        method="pearson",
        min_periods=2,
    )

    diversity = {}

    for strategy in performance_matrix.columns:

        correlations = correlation[strategy].drop(
            labels=[strategy],
            errors="ignore",
        )

        if correlations.empty:
            mean_abs_corr = 0.0
        else:
            mean_abs_corr = correlations.abs().mean()

        diversity[strategy] = (
            1.0 - mean_abs_corr
        )

    return pd.Series(diversity)


# ============================================================
# WEIGHT CALCULATION
# ============================================================

def calculate_weights(
    history,
    available_strategies,
):

    available = [
        s for s in available_strategies
        if s in history.columns
    ]

    if not available:
        return {}

    # --------------------------------------------------------
    # Historical performance
    # --------------------------------------------------------

    scores = {}

    for strategy in available:

        values = pd.to_numeric(
            history[strategy],
            errors="coerce",
        ).dropna()

        if len(values) < MIN_HISTORICAL_FOLDS:
            continue

        # Recent rolling window
        values = values.tail(
            ROLLING_FOLDS
        )

        mean_score = values.mean()

        # Penalize unstable strategies
        if len(values) >= 2:
            std_score = values.std(
                ddof=1
            )
        else:
            std_score = 0.0

        adjusted_score = (
            mean_score
            - STABILITY_PENALTY * std_score
        )

        scores[strategy] = adjusted_score

    # --------------------------------------------------------
    # If insufficient history:
    # equal weights
    # --------------------------------------------------------

    if not scores:

        equal_weight = (
            1.0 / len(available)
        )

        return {
            strategy: equal_weight
            for strategy in available
        }

    # --------------------------------------------------------
    # Convert scores to positive weights
    # --------------------------------------------------------

    score_series = pd.Series(scores)

    # Shift scores so all are non-negative
    minimum = score_series.min()

    shifted = (
        score_series - minimum
    )

    # Small epsilon prevents zero-only vectors
    shifted = shifted + 1e-9

    # --------------------------------------------------------
    # Apply maximum weight
    # --------------------------------------------------------

    raw = shifted / shifted.sum()

    weights = raw.copy()

    # Iterative capped normalization
    for _ in range(20):

        over = weights > MAX_STRATEGY_WEIGHT

        if not over.any():
            break

        excess = (
            weights[over]
            - MAX_STRATEGY_WEIGHT
        ).sum()

        weights[over] = MAX_STRATEGY_WEIGHT

        under = ~over

        if under.sum() == 0:
            break

        available_mass = weights[under].sum()

        if available_mass <= 0:
            break

        weights[under] += (
            weights[under]
            / available_mass
            * excess
        )

    # Final normalization
    total = weights.sum()

    if total > 0:
        weights = weights / total

    return weights.to_dict()


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_test(
    observed_difference,
    draws,
    simulations=MONTE_CARLO_SIMULATIONS,
):

    if draws <= 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(
        42
    )

    # Approximate null distribution for hit rate
    p = RANDOM_EXPECTED_HITS

    simulated_hits = rng.binomial(
        n=int(draws),
        p=p,
        size=simulations,
    ) / draws

    simulated_difference = (
        simulated_hits
        - RANDOM_EXPECTED_HITS
    )

    p_value = (
        np.mean(
            np.abs(simulated_difference)
            >= abs(observed_difference)
        )
    )

    lower = np.percentile(
        simulated_difference,
        2.5,
    )

    upper = np.percentile(
        simulated_difference,
        97.5,
    )

    return (
        p_value,
        lower,
        upper,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ADAPTIVE META-MODEL V8 TOP-6 WALK-FORWARD TEST"
    )
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        f"Maximum strategy weight: "
        f"{MAX_STRATEGY_WEIGHT:.2f}"
    )

    print(
        f"Minimum historical folds: "
        f"{MIN_HISTORICAL_FOLDS}"
    )

    print(
        f"Minimum historical draws: "
        f"{MIN_HISTORICAL_DRAWS}"
    )

    print(
        f"Minimum current draws: "
        f"{MIN_CURRENT_DRAWS}"
    )

    print(
        f"Rolling folds: "
        f"{ROLLING_FOLDS}"
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

    strategies = load_all_strategies()

    # --------------------------------------------------------
    # MATRICES
    # --------------------------------------------------------

    (
        performance_matrix,
        draws_matrix,
        hits_matrix,
    ) = build_matrices(
        strategies
    )

    print()
    print("=" * 70)
    print("PERFORMANCE DIFFERENCE MATRIX")
    print("=" * 70)
    print()

    print(
        performance_matrix.to_string(
            float_format=lambda x:
            f"{x:+.6f}"
        )
    )

    print()
    print("=" * 70)
    print("TEST DRAW MATRIX")
    print("=" * 70)
    print()

    print(
        draws_matrix.to_string()
    )

    # --------------------------------------------------------
    # INFORMATION DIVERSITY
    # --------------------------------------------------------

    information_diversity = (
        calculate_information_diversity(
            performance_matrix
        )
    )

    print()
    print("=" * 70)
    print("INFORMATION DIVERSITY")
    print("=" * 70)

    for strategy, value in (
        information_diversity
        .sort_values(
            ascending=False
        )
        .items()
    ):

        print(
            f"  {strategy:<12}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "V8 ROLLING WALK-FORWARD TEST"
    )
    print("=" * 70)

    fold_results = []
    weight_results = []

    folds = list(
        performance_matrix.index
    )

    for fold in folds:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        # ----------------------------------------------------
        # Historical folds only
        # ----------------------------------------------------

        historical_folds = [
            f for f in folds
            if f < fold
        ]

        # Rolling history
        if historical_folds:
            historical_folds = (
                historical_folds[
                    -ROLLING_FOLDS:
                ]
            )

        # ----------------------------------------------------
        # Current available strategies
        # ----------------------------------------------------

        available = []

        for strategy in performance_matrix.columns:

            current_value = (
                performance_matrix
                .loc[fold, strategy]
            )

            current_draws = (
                draws_matrix
                .loc[fold, strategy]
            )

            # ------------------------------------------------
            # CRITICAL FIX:
            # current_value is guaranteed scalar
            # because duplicate columns were removed.
            # ------------------------------------------------

            if pd.isna(current_value):
                continue

            if pd.isna(current_draws):
                continue

            if (
                current_draws
                < MIN_CURRENT_DRAWS
            ):
                continue

            available.append(
                strategy
            )

        print()
        print(
            "Available strategies: "
            + ", ".join(available)
        )

        if not available:
            print(
                "No strategies available."
            )
            continue

        # ----------------------------------------------------
        # Historical matrix
        # ----------------------------------------------------

        if historical_folds:

            history = (
                performance_matrix
                .loc[
                    historical_folds,
                    available,
                ]
            )

        else:

            history = pd.DataFrame(
                columns=available
            )

        # ----------------------------------------------------
        # Calculate weights
        # ----------------------------------------------------

        if len(historical_folds) < (
            MIN_HISTORICAL_FOLDS
        ):

            weights = {
                strategy:
                1.0 / len(available)
                for strategy in available
            }

        else:

            weights = calculate_weights(
                history,
                available,
            )

            # Make sure every available strategy
            # has a weight.
            for strategy in available:
                weights.setdefault(
                    strategy,
                    0.0,
                )

            # Normalize
            total = sum(
                weights.values()
            )

            if total <= 0:

                weights = {
                    strategy:
                    1.0 / len(available)
                    for strategy in available
                }

            else:

                weights = {
                    strategy:
                    value / total
                    for strategy, value
                    in weights.items()
                }

        # ----------------------------------------------------
        # Print weights
        # ----------------------------------------------------

        print()
        print("V8 strategy weights:")

        for strategy in performance_matrix.columns:

            weight = weights.get(
                strategy,
                0.0,
            )

            print(
                f"  {strategy:<12}: "
                f"{weight:.4f}"
            )

        # ----------------------------------------------------
        # Selected strategy
        # ----------------------------------------------------

        selected_strategy = max(
            weights,
            key=weights.get,
        )

        print()
        print(
            "Selected highest-weight strategy: "
            + selected_strategy.upper()
        )

        # ----------------------------------------------------
        # Current performance
        # ----------------------------------------------------

        current_performance = {}

        for strategy in available:

            value = (
                performance_matrix
                .loc[fold, strategy]
            )

            # value MUST be scalar
            value = float(value)

            current_performance[
                strategy
            ] = value

        # ----------------------------------------------------
        # Weighted difference
        # ----------------------------------------------------

        weighted_difference = sum(
            weights[strategy]
            * current_performance[strategy]
            for strategy in available
        )

        adaptive_hits = (
            RANDOM_EXPECTED_HITS
            + weighted_difference
        )

        selected_difference = (
            current_performance[
                selected_strategy
            ]
        )

        selected_hits = (
            RANDOM_EXPECTED_HITS
            + selected_difference
        )

        print()
        print(
            "Adaptive weighted difference: "
            f"{weighted_difference:+.6f}"
        )

        print(
            "Adaptive weighted hits: "
            f"{adaptive_hits:.6f}"
        )

        print()
        print(
            "Current fold performance:"
        )

        for strategy in performance_matrix.columns:

            value = (
                performance_matrix
                .loc[fold, strategy]
            )

            if pd.isna(value):
                text = "NaN"
            else:
                text = f"{float(value):+.6f}"

            print(
                f"  {strategy:<12}: "
                f"{text}"
            )

        # ----------------------------------------------------
        # Test draws
        # ----------------------------------------------------

        current_draw_values = []

        for strategy in available:

            value = (
                draws_matrix
                .loc[fold, strategy]
            )

            if pd.isna(value):
                continue

            current_draw_values.append(
                float(value)
            )

        if current_draw_values:
            test_draws = int(
                max(current_draw_values)
            )
        else:
            test_draws = 0

        # ----------------------------------------------------
        # Save fold result
        # ----------------------------------------------------

        fold_results.append(
            {
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
            }
        )

        # ----------------------------------------------------
        # Save weights
        # ----------------------------------------------------

        weight_row = {
            "fold": fold
        }

        for strategy in (
            performance_matrix.columns
        ):

            weight_row[
                strategy
            ] = weights.get(
                strategy,
                0.0,
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

        raise RuntimeError(
            "No fold results were generated."
        )

    print()
    print("=" * 70)
    print("FOLD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "adaptive_average_hits":
                    lambda x:
                    f"{x:+.6f}",
                "adaptive_difference":
                    lambda x:
                    f"{x:+.6f}",
                "selected_average_hits":
                    lambda x:
                    f"{x:+.6f}",
                "selected_difference":
                    lambda x:
                    f"{x:+.6f}",
            },
        )
    )

    # ========================================================
    # WEIGHTED FINAL EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("V8 FINAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # Weight by actual number of test draws.
    # --------------------------------------------------------

    total_draws = (
        results_df[
            "test_draws"
        ].sum()
    )

    if total_draws > 0:

        weighted_hits = (
            (
                results_df[
                    "adaptive_average_hits"
                ]
                * results_df[
                    "test_draws"
                ]
            ).sum()
            / total_draws
        )

    else:

        weighted_hits = (
            results_df[
                "adaptive_average_hits"
            ].mean()
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

    simple_mean_hits = (
        results_df[
            "adaptive_average_hits"
        ].mean()
    )

    simple_mean_difference = (
        simple_mean_hits
        - RANDOM_EXPECTED_HITS
    )

    print()
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
        f"{simple_mean_hits:.6f}"
    )

    print(
        f"Simple mean difference:  "
        f"{simple_mean_difference:+.6f}"
    )

    # ========================================================
    # FOLD CONSISTENCY
    # ========================================================

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

    print()
    print(
        "Fold consistency:"
    )

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
    print(
        "Selected strategies:"
    )

    selected_counts = (
        results_df[
            "selected_strategy"
        ]
        .value_counts()
    )

    for strategy in performance_matrix.columns:

        count = int(
            selected_counts.get(
                strategy,
                0,
            )
        )

        print(
            f"  {strategy:<12}: "
            f"{count}"
        )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    print()
    print("=" * 70)
    print("MONTE-CARLO NULL TEST")
    print("=" * 70)

    p_value, lower, upper = (
        monte_carlo_test(
            weighted_difference,
            int(total_draws),
        )
    )

    print()
    print(
        "Observed weighted difference: "
        f"{weighted_difference:+.6f}"
    )

    print(
        "Monte-Carlo p-value: "
        f"{p_value:.6f}"
    )

    print(
        "Random 95% null range: "
        f"[{lower:+.6f}, {upper:+.6f}]"
    )

    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        weighted_difference > 0
        and p_value < 0.05
    ):

        print(
            "V8 OUTPERFORMS THE RANDOM BASELINE "
            "WITH STATISTICAL SIGNIFICANCE."
        )

    elif weighted_difference > 0:

        print(
            "V8 IS ABOVE THE RANDOM BASELINE, "
            "BUT THE ADVANTAGE IS NOT "
            "STATISTICALLY SIGNIFICANT."
        )

    else:

        print(
            "V8 DOES NOT OUTPERFORM "
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
    # SAVE
    # ========================================================

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    weights_df.to_csv(
        OUTPUT_WEIGHTS,
        index=False,
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        OUTPUT_RESULTS
    )

    print(
        OUTPUT_WEIGHTS
    )

    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()