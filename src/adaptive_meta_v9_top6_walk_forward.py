"""
ADAPTIVE META-MODEL V9 TOP-6 WALK-FORWARD TEST

V9 goals:
- Fix V8 test_draws handling.
- Never discard a strategy merely because test_draws is missing
  when its performance difference is valid.
- Strict walk-forward weighting: fold t uses folds < t only.
- Maximum strategy weight.
- Shrinkage toward equal weights.
- Minimum historical folds.
- Monte-Carlo null test.
- Compare adaptive model against:
    1. random baseline
    2. simple mean of available strategies
    3. best available strategy

This is intended as the FINAL diagnostic version of the
adaptive-meta-model family.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

RANDOM_EXPECTED_HITS = 0.734694

MAX_STRATEGY_WEIGHT = 0.40
MIN_HISTORICAL_FOLDS = 2

SHRINKAGE = 0.50

MONTE_CARLO_SIMULATIONS = 10000

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

def clean_number(value):
    """
    Safely convert a value to float.

    Handles:
    - NaN
    - strings
    - pandas/numpy scalar values
    """
    try:
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return np.nan
            value = value.iloc[0]

        if isinstance(value, pd.DataFrame):
            if value.empty:
                return np.nan
            value = value.iloc[0, 0]

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def find_column(df, candidates):
    """
    Find the first matching column from candidates.
    """
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def extract_difference_column(df):
    """
    Detect the performance-difference column.
    """

    candidates = [
        "adaptive_difference",
        "selected_difference",
        "mean_difference",
        "difference",
        "performance_difference",
        "improvement",
    ]

    column = find_column(df, candidates)

    if column is not None:
        return column

    raise ValueError(
        f"Could not find performance difference column.\n"
        f"Available columns: {list(df.columns)}"
    )


def extract_fold_column(df):
    candidates = [
        "fold",
        "Fold",
        "fold_id",
    ]

    column = find_column(df, candidates)

    if column is None:
        raise ValueError(
            f"Could not find fold column.\n"
            f"Available columns: {list(df.columns)}"
        )

    return column


def extract_draw_column(df):
    """
    test_draws is optional.

    V9 deliberately does NOT require it for a strategy
    to be considered valid.
    """

    candidates = [
        "test_draws",
        "test_draw",
        "draws",
        "n_test",
        "test_count",
    ]

    return find_column(df, candidates)


# ============================================================
# LOAD STRATEGIES
# ============================================================

def load_strategy_results():

    results = {}

    print("=" * 70)
    print("LOADING STRATEGY RESULTS")
    print("=" * 70)

    for strategy, filename in STRATEGY_FILES.items():

        path = DATA_DIR / filename

        if not path.exists():
            print(f"{strategy:12s}: FILE NOT FOUND -> {filename}")
            continue

        df = pd.read_csv(path)

        fold_col = extract_fold_column(df)
        diff_col = extract_difference_column(df)
        draw_col = extract_draw_column(df)

        df = df.copy()

        df["_fold"] = pd.to_numeric(
            df[fold_col],
            errors="coerce"
        )

        df["_difference"] = pd.to_numeric(
            df[diff_col],
            errors="coerce"
        )

        if draw_col is not None:

            df["_test_draws"] = pd.to_numeric(
                df[draw_col],
                errors="coerce"
            )

        else:

            df["_test_draws"] = np.nan

        df = df.dropna(subset=["_fold"])

        df["_fold"] = df["_fold"].astype(int)

        results[strategy] = df

        print(
            f"{strategy:12s}: "
            f"{len(df)} rows -> {filename}"
        )

    return results


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_performance_matrix(results):

    folds = sorted(
        set(
            fold
            for df in results.values()
            for fold in df["_fold"].tolist()
        )
    )

    matrix = pd.DataFrame(
        index=folds,
        columns=STRATEGY_FILES.keys(),
        dtype=float,
    )

    draws = pd.DataFrame(
        index=folds,
        columns=STRATEGY_FILES.keys(),
        dtype=float,
    )

    for strategy, df in results.items():

        for _, row in df.iterrows():

            fold = int(row["_fold"])

            difference = clean_number(
                row["_difference"]
            )

            test_draws = clean_number(
                row["_test_draws"]
            )

            matrix.loc[fold, strategy] = difference

            if np.isfinite(test_draws):
                draws.loc[fold, strategy] = test_draws

    return matrix, draws


# ============================================================
# VALID STRATEGIES
# ============================================================

def available_strategies(
    performance_matrix,
    fold
):
    """
    A strategy is available when its performance is valid.

    IMPORTANT:
    test_draws is NOT required here.

    This fixes the V8 problem.
    """

    available = []

    for strategy in performance_matrix.columns:

        value = performance_matrix.loc[
            fold,
            strategy
        ]

        if np.isfinite(value):
            available.append(strategy)

    return available


# ============================================================
# HISTORICAL PERFORMANCE
# ============================================================

def historical_mean(
    performance_matrix,
    strategy,
    current_fold
):

    historical = performance_matrix.loc[
        performance_matrix.index < current_fold,
        strategy
    ]

    historical = historical.dropna()

    if len(historical) < MIN_HISTORICAL_FOLDS:
        return np.nan

    return float(historical.mean())


def historical_std(
    performance_matrix,
    strategy,
    current_fold
):

    historical = performance_matrix.loc[
        performance_matrix.index < current_fold,
        strategy
    ]

    historical = historical.dropna()

    if len(historical) < 2:
        return 0.0

    return float(historical.std(ddof=1))


# ============================================================
# V9 WEIGHTS
# ============================================================

def calculate_weights(
    performance_matrix,
    current_fold,
    available
):

    n = len(available)

    if n == 0:
        return {}

    if n == 1:
        return {available[0]: 1.0}

    raw_scores = {}

    for strategy in available:

        mean_perf = historical_mean(
            performance_matrix,
            strategy,
            current_fold
        )

        if not np.isfinite(mean_perf):
            mean_perf = 0.0

        std_perf = historical_std(
            performance_matrix,
            strategy,
            current_fold
        )

        # ----------------------------------------------------
        # Stability adjustment
        # ----------------------------------------------------

        stability_factor = 1.0 / (
            1.0 + max(std_perf, 0.0)
        )

        score = (
            mean_perf *
            stability_factor
        )

        raw_scores[strategy] = score

    # --------------------------------------------------------
    # Convert negative/positive scores into positive weights.
    #
    # Shift by minimum so negative historical performance
    # does not create negative probabilities.
    # --------------------------------------------------------

    values = np.array(
        list(raw_scores.values()),
        dtype=float
    )

    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    minimum = values.min()

    shifted = values - minimum

    # Small positive floor.
    shifted = shifted + 1e-6

    total = shifted.sum()

    if total <= 0:
        weights = np.ones(n) / n

    else:
        weights = shifted / total

    weights = dict(
        zip(available, weights)
    )

    # --------------------------------------------------------
    # Shrink toward equal weights
    # --------------------------------------------------------

    equal_weight = 1.0 / n

    for strategy in available:

        weights[strategy] = (
            SHRINKAGE * equal_weight
            +
            (1.0 - SHRINKAGE)
            * weights[strategy]
        )

    # --------------------------------------------------------
    # Apply maximum weight constraint.
    # --------------------------------------------------------

    weights = cap_weights(
        weights,
        MAX_STRATEGY_WEIGHT
    )

    return weights


def cap_weights(weights, max_weight):

    if not weights:
        return {}

    weights = {
        k: float(v)
        for k, v in weights.items()
    }

    # Iteratively cap large weights and redistribute excess.
    for _ in range(100):

        over = {
            k: v
            for k, v in weights.items()
            if v > max_weight
        }

        if not over:
            break

        excess = sum(
            v - max_weight
            for v in over.values()
        )

        for k in over:
            weights[k] = max_weight

        under = [
            k
            for k, v in weights.items()
            if v < max_weight - 1e-12
        ]

        if not under:
            break

        under_total = sum(
            weights[k]
            for k in under
        )

        if under_total <= 0:

            add = excess / len(under)

            for k in under:
                weights[k] += add

        else:

            for k in under:

                share = (
                    weights[k]
                    / under_total
                )

                weights[k] += (
                    excess * share
                )

    # Final normalization
    total = sum(weights.values())

    if total > 0:

        weights = {
            k: v / total
            for k, v in weights.items()
        }

    return weights


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_test(
    observed_difference,
    fold_results,
    simulations=10000
):

    rng = np.random.default_rng(42)

    observed = float(
        observed_difference
    )

    simulated = []

    for _ in range(simulations):

        null_differences = []

        for result in fold_results:

            draws = result["test_draws"]

            if draws is None:
                continue

            draws = int(draws)

            if draws <= 0:
                continue

            # Random baseline noise around zero.
            random_hits = rng.binomial(
                draws,
                RANDOM_EXPECTED_HITS
            ) / draws

            null_differences.append(
                random_hits
                - RANDOM_EXPECTED_HITS
            )

        if null_differences:

            simulated.append(
                np.mean(null_differences)
            )

    simulated = np.array(simulated)

    if len(simulated) == 0:
        return np.nan, np.nan, np.nan

    p_value = np.mean(
        simulated <= observed
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
        float(p_value),
        float(lower),
        float(upper)
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE META-MODEL V9 TOP-6 WALK-FORWARD TEST")
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
        f"Shrinkage: "
        f"{SHRINKAGE:.2f}"
    )

    print(
        f"Monte-Carlo simulations: "
        f"{MONTE_CARLO_SIMULATIONS}"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    results = load_strategy_results()

    if not results:
        raise RuntimeError(
            "No strategy result files found."
        )

    # --------------------------------------------------------
    # Matrices
    # --------------------------------------------------------

    performance_matrix, draw_matrix = (
        build_performance_matrix(results)
    )

    print()
    print("=" * 70)
    print("PERFORMANCE DIFFERENCE MATRIX")
    print("=" * 70)

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

    print(
        draw_matrix.to_string()
    )

    print()
    print("=" * 70)
    print("V9 WALK-FORWARD TEST")
    print("=" * 70)

    fold_results = []

    for fold in performance_matrix.index:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        available = available_strategies(
            performance_matrix,
            fold
        )

        print()
        print(
            "Available strategies: "
            + (
                ", ".join(available)
                if available
                else "NONE"
            )
        )

        if not available:

            print("No strategies available.")
            continue

        # ----------------------------------------------------
        # Weights
        # ----------------------------------------------------

        weights = calculate_weights(
            performance_matrix,
            fold,
            available
        )

        print()
        print("V9 strategy weights:")

        for strategy in performance_matrix.columns:

            weight = weights.get(
                strategy,
                0.0
            )

            print(
                f"  {strategy:12s}: "
                f"{weight:.4f}"
            )

        # ----------------------------------------------------
        # Current fold performance
        # ----------------------------------------------------

        current = {}

        for strategy in performance_matrix.columns:

            value = clean_number(
                performance_matrix.loc[
                    fold,
                    strategy
                ]
            )

            current[strategy] = value

        # ----------------------------------------------------
        # Adaptive weighted difference
        # ----------------------------------------------------

        adaptive_difference = 0.0
        weight_sum = 0.0

        for strategy, weight in weights.items():

            value = current.get(
                strategy,
                np.nan
            )

            if not np.isfinite(value):
                continue

            adaptive_difference += (
                weight * value
            )

            weight_sum += weight

        if weight_sum > 0:

            adaptive_difference /= (
                weight_sum
            )

        else:

            adaptive_difference = np.nan

        adaptive_hits = (
            RANDOM_EXPECTED_HITS
            + adaptive_difference
        )

        # ----------------------------------------------------
        # Selected strategy
        # ----------------------------------------------------

        selected_strategy = max(
            weights,
            key=weights.get
        )

        selected_difference = current[
            selected_strategy
        ]

        selected_hits = (
            RANDOM_EXPECTED_HITS
            + selected_difference
        )

        # ----------------------------------------------------
        # Number of test draws
        #
        # IMPORTANT:
        # Use the largest valid draw count among the
        # strategies participating in this fold.
        #
        # This prevents missing test_draws from destroying
        # otherwise valid fold results.
        # ----------------------------------------------------

        fold_draws = []

        for strategy in available:

            draws = clean_number(
                draw_matrix.loc[
                    fold,
                    strategy
                ]
            )

            if np.isfinite(draws) and draws > 0:
                fold_draws.append(int(draws))

        if fold_draws:

            test_draws = max(
                fold_draws
            )

        else:

            test_draws = None

        print()
        print(
            "Selected highest-weight strategy: "
            f"{selected_strategy.upper()}"
        )

        print()
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

        for strategy in performance_matrix.columns:

            value = current[strategy]

            if np.isfinite(value):

                print(
                    f"  {strategy:12s}: "
                    f"{value:+.6f}"
                )

            else:

                print(
                    f"  {strategy:12s}: NaN"
                )

        fold_results.append(
            {
                "fold": fold,
                "test_draws": test_draws,
                "adaptive_average_hits":
                    adaptive_hits,
                "adaptive_difference":
                    adaptive_difference,
                "selected_strategy":
                    selected_strategy,
                "selected_average_hits":
                    selected_hits,
                "selected_difference":
                    selected_difference,
                "weights": weights,
            }
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    if not fold_results:
        raise RuntimeError(
            "No valid folds were produced."
        )

    results_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in row.items()
                if k != "weights"
            }
            for row in fold_results
        ]
    )

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
    # FINAL EVALUATION
    # ========================================================

    adaptive_values = (
        results_df[
            "adaptive_average_hits"
        ]
        .dropna()
    )

    adaptive_differences = (
        results_df[
            "adaptive_difference"
        ]
        .dropna()
    )

    weighted_draws = (
        results_df[
            "test_draws"
        ]
        .fillna(0)
    )

    if (
        weighted_draws.sum() > 0
        and len(adaptive_differences)
        == len(weighted_draws)
    ):

        weighted_difference = np.average(
            adaptive_differences,
            weights=weighted_draws
        )

        weighted_hits = (
            RANDOM_EXPECTED_HITS
            + weighted_difference
        )

    else:

        weighted_hits = float(
            adaptive_values.mean()
        )

        weighted_difference = (
            weighted_hits
            - RANDOM_EXPECTED_HITS
        )

    simple_hits = float(
        adaptive_values.mean()
    )

    simple_difference = (
        simple_hits
        - RANDOM_EXPECTED_HITS
    )

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    above_random = int(
        (adaptive_differences > 0).sum()
    )

    below_random = int(
        (adaptive_differences < 0).sum()
    )

    # --------------------------------------------------------
    # Selected strategies
    # --------------------------------------------------------

    selected_counts = (
        results_df[
            "selected_strategy"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    observed_difference = (
        weighted_difference
    )

    p_value, mc_lower, mc_upper = (
        monte_carlo_test(
            observed_difference,
            fold_results,
            MONTE_CARLO_SIMULATIONS
        )
    )

    relative_improvement = (
        weighted_difference
        / RANDOM_EXPECTED_HITS
        * 100.0
    )

    print()
    print("=" * 70)
    print("V9 FINAL EVALUATION")
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
        f"Above random: {above_random}"
    )

    print(
        f"Below random: {below_random}"
    )

    print()
    print("Selected strategies:")

    for strategy in STRATEGY_FILES:

        count = int(
            selected_counts.get(
                strategy,
                0
            )
        )

        print(
            f"  {strategy:12s}: "
            f"{count}"
        )

    print()
    print("=" * 70)
    print("MONTE-CARLO NULL TEST")
    print("=" * 70)

    print(
        f"Observed weighted difference: "
        f"{observed_difference:+.6f}"
    )

    if np.isfinite(p_value):

        print(
            f"Monte-Carlo p-value: "
            f"{p_value:.6f}"
        )

        print(
            f"Random 95% null range: "
            f"[{mc_lower:+.6f}, "
            f"{mc_upper:+.6f}]"
        )

    else:

        print(
            "Monte-Carlo p-value: unavailable"
        )

    print()
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    statistically_supported = (
        weighted_difference > 0
        and np.isfinite(p_value)
        and p_value < 0.05
    )

    if statistically_supported:

        print(
            "V9 OUTPERFORMS THE RANDOM BASELINE "
            "WITH STATISTICAL SUPPORT."
        )

    elif weighted_difference > 0:

        print(
            "V9 IS ABOVE THE RANDOM BASELINE, "
            "BUT THE ADVANTAGE IS NOT "
            "STATISTICALLY SIGNIFICANT."
        )

    else:

        print(
            "V9 DOES NOT OUTPERFORM "
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

    if np.isfinite(p_value):

        print(
            f"Monte-Carlo p-value: "
            f"{p_value:.6f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_results = (
        DATA_DIR
        / "adaptive_meta_v9_top6_walk_forward_results.csv"
    )

    output_weights = (
        DATA_DIR
        / "adaptive_meta_v9_weights.csv"
    )

    results_df.to_csv(
        output_results,
        index=False
    )

    weight_rows = []

    for row in fold_results:

        for strategy, weight in row[
            "weights"
        ].items():

            weight_rows.append(
                {
                    "fold": row["fold"],
                    "strategy": strategy,
                    "weight": weight,
                }
            )

    weights_df = pd.DataFrame(
        weight_rows
    )

    weights_df.to_csv(
        output_weights,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(output_results)
    print(output_weights)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
