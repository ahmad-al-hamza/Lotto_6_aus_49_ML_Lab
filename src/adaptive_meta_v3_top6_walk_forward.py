"""
ADAPTIVE META-MODEL V3 TOP-6 WALK-FORWARD TEST

Purpose
-------
Combine multiple Top-6 strategies using stable historical performance.

Strategies:
    - meta_score
    - ensemble
    - recency
    - stability
    - diversity

V3 improvements:
    1. Walk-forward only
    2. Historical information only
    3. Recency-weighted historical performance
    4. Shrinkage toward zero
    5. Weight floor / ceiling
    6. Ignore unavailable strategies
    7. Soft weighting instead of hard strategy selection
    8. Monte-Carlo null test
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "processed"

RANDOM_EXPECTED = 6 * 6 / 49

STRATEGIES = [
    "meta_score",
    "ensemble",
    "recency",
    "stability",
    "diversity",
]

# Historical weighting
DECAY = 0.70

# Shrink historical performance toward zero
SHRINKAGE = 0.50

# Prevent one strategy from dominating
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.55

# Softmax temperature
TEMPERATURE = 0.025

# Monte Carlo
MC_SIMULATIONS = 10000

RANDOM_SEED = 42


# ============================================================
# FILES
# ============================================================

FILES = {
    "meta_score": RESULTS_DIR / "adaptive_meta_score_top6_walk_forward_results.csv",
    "ensemble": RESULTS_DIR / "ensemble_top6_walk_forward_results.csv",
    "recency": RESULTS_DIR / "recency_top6_walk_forward_results.csv",
    "stability": RESULTS_DIR / "stability_top6_walk_forward_results.csv",
    "diversity": RESULTS_DIR / "diversity_top6_walk_forward_results.csv",
}


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find the first existing column from a list of possible names.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_strategy(name, path):
    """
    Load one strategy result file and normalize columns.
    """

    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    fold_col = find_column(
        df,
        ["fold", "Fold", "fold_id"]
    )

    diff_col = find_column(
        df,
        [
            "difference",
            "adaptive_difference",
            "selected_difference",
            "mean_difference",
        ]
    )

    hits_col = find_column(
        df,
        [
            "average_hits",
            "adaptive_average_hits",
            "selected_average_hits",
            "mean_hits",
        ]
    )

    draws_col = find_column(
        df,
        [
            "test_draws",
            "draws",
            "testing_draws",
        ]
    )

    if fold_col is None:
        raise ValueError(
            f"{name}: Could not find fold column in {path}"
        )

    if diff_col is None and hits_col is None:
        raise ValueError(
            f"{name}: Could not find performance column in {path}"
        )

    result = pd.DataFrame()

    result["fold"] = pd.to_numeric(
        df[fold_col],
        errors="coerce"
    )

    if diff_col is not None:
        result["difference"] = pd.to_numeric(
            df[diff_col],
            errors="coerce"
        )

    else:
        hits = pd.to_numeric(
            df[hits_col],
            errors="coerce"
        )

        result["difference"] = (
            hits - RANDOM_EXPECTED
        )

    if hits_col is not None:
        result["average_hits"] = pd.to_numeric(
            df[hits_col],
            errors="coerce"
        )

    else:
        result["average_hits"] = (
            RANDOM_EXPECTED + result["difference"]
        )

    if draws_col is not None:
        result["test_draws"] = pd.to_numeric(
            df[draws_col],
            errors="coerce"
        )
    else:
        result["test_draws"] = np.nan

    result = result.dropna(subset=["fold"])

    result["fold"] = result["fold"].astype(int)

    result["strategy"] = name

    return result


# ============================================================
# LOAD ALL STRATEGIES
# ============================================================

def load_all():

    print("Loading strategy results...")

    frames = []

    for strategy in STRATEGIES:

        df = load_strategy(
            strategy,
            FILES[strategy]
        )

        if df.empty:
            continue

        print(
            f"Loaded {strategy:<11}: "
            f"{len(df)} fold results"
        )

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No strategy result files were loaded."
        )

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# PERFORMANCE MATRIX
# ============================================================

def create_matrix(all_results):

    matrix = all_results.pivot_table(
        index="fold",
        columns="strategy",
        values="difference",
        aggfunc="first"
    )

    matrix = matrix.reindex(
        columns=STRATEGIES
    )

    return matrix


def create_hits_matrix(all_results):

    matrix = all_results.pivot_table(
        index="fold",
        columns="strategy",
        values="average_hits",
        aggfunc="first"
    )

    return matrix.reindex(
        columns=STRATEGIES
    )


def create_draw_matrix(all_results):

    matrix = all_results.pivot_table(
        index="fold",
        columns="strategy",
        values="test_draws",
        aggfunc="first"
    )

    return matrix.reindex(
        columns=STRATEGIES
    )


# ============================================================
# V3 WEIGHT CALCULATION
# ============================================================

def calculate_weights(
    historical_values,
    available_strategies
):
    """
    Calculate stable V3 weights.

    historical_values:
        DataFrame indexed by fold and columns strategies.

    Only historical folds are used.
    """

    available = [
        s for s in STRATEGIES
        if s in available_strategies
    ]

    if not available:
        return {}

    scores = {}

    historical = historical_values[
        available
    ].copy()

    # --------------------------------------------------------
    # RECENCY WEIGHTING
    # --------------------------------------------------------

    valid_folds = list(historical.index)

    if not valid_folds:
        return {
            s: 1.0 / len(available)
            for s in available
        }

    weights_history = np.array([
        DECAY ** (
            len(valid_folds) - 1 - i
        )
        for i in range(len(valid_folds))
    ])

    weights_history = (
        weights_history /
        weights_history.sum()
    )

    # --------------------------------------------------------
    # RECENCY-WEIGHTED PERFORMANCE
    # --------------------------------------------------------

    for strategy in available:

        values = historical[
            strategy
        ].to_numpy(dtype=float)

        valid = np.isfinite(values)

        if not valid.any():
            scores[strategy] = 0.0
            continue

        vals = values[valid]

        w = weights_history[valid]

        w = w / w.sum()

        weighted_score = np.sum(
            vals * w
        )

        scores[strategy] = weighted_score

    scores_series = pd.Series(scores)

    # --------------------------------------------------------
    # SHRINKAGE
    # --------------------------------------------------------

    # Pull performance toward zero.
    scores_series = (
        scores_series *
        (1.0 - SHRINKAGE)
    )

    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    scaled = (
        scores_series /
        TEMPERATURE
    )

    scaled = scaled - scaled.max()

    exp_scores = np.exp(
        np.clip(scaled, -20, 20)
    )

    softmax_weights = (
        exp_scores /
        exp_scores.sum()
    )

    weights = pd.Series(
        softmax_weights,
        index=available
    )

    # --------------------------------------------------------
    # WEIGHT FLOOR
    # --------------------------------------------------------

    weights = weights.clip(
        lower=MIN_WEIGHT,
        upper=MAX_WEIGHT
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    weights = weights / weights.sum()

    return weights.to_dict()


# ============================================================
# ADAPTIVE SCORE
# ============================================================

def calculate_adaptive_difference(
    current_row,
    weights
):
    """
    Weighted current-fold performance.
    """

    weighted_values = []

    for strategy, weight in weights.items():

        value = current_row.get(
            strategy,
            np.nan
        )

        if pd.isna(value):
            continue

        weighted_values.append(
            weight * value
        )

    if not weighted_values:
        return np.nan

    return float(
        np.sum(weighted_values)
    )


# ============================================================
# SELECT BEST AVAILABLE STRATEGY
# ============================================================

def select_strategy(
    current_row,
    weights
):
    """
    Select highest-weight available strategy.
    """

    available = []

    for strategy, weight in weights.items():

        value = current_row.get(
            strategy,
            np.nan
        )

        if pd.notna(value):

            available.append(
                (
                    strategy,
                    weight
                )
            )

    if not available:
        return None

    available.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return available[0][0]


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_test(
    observed_difference,
    n_folds,
    seed=RANDOM_SEED
):
    """
    Null distribution for the mean difference.

    Uses the same random Top-6 theoretical expectation
    and simulates random Top-6 hit counts.
    """

    rng = np.random.default_rng(seed)

    if n_folds <= 0:
        return np.nan, np.array([])

    # Simulate 6 selected numbers against
    # 6 winning numbers from 49.

    # Each draw has hypergeometric distribution:
    #
    # population = 49
    # success = 6
    # draws = 6

    simulations = rng.hypergeometric(
        ngood=6,
        nbad=43,
        nsample=6,
        size=(
            MC_SIMULATIONS,
            n_folds
        )
    )

    simulated_mean_hits = (
        simulations.mean(axis=1)
    )

    simulated_difference = (
        simulated_mean_hits -
        RANDOM_EXPECTED
    )

    p_value = np.mean(
        simulated_difference <=
        observed_difference
    )

    return p_value, simulated_difference


# ============================================================
# MAIN WALK-FORWARD TEST
# ============================================================

def main():

    print("=" * 70)
    print(
        "ADAPTIVE META-MODEL V3 TOP-6 WALK-FORWARD TEST"
    )
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    all_results = load_all()

    performance = create_matrix(
        all_results
    )

    hits = create_hits_matrix(
        all_results
    )

    draws = create_draw_matrix(
        all_results
    )

    print()
    print("=" * 70)
    print("STRATEGY PERFORMANCE MATRIX")
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

    print()
    print("=" * 70)
    print("ADAPTIVE META-MODEL V3 WALK-FORWARD TEST")
    print("=" * 70)

    folds = sorted(
        performance.index
    )

    results = []

    for fold in folds:

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        # ----------------------------------------------------
        # Historical folds ONLY
        # ----------------------------------------------------

        historical = performance[
            performance.index < fold
        ]

        current = performance.loc[
            fold
        ]

        current_hits = (
            hits.loc[fold]
            if fold in hits.index
            else pd.Series(dtype=float)
        )

        current_draws = (
            draws.loc[fold]
            if fold in draws.index
            else pd.Series(dtype=float)
        )

        # ----------------------------------------------------
        # Available strategies in CURRENT fold
        # ----------------------------------------------------

        available = [
            s for s in STRATEGIES
            if (
                s in current.index and
                pd.notna(current[s])
            )
        ]

        print()
        print(
            "Available strategies: "
            + ", ".join(available)
        )

        # ----------------------------------------------------
        # Calculate historical weights
        # ----------------------------------------------------

        if historical.empty:

            weights = {
                s: 1.0 / len(available)
                for s in available
            }

        else:

            weights = calculate_weights(
                historical,
                available
            )

        print()
        print("V3 strategy weights:")

        for strategy in STRATEGIES:

            print(
                f"  {strategy:<11}: "
                f"{weights.get(strategy, 0.0):.4f}"
            )

        # ----------------------------------------------------
        # Adaptive weighted difference
        # ----------------------------------------------------

        adaptive_difference = (
            calculate_adaptive_difference(
                current,
                weights
            )
        )

        adaptive_hits = (
            RANDOM_EXPECTED +
            adaptive_difference
        )

        selected = select_strategy(
            current,
            weights
        )

        if selected is not None:

            selected_difference = float(
                current[selected]
            )

            selected_hits = (
                RANDOM_EXPECTED +
                selected_difference
            )

        else:

            selected_difference = np.nan
            selected_hits = np.nan

        # ----------------------------------------------------
        # Print current performance
        # ----------------------------------------------------

        print()
        print(
            f"Selected highest-weight strategy: "
            f"{selected}"
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

        for strategy in STRATEGIES:

            value = current.get(
                strategy,
                np.nan
            )

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
        # Test draws
        # ----------------------------------------------------

        valid_draws = [
            current_draws.get(
                s,
                np.nan
            )
            for s in available
            if pd.notna(
                current_draws.get(
                    s,
                    np.nan
                )
            )
        ]

        if valid_draws:

            # Use maximum available test size.
            test_draws = int(
                max(valid_draws)
            )

        else:

            test_draws = np.nan

        results.append(
            {
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
            }
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
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
    # WEIGHTED OVERALL PERFORMANCE
    # ========================================================

    valid = results_df[
        results_df[
            "adaptive_average_hits"
        ].notna()
    ].copy()

    valid_draws = valid[
        "test_draws"
    ].fillna(1)

    total_weight = (
        valid_draws.sum()
    )

    weighted_adaptive_hits = (
        (
            valid[
                "adaptive_average_hits"
            ] *
            valid_draws
        ).sum()
        /
        total_weight
    )

    weighted_difference = (
        weighted_adaptive_hits -
        RANDOM_EXPECTED
    )

    relative_improvement = (
        weighted_difference /
        RANDOM_EXPECTED
    ) * 100

    # ========================================================
    # SIMPLE MEAN
    # ========================================================

    simple_mean_hits = (
        valid[
            "adaptive_average_hits"
        ].mean()
    )

    simple_difference = (
        simple_mean_hits -
        RANDOM_EXPECTED
    )

    # ========================================================
    # CONSISTENCY
    # ========================================================

    above = int(
        (
            valid[
                "adaptive_difference"
            ] > 0
        ).sum()
    )

    below = int(
        (
            valid[
                "adaptive_difference"
            ] < 0
        ).sum()
    )

    # ========================================================
    # SELECTED STRATEGIES
    # ========================================================

    selected_counts = (
        valid[
            "selected_strategy"
        ]
        .value_counts()
        .to_dict()
    )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    mc_p, mc_distribution = (
        monte_carlo_test(
            weighted_difference,
            len(valid)
        )
    )

    if len(mc_distribution) > 0:

        lower = np.percentile(
            mc_distribution,
            2.5
        )

        upper = np.percentile(
            mc_distribution,
            97.5
        )

    else:

        lower = np.nan
        upper = np.nan

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("ADAPTIVE META-MODEL V3 SUMMARY")
    print("=" * 70)

    print(
        f"Weighted adaptive hits: "
        f"{weighted_adaptive_hits:.6f}"
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
            f"  {strategy:<11}: "
            f"{selected_counts.get(strategy, 0)}"
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
        f"{mc_p:.6f}"
    )

    print(
        f"Random 95% range: "
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
        and mc_p < 0.05
    ):

        print(
            "Adaptive Meta-Model V3 shows "
            "statistically significant improvement "
            "over the random baseline."
        )

    elif weighted_difference > 0:

        print(
            "Adaptive Meta-Model V3 is above "
            "the random baseline, but the improvement "
            "is not statistically significant."
        )

    else:

        print(
            "Adaptive Meta-Model V3 does not outperform "
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
        f"{mc_p:.6f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        RESULTS_DIR /
        "adaptive_meta_v3_top6_walk_forward_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED TO:")
    print("=" * 70)

    print(output_path)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()