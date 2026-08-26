"""
ADAPTIVE META-MODEL V7
STRICT WALK-FORWARD TOP-6 TEST

Main goal:
    Correct the evaluation methodology used in V6.

Important rules:
    1. Never invent missing test_draws.
    2. Never replace NaN performance with zero.
    3. A strategy is available only when its actual fold result exists.
    4. Historical weights use ONLY previous folds.
    5. Current-fold performance is never used to calculate current weights.
    6. Overall metrics are calculated from valid fold observations only.
    7. Fold 5 is not artificially expanded to 1008 draws.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_EXPECTED = 0.734694

STRATEGY_FILES = {
    "meta_score": "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": "recency_top6_walk_forward_results.csv",
    "stability": "stability_top6_walk_forward_results.csv",
    "diversity": "diversity_top6_walk_forward_results.csv",
    "ensemble": "ensemble_top6_walk_forward_results.csv",
}

MAX_WEIGHT = 0.40
MIN_HISTORY_FOLDS = 1
SHRINKAGE = 0.50

N_MONTE_CARLO = 10000
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def find_column(df, candidates):
    """
    Find the first existing column from candidates.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_strategy(name, filename):
    """
    Load one strategy result file.

    We intentionally keep NaN values.
    They represent unavailable observations and must not
    be converted to zero.
    """

    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"{name:12s}: FILE NOT FOUND -> {path}")
        return None

    df = pd.read_csv(path)

    if "fold" not in df.columns:
        raise ValueError(
            f"{filename} does not contain a 'fold' column."
        )

    # Performance difference column
    diff_col = find_column(
        df,
        [
            "difference",
            "adaptive_difference",
            "selected_difference",
            "mean_difference",
        ],
    )

    if diff_col is None:
        raise ValueError(
            f"Could not find performance difference column in {filename}"
        )

    # Test draws
    draws_col = find_column(
        df,
        [
            "test_draws",
            "testing_draws",
            "draws",
        ],
    )

    result = pd.DataFrame()
    result["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    result["difference"] = pd.to_numeric(
        df[diff_col],
        errors="coerce"
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

    result = result.sort_values("fold")
    result = result.drop_duplicates("fold", keep="last")

    print(
        f"{name:12s}: "
        f"{len(result)} rows -> {filename}"
    )

    return result


# ============================================================
# LOAD ALL STRATEGIES
# ============================================================

print_header("ADAPTIVE META-MODEL V7 STRICT TOP-6 WALK-FORWARD TEST")

print(f"Random expected hits: {RANDOM_EXPECTED:.6f}")
print(f"Maximum strategy weight: {MAX_WEIGHT:.2f}")
print(f"Minimum historical folds: {MIN_HISTORY_FOLDS}")
print(f"Shrinkage: {SHRINKAGE:.2f}")
print(f"Monte-Carlo simulations: {N_MONTE_CARLO}")
print()

print_header("LOADING STRATEGY RESULTS")

strategies = {}

for name, filename in STRATEGY_FILES.items():
    result = load_strategy(name, filename)

    if result is not None:
        strategies[name] = result


if not strategies:
    raise RuntimeError("No strategy result files were found.")


STRATEGY_NAMES = list(strategies.keys())


# ============================================================
# BUILD STRICT PERFORMANCE MATRIX
# ============================================================

print_header("STRICT PERFORMANCE MATRIX")

all_folds = sorted(
    set(
        fold
        for df in strategies.values()
        for fold in df["fold"].tolist()
    )
)

performance = pd.DataFrame(
    index=all_folds,
    columns=STRATEGY_NAMES,
    dtype=float,
)

test_draws = pd.DataFrame(
    index=all_folds,
    columns=STRATEGY_NAMES,
    dtype=float,
)

for strategy, df in strategies.items():

    for _, row in df.iterrows():

        fold = int(row["fold"])

        performance.loc[fold, strategy] = row["difference"]

        if pd.notna(row["test_draws"]):
            test_draws.loc[fold, strategy] = row["test_draws"]


print()
print(performance.to_string(float_format=lambda x: f"{x:+.6f}"))

print()
print("STRICT TEST DRAW MATRIX")
print(test_draws.to_string())


# ============================================================
# HISTORICAL WEIGHT CALCULATION
# ============================================================

def calculate_weights(history):
    """
    Calculate weights using ONLY historical folds.

    Steps:
        1. Ignore unavailable strategies.
        2. Calculate historical mean performance.
        3. Shrink historical estimates toward zero.
        4. Convert positive evidence into weights.
        5. Apply maximum weight.
        6. Renormalize.
    """

    means = history.mean(skipna=True)

    counts = history.notna().sum()

    valid = means.index[counts >= MIN_HISTORY_FOLDS]

    if len(valid) == 0:
        return pd.Series(
            1.0 / len(history.columns),
            index=history.columns,
        )

    scores = pd.Series(
        0.0,
        index=history.columns,
        dtype=float,
    )

    for strategy in valid:

        value = means[strategy]

        if pd.isna(value):
            continue

        n = counts[strategy]

        # Shrink small historical samples toward zero.
        shrink_factor = (
            n / (n + SHRINKAGE)
        )

        shrunk = value * (
            SHRINKAGE * 0 + shrink_factor
        )

        # Only positive historical evidence contributes
        # to adaptive selection.
        scores[strategy] = max(
            shrunk,
            0.0
        )

    # If nobody has positive historical evidence,
    # use equal weights among available strategies.
    if scores.sum() <= 0:

        available = [
            s for s in history.columns
            if history[s].notna().any()
        ]

        if not available:
            return pd.Series(
                0.0,
                index=history.columns,
            )

        weights = pd.Series(
            0.0,
            index=history.columns,
        )

        weights.loc[available] = (
            1.0 / len(available)
        )

        return weights

    # Normalize
    weights = scores / scores.sum()

    # Maximum weight cap
    weights = weights.clip(
        upper=MAX_WEIGHT
    )

    # Remove tiny numerical values
    weights[weights < 1e-12] = 0.0

    # Renormalize
    total = weights.sum()

    if total > 0:
        weights = weights / total

    return weights


# ============================================================
# WALK-FORWARD
# ============================================================

print_header("V7 STRICT WALK-FORWARD TEST")

fold_results = []
weight_results = []

for fold in all_folds:

    print()
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    # --------------------------------------------------------
    # Historical data ONLY
    # --------------------------------------------------------

    historical = performance.loc[
        performance.index < fold
    ]

    weights = calculate_weights(
        historical
    )

    # --------------------------------------------------------
    # Current fold availability
    # --------------------------------------------------------

    current = performance.loc[fold]

    available = [
        strategy
        for strategy in STRATEGY_NAMES
        if pd.notna(current[strategy])
    ]

    print()
    print(
        "Available strategies: "
        + ", ".join(available)
    )

    if not available:
        print("No valid strategy available. Skipping fold.")
        continue

    # --------------------------------------------------------
    # Restrict weights to available strategies
    # --------------------------------------------------------

    current_weights = weights.copy()

    for strategy in STRATEGY_NAMES:

        if strategy not in available:
            current_weights[strategy] = 0.0

    total = current_weights.sum()

    if total <= 0:

        current_weights[:] = 0.0

        for strategy in available:
            current_weights[strategy] = (
                1.0 / len(available)
            )

    else:
        current_weights = (
            current_weights / total
        )

    # --------------------------------------------------------
    # Print weights
    # --------------------------------------------------------

    print()
    print("V7 STRICT strategy weights:")

    for strategy in STRATEGY_NAMES:

        print(
            f"  {strategy:12s}: "
            f"{current_weights[strategy]:.4f}"
        )

    selected = current_weights.idxmax()

    print()
    print(
        "Selected highest-weight strategy: "
        + selected.upper()
    )

    # --------------------------------------------------------
    # Adaptive weighted difference
    # --------------------------------------------------------

    adaptive_difference = 0.0

    for strategy in available:

        adaptive_difference += (
            current_weights[strategy]
            * current[strategy]
        )

    adaptive_hits = (
        RANDOM_EXPECTED
        + adaptive_difference
    )

    selected_difference = current[selected]
    selected_hits = (
        RANDOM_EXPECTED
        + selected_difference
    )

    # --------------------------------------------------------
    # Strict test draws
    #
    # IMPORTANT:
    # We use only draw counts belonging to strategies
    # actually participating in this fold.
    #
    # No max() fallback.
    # --------------------------------------------------------

    available_draws = []

    for strategy in available:

        d = test_draws.loc[
            fold,
            strategy
        ]

        if pd.notna(d) and d > 0:
            available_draws.append(
                float(d)
            )

    if available_draws:

        # Use the minimum valid draw count so that a fold
        # is never artificially expanded by another strategy.
        fold_test_draws = int(
            min(available_draws)
        )

    else:

        fold_test_draws = None

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

    for strategy in STRATEGY_NAMES:

        value = current[strategy]

        if pd.isna(value):

            print(
                f"  {strategy:12s}: NaN"
            )

        else:

            print(
                f"  {strategy:12s}: "
                f"{value:+.6f}"
            )

    # --------------------------------------------------------
    # Save fold result
    # --------------------------------------------------------

    fold_results.append(
        {
            "fold": fold,
            "test_draws": fold_test_draws,
            "adaptive_average_hits": adaptive_hits,
            "adaptive_difference": adaptive_difference,
            "selected_strategy": selected,
            "selected_average_hits": selected_hits,
            "selected_difference": selected_difference,
            "available_strategies": ",".join(
                available
            ),
        }
    )

    weight_row = {
        "fold": fold
    }

    for strategy in STRATEGY_NAMES:
        weight_row[
            strategy
        ] = current_weights[strategy]

    weight_results.append(
        weight_row
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

weights_df = pd.DataFrame(
    weight_results
)

if results_df.empty:
    raise RuntimeError(
        "No valid folds were available."
    )


# ============================================================
# WEIGHTED OVERALL METRIC
# ============================================================

# IMPORTANT:
# We weight folds by their ACTUAL test draw count.
#
# This prevents a 7-draw fold from having the same influence
# as a 1008-draw fold.

valid_weight_rows = results_df[
    results_df["test_draws"].notna()
    & (results_df["test_draws"] > 0)
].copy()

if not valid_weight_rows.empty:

    weighted_adaptive_hits = np.average(
        valid_weight_rows[
            "adaptive_average_hits"
        ],
        weights=valid_weight_rows[
            "test_draws"
        ],
    )

else:

    weighted_adaptive_hits = (
        results_df[
            "adaptive_average_hits"
        ].mean()
    )


weighted_difference = (
    weighted_adaptive_hits
    - RANDOM_EXPECTED
)

relative_improvement = (
    weighted_difference
    / RANDOM_EXPECTED
) * 100


# ============================================================
# SIMPLE MEAN
# ============================================================

simple_mean_hits = (
    results_df[
        "adaptive_average_hits"
    ].mean()
)

simple_mean_difference = (
    simple_mean_hits
    - RANDOM_EXPECTED
)


# ============================================================
# FOLD CONSISTENCY
# ============================================================

above_random = int(
    (
        results_df[
            "adaptive_difference"
        ] > 0
    ).sum()
)

below_random = int(
    (
        results_df[
            "adaptive_difference"
        ] < 0
    ).sum()
)


# ============================================================
# SELECTED STRATEGIES
# ============================================================

selected_counts = (
    results_df[
        "selected_strategy"
    ]
    .value_counts()
)

# Ensure all strategies appear
for strategy in STRATEGY_NAMES:

    if strategy not in selected_counts.index:
        selected_counts.loc[strategy] = 0

selected_counts = (
    selected_counts
    .reindex(
        STRATEGY_NAMES,
        fill_value=0
    )
)


# ============================================================
# MONTE-CARLO NULL TEST
# ============================================================

print_header("MONTE-CARLO NULL TEST")

rng = np.random.default_rng(
    RANDOM_SEED
)

observed = weighted_difference

null_values = np.zeros(
    N_MONTE_CARLO
)

# The null distribution is generated around zero
# using the empirical fold-level scale.

fold_differences = (
    valid_weight_rows[
        "adaptive_difference"
    ].to_numpy()
)

if len(fold_differences) >= 2:

    null_std = np.std(
        fold_differences,
        ddof=1
    )

else:

    null_std = 0.0


if null_std > 0:

    null_values = rng.normal(
        loc=0.0,
        scale=null_std,
        size=N_MONTE_CARLO,
    )

else:

    null_values[:] = 0.0


# Two-sided empirical p-value
p_value = (
    np.mean(
        np.abs(null_values)
        >= abs(observed)
    )
)


null_low = np.percentile(
    null_values,
    2.5
)

null_high = np.percentile(
    null_values,
    97.5
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
    f"Random 95% null range: "
    f"[{null_low:+.6f}, {null_high:+.6f}]"
)


# ============================================================
# PRINT FOLD RESULTS
# ============================================================

print_header("FOLD RESULTS")

display_columns = [
    "fold",
    "test_draws",
    "adaptive_average_hits",
    "adaptive_difference",
    "selected_strategy",
    "selected_average_hits",
    "selected_difference",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "adaptive_average_hits":
                lambda x: f"{x:+.6f}",
            "adaptive_difference":
                lambda x: f"{x:+.6f}",
            "selected_average_hits":
                lambda x: f"{x:+.6f}",
            "selected_difference":
                lambda x: f"{x:+.6f}",
        },
    )
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print_header("V7 STRICT FINAL EVALUATION")

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

for strategy in STRATEGY_NAMES:

    print(
        f"  {strategy:12s}: "
        f"{int(selected_counts[strategy])}"
    )

print()

print_header("FINAL CONCLUSION")

if (
    weighted_difference > 0
    and p_value < 0.05
):

    print(
        "V7 STRICT OUTPERFORMS "
        "THE RANDOM BASELINE."
    )

elif weighted_difference > 0:

    print(
        "V7 STRICT IS ABOVE THE RANDOM "
        "BASELINE, BUT THE ADVANTAGE IS "
        "NOT STATISTICALLY SIGNIFICANT."
    )

else:

    print(
        "V7 STRICT DOES NOT OUTPERFORM "
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


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_path = (
    RESULTS_DIR
    / "adaptive_meta_v7_top6_walk_forward_results.csv"
)

weights_path = (
    RESULTS_DIR
    / "adaptive_meta_v7_weights.csv"
)

results_df.to_csv(
    results_path,
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

print(results_path)
print(weights_path)

print("=" * 70)
print("DONE")
print("=" * 70)
