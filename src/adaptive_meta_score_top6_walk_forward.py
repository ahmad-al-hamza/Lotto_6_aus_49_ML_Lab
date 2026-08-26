"""
ADAPTIVE META-SCORE TOP-6 WALK-FORWARD TEST

Combines:
    1. Meta-Score
    2. Ensemble
    3. Recency
    4. Stability
    5. Diversity

The strategy weights are learned ONLY from previous folds.
No future test-fold information is used to determine weights.

Expected input files:
    data/processed/meta_score_top6_walk_forward_results.csv
    data/processed/ensemble_top6_walk_forward_results.csv
    data/processed/recency_top6_walk_forward_results.csv
    data/processed/stability_top6_walk_forward_results.csv
    data/processed/diversity_top6_walk_forward_results.csv

Output:
    data/processed/adaptive_meta_score_top6_walk_forward_results.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

META_FILE = DATA_DIR / "meta_score_top6_walk_forward_results.csv"
ENSEMBLE_FILE = DATA_DIR / "ensemble_top6_walk_forward_results.csv"
RECENCY_FILE = DATA_DIR / "recency_top6_walk_forward_results.csv"
STABILITY_FILE = DATA_DIR / "stability_top6_walk_forward_results.csv"
DIVERSITY_FILE = DATA_DIR / "diversity_top6_walk_forward_results.csv"

OUTPUT_FILE = (
    DATA_DIR / "adaptive_meta_score_top6_walk_forward_results.csv"
)

NUMBER_COUNT = 6
NUMBER_RANGE = 49

RANDOM_EXPECTED = NUMBER_COUNT * NUMBER_COUNT / NUMBER_RANGE

STRATEGIES = [
    "meta_score",
    "ensemble",
    "recency",
    "stability",
    "diversity",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    """Convert value to finite float."""
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def load_results(path, strategy_name):
    """
    Load a strategy result file and normalize its columns.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "fold" not in df.columns:
        raise ValueError(
            f"{path.name} does not contain a 'fold' column."
        )

    # Most result files use "difference".
    if "difference" not in df.columns:

        if "difference_percent" in df.columns:
            df["difference"] = (
                pd.to_numeric(
                    df["difference_percent"],
                    errors="coerce"
                ) / 100.0
            )

        elif "average_hits" in df.columns:
            df["difference"] = (
                pd.to_numeric(
                    df["average_hits"],
                    errors="coerce"
                )
                - RANDOM_EXPECTED
            )

        else:
            raise ValueError(
                f"Cannot determine performance difference "
                f"from {path.name}"
            )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="coerce"
    )

    df["difference"] = pd.to_numeric(
        df["difference"],
        errors="coerce"
    )

    df["strategy"] = strategy_name

    return df[
        [
            "fold",
            "difference",
            "strategy",
        ]
    ].copy()


def softmax(values, temperature=1.0):
    """
    Convert strategy scores into positive normalized weights.

    Temperature prevents one strategy from immediately
    receiving 100% of the weight.
    """

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    temperature = max(float(temperature), 1e-6)

    scaled = values / temperature

    scaled -= np.max(scaled)

    exp_values = np.exp(
        np.clip(scaled, -50, 50)
    )

    total = exp_values.sum()

    if total <= 0 or not np.isfinite(total):
        return np.ones(len(values)) / len(values)

    return exp_values / total


def calculate_adaptive_weights(
    historical_performance,
    temperature=0.025,
    decay=0.85,
):
    """
    Calculate strategy weights using ONLY historical folds.

    More recent historical folds receive larger weights.

    historical_performance:
        DataFrame:
            fold | strategy columns

    Returns:
        dict(strategy -> weight)
    """

    if historical_performance.empty:
        equal_weight = 1.0 / len(STRATEGIES)

        return {
            strategy: equal_weight
            for strategy in STRATEGIES
        }

    scores = []

    for strategy in STRATEGIES:

        if strategy not in historical_performance.columns:
            scores.append(0.0)
            continue

        series = pd.to_numeric(
            historical_performance[strategy],
            errors="coerce"
        ).dropna()

        if series.empty:
            scores.append(0.0)
            continue

        # Most recent observation gets the largest weight.
        n = len(series)

        decay_weights = np.array(
            [
                decay ** (n - 1 - i)
                for i in range(n)
            ],
            dtype=float,
        )

        weighted_score = np.average(
            series.values,
            weights=decay_weights,
        )

        scores.append(
            safe_float(weighted_score)
        )

    weights = softmax(
        scores,
        temperature=temperature,
    )

    return {
        strategy: float(weight)
        for strategy, weight in zip(
            STRATEGIES,
            weights,
        )
    }


def calculate_meta_expected_difference(
    weights,
    current_scores,
):
    """
    Calculate expected performance using historical weights.

    IMPORTANT:
    This function is only for evaluation/reporting.
    It does not modify the weights.
    """

    total = 0.0
    weight_sum = 0.0

    for strategy in STRATEGIES:

        value = current_scores.get(
            strategy,
            np.nan,
        )

        if pd.isna(value):
            continue

        weight = weights.get(
            strategy,
            0.0,
        )

        total += weight * float(value)
        weight_sum += weight

    if weight_sum <= 0:
        return np.nan

    return total / weight_sum


def choose_strategy(weights):
    """Return highest-weight strategy."""

    return max(
        weights,
        key=weights.get,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ADAPTIVE META-SCORE TOP-6 WALK-FORWARD TEST")
print("=" * 70)

print(
    f"Random expected hits: "
    f"{RANDOM_EXPECTED:.6f}"
)

print("\nLoading strategy results...")

meta_df = load_results(
    META_FILE,
    "meta_score",
)

ensemble_df = load_results(
    ENSEMBLE_FILE,
    "ensemble",
)

recency_df = load_results(
    RECENCY_FILE,
    "recency",
)

stability_df = load_results(
    STABILITY_FILE,
    "stability",
)

diversity_df = load_results(
    DIVERSITY_FILE,
    "diversity",
)

all_results = pd.concat(
    [
        meta_df,
        ensemble_df,
        recency_df,
        stability_df,
        diversity_df,
    ],
    ignore_index=True,
)


# ============================================================
# CREATE PERFORMANCE MATRIX
# ============================================================

performance_matrix = (
    all_results
    .pivot_table(
        index="fold",
        columns="strategy",
        values="difference",
        aggfunc="first",
    )
    .reset_index()
)

for strategy in STRATEGIES:

    if strategy not in performance_matrix.columns:
        performance_matrix[strategy] = np.nan

performance_matrix = performance_matrix[
    ["fold"] + STRATEGIES
]

performance_matrix = performance_matrix.sort_values(
    "fold"
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("STRATEGY PERFORMANCE MATRIX")
print("=" * 70)

print(
    performance_matrix.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}",
    )
)


# ============================================================
# WALK-FORWARD META MODEL
# ============================================================

print("\n")
print("=" * 70)
print("ADAPTIVE META-SCORE WALK-FORWARD TEST")
print("=" * 70)

adaptive_rows = []

selected_counts = {
    strategy: 0
    for strategy in STRATEGIES
}

for position, row in performance_matrix.iterrows():

    fold = int(row["fold"])

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    # --------------------------------------------------------
    # CRITICAL:
    # Only folds BEFORE the current fold are available.
    # --------------------------------------------------------

    historical = performance_matrix[
        performance_matrix["fold"] < fold
    ].copy()

    if historical.empty:

        weights = {
            strategy: 1.0 / len(STRATEGIES)
            for strategy in STRATEGIES
        }

    else:

        weights = calculate_adaptive_weights(
            historical_performance=historical[
                STRATEGIES
            ],
            temperature=0.025,
            decay=0.85,
        )

    selected_strategy = choose_strategy(
        weights
    )

    selected_counts[
        selected_strategy
    ] += 1

    # Current fold performance is used ONLY for evaluation.
    current_scores = {
        strategy: safe_float(
            row[strategy],
            default=np.nan,
        )
        for strategy in STRATEGIES
    }

    expected_difference = (
        calculate_meta_expected_difference(
            weights,
            current_scores,
        )
    )

    # Actual adaptive result:
    # weighted combination of all available strategies.
    adaptive_difference = expected_difference

    print("\nHistorical strategy weights:")

    for strategy in STRATEGIES:

        print(
            f"  {strategy:<10}: "
            f"{weights[strategy]:.4f}"
        )

    print(
        f"\nSelected strategy: "
        f"{selected_strategy.upper()}"
    )

    print(
        f"Adaptive expected difference: "
        f"{adaptive_difference:+.6f}"
    )

    print("\nCurrent fold performance:")

    for strategy in STRATEGIES:

        value = current_scores[strategy]

        if np.isnan(value):

            print(
                f"  {strategy:<10}: NaN"
            )

        else:

            print(
                f"  {strategy:<10}: "
                f"{value:+.6f}"
            )

    # --------------------------------------------------------
    # Calculate actual average hits.
    # --------------------------------------------------------

    adaptive_average_hits = (
        RANDOM_EXPECTED
        + adaptive_difference
    )

    adaptive_rows.append(
        {
            "fold": fold,
            "adaptive_difference": adaptive_difference,
            "adaptive_average_hits": adaptive_average_hits,
            "random_expected": RANDOM_EXPECTED,
            "selected_strategy": selected_strategy,

            "weight_meta_score":
                weights["meta_score"],

            "weight_ensemble":
                weights["ensemble"],

            "weight_recency":
                weights["recency"],

            "weight_stability":
                weights["stability"],

            "weight_diversity":
                weights["diversity"],

            "meta_score_difference":
                current_scores["meta_score"],

            "ensemble_difference":
                current_scores["ensemble"],

            "recency_difference":
                current_scores["recency"],

            "stability_difference":
                current_scores["stability"],

            "diversity_difference":
                current_scores["diversity"],
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    adaptive_rows
)


# ============================================================
# SUMMARY
# ============================================================

mean_adaptive_difference = (
    results_df["adaptive_difference"]
    .mean()
)

mean_random_difference = 0.0

mean_improvement_percent = (
    mean_adaptive_difference
    / RANDOM_EXPECTED
    * 100.0
)

mean_adaptive_hits = (
    results_df["adaptive_average_hits"]
    .mean()
)

above_random = int(
    (
        results_df["adaptive_difference"]
        > 0
    ).sum()
)

below_random = int(
    (
        results_df["adaptive_difference"]
        < 0
    ).sum()
)


print("\n")
print("=" * 70)
print("ADAPTIVE META-SCORE SUMMARY")
print("=" * 70)

print(
    f"Mean adaptive hits: "
    f"{mean_adaptive_hits:.6f}"
)

print(
    f"Mean random expected: "
    f"{RANDOM_EXPECTED:.6f}"
)

print(
    f"Mean adaptive difference: "
    f"{mean_adaptive_difference:+.6f}"
)

print(
    f"Mean improvement: "
    f"{mean_improvement_percent:+.3f}%"
)

print("\nFold consistency:")

print(
    f"Above random: {above_random}"
)

print(
    f"Below random: {below_random}"
)

print("\nSelected strategies:")

for strategy in STRATEGIES:

    print(
        f"  {strategy:<10}: "
        f"{selected_counts[strategy]}"
    )


# ============================================================
# PERFORMANCE BY FOLD
# ============================================================

print("\n")
print("=" * 70)
print("FOLD RESULTS")
print("=" * 70)

display_columns = [
    "fold",
    "adaptive_average_hits",
    "random_expected",
    "adaptive_difference",
    "selected_strategy",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}",
    )
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

if (
    mean_adaptive_difference > 0
    and above_random > below_random
):

    print(
        "\nAdaptive Meta-Score is above "
        "the random baseline."
    )

    print(
        f"Average advantage: "
        f"{mean_adaptive_difference:+.6f}"
    )

    print(
        f"Relative improvement: "
        f"{mean_improvement_percent:+.3f}%"
    )

else:

    print(
        "\nAdaptive Meta-Score did not "
        "outperform the random baseline "
        "consistently."
    )


# ============================================================
# SAVE
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n")
print("=" * 70)
print("RESULTS SAVED TO:")
print("=" * 70)

print(OUTPUT_FILE)

print("=" * 70)
print("DONE")
print("=" * 70)
