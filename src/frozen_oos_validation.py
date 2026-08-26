"""
FROZEN OUT-OF-SAMPLE VALIDATION
================================

Final validation stage after V10.

IMPORTANT:
- No model training.
- No parameter optimization.
- No weight optimization.
- No strategy modification.
- No use of previous test folds as new observations.

This script evaluates FROZEN strategies on completely unseen
out-of-sample draws.

Expected input:

data/processed/frozen_oos_strategy_results.csv

The CSV should contain one row per unseen OOS draw/fold and
the performance of each frozen strategy.

Expected strategy columns:

    meta_score
    recency
    stability
    diversity
    ensemble

Optionally, V10 can be supplied directly as:

    V10

or calculated from frozen V10 weights.

Usage:

    python src/frozen_oos_validation.py

or:

    python src/frozen_oos_validation.py --input data/processed/frozen_oos_strategy_results.csv

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_EXPECTED_HITS = 0.734694

N_SIMULATIONS = 10_000

CONFIDENCE_LEVEL = 0.95

MAX_WEIGHT = 0.40

STRATEGIES = [
    "meta_score",
    "recency",
    "stability",
    "diversity",
    "ensemble",
]

OUTPUT_DIR = Path("data/processed")

DEFAULT_INPUT = OUTPUT_DIR / "frozen_oos_strategy_results.csv"

V10_WEIGHTS_FILE = OUTPUT_DIR / "adaptive_meta_v10_weights.csv"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names while preserving original data.
    """

    df = df.copy()

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    print("Actual CSV Columns:", list(df.columns))
    return df


def find_strategy_column(
    df: pd.DataFrame,
    strategy: str,
) -> str | None:
    """
    Find a strategy column using flexible naming matching.
    """
    strategy_lower = strategy.lower()

    # 1. المطابقة المباشرة أو مع اللاحقات الشائعة
    candidates = [
        strategy_lower,
        f"{strategy_lower}_difference",
        f"{strategy_lower}_diff",
        f"{strategy_lower}_mean_difference",
        f"{strategy_lower}_performance",
        f"{strategy_lower}_average_hits",
        f"{strategy_lower}_hits",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # 2. البحث عن اسم الاستراتيجية كجزء من اسم العمود (مثل strategy_meta_score أو meta_score_val)
    for col in df.columns:
        if strategy_lower in col.lower():
            return col

    return None


def load_oos_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print()
        print("ERROR: OOS input file was not found:")
        print(f"  {path}")
        print()
        print("Create the frozen OOS strategy results first.")
        print()
        sys.exit(1)

    df = pd.read_csv(path)

    if df.empty:
        print("ERROR: OOS file is empty.")
        sys.exit(1)

    df = normalize_columns(df)

    # تحويل البيانات من Long Format إلى Wide Format إذا كان عمود strategy موجوداً
    if "strategy" in df.columns and "hits" in df.columns:
        # إذا كان عمود التاريخ أو الترتيب موجوداً نستخدمه كـ index
        index_col = "date" if "date" in df.columns else None
        
        if index_col:
            df = df.pivot(index=index_col, columns="strategy", values="hits").reset_index()
        else:
            # في حال عدم وجود تاريخ ننشئ معرفاً لكل جولة/توقع
            df['draw_id'] = df.groupby('strategy').cumcount()
            df = df.pivot(index='draw_id', columns='strategy', values='hits').reset_index()

    return df


def convert_hits_to_difference(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert a hits/average-hits column into difference from random.

    If the values already look like differences, keep them unchanged.
    """

    values = pd.to_numeric(series, errors="coerce")

    valid = values.dropna()

    if valid.empty:
        return values

    # Differences normally fall around [-1, +1]
    # Hits are expected to be positive and close to RANDOM_EXPECTED_HITS.
    #
    # If the median is near the random baseline, interpret as hits.
    median_value = float(valid.median())

    if (
        "difference" not in column_name
        and "diff" not in column_name
        and median_value >= 0.0
        and median_value <= 1.0
    ):
        return values - RANDOM_EXPECTED_HITS

    return values


# ============================================================
# LOAD FROZEN V10 WEIGHTS
# ============================================================

def load_v10_weights(path: Path) -> dict[str, float] | None:
    """
    Try to load the LAST frozen V10 weight vector.

    This is intentionally NOT optimized on OOS data.
    """

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    df = normalize_columns(df)

    # --------------------------------------------------------
    # Case 1:
    # One row containing strategy columns
    # --------------------------------------------------------

    direct = {}

    for strategy in STRATEGIES:
        if strategy in df.columns:
            value = pd.to_numeric(
                df.iloc[-1][strategy],
                errors="coerce",
            )

            if pd.notna(value):
                direct[strategy] = float(value)

    if direct:
        total = sum(direct.values())

        if total > 0:
            return {
                k: v / total
                for k, v in direct.items()
            }

    # --------------------------------------------------------
    # Case 2:
    # Columns like:
    #
    # strategy / weight
    #
    # --------------------------------------------------------

    if "strategy" in df.columns and "weight" in df.columns:

        temp = df.copy()

        temp["strategy"] = temp["strategy"].astype(str).str.lower()

        temp["weight"] = pd.to_numeric(
            temp["weight"],
            errors="coerce",
        )

        temp = temp.dropna(
            subset=["strategy", "weight"]
        )

        if not temp.empty:

            # Use the latest occurrence of each strategy.
            latest = (
                temp
                .groupby("strategy", as_index=False)
                .tail(1)
            )

            weights = {}

            for _, row in latest.iterrows():

                strategy = row["strategy"]

                if strategy in STRATEGIES:
                    weights[strategy] = float(row["weight"])

            if weights:
                total = sum(weights.values())

                if total > 0:
                    return {
                        k: v / total
                        for k, v in weights.items()
                    }

    # --------------------------------------------------------
    # Case 3:
    # One row per fold with strategy columns
    # --------------------------------------------------------

    direct = {}

    for strategy in STRATEGIES:

        matching = [
            c for c in df.columns
            if c == strategy
            or c.endswith(f"_{strategy}")
            or c.startswith(f"{strategy}_")
        ]

        if matching:

            column = matching[-1]

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if not values.empty:
                direct[strategy] = float(values.iloc[-1])

    if direct:

        total = sum(direct.values())

        if total > 0:

            return {
                k: v / total
                for k, v in direct.items()
            }

    return None


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_difference_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:

    matrix = pd.DataFrame(index=df.index)

    for strategy in STRATEGIES:

        column = find_strategy_column(
            df,
            strategy,
        )

        if column is None:
            matrix[strategy] = np.nan
            continue

        matrix[strategy] = convert_hits_to_difference(
            df[column],
            column,
        )

    return matrix


# ============================================================
# V10 CALCULATION
# ============================================================

def apply_v10_weights(
    matrix: pd.DataFrame,
    weights: dict[str, float] | None,
) -> pd.Series:

    result = pd.Series(
        np.nan,
        index=matrix.index,
        dtype=float,
    )

    if not weights:
        return result

    available_weights = {
        strategy: weight
        for strategy, weight in weights.items()
        if strategy in matrix.columns
    }

    if not available_weights:
        return result

    for idx in matrix.index:

        row = matrix.loc[idx]

        valid = {
            strategy: weight
            for strategy, weight in available_weights.items()
            if pd.notna(row[strategy])
        }

        if not valid:
            continue

        total_weight = sum(valid.values())

        if total_weight <= 0:
            continue

        weighted_value = sum(
            row[strategy] * weight
            for strategy, weight in valid.items()
        )

        result.loc[idx] = weighted_value / total_weight

    return result


# ============================================================
# STATISTICS
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    simulations: int = N_SIMULATIONS,
) -> tuple[float, float]:

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(42)

    means = np.empty(simulations)

    n = len(values)

    for i in range(simulations):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = np.mean(sample)

    alpha = 1.0 - CONFIDENCE_LEVEL

    low = np.quantile(
        means,
        alpha / 2,
    )

    high = np.quantile(
        means,
        1 - alpha / 2,
    )

    return float(low), float(high)


def permutation_test(
    values: np.ndarray,
    simulations: int = N_SIMULATIONS,
) -> float:

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    observed = float(np.mean(values))

    rng = np.random.default_rng(42)

    n = len(values)

    null_means = np.empty(simulations)

    for i in range(simulations):

        signs = rng.choice(
            [-1.0, 1.0],
            size=n,
        )

        null_means[i] = np.mean(
            values * signs
        )

    p_value = (
        np.sum(
            np.abs(null_means)
            >= abs(observed)
        )
        + 1
    ) / (
        simulations + 1
    )

    return float(p_value)


# ============================================================
# MODEL SUMMARY
# ============================================================

def summarize_model(
    name: str,
    differences: pd.Series,
) -> dict:

    values = pd.to_numeric(
        differences,
        errors="coerce",
    ).dropna().to_numpy()

    if len(values) == 0:

        return {
            "model": name,
            "observations": 0,
            "mean_difference": np.nan,
            "relative_improvement_pct": np.nan,
            "median_difference": np.nan,
            "std_difference": np.nan,
            "above_random": 0,
            "below_random": 0,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "permutation_p_value": np.nan,
        }

    mean_difference = float(np.mean(values))

    mean_hits = (
        RANDOM_EXPECTED_HITS
        + mean_difference
    )

    relative_improvement = (
        mean_difference
        / RANDOM_EXPECTED_HITS
        * 100.0
    )

    ci_low, ci_high = bootstrap_mean_ci(
        values
    )

    p_value = permutation_test(
        values
    )

    return {
        "model": name,
        "observations": len(values),
        "mean_hits": mean_hits,
        "mean_difference": mean_difference,
        "relative_improvement_pct": relative_improvement,
        "median_difference": float(
            np.median(values)
        ),
        "std_difference": float(
            np.std(values, ddof=1)
        ) if len(values) > 1 else 0.0,
        "above_random": int(
            np.sum(values > 0)
        ),
        "below_random": int(
            np.sum(values < 0)
        ),
        "equal_random": int(
            np.sum(values == 0)
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "permutation_p_value": p_value,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Frozen out-of-sample validation of "
            "strategies and V10."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Frozen OOS strategy results CSV.",
    )

    parser.add_argument(
        "--v10-weights",
        type=str,
        default=str(V10_WEIGHTS_FILE),
        help="Frozen V10 weights CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print_header(
        "FROZEN OUT-OF-SAMPLE VALIDATION"
    )

    print(
        f"Random expected hits: {RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        f"Bootstrap/permutation simulations: "
        f"{N_SIMULATIONS:,}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "No model optimization is performed."
    )
    print(
        "No parameters are changed."
    )
    print(
        "No OOS information is used to create weights."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print_header(
        "LOADING COMPLETELY UNSEEN OOS DATA"
    )

    input_path = Path(args.input)

    print(
        f"Input: {input_path}"
    )

    df = load_oos_data(
        input_path
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # BUILD MATRIX
    # --------------------------------------------------------

    print_header(
        "FROZEN STRATEGY PERFORMANCE MATRIX"
    )

    matrix = build_difference_matrix(
        df
    )

    print(
        matrix.to_string()
    )

    available = [
        strategy
        for strategy in STRATEGIES
        if matrix[strategy].notna().any()
    ]

    print()
    print(
        "Available frozen strategies:"
    )

    for strategy in available:
        print(
            f"  {strategy}"
        )

    if not available:

        print()
        print(
            "ERROR: No strategy columns were found."
        )

        print()
        print(
            "Expected columns:"
        )

        for strategy in STRATEGIES:
            print(
                f"  {strategy}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # LOAD V10 WEIGHTS
    # --------------------------------------------------------

    print_header(
        "LOADING FROZEN V10 WEIGHTS"
    )

    v10_weights = load_v10_weights(
        Path(args.v10_weights)
    )

    if v10_weights:

        print(
            "Frozen V10 weights:"
        )

        for strategy in STRATEGIES:

            weight = v10_weights.get(
                strategy,
                0.0,
            )

            print(
                f"  {strategy:12s}: "
                f"{weight:.4f}"
            )

    else:

        print(
            "WARNING: V10 weights could not be loaded."
        )

        print(
            "V10 will not be calculated."
        )

    # --------------------------------------------------------
    # CALCULATE V10
    # --------------------------------------------------------

    matrix_with_v10 = matrix.copy()

    if v10_weights:

        matrix_with_v10["V10"] = apply_v10_weights(
            matrix,
            v10_weights,
        )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    print_header(
        "FROZEN OOS MODEL EVALUATION"
    )

    summaries = []

    for strategy in available:

        summary = summarize_model(
            strategy,
            matrix_with_v10[strategy],
        )

        summaries.append(
            summary
        )

    if "V10" in matrix_with_v10.columns:

        summary = summarize_model(
            "V10",
            matrix_with_v10["V10"],
        )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    if not summary_df.empty:

        summary_df = summary_df.sort_values(
            "mean_difference",
            ascending=False,
        ).reset_index(
            drop=True
        )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    print_header(
        "BEST FROZEN OOS MODEL"
    )

    valid_summary = summary_df[
        summary_df["observations"] > 0
    ]

    if valid_summary.empty:

        print(
            "No valid OOS model results."
        )

        sys.exit(1)

    best = valid_summary.iloc[0]

    print(
        f"Model:                 "
        f"{best['model']}"
    )

    print(
        f"Observations:          "
        f"{int(best['observations'])}"
    )

    print(
        f"Mean hits:             "
        f"{best['mean_hits']:.6f}"
    )

    print(
        f"Random expected hits:  "
        f"{RANDOM_EXPECTED_HITS:.6f}"
    )

    print(
        f"Mean difference:       "
        f"{best['mean_difference']:+.6f}"
    )

    print(
        f"Relative improvement:  "
        f"{best['relative_improvement_pct']:+.3f}%"
    )

    print(
        f"Bootstrap 95% CI:      "
        f"[{best['bootstrap_ci_low']:+.6f}, "
        f"{best['bootstrap_ci_high']:+.6f}]"
    )

    print(
        f"Permutation p-value:   "
        f"{best['permutation_p_value']:.6f}"
    )

    # --------------------------------------------------------
    # ROBUSTNESS DECISION
    # --------------------------------------------------------

    print_header(
        "FINAL OUT-OF-SAMPLE DECISION"
    )

    mean_difference = float(
        best["mean_difference"]
    )

    ci_low = float(
        best["bootstrap_ci_low"]
    )

    p_value = float(
        best["permutation_p_value"]
    )

    observations = int(
        best["observations"]
    )

    statistically_positive = (
        mean_difference > 0
        and ci_low > 0
        and p_value < 0.05
    )

    if statistically_positive:

        print(
            "============================================================"
        )
        print(
            "POSITIVE OUT-OF-SAMPLE SIGNAL"
        )
        print(
            "============================================================"
        )

        print()
        print(
            "The frozen model shows a statistically "
            "positive OOS advantage."
        )

        print()
        print(
            "This is the first result that would justify "
            "further independent validation."
        )

    else:

        print(
            "NO STATISTICALLY ROBUST OUT-OF-SAMPLE SIGNAL."
        )

        print()
        print(
            "The frozen strategies do not provide "
            "sufficient evidence of a genuine advantage."
        )

        print()
        print(
            "DO NOT create V11."
        )

        print(
            "DO NOT tune the model using these OOS results."
        )

        print(
            "The correct action is to keep the model frozen "
            "and collect additional unseen observations."
        )

    # --------------------------------------------------------
    # ADDITIONAL INTERPRETATION
    # --------------------------------------------------------

    print_header(
        "INTERPRETATION"
    )

    if observations < 20:

        print(
            f"WARNING: Only {observations} OOS observations "
            "are available."
        )

        print(
            "This is too small for a strong final conclusion."
        )

        print(
            "Treat this as an initial OOS checkpoint."
        )

    elif observations < 50:

        print(
            f"OOS observations: {observations}"
        )

        print(
            "Evidence is still limited."
        )

    else:

        print(
            f"OOS observations: {observations}"
        )

        print(
            "The OOS sample is substantially more informative."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "frozen_oos_validation_summary.csv"
    )

    matrix_path = (
        output_dir
        / "frozen_oos_validation_matrix.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    matrix_with_v10.to_csv(
        matrix_path,
        index=False,
    )

    print_header(
        "RESULTS SAVED"
    )

    print(
        summary_path
    )

    print(
        matrix_path
    )

    print()
    print(
        "DONE"
    )


if __name__ == "__main__":
    main()