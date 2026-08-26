# adaptive_meta_final_validation.py

"""
FINAL VALIDATION

Purpose:
Compare V6-V10 and the original strategies using the SAME folds.

This is intentionally NOT another adaptive weighting model.

We want to answer:
1. Is any strategy consistently above random?
2. Is V6 genuinely better than the others?
3. Does the apparent improvement survive bootstrap/permutation testing?
4. Are we simply overfitting the available folds?

If no model passes the validation criteria, STOP model development
and collect more out-of-sample draws.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANDOM_EXPECTED = 0.734694

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42

# Minimum acceptable criteria
MIN_MEAN_ADVANTAGE = 0.0
MAX_P_VALUE = 0.05

# ============================================================
# FILES
# ============================================================

FILES = {
    "V6": "adaptive_meta_v6_top6_walk_forward_results.csv",
    "V7": "adaptive_meta_v7_top6_walk_forward_results.csv",
    "V8": "adaptive_meta_v8_top6_walk_forward_results.csv",
    "V9": "adaptive_meta_v9_top6_walk_forward_results.csv",
    "V10": "adaptive_meta_v10_top6_walk_forward_results.csv",

    "meta_score": "adaptive_meta_score_top6_walk_forward_results.csv",
    "recency": "recency_top6_walk_forward_results.csv",
    "stability": "stability_top6_walk_forward_results.csv",
    "diversity": "diversity_top6_walk_forward_results.csv",
    "ensemble": "ensemble_top6_walk_forward_results.csv",
}


# ============================================================
# HELPERS
# ============================================================

def load_result(name, filename):
    """Load one result CSV if it exists."""

    path = PROCESSED_DIR / filename

    if not path.exists():
        print(f"WARNING: {name} file not found:")
        print(f"  {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read {name}: {exc}")
        return None

    print(
        f"{name:12s}: "
        f"{len(df)} rows -> {filename}"
    )

    return df


def find_column(df, candidates):
    """Return the first matching column name."""

    for column in candidates:
        if column in df.columns:
            return column

    return None


def extract_fold_results(name, df):
    """Extract standardized fold-level results."""

    if df is None or df.empty:
        return None

    fold_col = find_column(
        df,
        [
            "fold",
            "Fold",
            "FOLD",
        ],
    )

    diff_col = find_column(
        df,
        [
            "adaptive_difference",
            "weighted_difference",
            "difference",
            "selected_difference",
        ],
    )

    hits_col = find_column(
        df,
        [
            "adaptive_average_hits",
            "weighted_adaptive_hits",
            "adaptive_hits",
            "average_hits",
        ],
    )

    draws_col = find_column(
        df,
        [
            "test_draws",
            "draws",
            "test_count",
        ],
    )

    if fold_col is None:
        print(
            f"WARNING: {name}: fold column not found. "
            f"Available columns: {list(df.columns)}"
        )
        return None

    if diff_col is None:
        print(
            f"WARNING: {name}: difference column not found. "
            f"Available columns: {list(df.columns)}"
        )
        return None

    result = pd.DataFrame()

    result["fold"] = pd.to_numeric(
        df[fold_col],
        errors="coerce",
    )

    result["difference"] = pd.to_numeric(
        df[diff_col],
        errors="coerce",
    )

    if hits_col is not None:
        result["hits"] = pd.to_numeric(
            df[hits_col],
            errors="coerce",
        )
    else:
        result["hits"] = (
            RANDOM_EXPECTED
            + result["difference"]
        )

    if draws_col is not None:
        result["draws"] = pd.to_numeric(
            df[draws_col],
            errors="coerce",
        )
    else:
        result["draws"] = np.nan

    result = result.dropna(
        subset=[
            "fold",
            "difference",
        ]
    )

    if result.empty:
        print(
            f"WARNING: {name}: no valid fold results."
        )
        return None

    result["model"] = name

    return result[
        [
            "model",
            "fold",
            "draws",
            "hits",
            "difference",
        ]
    ]


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_mean_difference(values, rng):
    """
    Bootstrap the mean fold-level difference.

    Returns:
        observed mean,
        lower 95% percentile,
        upper 95% percentile
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    observed = float(
        np.mean(values)
    )

    boot = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for i in range(N_BOOTSTRAP):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot[i] = np.mean(sample)

    lower = float(
        np.percentile(
            boot,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            boot,
            97.5,
        )
    )

    return observed, lower, upper


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_p_value(values, rng):
    """
    Two-sided sign-flip permutation test.

    H0:
        Mean fold-level difference = 0

    This is appropriate when the unit being tested is the
    independent fold-level difference.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    observed = abs(
        np.mean(values)
    )

    count = 0

    for _ in range(N_BOOTSTRAP):
        signs = rng.choice(
            [-1.0, 1.0],
            size=len(values),
        )

        simulated = abs(
            np.mean(values * signs)
        )

        if simulated >= observed:
            count += 1

    return (
        count + 1
    ) / (
        N_BOOTSTRAP + 1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL META-MODEL VALIDATION")
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print(
        f"Bootstrap/permutation simulations: "
        f"{N_BOOTSTRAP:,}"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING RESULTS")
    print("=" * 70)

    raw_results = {}

    for name, filename in FILES.items():

        raw_results[name] = load_result(
            name,
            filename,
        )

    print()

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    all_results = []

    for name, df in raw_results.items():

        extracted = extract_fold_results(
            name,
            df,
        )

        if extracted is not None:
            all_results.append(
                extracted
            )

    if not all_results:
        print(
            "ERROR: No usable result files."
        )
        return

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CHECK DUPLICATE FOLDS
    # --------------------------------------------------------

    duplicate_mask = results.duplicated(
        subset=["model", "fold"],
        keep=False,
    )

    if duplicate_mask.any():

        print("=" * 70)
        print("WARNING: DUPLICATE MODEL/FOLD ROWS")
        print("=" * 70)

        duplicates = results[
            duplicate_mask
        ].sort_values(
            ["model", "fold"]
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        print()
        print(
            "Keeping the first row for each "
            "model/fold combination."
        )

        results = results.drop_duplicates(
            subset=["model", "fold"],
            keep="first",
        )

        print()

    # --------------------------------------------------------
    # FOLD MATRIX
    # --------------------------------------------------------

    print("=" * 70)
    print("FOLD DIFFERENCE MATRIX")
    print("=" * 70)

    matrix = results.pivot_table(
        index="fold",
        columns="model",
        values="difference",
        aggfunc="first",
    )

    print(
        matrix.to_string(
            float_format=lambda x:
            f"{x:+.6f}"
        )
    )

    print()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    summary_rows = []

    for model in sorted(
        results["model"].unique()
    ):

        subset = results[
            results["model"] == model
        ].copy()

        differences = subset[
            "difference"
        ].dropna().to_numpy(
            dtype=float
        )

        if len(differences) == 0:
            continue

        mean_diff = float(
            np.mean(differences)
        )

        median_diff = float(
            np.median(differences)
        )

        std_diff = float(
            np.std(
                differences,
                ddof=1,
            )
            if len(differences) > 1
            else 0.0
        )

        above = int(
            np.sum(
                differences > 0
            )
        )

        below = int(
            np.sum(
                differences < 0
            )
        )

        equal = int(
            np.sum(
                differences == 0
            )
        )

        mean_hits = (
            RANDOM_EXPECTED
            + mean_diff
        )

        relative = (
            mean_diff
            / RANDOM_EXPECTED
            * 100.0
        )

        boot_mean, ci_low, ci_high = (
            bootstrap_mean_difference(
                differences,
                rng,
            )
        )

        p_value = (
            permutation_p_value(
                differences,
                rng,
            )
        )

        summary_rows.append(
            {
                "model": model,
                "folds": len(differences),
                "mean_hits": mean_hits,
                "mean_difference": mean_diff,
                "relative_improvement_pct": relative,
                "median_difference": median_diff,
                "std_difference": std_diff,
                "above_random": above,
                "below_random": below,
                "equal_random": equal,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_p_value": p_value,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    if summary.empty:
        print(
            "ERROR: No valid summary statistics."
        )
        return

    summary = summary.sort_values(
        "mean_difference",
        ascending=False,
    ).reset_index(
        drop=True
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    print()

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best = summary.iloc[0]

    print("=" * 70)
    print("BEST OBSERVED MODEL")
    print("=" * 70)

    print(
        f"Model:                 {best['model']}"
    )

    print(
        f"Folds:                 {int(best['folds'])}"
    )

    print(
        f"Mean hits:             "
        f"{best['mean_hits']:.6f}"
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

    print()

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    significant = (
        best["permutation_p_value"]
        < MAX_P_VALUE
    )

    positive_ci = (
        best["bootstrap_ci_low"]
        > 0
    )

    positive_mean = (
        best["mean_difference"]
        > MIN_MEAN_ADVANTAGE
    )

    print("=" * 70)
    print("FINAL DECISION")
    print("=" * 70)

    if (
        positive_mean
        and positive_ci
        and significant
    ):

        print(
            "SIGNAL DETECTED."
        )

        print()
        print(
            "The best model shows:"
        )

        print(
            "  - positive mean advantage"
        )

        print(
            "  - bootstrap confidence interval "
            "above zero"
        )

        print(
            "  - permutation p-value < 0.05"
        )

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Freeze this model and perform "
            "a completely unseen holdout test."
        )

    else:

        print(
            "NO STATISTICALLY ROBUST SIGNAL."
        )

        print()
        print(
            "The current evidence does NOT "
            "justify another adaptive meta-model."
        )

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "STOP MODEL ITERATION."
        )

        print(
            "Collect new completely unseen "
            "out-of-sample draws and test "
            "the frozen strategies."
        )

    print()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR
        / "adaptive_meta_final_validation.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    fold_output = (
        PROCESSED_DIR
        / "adaptive_meta_final_fold_matrix.csv"
    )

    matrix.to_csv(
        fold_output
    )

    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(output_path)
    print(fold_output)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()