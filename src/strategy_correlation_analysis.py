"""
STRATEGY CORRELATION ANALYSIS

Analyzes whether the existing Top-6 strategies provide
independent information or are largely producing similar results.

Strategies:
- meta_score
- ensemble
- recency
- stability
- diversity

Outputs:
1. Fold-by-fold performance correlation
2. Pearson correlation
3. Spearman correlation
4. Jaccard similarity where Top-6 sets are available
5. Average number of common numbers
6. Strategy performance summary
7. Recommendation for next modeling step
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("data/processed")

STRATEGIES = [
    "meta_score",
    "ensemble",
    "recency",
    "stability",
    "diversity",
]

RANDOM_EXPECTED = 6 / 49


# ============================================================
# FILE LOCATOR
# ============================================================

def find_result_file(strategy):
    """
    Find the most recent result CSV belonging to a strategy.
    """

    candidates = list(RESULTS_DIR.glob(f"*{strategy}*walk_forward*.csv"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_strategy_results():
    """
    Load fold-level performance results.
    """

    results = {}

    print("=" * 70)
    print("LOADING STRATEGY RESULTS")
    print("=" * 70)

    for strategy in STRATEGIES:

        path = find_result_file(strategy)

        if path is None:
            print(f"{strategy:12s}: NOT FOUND")
            continue

        try:
            df = pd.read_csv(path)

            results[strategy] = df

            print(
                f"{strategy:12s}: "
                f"{len(df)} rows -> {path.name}"
            )

        except Exception as exc:
            print(
                f"{strategy:12s}: ERROR -> {exc}"
            )

    print()

    return results


# ============================================================
# DETECT PERFORMANCE COLUMN
# ============================================================

def detect_difference_column(df):
    """
    Detect the column containing performance difference.
    """

    candidates = [
        "difference",
        "adaptive_difference",
        "selected_difference",
        "mean_difference",
        "difference_percent",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def detect_hits_column(df):
    """
    Detect average hits column.
    """

    candidates = [
        "average_hits",
        "adaptive_average_hits",
        "selected_average_hits",
        "mean_hits",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def detect_fold_column(df):
    """
    Detect fold column.
    """

    for column in ["fold", "Fold"]:
        if column in df.columns:
            return column

    return None


# ============================================================
# PERFORMANCE MATRIX
# ============================================================

def build_performance_matrix(results):
    """
    Build:

               strategy A strategy B ...
    fold 1
    fold 2
    ...
    """

    matrix = {}

    for strategy, df in results.items():

        diff_col = detect_difference_column(df)

        if diff_col is None:
            print(
                f"WARNING: no difference column "
                f"found for {strategy}"
            )
            continue

        fold_col = detect_fold_column(df)

        if fold_col is None:
            continue

        temp = df[[fold_col, diff_col]].copy()

        temp[fold_col] = pd.to_numeric(
            temp[fold_col],
            errors="coerce"
        )

        temp[diff_col] = pd.to_numeric(
            temp[diff_col],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[fold_col]
        )

        temp = temp.set_index(fold_col)[diff_col]

        matrix[strategy] = temp

    if not matrix:
        return pd.DataFrame()

    performance = pd.DataFrame(matrix)

    performance.index.name = "fold"

    return performance.sort_index()


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlations(performance):
    """
    Calculate Pearson and Spearman correlations.
    """

    print("=" * 70)
    print("PEARSON CORRELATION")
    print("=" * 70)

    pearson = performance.corr(
        method="pearson",
        min_periods=2
    )

    print(
        pearson.to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    print()

    print("=" * 70)
    print("SPEARMAN CORRELATION")
    print("=" * 70)

    spearman = performance.corr(
        method="spearman",
        min_periods=2
    )

    print(
        spearman.to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    print()

    return pearson, spearman


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def performance_summary(performance):
    """
    Calculate basic statistics for every strategy.
    """

    rows = []

    for strategy in performance.columns:

        values = performance[strategy].dropna()

        if len(values) == 0:
            continue

        rows.append({
            "strategy": strategy,
            "folds": len(values),
            "mean_difference": values.mean(),
            "median_difference": values.median(),
            "std_difference": values.std(ddof=1)
            if len(values) > 1 else np.nan,
            "above_random": int(
                (values > 0).sum()
            ),
            "below_random": int(
                (values < 0).sum()
            ),
            "best_fold": values.max(),
            "worst_fold": values.min(),
        })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            "mean_difference",
            ascending=False
        )

    return summary


# ============================================================
# TOP-6 EXTRACTION
# ============================================================

def find_top6_columns(df):
    """
    Try to detect columns representing selected Top-6 numbers.

    Supported formats include:

    number
    rank
    selected numbers
    top_6
    etc.

    Also attempts to parse rows containing
    textual lists such as:

        [6, 26, 38, 43, 42, 32]
    """

    candidates = [
        "top6",
        "top_6",
        "selected",
        "selected_numbers",
        "numbers",
        "prediction",
        "predicted_numbers",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    return None


def parse_top6(value):
    """
    Convert a Top-6 representation into a set.
    """

    if pd.isna(value):
        return None

    if isinstance(value, (list, tuple, set)):
        values = value

    else:
        text = str(value)

        # Remove brackets
        text = (
            text.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
        )

        parts = [
            p.strip()
            for p in text.split(",")
        ]

        values = []

        for part in parts:

            try:
                values.append(int(float(part)))
            except Exception:
                continue

    values = {
        int(v)
        for v in values
        if 1 <= int(v) <= 49
    }

    if not values:
        return None

    return values


def extract_top6_sets(results):
    """
    Extract Top-6 sets when available.
    """

    top6 = {}

    print("=" * 70)
    print("TOP-6 EXTRACTION")
    print("=" * 70)

    for strategy, df in results.items():

        column = find_top6_columns(df)

        if column is None:
            print(
                f"{strategy:12s}: "
                f"Top-6 column not found"
            )
            continue

        fold_col = detect_fold_column(df)

        if fold_col is None:
            continue

        strategy_sets = {}

        for _, row in df.iterrows():

            fold = row[fold_col]

            parsed = parse_top6(
                row[column]
            )

            if parsed is not None:
                strategy_sets[int(fold)] = parsed

        if strategy_sets:

            top6[strategy] = strategy_sets

            print(
                f"{strategy:12s}: "
                f"{len(strategy_sets)} folds"
            )

        else:

            print(
                f"{strategy:12s}: "
                f"no valid Top-6 data"
            )

    print()

    return top6


# ============================================================
# JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(a, b):
    """
    Jaccard similarity:

        intersection / union
    """

    if not a or not b:
        return np.nan

    union = a | b

    if not union:
        return np.nan

    return len(a & b) / len(union)


# ============================================================
# TOP-6 OVERLAP ANALYSIS
# ============================================================

def analyze_top6_overlap(top6):
    """
    Calculate pairwise Top-6 overlap.
    """

    if len(top6) < 2:
        print(
            "Not enough Top-6 data "
            "for overlap analysis."
        )
        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    strategies = list(top6.keys())

    jaccard_rows = []

    common_rows = []

    for a, b in itertools.combinations(
        strategies,
        2
    ):

        common_jaccard = []
        common_count = []

        common_folds = sorted(
            set(top6[a])
            & set(top6[b])
        )

        for fold in common_folds:

            set_a = top6[a][fold]
            set_b = top6[b][fold]

            common_jaccard.append(
                jaccard_similarity(
                    set_a,
                    set_b
                )
            )

            common_count.append(
                len(set_a & set_b)
            )

        if not common_folds:
            continue

        jaccard_rows.append({
            "strategy_a": a,
            "strategy_b": b,
            "folds": len(common_folds),
            "mean_jaccard": np.mean(
                common_jaccard
            ),
            "max_jaccard": np.max(
                common_jaccard
            ),
            "min_jaccard": np.min(
                common_jaccard
            ),
        })

        common_rows.append({
            "strategy_a": a,
            "strategy_b": b,
            "folds": len(common_folds),
            "mean_common_numbers": np.mean(
                common_count
            ),
            "max_common_numbers": np.max(
                common_count
            ),
            "min_common_numbers": np.min(
                common_count
            ),
        })

    jaccard_df = pd.DataFrame(
        jaccard_rows
    )

    common_df = pd.DataFrame(
        common_rows
    )

    return jaccard_df, common_df


# ============================================================
# INFORMATION DIVERSITY SCORE
# ============================================================

def calculate_information_diversity(
    correlation,
    strategy_names
):
    """
    Estimate how much independent information
    each strategy provides.

    Higher = less correlated with the others.
    """

    rows = []

    for strategy in strategy_names:

        correlations = []

        for other in strategy_names:

            if other == strategy:
                continue

            if (
                strategy in correlation.index
                and other in correlation.columns
            ):

                value = correlation.loc[
                    strategy,
                    other
                ]

                if pd.notna(value):
                    correlations.append(
                        abs(value)
                    )

        if correlations:

            mean_abs_corr = np.mean(
                correlations
            )

            diversity = 1.0 - mean_abs_corr

        else:

            mean_abs_corr = np.nan
            diversity = np.nan

        rows.append({
            "strategy": strategy,
            "mean_absolute_correlation":
                mean_abs_corr,
            "information_diversity":
                diversity,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            "information_diversity",
            ascending=False
        )

    return df


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_recommendation(
    summary,
    diversity
):
    """
    Generate a conservative recommendation.
    """

    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if summary.empty:

        print(
            "Insufficient data for recommendation."
        )

        return

    best = summary.iloc[0]

    print(
        f"Best mean strategy: "
        f"{best['strategy']}"
    )

    print(
        f"Mean difference: "
        f"{best['mean_difference']:+.6f}"
    )

    print()

    if not diversity.empty:

        most_independent = diversity.iloc[0]

        print(
            "Most information-diverse strategy: "
            f"{most_independent['strategy']}"
        )

        print(
            "Information diversity: "
            f"{most_independent['information_diversity']:.4f}"
        )

    print()

    positive = summary[
        summary["mean_difference"] > 0
    ]

    if len(positive) == 0:

        print(
            "No strategy shows a positive "
            "mean advantage."
        )

        print(
            "Do NOT build another meta-model yet."
        )

    elif len(positive) == 1:

        print(
            "Only one strategy has a positive "
            "mean advantage."
        )

        print(
            "A V4 ensemble should NOT be assumed "
            "to improve performance."
        )

        print(
            "First verify the result on an "
            "additional untouched holdout period."
        )

    else:

        print(
            f"{len(positive)} strategies have "
            "positive mean differences."
        )

        print(
            "A V4 model may be justified, "
            "but only if the strategies provide "
            "sufficiently independent information."
        )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    performance,
    summary,
    pearson,
    spearman,
    diversity,
    jaccard,
    common
):
    """
    Save all analysis outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    performance.to_csv(
        RESULTS_DIR /
        "strategy_correlation_performance_matrix.csv"
    )

    summary.to_csv(
        RESULTS_DIR /
        "strategy_performance_summary.csv",
        index=False
    )

    pearson.to_csv(
        RESULTS_DIR /
        "strategy_pearson_correlation.csv"
    )

    spearman.to_csv(
        RESULTS_DIR /
        "strategy_spearman_correlation.csv"
    )

    diversity.to_csv(
        RESULTS_DIR /
        "strategy_information_diversity.csv",
        index=False
    )

    if not jaccard.empty:

        jaccard.to_csv(
            RESULTS_DIR /
            "strategy_top6_jaccard.csv",
            index=False
        )

    if not common.empty:

        common.to_csv(
            RESULTS_DIR /
            "strategy_top6_overlap.csv",
            index=False
        )

    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        RESULTS_DIR /
        "strategy_performance_summary.csv"
    )

    print(
        RESULTS_DIR /
        "strategy_pearson_correlation.csv"
    )

    print(
        RESULTS_DIR /
        "strategy_spearman_correlation.csv"
    )

    print(
        RESULTS_DIR /
        "strategy_information_diversity.csv"
    )

    if not jaccard.empty:

        print(
            RESULTS_DIR /
            "strategy_top6_jaccard.csv"
        )

    if not common.empty:

        print(
            RESULTS_DIR /
            "strategy_top6_overlap.csv"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STRATEGY CORRELATION ANALYSIS")
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    results = load_strategy_results()

    if len(results) < 2:

        print(
            "ERROR: At least two strategy "
            "result files are required."
        )

        return

    # --------------------------------------------------------
    # Performance matrix
    # --------------------------------------------------------

    performance = build_performance_matrix(
        results
    )

    print("=" * 70)
    print("STRATEGY PERFORMANCE MATRIX")
    print("=" * 70)

    print(
        performance.to_string(
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    print()

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    pearson, spearman = calculate_correlations(
        performance
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = performance_summary(
        performance
    )

    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    print()

    # --------------------------------------------------------
    # Information diversity
    # --------------------------------------------------------

    diversity = calculate_information_diversity(
        pearson,
        list(performance.columns)
    )

    print("=" * 70)
    print("INFORMATION DIVERSITY")
    print("=" * 70)

    print(
        diversity.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()

    # --------------------------------------------------------
    # Top-6 overlap
    # --------------------------------------------------------

    top6 = extract_top6_sets(
        results
    )

    jaccard, common = analyze_top6_overlap(
        top6
    )

    if not jaccard.empty:

        print("=" * 70)
        print("TOP-6 JACCARD SIMILARITY")
        print("=" * 70)

        print(
            jaccard.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        print()

    if not common.empty:

        print("=" * 70)
        print("TOP-6 COMMON NUMBERS")
        print("=" * 70)

        print(
            common.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        print()

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    generate_recommendation(
        summary,
        diversity
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        performance,
        summary,
        pearson,
        spearman,
        diversity,
        jaccard,
        common
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()