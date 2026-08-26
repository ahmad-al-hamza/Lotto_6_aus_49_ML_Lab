"""
ADAPTIVE META-MODEL TOP-6 WALK-FORWARD TEST

Combines:
    1. Ensemble
    2. Recency
    3. Stability
    4. Diversity

The meta-model learns which strategy performed better in previous
walk-forward folds and adapts the weights for the next fold.

IMPORTANT:
- No future test information is used.
- Strategy selection for fold N uses only folds < N.
- Random baseline = 6/49.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/processed/lotto_6aus49_clean.csv")

RESULT_FILES = {
    "ensemble": Path(
        "data/processed/ensemble_top6_walk_forward_results.csv"
    ),
    "recency": Path(
        "data/processed/recency_top6_walk_forward_results.csv"
    ),
    "stability": Path(
        "data/processed/stability_top6_walk_forward_results.csv"
    ),
    "diversity": Path(
        "data/processed/diversity_top6_walk_forward_results.csv"
    ),
}

OUTPUT_PATH = Path(
    "data/processed/adaptive_meta_top6_walk_forward_results.csv"
)

NUMBERS = list(range(1, 50))
TOP_K = 6

RANDOM_EXPECTED = TOP_K * TOP_K / 49.0


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_dataset():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    number_cols = [f"n{i}" for i in range(1, 7)]

    missing = [c for c in number_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing number columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    return df


def load_strategy_results():
    """
    Load the already calculated walk-forward results.

    Expected columns:
        fold
        average_hits
        random_expected
        difference
        difference_percent
        empirical_p

    The function is intentionally tolerant of small differences
    in column naming.
    """

    results = {}

    for strategy, path in RESULT_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file:\n{path}\n\n"
                f"Run the {strategy} walk-forward script first."
            )

        df = pd.read_csv(path)

        # Normalize column names
        df.columns = [
            c.strip().lower().replace(" ", "_")
            for c in df.columns
        ]

        if "fold" not in df.columns:
            raise ValueError(
                f"{path} does not contain a 'fold' column."
            )

        results[strategy] = df

        print(
            f"Loaded {strategy:10s}: "
            f"{len(df)} fold results"
        )

    return results


# ============================================================
# BUILD PERFORMANCE MATRIX
# ============================================================

def build_performance_matrix(results):
    """
    Creates:

        fold | ensemble | recency | stability | diversity

    Each value represents the strategy's performance
    relative to random expectation.

    Positive = better than random
    Negative = worse than random
    """

    frames = []

    for strategy, df in results.items():

        temp = df[["fold", "difference"]].copy()

        temp = temp.rename(
            columns={"difference": strategy}
        )

        frames.append(temp)

    performance = frames[0]

    for frame in frames[1:]:
        performance = performance.merge(
            frame,
            on="fold",
            how="outer"
        )

    performance = performance.sort_values("fold")

    return performance


# ============================================================
# ADAPTIVE WEIGHTS
# ============================================================

def calculate_adaptive_weights(
    performance,
    current_fold,
    decay=0.80,
):
    """
    Calculate strategy weights using ONLY previous folds.

    More recent folds receive more weight.

    Example:

        previous fold:
            0.80

        two folds ago:
            0.80^2

        three folds ago:
            0.80^3

    Negative performance is penalized.

    If there is not enough history, equal weights are used.
    """

    strategies = [
        "ensemble",
        "recency",
        "stability",
        "diversity",
    ]

    history = performance[
        performance["fold"] < current_fold
    ].copy()

    if history.empty:
        return {
            strategy: 1.0 / len(strategies)
            for strategy in strategies
        }

    # Most recent fold gets highest weight
    history = history.sort_values("fold")

    weighted_scores = {}

    for strategy in strategies:

        values = history[strategy].astype(float).values

        # Ignore missing values
        values = values[~np.isnan(values)]

        if len(values) == 0:
            weighted_scores[strategy] = 0.0
            continue

        weights = np.array([
            decay ** i
            for i in range(len(values) - 1, -1, -1)
        ])

        score = np.sum(values * weights) / np.sum(weights)

        weighted_scores[strategy] = score

    # --------------------------------------------------------
    # Convert performance into positive weights
    # --------------------------------------------------------

    # Shift so all values are positive.
    scores = np.array(
        list(weighted_scores.values()),
        dtype=float
    )

    minimum = scores.min()

    if minimum <= 0:
        scores = scores - minimum + 0.001

    # Avoid one strategy receiving all weight.
    scores = np.maximum(scores, 0.001)

    total = scores.sum()

    weights = scores / total

    return dict(
        zip(strategies, weights)
    )


# ============================================================
# META PERFORMANCE
# ============================================================

def calculate_meta_score(
    performance,
    current_fold,
    weights,
):
    """
    Calculate the expected performance of the adaptive
    strategy for the current fold.

    This is used only for analysis.

    It NEVER uses the current fold to calculate weights.
    """

    row = performance[
        performance["fold"] == current_fold
    ]

    if row.empty:
        return np.nan

    row = row.iloc[0]

    score = 0.0

    for strategy, weight in weights.items():
        value = row[strategy]

        if pd.notna(value):
            score += weight * value

    return score


# ============================================================
# BEST STRATEGY
# ============================================================

def choose_best_strategy(weights):
    return max(
        weights,
        key=weights.get
    )


# ============================================================
# WALK FORWARD META TEST
# ============================================================

def run_meta_walk_forward(performance):

    print_header(
        "ADAPTIVE META-MODEL WALK-FORWARD TEST"
    )

    folds = sorted(
        performance["fold"].unique()
    )

    records = []

    for fold in folds:

        print_header(f"FOLD {fold}")

        weights = calculate_adaptive_weights(
            performance,
            current_fold=fold,
            decay=0.80,
        )

        selected_strategy = choose_best_strategy(
            weights
        )

        meta_score = calculate_meta_score(
            performance,
            current_fold=fold,
            weights=weights,
        )

        print("Historical strategy weights:")

        for strategy, weight in weights.items():
            print(
                f"  {strategy:10s}: "
                f"{weight:.4f}"
            )

        print()
        print(
            f"Selected strategy: "
            f"{selected_strategy.upper()}"
        )

        print(
            f"Meta expected difference: "
            f"{meta_score:+.6f}"
        )

        # Actual current-fold performances
        row = performance[
            performance["fold"] == fold
        ].iloc[0]

        print()
        print("Current fold performance:")

        for strategy in weights:
            print(
                f"  {strategy:10s}: "
                f"{row[strategy]:+.6f}"
            )

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # We do NOT use current fold performance to choose
        # the strategy.
        #
        # We only use it to evaluate the decision afterwards.
        # ----------------------------------------------------

        selected_difference = row[
            selected_strategy
        ]

        records.append({
            "fold": fold,
            "selected_strategy": selected_strategy,
            "ensemble_weight": weights["ensemble"],
            "recency_weight": weights["recency"],
            "stability_weight": weights["stability"],
            "diversity_weight": weights["diversity"],
            "selected_difference": selected_difference,
            "meta_score": meta_score,
        })

    return pd.DataFrame(records)


# ============================================================
# BASELINE COMPARISON
# ============================================================

def evaluate_meta(meta_results):

    print_header("META-MODEL SUMMARY")

    mean_difference = (
        meta_results["selected_difference"]
        .mean()
    )

    mean_random = 0.0

    print(
        f"Mean adaptive difference: "
        f"{mean_difference:+.6f}"
    )

    print(
        f"Mean random difference: "
        f"{mean_random:+.6f}"
    )

    print(
        f"Mean improvement: "
        f"{mean_difference * 100:+.3f}%"
    )

    above = (
        meta_results["selected_difference"] > 0
    ).sum()

    below = (
        meta_results["selected_difference"] < 0
    ).sum()

    print()
    print("Fold consistency:")
    print(f"Above random: {above}")
    print(f"Below random: {below}")

    print()
    print("Selected strategies:")

    counts = (
        meta_results["selected_strategy"]
        .value_counts()
    )

    for strategy, count in counts.items():
        print(
            f"  {strategy:10s}: {count}"
        )

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    if mean_difference > 0:
        print()
        print(
            "META-MODEL RESULT:"
        )
        print(
            "Adaptive strategy performed above "
            "the random baseline in aggregate."
        )
    else:
        print()
        print(
            "META-MODEL RESULT:"
        )
        print(
            "Adaptive strategy did not outperform "
            "the random baseline."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ADAPTIVE META-MODEL TOP-6 WALK-FORWARD TEST"
    )

    print(
        "Random expected hits:",
        f"{RANDOM_EXPECTED:.6f}"
    )

    # Load raw dataset
    load_dataset()

    print()

    # Load previous strategy results
    results = load_strategy_results()

    print()

    # Build historical performance matrix
    performance = build_performance_matrix(
        results
    )

    print_header(
        "STRATEGY PERFORMANCE MATRIX"
    )

    print(
        performance.to_string(
            index=False
        )
    )

    # Run adaptive walk-forward
    meta_results = run_meta_walk_forward(
        performance
    )

    # Evaluate
    evaluate_meta(
        meta_results
    )

    # Save
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    meta_results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED TO:")
    print(OUTPUT_PATH)
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
