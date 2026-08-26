import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/number_features_v3.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lottery parameters
NUMBERS = 49
NUMBERS_PER_DRAW = 6

# Windows to test
FREQ_WINDOWS = [5, 10, 20, 50, 100, 200]

# Walk-forward folds
N_FOLDS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RECENCY EFFECT TEST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["date", "number"]).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")

required_columns = [
    "date",
    "number",
    "target",
    "gap",
    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "freq_100",
    "freq_200",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("\nAll required columns found.")


# ============================================================
# DRAW-LEVEL DATA
# ============================================================

draws = (
    df.groupby("date")["number"]
    .apply(lambda x: set(x.astype(int)))
    .reset_index()
)

draws.columns = ["date", "numbers"]

draws = draws.sort_values("date").reset_index(drop=True)

print(f"Total draws: {len(draws)}")


# ============================================================
# BASIC RECENCY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BASIC GAP ANALYSIS")
print("=" * 70)

gap_stats = (
    df.groupby("gap")["target"]
    .agg(
        count="count",
        positives="sum",
        hit_rate="mean",
    )
    .reset_index()
)

gap_stats["hit_rate_percent"] = gap_stats["hit_rate"] * 100

print("\nGap statistics:")
print(gap_stats.head(30).to_string(index=False))


# ============================================================
# GAP BUCKETS
# ============================================================

def gap_bucket(gap):
    if gap <= 1:
        return "1"
    elif gap <= 2:
        return "2"
    elif gap <= 3:
        return "3"
    elif gap <= 5:
        return "4-5"
    elif gap <= 10:
        return "6-10"
    elif gap <= 20:
        return "11-20"
    elif gap <= 50:
        return "21-50"
    elif gap <= 100:
        return "51-100"
    else:
        return "101+"


df["gap_bucket"] = df["gap"].apply(gap_bucket)

bucket_order = [
    "1",
    "2",
    "3",
    "4-5",
    "6-10",
    "11-20",
    "21-50",
    "51-100",
    "101+",
]

bucket_stats = (
    df.groupby("gap_bucket", observed=False)["target"]
    .agg(
        samples="count",
        positives="sum",
        hit_rate="mean",
    )
    .reindex(bucket_order)
    .reset_index()
)

bucket_stats["hit_rate_percent"] = bucket_stats["hit_rate"] * 100

print("\n" + "=" * 70)
print("GAP BUCKET ANALYSIS")
print("=" * 70)

print(bucket_stats.to_string(index=False))


# ============================================================
# EXPECTED RANDOM RATE
# ============================================================

random_rate = NUMBERS_PER_DRAW / NUMBERS

print("\nRandom expected probability:")
print(f"{random_rate:.6f}")
print(f"{random_rate * 100:.4f}%")


# ============================================================
# RECENCY VS RANDOM
# ============================================================

bucket_stats["difference"] = (
    bucket_stats["hit_rate"] - random_rate
)

bucket_stats["difference_percent"] = (
    bucket_stats["difference"] / random_rate * 100
)

print("\n" + "=" * 70)
print("GAP EFFECT VS RANDOM")
print("=" * 70)

print(
    bucket_stats[
        [
            "gap_bucket",
            "hit_rate_percent",
            "difference",
            "difference_percent",
        ]
    ].to_string(index=False)
)


# ============================================================
# FREQUENCY WINDOW ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("RECENCY / FREQUENCY WINDOW ANALYSIS")
print("=" * 70)

frequency_results = []

for window in FREQ_WINDOWS:

    col = f"freq_{window}"

    stats = (
        df.groupby(col)["target"]
        .agg(
            samples="count",
            positives="sum",
            hit_rate="mean",
        )
        .reset_index()
    )

    stats["window"] = window

    weighted_hit_rate = (
        df["target"].mean()
    )

    frequency_results.append(
        {
            "window": window,
            "mean_frequency": df[col].mean(),
            "median_frequency": df[col].median(),
            "max_frequency": df[col].max(),
            "overall_hit_rate": weighted_hit_rate,
        }
    )

frequency_results = pd.DataFrame(frequency_results)

print(
    frequency_results.to_string(index=False)
)


# ============================================================
# FREQUENCY BUCKET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FREQUENCY EFFECT")
print("=" * 70)

frequency_bucket_results = []

for window in FREQ_WINDOWS:

    col = f"freq_{window}"

    temp = df.copy()

    # Quantile bins
    try:
        temp["freq_bucket"] = pd.qcut(
            temp[col],
            q=5,
            duplicates="drop",
        )
    except Exception:
        continue

    grouped = (
        temp.groupby("freq_bucket", observed=False)["target"]
        .agg(
            samples="count",
            positives="sum",
            hit_rate="mean",
        )
        .reset_index()
    )

    grouped["window"] = window
    grouped["difference"] = (
        grouped["hit_rate"] - random_rate
    )

    grouped["difference_percent"] = (
        grouped["difference"] / random_rate * 100
    )

    frequency_bucket_results.append(grouped)

frequency_bucket_results = pd.concat(
    frequency_bucket_results,
    ignore_index=True,
)

frequency_bucket_results["hit_rate_percent"] = (
    frequency_bucket_results["hit_rate"] * 100
)

print(
    frequency_bucket_results[
        [
            "window",
            "freq_bucket",
            "samples",
            "hit_rate_percent",
            "difference_percent",
        ]
    ].to_string(index=False)
)


# ============================================================
# HOT / COLD TEST
# ============================================================

print("\n" + "=" * 70)
print("HOT / COLD NUMBER TEST")
print("=" * 70)

hot_cold_results = []

for window in FREQ_WINDOWS:

    col = f"freq_{window}"

    # For each draw, divide numbers into:
    # bottom 20% = cold
    # middle 60%
    # top 20% = hot

    temp = df.copy()

    temp["rank_percentile"] = (
        temp.groupby("date")[col]
        .rank(
            pct=True,
            method="average",
        )
    )

    temp["category"] = np.select(
        [
            temp["rank_percentile"] <= 0.20,
            temp["rank_percentile"] >= 0.80,
        ],
        [
            "cold",
            "hot",
        ],
        default="middle",
    )

    grouped = (
        temp.groupby("category")["target"]
        .agg(
            samples="count",
            positives="sum",
            hit_rate="mean",
        )
        .reset_index()
    )

    grouped["window"] = window

    grouped["hit_rate_percent"] = (
        grouped["hit_rate"] * 100
    )

    grouped["difference_percent"] = (
        (grouped["hit_rate"] - random_rate)
        / random_rate
        * 100
    )

    hot_cold_results.append(grouped)

hot_cold_results = pd.concat(
    hot_cold_results,
    ignore_index=True,
)

print(
    hot_cold_results[
        [
            "window",
            "category",
            "samples",
            "hit_rate_percent",
            "difference_percent",
        ]
    ].to_string(index=False)
)


# ============================================================
# WALK-FORWARD RECENCY TEST
# ============================================================

print("\n" + "=" * 70)
print("WALK-FORWARD RECENCY TEST")
print("=" * 70)

n_draws = len(draws)

fold_size = n_draws // (N_FOLDS + 1)

walk_forward_results = []

for fold in range(N_FOLDS):

    train_end = fold_size * (fold + 1)
    test_end = fold_size * (fold + 2)

    if fold == N_FOLDS - 1:
        test_end = n_draws

    train_dates = draws.iloc[:train_end]["date"]

    test_dates = draws.iloc[train_end:test_end]["date"]

    if len(test_dates) == 0:
        continue

    train_end_date = train_dates.iloc[-1]
    test_start_date = test_dates.iloc[0]
    test_end_date = test_dates.iloc[-1]

    print("\n" + "-" * 70)
    print(f"Fold {fold + 1}")

    print(
        f"Training: {train_dates.iloc[0]} -> {train_end_date}"
    )

    print(
        f"Testing:  {test_start_date} -> {test_end_date}"
    )

    test_df = df[
        df["date"].isin(test_dates)
    ].copy()

    print(f"Test samples: {len(test_df)}")

    # Overall hit rate
    overall_rate = test_df["target"].mean()

    print(
        f"Overall hit rate: {overall_rate:.6f}"
    )

    print(
        f"Difference from random: "
        f"{overall_rate - random_rate:+.6f}"
    )

    # Gap correlation
    gap_corr = test_df[
        ["gap", "target"]
    ].corr().iloc[0, 1]

    print(
        f"Gap / target correlation: {gap_corr:.6f}"
    )

    # Frequency correlations
    for window in FREQ_WINDOWS:

        col = f"freq_{window}"

        corr = test_df[
            [col, "target"]
        ].corr().iloc[0, 1]

        print(
            f"{col:10s} correlation: {corr:+.6f}"
        )

    walk_forward_results.append(
        {
            "fold": fold + 1,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "test_samples": len(test_df),
            "hit_rate": overall_rate,
            "random_expected": random_rate,
            "difference": overall_rate - random_rate,
            "gap_correlation": gap_corr,
        }
    )


walk_forward_results = pd.DataFrame(
    walk_forward_results
)


# ============================================================
# PER-FOLD CORRELATION TABLE
# ============================================================

print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

if len(walk_forward_results) > 0:

    print(
        walk_forward_results.to_string(index=False)
    )

    print("\nOverall:")
    print(
        f"Mean hit rate: "
        f"{walk_forward_results['hit_rate'].mean():.6f}"
    )

    print(
        f"Mean difference: "
        f"{walk_forward_results['difference'].mean():+.6f}"
    )

    print(
        f"Mean gap correlation: "
        f"{walk_forward_results['gap_correlation'].mean():+.6f}"
    )


# ============================================================
# NUMBER-LEVEL RECENCY EFFECT
# ============================================================

print("\n" + "=" * 70)
print("NUMBER-LEVEL RECENCY EFFECT")
print("=" * 70)

number_stats = (
    df.groupby("number")
    .agg(
        mean_gap=("gap", "mean"),
        median_gap=("gap", "median"),
        mean_freq_5=("freq_5", "mean"),
        mean_freq_20=("freq_20", "mean"),
        mean_freq_100=("freq_100", "mean"),
        hit_rate=("target", "mean"),
        appearances=("target", "sum"),
    )
    .reset_index()
)

number_stats["difference_from_random"] = (
    number_stats["hit_rate"] - random_rate
)

number_stats["difference_percent"] = (
    number_stats["difference_from_random"]
    / random_rate
    * 100
)

print(
    number_stats.sort_values(
        "hit_rate",
        ascending=False,
    ).to_string(index=False)
)


# ============================================================
# TOP / BOTTOM NUMBERS
# ============================================================

print("\nTop 10 numbers by hit rate:")
print(
    number_stats
    .sort_values("hit_rate", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nBottom 10 numbers by hit rate:")
print(
    number_stats
    .sort_values("hit_rate", ascending=True)
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

gap_stats.to_csv(
    f"{OUTPUT_DIR}/recency_gap_stats.csv",
    index=False,
)

bucket_stats.to_csv(
    f"{OUTPUT_DIR}/recency_gap_buckets.csv",
    index=False,
)

frequency_results.to_csv(
    f"{OUTPUT_DIR}/recency_frequency_summary.csv",
    index=False,
)

frequency_bucket_results.to_csv(
    f"{OUTPUT_DIR}/recency_frequency_buckets.csv",
    index=False,
)

hot_cold_results.to_csv(
    f"{OUTPUT_DIR}/recency_hot_cold.csv",
    index=False,
)

walk_forward_results.to_csv(
    f"{OUTPUT_DIR}/recency_walk_forward.csv",
    index=False,
)

number_stats.to_csv(
    f"{OUTPUT_DIR}/recency_number_stats.csv",
    index=False,
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL RECENCY ANALYSIS")
print("=" * 70)

mean_difference = (
    walk_forward_results["difference"].mean()
    if len(walk_forward_results) > 0
    else np.nan
)

mean_gap_corr = (
    walk_forward_results["gap_correlation"].mean()
    if len(walk_forward_results) > 0
    else np.nan
)

print(
    f"\nRandom expected hit rate: {random_rate:.6f}"
)

print(
    f"Overall target rate:      {df['target'].mean():.6f}"
)

print(
    f"Mean walk-forward difference: "
    f"{mean_difference:+.6f}"
)

print(
    f"Mean gap correlation: "
    f"{mean_gap_corr:+.6f}"
)

if abs(mean_difference) < 0.01:
    print(
        "\nConclusion: "
        "No strong recency effect detected."
    )
else:
    print(
        "\nConclusion: "
        "A possible recency effect exists "
        "and requires further statistical testing."
    )

print("\nResults saved to:")
print(f"{OUTPUT_DIR}/recency_gap_stats.csv")
print(f"{OUTPUT_DIR}/recency_gap_buckets.csv")
print(f"{OUTPUT_DIR}/recency_frequency_summary.csv")
print(f"{OUTPUT_DIR}/recency_frequency_buckets.csv")
print(f"{OUTPUT_DIR}/recency_hot_cold.csv")
print(f"{OUTPUT_DIR}/recency_walk_forward.csv")
print(f"{OUTPUT_DIR}/recency_number_stats.csv")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)