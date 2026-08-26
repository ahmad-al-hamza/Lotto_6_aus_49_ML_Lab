import pandas as pd
import numpy as np

from xgboost import XGBClassifier


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/processed/number_features_v3.csv"

SPLIT_DATE = "2017-05-06"

RANDOM_STATE = 42

TOP_N = 6


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:")
print(df.shape)

print()


# ============================================================
# 3. Features
# ============================================================

FEATURES = [
    "number",

    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "freq_100",
    "freq_200",

    "gap",
    "recent_vs_long",

    "previous_draw_sum",
    "previous_draw_mean",
    "previous_draw_std",
    "previous_draw_range",

    "previous_odd_count",
    "previous_even_count",

    "previous_consecutive_pairs",
    "previous_consecutive_max",

    "draw_index",

    "year",
    "month",
    "day_of_week",
]

TARGET = "target"


# ============================================================
# 4. Clean data
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()


# ============================================================
# 5. Time-based split
# ============================================================

split_date = pd.Timestamp(SPLIT_DATE)

train_df = df[
    df["date"] < split_date
].copy()

test_df = df[
    df["date"] >= split_date
].copy()


print("Split date:", split_date)

print(
    "Training dates:",
    train_df["date"].min(),
    "→",
    train_df["date"].max()
)

print(
    "Testing dates:",
    test_df["date"].min(),
    "→",
    test_df["date"].max()
)

print()

print(
    "Training draws:",
    train_df["date"].nunique()
)

print(
    "Test draws:",
    test_df["date"].nunique()
)

print()


# ============================================================
# 6. Training data
# ============================================================

X_train = train_df[FEATURES]

y_train = train_df[TARGET]


# ============================================================
# 7. Class imbalance
# ============================================================

positive = y_train.sum()

negative = len(y_train) - positive

scale_pos_weight = negative / positive

print("Positive samples:", positive)

print("Negative samples:", negative)

print(
    "Scale pos weight:",
    scale_pos_weight
)

print()


# ============================================================
# 8. XGBoost model
# ============================================================

model = XGBClassifier(
    n_estimators=500,

    max_depth=5,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=5,

    gamma=0,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# 9. Train
# ============================================================

print("Training XGBoost V3...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")

print()


# ============================================================
# 10. Predict probabilities
# ============================================================

test_df = test_df.copy()

test_df["probability"] = model.predict_proba(
    test_df[FEATURES]
)[:, 1]


# ============================================================
# 11. Generate Top-6 predictions
# ============================================================

results = []

test_dates = sorted(
    test_df["date"].unique()
)

print("Generating Top-6 predictions...")

print()


for date in test_dates:

    day_df = test_df[
        test_df["date"] == date
    ].copy()

    # Sort by predicted probability
    day_df = day_df.sort_values(
        "probability",
        ascending=False
    )

    # Top 6 numbers
    top6 = (
        day_df
        .head(TOP_N)["number"]
        .astype(int)
        .tolist()
    )

    # Actual numbers
    actual = (
        day_df[
            day_df["target"] == 1
        ]["number"]
        .astype(int)
        .tolist()
    )

    # Calculate hits
    hits = len(
        set(top6) & set(actual)
    )

    results.append(
        {
            "date": date,
            "top6": top6,
            "actual": actual,
            "hits": hits
        }
    )


results_df = pd.DataFrame(results)


# ============================================================
# 12. Backtest statistics
# ============================================================

total_hits = results_df["hits"].sum()

average_hits = results_df["hits"].mean()

maximum_hits = results_df["hits"].max()


print("=" * 50)

print("XGBoost V3 Top-6 Backtest")

print("=" * 50)

print(
    "Test draws:",
    len(results_df)
)

print(
    "Average hits:",
    average_hits
)

print(
    "Total hits:",
    total_hits
)

print(
    "Maximum hits:",
    maximum_hits
)

print()


# ============================================================
# 13. Hit distribution
# ============================================================

hit_distribution = (
    results_df["hits"]
    .value_counts()
    .sort_index()
)


print("Hit distribution")

print("-" * 50)

print(hit_distribution)

print()


# ============================================================
# 14. Hit percentages
# ============================================================

hit_percentages = (
    results_df["hits"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


print("Hit percentages")

print("-" * 50)

print(hit_percentages)

print()


# ============================================================
# 15. Random expected value
# ============================================================

random_expected = (
    TOP_N * TOP_N / 49
)


print("Expected random hits:")

print(random_expected)

print()


# ============================================================
# 16. Difference from random
# ============================================================

difference = (
    average_hits - random_expected
)

percentage_difference = (
    difference / random_expected
) * 100


print("Difference from random:")

print(difference)

print(
    "Percentage difference:",
    percentage_difference,
    "%"
)

print()


# ============================================================
# 17. First predictions
# ============================================================

print("First 20 predictions")

print("-" * 50)

print(
    results_df.head(20).to_string(
        index=False
    )
)

print()


# ============================================================
# 18. Best predictions
# ============================================================

print("Highest hit predictions")

print("-" * 50)

best_predictions = (
    results_df
    .sort_values(
        "hits",
        ascending=False
    )
    .head(20)
)

print(
    best_predictions.to_string(
        index=False
    )
)

print()


# ============================================================
# 19. Save results
# ============================================================

OUTPUT_PATH = (
    "data/processed/"
    "xgboost_v3_top6_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("Results saved to:")

print(OUTPUT_PATH)

print()

print("Done.")