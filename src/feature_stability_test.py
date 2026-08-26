import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/number_features_v3.csv"
OUTPUT_PATH = "data/processed/feature_stability_results.csv"

RANDOM_SEED = 42

# عدد السحوبات في كل فترة اختبار
TEST_DRAWS = 483

# ============================================================
# FEATURES
# ============================================================

FEATURE_GROUPS = {
    "frequency": [
        "freq_5",
        "freq_10",
        "freq_20",
        "freq_50",
        "freq_100",
        "freq_200",
        "gap",
        "recent_vs_long",
    ],

    "previous_draw": [
        "previous_draw_sum",
        "previous_draw_mean",
        "previous_draw_std",
        "previous_draw_range",
        "previous_odd_count",
        "previous_even_count",
        "previous_consecutive_pairs",
        "previous_consecutive_max",
    ],

    "calendar": [
        "year",
        "month",
        "day_of_week",
        "draw_index",
    ],
}

FEATURES = (
    FEATURE_GROUPS["frequency"]
    + FEATURE_GROUPS["previous_draw"]
    + FEATURE_GROUPS["calendar"]
)

# ============================================================
# RANDOM EXPECTED
# ============================================================

RANDOM_EXPECTED = 6 * 6 / 49


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE STABILITY TEST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["date", "number"]).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    raise ValueError(
        f"Missing features in dataset: {missing_features}"
    )

print("\nAll required features found.")


# ============================================================
# DRAW INFORMATION
# ============================================================

dates = sorted(df["date"].unique())

print(f"Total draws: {len(dates)}")

if len(dates) < TEST_DRAWS * 6:
    raise ValueError("Not enough draws for stability test.")


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

def create_model(scale_pos_weight):

    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


# ============================================================
# TOP-6 BACKTEST
# ============================================================

def calculate_top6_hits(test_df, probabilities):

    temp = test_df[["date", "number", "target"]].copy()

    temp["probability"] = probabilities

    results = []

    for date, group in temp.groupby("date"):

        group = group.sort_values(
            "probability",
            ascending=False
        )

        top6 = group.head(6)["number"].tolist()

        actual = group.loc[
            group["target"] == 1,
            "number"
        ].tolist()

        hits = len(set(top6) & set(actual))

        results.append({
            "date": date,
            "top6": top6,
            "actual": actual,
            "hits": hits
        })

    result_df = pd.DataFrame(results)

    return result_df


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, features):

    importance = model.feature_importances_

    result = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    result = result.sort_values(
        "importance",
        ascending=False
    )

    return result


# ============================================================
# DEFINE WALK-FORWARD FOLDS
# ============================================================

print("\n" + "=" * 70)
print("CREATING WALK-FORWARD FOLDS")
print("=" * 70)

folds = []

# Start testing after enough historical data exists.
# We use 5 equally sized test periods.

available_dates = dates

total_dates = len(available_dates)

# Leave enough historical data for the first training period.
first_test_start = total_dates - TEST_DRAWS * 5

if first_test_start < 500:
    first_test_start = 500

for fold in range(5):

    test_start_idx = first_test_start + fold * TEST_DRAWS
    test_end_idx = test_start_idx + TEST_DRAWS

    if test_end_idx > total_dates:
        break

    train_dates = available_dates[:test_start_idx]
    test_dates = available_dates[test_start_idx:test_end_idx]

    folds.append({
        "fold": fold + 1,
        "train_dates": train_dates,
        "test_dates": test_dates
    })


# ============================================================
# RESULTS
# ============================================================

fold_results = []

importance_results = []

# ============================================================
# RUN FOLDS
# ============================================================

for fold_info in folds:

    fold = fold_info["fold"]

    train_dates = fold_info["train_dates"]
    test_dates = fold_info["test_dates"]

    train_start = train_dates[0]
    train_end = train_dates[-1]

    test_start = test_dates[0]
    test_end = test_dates[-1]

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    print(
        f"Training: {train_start} -> {train_end}"
    )

    print(
        f"Testing:  {test_start} -> {test_end}"
    )

    train_mask = df["date"].isin(train_dates)
    test_mask = df["date"].isin(test_dates)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # --------------------------------------------------------
    # CLASS BALANCE
    # --------------------------------------------------------

    positives = y_train.sum()
    negatives = len(y_train) - positives

    scale_pos_weight = negatives / positives

    print(f"Positive samples: {positives}")
    print(f"Negative samples: {negatives}")
    print(f"Scale pos weight: {scale_pos_weight:.6f}")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = create_model(scale_pos_weight)

    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_prob = model.predict_proba(
        X_train
    )[:, 1]

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    train_auc = roc_auc_score(
        y_train,
        train_prob
    )

    test_auc = roc_auc_score(
        y_test,
        test_prob
    )

    # --------------------------------------------------------
    # TOP-6
    # --------------------------------------------------------

    top6_results = calculate_top6_hits(
        test_df,
        test_prob
    )

    average_hits = top6_results["hits"].mean()
    total_hits = top6_results["hits"].sum()
    maximum_hits = top6_results["hits"].max()

    difference = average_hits - RANDOM_EXPECTED

    difference_percent = (
        difference / RANDOM_EXPECTED
    ) * 100

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nResults")
    print("-" * 40)

    print(f"Training ROC-AUC: {train_auc:.6f}")
    print(f"Test ROC-AUC:     {test_auc:.6f}")

    print(f"Average hits:     {average_hits:.6f}")
    print(f"Total hits:       {total_hits}")
    print(f"Maximum hits:     {maximum_hits}")

    print(f"Random expected:  {RANDOM_EXPECTED:.6f}")

    print(f"Difference:       {difference:+.6f}")
    print(f"Difference %:     {difference_percent:+.3f}%")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = get_feature_importance(
        model,
        FEATURES
    )

    print("\nTop 10 Features")
    print("-" * 40)

    print(
        importance.head(10).to_string(index=False)
    )

    # Save importance

    for _, row in importance.iterrows():

        importance_results.append({
            "fold": fold,
            "feature": row["feature"],
            "importance": row["importance"]
        })

    # --------------------------------------------------------
    # FOLD SUMMARY
    # --------------------------------------------------------

    fold_results.append({
        "fold": fold,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_draws": len(train_dates),
        "test_draws": len(test_dates),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "average_hits": average_hits,
        "total_hits": total_hits,
        "maximum_hits": maximum_hits,
        "random_expected": RANDOM_EXPECTED,
        "difference": difference,
        "difference_percent": difference_percent
    })


# ============================================================
# RESULTS DATAFRAMES
# ============================================================

results_df = pd.DataFrame(fold_results)

importance_df = pd.DataFrame(
    importance_results
)


# ============================================================
# FEATURE STABILITY
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE STABILITY")
print("=" * 70)

importance_pivot = importance_df.pivot(
    index="feature",
    columns="fold",
    values="importance"
)

importance_pivot.columns = [
    f"fold_{c}"
    for c in importance_pivot.columns
]

importance_pivot["mean_importance"] = (
    importance_pivot.mean(axis=1)
)

importance_pivot["std_importance"] = (
    importance_pivot.std(axis=1)
)

importance_pivot["cv"] = (
    importance_pivot["std_importance"]
    / importance_pivot["mean_importance"]
)

importance_pivot = importance_pivot.sort_values(
    "mean_importance",
    ascending=False
)

print(
    importance_pivot.to_string()
)


# ============================================================
# FEATURE RANK STABILITY
# ============================================================

rank_df = importance_df.copy()

rank_df["rank"] = rank_df.groupby(
    "fold"
)["importance"].rank(
    ascending=False,
    method="average"
)

rank_pivot = rank_df.pivot(
    index="feature",
    columns="fold",
    values="rank"
)

rank_pivot.columns = [
    f"fold_{c}"
    for c in rank_pivot.columns
]

rank_pivot["mean_rank"] = rank_pivot.mean(axis=1)

rank_pivot["std_rank"] = rank_pivot.std(axis=1)

rank_pivot = rank_pivot.sort_values(
    "mean_rank"
)

print("\n")
print("=" * 70)
print("FEATURE RANK STABILITY")
print("=" * 70)

print(
    rank_pivot.to_string()
)


# ============================================================
# SPEARMAN CORRELATION BETWEEN FOLDS
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANCE CORRELATION BETWEEN FOLDS")
print("=" * 70)

correlation_matrix = importance_pivot[
    [c for c in importance_pivot.columns if c.startswith("fold_")]
].T.corr(method="spearman")

print(
    correlation_matrix.to_string()
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

print(
    results_df[
        [
            "fold",
            "test_draws",
            "train_auc",
            "test_auc",
            "average_hits",
            "random_expected",
            "difference",
            "difference_percent",
            "maximum_hits"
        ]
    ].to_string(index=False)
)

print("\nOverall")
print("-" * 40)

mean_test_auc = results_df["test_auc"].mean()
std_test_auc = results_df["test_auc"].std()

mean_hits = results_df["average_hits"].mean()
std_hits = results_df["average_hits"].std()

mean_difference = results_df["difference"].mean()

print(f"Mean Test ROC-AUC: {mean_test_auc:.6f}")
print(f"Std Test ROC-AUC:  {std_test_auc:.6f}")

print(f"Mean Average Hits: {mean_hits:.6f}")
print(f"Std Average Hits:  {std_hits:.6f}")

print(f"Random Expected:   {RANDOM_EXPECTED:.6f}")

print(
    f"Mean Difference:   {mean_difference:+.6f}"
)


# ============================================================
# BEST / WORST FOLD
# ============================================================

best_fold = results_df.loc[
    results_df["average_hits"].idxmax()
]

worst_fold = results_df.loc[
    results_df["average_hits"].idxmin()
]

print("\n")
print("=" * 70)
print("BEST / WORST FOLD")
print("=" * 70)

print(
    f"Best fold:  {int(best_fold['fold'])}"
)

print(
    f"Average hits: {best_fold['average_hits']:.6f}"
)

print(
    f"Test AUC: {best_fold['test_auc']:.6f}"
)

print()

print(
    f"Worst fold: {int(worst_fold['fold'])}"
)

print(
    f"Average hits: {worst_fold['average_hits']:.6f}"
)

print(
    f"Test AUC: {worst_fold['test_auc']:.6f}"
)


# ============================================================
# TOP STABLE FEATURES
# ============================================================

stable_features = importance_pivot[
    [
        "mean_importance",
        "std_importance",
        "cv"
    ]
].copy()

stable_features = stable_features.sort_values(
    "cv"
)

print("\n")
print("=" * 70)
print("MOST STABLE FEATURES")
print("=" * 70)

print(
    stable_features.head(10).to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

importance_output = (
    "data/processed/"
    "feature_stability_importance.csv"
)

importance_df.to_csv(
    importance_output,
    index=False
)

rank_output = (
    "data/processed/"
    "feature_stability_ranks.csv"
)

rank_pivot.to_csv(
    rank_output
)

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_PATH)
print(importance_output)
print(rank_output)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)