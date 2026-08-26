import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/number_features_v3.csv"

SPLIT_DATE = "2017-05-06"

RANDOM_STATE = 42

# XGBoost parameters
N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


# ============================================================
# Feature Groups
# ============================================================

V2_FEATURES = [
    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "freq_100",
    "freq_200",
    "gap",
    "recent_vs_long",
]

PREVIOUS_DRAW_FEATURES = [
    "previous_draw_sum",
    "previous_draw_mean",
    "previous_draw_std",
    "previous_draw_range",
    "previous_odd_count",
    "previous_even_count",
    "previous_consecutive_pairs",
    "previous_consecutive_max",
]

CALENDAR_FEATURES = [
    "year",
    "month",
    "day_of_week",
    "draw_index",
]


# Four experiments
EXPERIMENTS = {
    "V2 Baseline": V2_FEATURES,

    "V2 + Previous Draw":
        V2_FEATURES + PREVIOUS_DRAW_FEATURES,

    "V2 + Calendar":
        V2_FEATURES + CALENDAR_FEATURES,

    "V2 + Previous Draw + Calendar":
        V2_FEATURES + PREVIOUS_DRAW_FEATURES + CALENDAR_FEATURES,
}


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("ABLATION TEST")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")

print(f"Date range:")
print(f"{df['date'].min()} -> {df['date'].max()}")


# ============================================================
# Train / Test Split
# ============================================================

split_date = pd.Timestamp(SPLIT_DATE)

train_df = df[df["date"] < split_date].copy()

test_df = df[df["date"] >= split_date].copy()

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Split date: {split_date}")

print(
    f"Training dates: "
    f"{train_df['date'].min()} -> {train_df['date'].max()}"
)

print(
    f"Testing dates: "
    f"{test_df['date'].min()} -> {test_df['date'].max()}"
)

print(f"Training draws: {train_df['date'].nunique()}")
print(f"Test draws: {test_df['date'].nunique()}")

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")


# ============================================================
# Random expected hits
# ============================================================

RANDOM_EXPECTED = 6 * 6 / 49

print(f"\nRandom expected Top-6 hits: {RANDOM_EXPECTED:.6f}")


# ============================================================
# Helper: Calculate Top-6 Hits
# ============================================================

def calculate_top6_results(test_data, probabilities):

    results = []

    temp = test_data[
        ["date", "number", "target"]
    ].copy()

    temp["probability"] = probabilities

    for date, group in temp.groupby("date", sort=True):

        # Select 6 numbers with highest predicted probability
        top6 = (
            group
            .sort_values("probability", ascending=False)
            .head(6)
        )

        predicted_numbers = top6["number"].tolist()

        actual_numbers = (
            group.loc[group["target"] == 1, "number"]
            .tolist()
        )

        hits = len(
            set(predicted_numbers)
            & set(actual_numbers)
        )

        results.append({
            "date": date,
            "top6": predicted_numbers,
            "actual": actual_numbers,
            "hits": hits
        })

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# Run Experiments
# ============================================================

all_results = []

experiment_details = {}

for experiment_name, features in EXPERIMENTS.items():

    print("\n")
    print("=" * 60)
    print(experiment_name)
    print("=" * 60)

    print("\nFeatures:")

    for feature in features:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Check features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        print("\nERROR: Missing features:")
        print(missing_features)

        continue

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X_train = train_df[features].copy()
    y_train = train_df["target"].copy()

    X_test = test_df[features].copy()
    y_test = test_df["target"].copy()

    # --------------------------------------------------------
    # Scale positive class
    # --------------------------------------------------------

    positive = (y_train == 1).sum()
    negative = (y_train == 0).sum()

    scale_pos_weight = negative / positive

    print(f"\nPositive samples: {positive}")
    print(f"Negative samples: {negative}")
    print(
        f"Scale pos weight: "
        f"{scale_pos_weight:.6f}"
    )

    # --------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    model = xgb.XGBClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_probabilities = model.predict_proba(
        X_train
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    train_auc = roc_auc_score(
        y_train,
        train_probabilities
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    print(f"\nTraining ROC-AUC: {train_auc:.6f}")
    print(f"Test ROC-AUC:     {test_auc:.6f}")

    # --------------------------------------------------------
    # Top-6 Backtest
    # --------------------------------------------------------

    top6_results = calculate_top6_results(
        test_df,
        test_probabilities
    )

    average_hits = top6_results["hits"].mean()

    total_hits = top6_results["hits"].sum()

    maximum_hits = top6_results["hits"].max()

    difference = (
        average_hits
        - RANDOM_EXPECTED
    )

    percentage_difference = (
        difference
        / RANDOM_EXPECTED
        * 100
    )

    # --------------------------------------------------------
    # Hit distribution
    # --------------------------------------------------------

    hit_distribution = (
        top6_results["hits"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Save details
    # --------------------------------------------------------

    experiment_details[experiment_name] = {
        "model": model,
        "top6_results": top6_results,
        "features": features,
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nTop-6 Backtest")
    print("-" * 40)

    print(
        f"Test draws:       "
        f"{len(top6_results)}"
    )

    print(
        f"Average hits:     "
        f"{average_hits:.6f}"
    )

    print(
        f"Total hits:       "
        f"{total_hits}"
    )

    print(
        f"Maximum hits:     "
        f"{maximum_hits}"
    )

    print(
        f"Random expected:  "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print(
        f"Difference:       "
        f"{difference:+.6f}"
    )

    print(
        f"Difference %:     "
        f"{percentage_difference:+.3f}%"
    )

    print("\nHit distribution:")
    print(hit_distribution)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(
        ascending=False
    )

    print("\nTop Features:")

    print(
        importance.head(10).to_string()
    )

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    all_results.append({

        "experiment": experiment_name,

        "features_count": len(features),

        "train_auc": train_auc,

        "test_auc": test_auc,

        "average_hits": average_hits,

        "total_hits": total_hits,

        "maximum_hits": maximum_hits,

        "random_expected": RANDOM_EXPECTED,

        "difference": difference,

        "difference_percent": percentage_difference,
    })


# ============================================================
# Final Comparison
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print("\n\n")
print("=" * 90)
print("ABLATION TEST — FINAL COMPARISON")
print("=" * 90)

if len(results_df) > 0:

    display_columns = [
        "experiment",
        "features_count",
        "train_auc",
        "test_auc",
        "average_hits",
        "random_expected",
        "difference",
        "difference_percent",
        "maximum_hits",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # Best model according to average hits
    # --------------------------------------------------------

    best_idx = results_df[
        "average_hits"
    ].idxmax()

    best = results_df.loc[
        best_idx
    ]

    print("\n")
    print("=" * 60)
    print("BEST EXPERIMENT")
    print("=" * 60)

    print(
        f"Experiment: "
        f"{best['experiment']}"
    )

    print(
        f"Average hits: "
        f"{best['average_hits']:.6f}"
    )

    print(
        f"Random expected: "
        f"{best['random_expected']:.6f}"
    )

    print(
        f"Difference: "
        f"{best['difference']:+.6f}"
    )

    print(
        f"Difference %: "
        f"{best['difference_percent']:+.3f}%"
    )

    print(
        f"Test ROC-AUC: "
        f"{best['test_auc']:.6f}"
    )

    # --------------------------------------------------------
    # Best model above random
    # --------------------------------------------------------

    above_random = results_df[
        results_df["average_hits"]
        > RANDOM_EXPECTED
    ]

    print("\n")
    print("=" * 60)
    print("MODELS ABOVE RANDOM")
    print("=" * 60)

    if len(above_random) == 0:

        print(
            "No experiment exceeded "
            "the random expected value."
        )

    else:

        print(
            above_random[
                [
                    "experiment",
                    "average_hits",
                    "test_auc",
                    "difference_percent",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    output_path = (
        "data/processed/"
        "ablation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print(
        f"Comparison saved to:\n"
        f"{output_path}"
    )


else:

    print(
        "No experiments were completed."
    )


print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)