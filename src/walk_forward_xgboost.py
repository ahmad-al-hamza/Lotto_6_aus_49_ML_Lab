import pandas as pd
import numpy as np

from xgboost import XGBClassifier


# =========================
# Load dataset
# =========================

df = pd.read_csv(
    "data/processed/number_features_v2.csv"
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "number"]
).reset_index(drop=True)


# =========================
# Features
# =========================

features = [
    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "freq_100",
    "freq_200",
    "gap",
    "recent_vs_long"
]

number_columns = [
    "n1", "n2", "n3",
    "n4", "n5", "n6"
]


# =========================
# Load actual Lotto data
# =========================

lotto = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

lotto["date"] = pd.to_datetime(
    lotto["date"]
)


# =========================
# Unique dates
# =========================

dates = np.sort(
    df["date"].unique()
)

n_dates = len(dates)

print("Total dates:", n_dates)


# =========================
# Walk-Forward settings
# =========================

N_FOLDS = 5

initial_train_ratio = 0.50

test_ratio = 0.10

initial_train_size = int(
    n_dates * initial_train_ratio
)

test_size = int(
    n_dates * test_ratio
)


# =========================
# Results
# =========================

fold_results = []


# =========================
# Walk-Forward
# =========================

for fold in range(N_FOLDS):

    train_end = (
        initial_train_size
        + fold * test_size
    )

    test_end = (
        train_end
        + test_size
    )

    if test_end > n_dates:
        break

    train_dates = dates[:train_end]

    test_dates = dates[
        train_end:test_end
    ]

    train_mask = df["date"].isin(
        train_dates
    )

    test_mask = df["date"].isin(
        test_dates
    )

    X_train = df.loc[
        train_mask,
        features
    ]

    y_train = df.loc[
        train_mask,
        "target"
    ]

    test_df = df.loc[
        test_mask
    ].copy()

    X_test = test_df[features]


    # =========================
    # Model
    # =========================

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


    # =========================
    # Train
    # =========================

    model.fit(
        X_train,
        y_train
    )


    # =========================
    # Predict
    # =========================

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_df["probability"] = probabilities


    # =========================
    # Top-6 predictions
    # =========================

    results = []


    for date, group in test_df.groupby(
        "date"
    ):

        top6 = (
            group
            .sort_values(
                "probability",
                ascending=False
            )
            .head(6)
            ["number"]
            .tolist()
        )


        actual_row = lotto[
            lotto["date"] == date
        ]

        if actual_row.empty:
            continue


        actual = (
            actual_row
            .iloc[0][number_columns]
            .astype(int)
            .tolist()
        )


        hits = len(
            set(top6)
            &
            set(actual)
        )


        results.append(
            hits
        )


    # =========================
    # Fold statistics
    # =========================

    average_hits = np.mean(results)

    total_hits = np.sum(results)

    fold_results.append({
        "fold": fold + 1,
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
        "test_draws": len(results),
        "average_hits": average_hits,
        "total_hits": total_hits
    })


    # =========================
    # Print fold
    # =========================

    print("\nFold", fold + 1)
    print("-" * 30)

    print(
        "Training:",
        train_dates[0],
        "→",
        train_dates[-1]
    )

    print(
        "Testing:",
        test_dates[0],
        "→",
        test_dates[-1]
    )

    print(
        "Test draws:",
        len(results)
    )

    print(
        "Average hits:",
        average_hits
    )

    print(
        "Total hits:",
        total_hits
    )


# =========================
# Final summary
# =========================

results_df = pd.DataFrame(
    fold_results
)


print("\n")
print("=" * 50)
print("WALK-FORWARD SUMMARY")
print("=" * 50)

print(
    results_df[
        [
            "fold",
            "test_draws",
            "average_hits",
            "total_hits"
        ]
    ].to_string(
        index=False
    )
)


# =========================
# Overall statistics
# =========================

print("\nOverall")
print("-" * 30)

print(
    "Mean fold performance:",
    results_df[
        "average_hits"
    ].mean()
)

print(
    "Std fold performance:",
    results_df[
        "average_hits"
    ].std()
)

print(
    "Best fold:",
    results_df[
        "average_hits"
    ].max()
)

print(
    "Worst fold:",
    results_df[
        "average_hits"
    ].min()
)

print(
    "Random expected:",
    36 / 49
)


# =========================
# Save
# =========================

results_df.to_csv(
    "data/processed/"
    "walk_forward_xgboost_results.csv",
    index=False
)