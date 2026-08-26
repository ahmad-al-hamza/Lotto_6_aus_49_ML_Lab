import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


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

X = df[features]
y = df["target"]


# =========================
# Time-Series split
# =========================

unique_dates = df["date"].unique()

split_date = unique_dates[
    int(len(unique_dates) * 0.8)
]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]


print("Split date:", split_date)

print(
    "Training draws:",
    df.loc[train_mask, "date"].nunique()
)

print(
    "Test draws:",
    df.loc[test_mask, "date"].nunique()
)


# =========================
# Train model
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# =========================
# Predict probabilities
# =========================

test_df = df[test_mask].copy()

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_df["probability"] = test_probabilities


# =========================
# Actual Lotto numbers
# =========================

lotto_df = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

lotto_df["date"] = pd.to_datetime(
    lotto_df["date"]
)

number_columns = [
    "n1", "n2", "n3",
    "n4", "n5", "n6"
]


# =========================
# Top-6 Backtest
# =========================

results = []

for date, group in test_df.groupby("date"):

    # Model ranking
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

    # Actual draw
    actual_row = lotto_df[
        lotto_df["date"] == date
    ]

    if actual_row.empty:
        continue

    actual_numbers = (
        actual_row.iloc[0][number_columns]
        .astype(int)
        .tolist()
    )

    hits = len(
        set(top6) &
        set(actual_numbers)
    )

    results.append({
        "date": date,
        "top6": top6,
        "actual": actual_numbers,
        "hits": hits
    })


results_df = pd.DataFrame(results)


# =========================
# Results
# =========================

print("\nTop-6 Backtest")
print("-" * 30)

print(
    "Test draws:",
    len(results_df)
)

print(
    "Average hits:",
    results_df["hits"].mean()
)

print(
    "Total hits:",
    results_df["hits"].sum()
)

print(
    "Maximum hits:",
    results_df["hits"].max()
)


# =========================
# Hit distribution
# =========================

print("\nHit distribution")
print("-" * 30)

print(
    results_df["hits"]
    .value_counts()
    .sort_index()
)


# =========================
# Hit percentages
# =========================

print("\nHit percentages")
print("-" * 30)

print(
    results_df["hits"]
    .value_counts(
        normalize=True
    )
    .sort_index() * 100
)


# =========================
# First examples
# =========================

print("\nFirst 10 predictions")
print("-" * 30)

print(
    results_df.head(10)
    .to_string(index=False)
)
results_df.to_csv(
    "data/processed/top6_results.csv",
    index=False
)