import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)


# =========================
# Load dataset
# =========================

df = pd.read_csv(
    "data/processed/number_features.csv"
)

df["date"] = pd.to_datetime(df["date"])


# =========================
# Features / Target
# =========================

y = df["target"]


# =========================
# Time-Series split
# =========================

unique_dates = df["date"].unique()

split_date = unique_dates[
    int(len(unique_dates) * 0.8)
]

test_mask = df["date"] >= split_date

y_test = y[test_mask]


# =========================
# Random predictions
# =========================

np.random.seed(42)

random_probabilities = np.random.random(
    len(y_test)
)

random_predictions = (
    random_probabilities >= 0.5
).astype(int)


# =========================
# Results
# =========================

print("Random Baseline")
print("-" * 30)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        random_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        random_predictions,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        random_predictions,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        random_probabilities
    )
)