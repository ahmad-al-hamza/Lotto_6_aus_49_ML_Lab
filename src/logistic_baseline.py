import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# =========================
# Load dataset
# =========================

df = pd.read_csv(
    "data/processed/number_features.csv"
)


# =========================
# Features
# =========================

features = [
    "freq_10",
    "freq_50",
    "freq_100",
    "gap"
]

X = df[features]
y = df["target"]


# =========================
# Time-Series Split by date
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
y_test = y[test_mask]

print("Split date:", split_date)
print("Training draws:", df.loc[train_mask, "date"].nunique())
print("Test draws:", df.loc[test_mask, "date"].nunique())


print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# =========================
# Model
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


# =========================
# Predictions
# =========================

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# =========================
# Evaluation
# =========================

print("\nModel Results")
print("-" * 30)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        predictions,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        predictions,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        probabilities
    )
)