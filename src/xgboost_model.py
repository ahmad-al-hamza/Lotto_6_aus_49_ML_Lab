import pandas as pd

from xgboost import XGBClassifier
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
# Time-Series Split
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

print(
    "Training draws:",
    df.loc[train_mask, "date"].nunique()
)

print(
    "Test draws:",
    df.loc[test_mask, "date"].nunique()
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Test samples:",
    len(X_test)
)


# =========================
# XGBoost
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
# Training
# =========================

model.fit(
    X_train,
    y_train
)


# =========================
# Predictions
# =========================

y_probability = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_probability >= 0.5
).astype(int)


# =========================
# Results
# =========================

print("\nXGBoost Results")
print("-" * 30)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_probability
    )
)


# =========================
# Feature Importance
# =========================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(
    ascending=False
)

print("\nFeature Importance")
print("-" * 30)

print(
    importance
)