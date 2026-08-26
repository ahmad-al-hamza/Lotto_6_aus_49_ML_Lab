import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/processed/number_features_v3.csv"

SPLIT_DATE = "2017-05-06"

RANDOM_STATE = 42


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:")
print(df.shape)

print()


# ============================================================
# 3. Define features
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
# 4. Remove rows with missing values
# ============================================================

df = df.dropna(subset=FEATURES + [TARGET]).copy()


# ============================================================
# 5. Time-based split
# ============================================================

split_date = pd.Timestamp(SPLIT_DATE)

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()


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

print("Training draws:",
      train_df["date"].nunique())

print("Test draws:",
      test_df["date"].nunique())

print()

print("Training samples:", len(train_df))
print("Test samples:", len(test_df))

print()


# ============================================================
# 6. Prepare X / y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# 7. Class imbalance
# ============================================================

positive = y_train.sum()
negative = len(y_train) - positive

scale_pos_weight = negative / positive

print("Positive samples:", positive)
print("Negative samples:", negative)
print("Scale pos weight:", scale_pos_weight)

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
# 10. Predictions
# ============================================================

y_prob_train = model.predict_proba(X_train)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]


# Default classification threshold
threshold = 0.5

y_pred = (y_prob_test >= threshold).astype(int)


# ============================================================
# 11. Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc_train = roc_auc_score(
    y_train,
    y_prob_train
)

roc_auc_test = roc_auc_score(
    y_test,
    y_prob_test
)


# ============================================================
# 12. Results
# ============================================================

print("=" * 50)
print("XGBoost V3 Results")
print("=" * 50)

print(f"Accuracy:       {accuracy:.6f}")
print(f"Precision:      {precision:.6f}")
print(f"Recall:         {recall:.6f}")
print(f"Training ROC-AUC: {roc_auc_train:.6f}")
print(f"Test ROC-AUC:     {roc_auc_test:.6f}")

print()


# ============================================================
# 13. Classification report
# ============================================================

print("Classification Report")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 14. Feature importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
)

importance = importance.sort_values(
    ascending=False
)


print("Feature Importance")
print("-" * 50)

print(
    importance.to_string()
)

print()


# ============================================================
# 15. Top 10 features
# ============================================================

print("Top 10 Features")
print("-" * 50)

print(
    importance.head(10).to_string()
)

print()


# ============================================================
# 16. Probability statistics
# ============================================================

print("Prediction Probability Statistics")
print("-" * 50)

print(
    pd.Series(y_prob_test).describe()
)

print()


# ============================================================
# 17. Positive prediction statistics
# ============================================================

print("Classification Statistics")
print("-" * 50)

print("Actual positive:", int(y_test.sum()))
print("Predicted positive:", int(y_pred.sum()))

print(
    "Actual positive %:",
    round(y_test.mean() * 100, 4)
)

print(
    "Predicted positive %:",
    round(y_pred.mean() * 100, 4)
)

print()


print("Done.")