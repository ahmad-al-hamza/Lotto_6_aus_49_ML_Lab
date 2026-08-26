"""
FROZEN NEXT DRAW PREDICTION

Purpose:
- Detect the next Lotto 6aus49 draw date (Wednesday/Saturday).
- Use ONLY draws up to the latest known draw for prediction.
- Keep V10 strategy weights frozen.
- Reproduce the RECENCY strategy exactly from recency_top6_walk_forward.py.
- Since the frozen V10 weights currently select RECENCY with weight 1.0,
  the final prediction is the RECENCY Top-6 prediction.
- Save the prediction to CSV.

No V11 / no optimization / no OOS tuning.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "results.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "lotto_6aus49_clean.csv"

V10_WEIGHTS = ROOT / "data" / "processed" / "adaptive_meta_v10_weights.csv"

RECENCY_MODULE = ROOT / "src" / "recency_top6_walk_forward.py"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "next_draw_predictions.csv"

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

N_NUMBERS = 49
TOP_K = 6
WINDOW_SIZE = 1000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location(
        path.stem,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data() -> pd.DataFrame:
    """
    Prefer the cleaned dataset because it is the dataset used by
    recency_top6_walk_forward.py. Fall back to raw results.csv.
    """

    path = CLEAN_DATA if CLEAN_DATA.exists() else RAW_DATA

    if not path.exists():
        raise FileNotFoundError(
            "Could not find lottery data.\n"
            f"Expected either:\n  {CLEAN_DATA}\n  {RAW_DATA}"
        )

    df = pd.read_csv(path)

    # Normalize date column.
    date_col = None
    for candidate in ("date", "Date"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError("Could not find date column.")

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize number columns.
    missing = [c for c in NUMBER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing number columns: {missing}")

    df = df.dropna(subset=["date"] + NUMBER_COLS).copy()

    for col in NUMBER_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    # Sort numbers within each draw, exactly as the cleaned dataset expects.
    df[NUMBER_COLS] = df[NUMBER_COLS].apply(
        lambda row: sorted(row.tolist()),
        axis=1,
        result_type="expand",
    )

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def get_next_draw_date(last_date: pd.Timestamp) -> pd.Timestamp:
    """
    Lotto 6aus49 draws are Wednesday and Saturday.

    Weekday:
      Monday=0 ... Wednesday=2 ... Saturday=5
    """

    weekday = int(last_date.weekday())

    if weekday < 2:
        days = 2 - weekday
    elif weekday < 5:
        days = 5 - weekday
    else:
        days = 2 + (7 - weekday)

    return last_date + timedelta(days=days)


def load_v10_weights() -> dict[str, float]:
    strategies = [
        "meta_score",
        "recency",
        "stability",
        "diversity",
        "ensemble",
    ]

    if not V10_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Frozen V10 weights not found: {V10_WEIGHTS}"
        )

    weights_df = pd.read_csv(V10_WEIGHTS)

    missing = [s for s in strategies if s not in weights_df.columns]
    if missing:
        raise ValueError(
            f"V10 weights file is missing columns: {missing}"
        )

    if weights_df.empty:
        raise ValueError("V10 weights file is empty.")

    # The V10 validation produced fold-level weights.
    # Use the LAST available frozen weight row, i.e. the most recent
    # frozen decision, without re-optimizing it.
    row = weights_df.iloc[-1]

    weights = {
        s: float(row[s]) if pd.notna(row[s]) else 0.0
        for s in strategies
    }

    total = sum(max(v, 0.0) for v in weights.values())

    if total <= 0:
        raise ValueError("Frozen V10 weights contain no positive weight.")

    # Normalize only to protect against CSV rounding. This is not optimization.
    weights = {
        s: max(weights[s], 0.0) / total
        for s in strategies
    }

    return weights


# ============================================================
# EXACT RECENCY PREDICTION
# ============================================================

def create_recency_training_samples(history: pd.DataFrame):
    """
    Same feature construction as recency_top6_walk_forward.py.
    """

    numbers = history[NUMBER_COLS].values

    X = []
    y = []

    if len(numbers) < 101:
        return np.empty((0, 9)), np.empty((0,))

    for i in range(100, len(numbers)):

        previous = numbers[:i]
        features = []

        for number in range(1, N_NUMBERS + 1):

            mask = previous == number

            total_frequency = mask.sum()
            freq_20 = mask[-20:].sum()
            freq_50 = mask[-50:].sum()
            freq_100 = mask[-100:].sum()

            positions = np.where(mask)[0]

            if len(positions) > 0:
                gap = i - positions[-1]
            else:
                gap = i

            rate_20 = freq_20 / 20.0
            rate_50 = freq_50 / 50.0
            rate_total = total_frequency / float(i)
            momentum = rate_20 - rate_50

            features.append([
                total_frequency,
                freq_20,
                freq_50,
                freq_100,
                gap,
                rate_20,
                rate_50,
                rate_total,
                momentum,
            ])

        target_draw = numbers[i]

        for number in range(1, N_NUMBERS + 1):
            X.append(features[number - 1])
            y.append(1 if number in target_draw else 0)

    return np.asarray(X), np.asarray(y)


def calculate_recency_scores(history: pd.DataFrame, model):
    """
    Same scoring logic as recency_top6_walk_forward.py.
    """

    numbers = history[NUMBER_COLS].values
    i = len(numbers)

    features = []

    for number in range(1, N_NUMBERS + 1):

        mask = numbers == number

        total_frequency = mask.sum()
        freq_20 = mask[-20:].sum()
        freq_50 = mask[-50:].sum()
        freq_100 = mask[-100:].sum()

        positions = np.where(mask)[0]

        if len(positions) > 0:
            gap = i - positions[-1]
        else:
            gap = i

        rate_20 = freq_20 / 20.0
        rate_50 = freq_50 / 50.0
        rate_total = total_frequency / float(i)
        momentum = rate_20 - rate_50

        features.append([
            total_frequency,
            freq_20,
            freq_50,
            freq_100,
            gap,
            rate_20,
            rate_50,
            rate_total,
            momentum,
        ])

    X = np.asarray(features)

    return model.predict_proba(X)[:, 1]


def predict_recency(history: pd.DataFrame):
    """
    Train the frozen RECENCY model on all known historical draws,
    then produce the next-draw Top-6.
    """

    # Import the exact sklearn objects/constants from the existing strategy.
    recency = load_module(RECENCY_MODULE)

    X_train, y_train = create_recency_training_samples(history)

    if len(X_train) == 0:
        raise RuntimeError("Not enough historical data for RECENCY training.")

    model = recency.Pipeline([
        ("scaler", recency.StandardScaler()),
        (
            "classifier",
            recency.LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=getattr(recency, "RANDOM_SEED", RANDOM_SEED),
            ),
        ),
    ])

    model.fit(X_train, y_train)

    scores = calculate_recency_scores(history, model)

    ranking = np.argsort(scores)[::-1]
    selected = (ranking[:TOP_K] + 1).tolist()

    score_table = pd.DataFrame({
        "number": np.arange(1, N_NUMBERS + 1),
        "score": scores,
    }).sort_values("score", ascending=False)

    return selected, score_table


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FROZEN NEXT DRAW PREDICTION")
    print("=" * 70)

    print("""
IMPORTANT:
No model optimization is performed.
No parameters are changed.
No OOS information is used.
V10 weights remain frozen.
The current frozen V10 decision is used as-is.
""")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING HISTORICAL DATA")
    print("=" * 70)

    df = load_data()

    last_date = df["date"].iloc[-1]
    next_date = get_next_draw_date(last_date)

    print(f"Total known draws: {len(df)}")
    print(f"Last known draw:   {last_date.date()}")
    print(
        f"Historical range:  "
        f"{df['date'].min().date()} -> {last_date.date()}"
    )
    print(f"Next draw:         {next_date.date()}")

    if next_date.weekday() not in (2, 5):
        raise RuntimeError("Calculated next draw is not Wednesday/Saturday.")

    # --------------------------------------------------------
    # V10 WEIGHTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING FROZEN V10 WEIGHTS")
    print("=" * 70)

    weights = load_v10_weights()

    for strategy, weight in weights.items():
        print(f"  {strategy:<12}: {weight:.4f}")

    selected_strategy = max(weights, key=weights.get)

    print(f"\nFrozen V10 selected strategy: {selected_strategy.upper()}")

    # --------------------------------------------------------
    # CURRENT FROZEN V10
    # --------------------------------------------------------

    # Current V10 frozen weights from the validation are expected to
    # select RECENCY. We deliberately do not manufacture predictions
    # for strategies whose original files do not expose a prediction API.
    if selected_strategy != "recency":
        raise RuntimeError(
            "The current frozen V10 weights do not select RECENCY.\n"
            f"Selected strategy: {selected_strategy}\n"
            "This script intentionally refuses to silently substitute "
            "another strategy."
        )

    print("\n" + "=" * 70)
    print("FROZEN RECENCY PREDICTION")
    print("=" * 70)

    prediction, score_table = predict_recency(df)

    print("\nTop candidates:")
    print(
        score_table.head(10).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("FROZEN V10 FINAL PREDICTION")
    print("=" * 70)

    print(f"Prediction date: {next_date.date()}")
    print(f"Strategy:         {selected_strategy.upper()}")
    print(f"Frozen weight:    {weights[selected_strategy]:.4f}")
    print(f"TOP-6:            {prediction}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "prediction_date": next_date.date().isoformat(),
        "last_known_date": last_date.date().isoformat(),
        "strategy": selected_strategy,
        "weight": weights[selected_strategy],
        "pred_n1": prediction[0],
        "pred_n2": prediction[1],
        "pred_n3": prediction[2],
        "pred_n4": prediction[3],
        "pred_n5": prediction[4],
        "pred_n6": prediction[5],
    }

    result = pd.DataFrame([row])

    # Replace an existing prediction for the same date rather than
    # creating duplicate rows.
    if OUTPUT_FILE.exists():
        old = pd.read_csv(OUTPUT_FILE)

        if "prediction_date" in old.columns:
            old = old[
                old["prediction_date"].astype(str)
                != next_date.date().isoformat()
            ]

            result = pd.concat(
                [old, result],
                ignore_index=True,
            )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nPrediction saved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
