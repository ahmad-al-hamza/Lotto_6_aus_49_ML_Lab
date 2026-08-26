"""
CREATE FROZEN OUT-OF-SAMPLE STRATEGY RESULTS
================================================

Purpose
-------
Create the completely unseen OOS dataset that will be consumed by:

    src/frozen_oos_validation.py

IMPORTANT
---------
This script does NOT optimize any model.

It does NOT:
    - change strategy parameters
    - search for better windows
    - create new adaptive weights
    - use OOS draws for training
    - use OOS performance to select a strategy

The five frozen strategies are:

    1. meta_score
    2. recency
    3. stability
    4. diversity
    5. ensemble

The current historical dataset ends at 2026-08-19.

Any draw after that date is considered completely unseen OOS data.

The script intentionally FAILS if no genuinely new draws exist.
It never manufactures OOS results.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RAW_PATH = Path("data/raw/results.csv")

CLEAN_PATH = Path(
    "data/processed/lotto_6aus49_clean.csv"
)

OUTPUT_PATH = Path(
    "data/processed/frozen_oos_strategy_results.csv"
)

NUMBER_COLUMNS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
]

MIN_NUMBER = 1
MAX_NUMBER = 49
TOP_K = 6

# ------------------------------------------------------------
# IMPORTANT:
# This is the last date contained in the historical dataset
# used by V6-V10 and the final validation.
#
# Your current dataset ends at 2026-08-19.
# ------------------------------------------------------------

HISTORICAL_CUTOFF = pd.Timestamp("2026-08-19")

RANDOM_EXPECTED = TOP_K * 6 / MAX_NUMBER


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_columns(df, path):
    required = ["date"] + NUMBER_COLUMNS

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )


def load_results(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n  {path}"
        )

    df = pd.read_csv(path)

    validate_columns(df, path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["date"] + NUMBER_COLUMNS
    ).copy()

    # Sort numbers inside every draw.
    df[NUMBER_COLUMNS] = df[NUMBER_COLUMNS].apply(
        lambda row: sorted(row),
        axis=1,
        result_type="expand"
    )

    # Sort chronologically.
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def validate_numbers(df):
    for col in NUMBER_COLUMNS:
        invalid = (
            (df[col] < MIN_NUMBER)
            | (df[col] > MAX_NUMBER)
        )

        if invalid.any():
            rows = df.loc[invalid, ["date", col]]

            raise ValueError(
                f"Invalid lottery numbers found:\n{rows.head(10)}"
            )

    # Every draw must contain six distinct numbers.
    duplicates = []

    for idx, row in df[NUMBER_COLUMNS].iterrows():

        values = [
            int(x)
            for x in row
        ]

        if len(set(values)) != TOP_K:
            duplicates.append(idx)

    if duplicates:
        raise ValueError(
            "Some draws contain duplicate numbers. "
            f"Rows: {duplicates[:10]}"
        )


def extract_oos(df):
    """
    Return only draws strictly after the historical cutoff.
    """

    oos = df[
        df["date"] > HISTORICAL_CUTOFF
    ].copy()

    return oos.reset_index(drop=True)


def create_draw_matrix(df):
    """
    Convert draws into a (n_draws, 49) binary matrix.
    """

    matrix = np.zeros(
        (len(df), MAX_NUMBER),
        dtype=np.int8
    )

    for i, row in enumerate(
        df[NUMBER_COLUMNS].itertuples(index=False)
    ):

        for number in row:

            number = int(number)

            matrix[
                i,
                number - 1
            ] = 1

    return matrix


# ============================================================
# FROZEN STRATEGY INTERFACE
# ============================================================

def validate_frozen_strategy_file(path, strategy_name):
    """
    Validate a strategy prediction file.

    Expected format:

        date,n1,n2,n3,n4,n5,n6

    where n1..n6 are the six numbers predicted by the
    frozen strategy for that date.

    No scores or optimization parameters are accepted here.
    """

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = [
        "date",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{strategy_name}: missing columns {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    ).copy()

    df[NUMBER_COLUMNS] = df[NUMBER_COLUMNS].apply(
        lambda row: sorted(row),
        axis=1,
        result_type="expand"
    )

    validate_numbers(df)

    # OOS-only enforcement.
    df = df[
        df["date"] > HISTORICAL_CUTOFF
    ].copy()

    df = (
        df.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"{strategy_name}: "
            "prediction file contains no dates after "
            f"{HISTORICAL_CUTOFF.date()}."
        )

    return df


# ============================================================
# HIT CALCULATION
# ============================================================

def calculate_hits(prediction_df, actual_oos):
    """
    Match frozen predictions against actual OOS draws.

    Returns one row per OOS draw.
    """

    pred = prediction_df.copy()
    actual = actual_oos.copy()

    pred = pred[
        [
            "date",
            *NUMBER_COLUMNS
        ]
    ].copy()

    actual = actual[
        [
            "date",
            *NUMBER_COLUMNS
        ]
    ].copy()

    merged = actual.merge(
        pred,
        on="date",
        how="inner",
        suffixes=(
            "_actual",
            "_pred"
        )
    )

    if merged.empty:
        return pd.DataFrame()

    rows = []

    for _, row in merged.iterrows():

        actual_numbers = {
            int(row[f"{col}_actual"])
            for col in NUMBER_COLUMNS
        }

        predicted_numbers = {
            int(row[f"{col}_pred"])
            for col in NUMBER_COLUMNS
        }

        hits = len(
            actual_numbers.intersection(
                predicted_numbers
            )
        )

        rows.append(
            {
                "date": row["date"],
                "hits": hits,

                "pred_n1": int(row["n1_pred"]),
                "pred_n2": int(row["n2_pred"]),
                "pred_n3": int(row["n3_pred"]),
                "pred_n4": int(row["n4_pred"]),
                "pred_n5": int(row["n5_pred"]),
                "pred_n6": int(row["n6_pred"]),

                "actual_n1": int(row["n1_actual"]),
                "actual_n2": int(row["n2_actual"]),
                "actual_n3": int(row["n3_actual"]),
                "actual_n4": int(row["n4_actual"]),
                "actual_n5": int(row["n5_actual"]),
                "actual_n6": int(row["n6_actual"]),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "CREATE FROZEN OUT-OF-SAMPLE STRATEGY RESULTS"
    )

    print(
        f"Historical cutoff: {HISTORICAL_CUTOFF.date()}"
    )

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print()
    print("IMPORTANT:")
    print("No model optimization is performed.")
    print("No parameters are changed.")
    print("No OOS information is used for training.")
    print("No strategy is selected using OOS performance.")

    # --------------------------------------------------------
    # Load historical/updated raw data
    # --------------------------------------------------------

    print_header(
        "LOADING DATA"
    )

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found:\n  {RAW_PATH}"
        )

    print(
        f"Raw input: {RAW_PATH}"
    )

    raw_df = load_results(
        RAW_PATH
    )

    validate_numbers(
        raw_df
    )

    print(
        f"Total raw draws: {len(raw_df)}"
    )

    print(
        f"Raw date range: "
        f"{raw_df['date'].min().date()} -> "
        f"{raw_df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Verify historical dataset
    # --------------------------------------------------------

    print_header(
        "VERIFYING HISTORICAL CUTOFF"
    )

    historical = raw_df[
        raw_df["date"] <= HISTORICAL_CUTOFF
    ].copy()

    oos = extract_oos(
        raw_df
    )

    print(
        f"Historical draws: {len(historical)}"
    )

    print(
        f"Historical last date: "
        f"{historical['date'].max().date()}"
    )

    print(
        f"Unseen OOS draws: {len(oos)}"
    )

    if historical.empty:
        raise RuntimeError(
            "Historical dataset is empty."
        )

    if historical["date"].max() != HISTORICAL_CUTOFF:
        raise RuntimeError(
            "Historical cutoff does not exist in the dataset.\n"
            f"Expected: {HISTORICAL_CUTOFF.date()}\n"
            f"Actual last historical date: "
            f"{historical['date'].max().date()}"
        )

    # --------------------------------------------------------
    # CRITICAL SAFETY CHECK
    # --------------------------------------------------------

    if oos.empty:

        print()
        print("=" * 70)
        print("NO NEW OOS DATA")
        print("=" * 70)

        print()
        print(
            "There are no draws after "
            f"{HISTORICAL_CUTOFF.date()}."
        )

        print()
        print(
            "Do NOT create frozen OOS results manually."
        )

        print(
            "Update data/raw/results.csv with genuinely "
            "new lottery draws first."
        )

        sys.exit(1)

    print()
    print(
        "OOS date range:"
    )

    print(
        f"{oos['date'].min().date()} -> "
        f"{oos['date'].max().date()}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # We do NOT calculate strategy predictions here.
    #
    # The five strategies must be run in FROZEN mode:
    #
    #   training = ALL draws <= 2026-08-19
    #   prediction = draws > 2026-08-19
    #
    # Their prediction files must then be placed in:
    #
    # data/processed/frozen_predictions/
    #
    # --------------------------------------------------------

    print_header(
        "LOADING FROZEN STRATEGY PREDICTIONS"
    )

    prediction_dir = Path(
        "data/processed/frozen_predictions"
    )

    strategies = [
        "meta_score",
        "recency",
        "stability",
        "diversity",
        "ensemble",
    ]

    strategy_predictions = {}

    for strategy in strategies:

        path = prediction_dir / (
            f"{strategy}_frozen_predictions.csv"
        )

        print(
            f"{strategy:<12}: {path}"
        )

        result = validate_frozen_strategy_file(
            path,
            strategy
        )

        if result is None:

            print(
                " " * 14
                + "MISSING"
            )

        else:

            print(
                " " * 14
                + f"{len(result)} predictions"
            )

            strategy_predictions[
                strategy
            ] = result

    # --------------------------------------------------------
    # Do not silently continue with missing strategies.
    # --------------------------------------------------------

    missing = [
        strategy
        for strategy in strategies
        if strategy not in strategy_predictions
    ]

    if missing:

        print_header(
            "FROZEN PREDICTIONS ARE INCOMPLETE"
        )

        print(
            "Missing frozen prediction files:"
        )

        for strategy in missing:
            print(
                f"  - {strategy}"
            )

        print()
        print(
            "Expected directory:"
        )

        print(
            f"  {prediction_dir}"
        )

        print()
        print(
            "This script will NOT create fake or "
            "historically reconstructed OOS results."
        )

        print()
        print(
            "Run each frozen strategy using:"
        )

        print(
            "training <= 2026-08-19"
        )

        print(
            "prediction > 2026-08-19"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Calculate OOS results
    # --------------------------------------------------------

    print_header(
        "CALCULATING FROZEN OOS RESULTS"
    )

    all_results = []

    for strategy in strategies:

        predictions = strategy_predictions[
            strategy
        ]

        results = calculate_hits(
            predictions,
            oos
        )

        if results.empty:
            raise RuntimeError(
                f"{strategy}: no matching OOS dates."
            )

        results.insert(
            0,
            "strategy",
            strategy
        )

        all_results.append(
            results
        )

        average_hits = (
            results["hits"].mean()
        )

        difference = (
            average_hits
            - RANDOM_EXPECTED
        )

        difference_pct = (
            difference
            / RANDOM_EXPECTED
            * 100
        )

        print()
        print(
            f"{strategy.upper()}"
        )

        print(
            f"  OOS draws:       {len(results)}"
        )

        print(
            f"  Average hits:    "
            f"{average_hits:.6f}"
        )

        print(
            f"  Difference:      "
            f"{difference:+.6f}"
        )

        print(
            f"  Difference %:    "
            f"{difference_pct:+.3f}%"
        )

        print(
            f"  Maximum hits:    "
            f"{results['hits'].max()}"
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    print_header(
        "FINAL INTEGRITY CHECK"
    )

    for strategy in strategies:

        strategy_df = final_df[
            final_df["strategy"] == strategy
        ]

        dates = set(
            strategy_df["date"]
        )

        expected_dates = set(
            oos["date"]
        )

        if dates != expected_dates:

            missing_dates = (
                expected_dates - dates
            )

            extra_dates = (
                dates - expected_dates
            )

            raise RuntimeError(
                f"{strategy}: OOS date mismatch.\n"
                f"Missing: {sorted(missing_dates)[:10]}\n"
                f"Extra: {sorted(extra_dates)[:10]}"
            )

        if len(strategy_df) != len(oos):

            raise RuntimeError(
                f"{strategy}: expected "
                f"{len(oos)} predictions but got "
                f"{len(strategy_df)}."
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df = final_df.sort_values(
        [
            "date",
            "strategy"
        ]
    ).reset_index(
        drop=True
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header(
        "FROZEN OOS SUMMARY"
    )

    summary = []

    for strategy in strategies:

        strategy_df = final_df[
            final_df["strategy"] == strategy
        ]

        average_hits = (
            strategy_df["hits"].mean()
        )

        difference = (
            average_hits
            - RANDOM_EXPECTED
        )

        difference_pct = (
            difference
            / RANDOM_EXPECTED
            * 100
        )

        summary.append(
            {
                "strategy": strategy,
                "oos_draws": len(strategy_df),
                "average_hits": average_hits,
                "difference": difference,
                "difference_pct": difference_pct,
                "max_hits": strategy_df["hits"].max(),
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
