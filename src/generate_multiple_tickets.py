"""
CREATE FROZEN OUT-OF-SAMPLE PREDICTIONS V2

Purpose
-------
Create completely frozen predictions for the five final strategies:

    1. meta_score
    2. recency
    3. stability
    4. diversity
    5. ensemble

Historical cutoff:
    2026-08-19

IMPORTANT
---------
Training is performed ONLY on historical data <= cutoff.

The unseen OOS draw(s) are NEVER passed into the prediction logic.

No OOS optimization.
No OOS weight fitting.
No strategy selection using OOS results.

The output of this script is intended for:

    frozen_oos_validation.py
"""

from pathlib import Path
import importlib.util
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA = Path("data/raw/results.csv")

CUTOFF_DATE = pd.Timestamp("2026-08-19")

OUTPUT_DIR = Path(
    "data/processed/frozen_predictions"
)

NUMBER_COLUMNS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
]

STRATEGIES = [
    "meta_score",
    "recency",
    "stability",
    "diversity",
    "ensemble",
]

STRATEGY_FILES = {
    "meta_score":
        Path("src/meta_score_top6_walk_forward.py"),

    "recency":
        Path("src/recency_top6_walk_forward.py"),

    "stability":
        Path("src/stability_top6_walk_forward.py"),

    "diversity":
        Path("src/diversity_top6_walk_forward.py"),

    "ensemble":
        Path("src/ensemble_top6_walk_forward.py"),
}


# ============================================================
# HELPERS
# ============================================================

def load_module(path, name):
    """
    Load a Python source file without executing its main block.
    """

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def create_draw_matrix(df):
    """
    Convert lottery draws into a 0/1 matrix.

    Shape:
        draws x 49
    """

    matrix = np.zeros(
        (len(df), 49),
        dtype=np.int8,
    )

    for i, row in enumerate(
        df[NUMBER_COLUMNS].itertuples(
            index=False
        )
    ):

        for number in row:

            number = int(number)

            if 1 <= number <= 49:
                matrix[i, number - 1] = 1

    return matrix


def normalize_top6(result):
    """
    Convert various possible prediction outputs
    into exactly six integers.

    Accepted forms:

        [1, 2, 3, 4, 5, 6]

        numpy array

        pandas Series

        DataFrame containing a number column

        DataFrame indexed by number

        tuple/list whose first element is Top-6
    """

    # --------------------------------------------------------
    # Direct list / tuple / ndarray / Series
    # --------------------------------------------------------

    if isinstance(
        result,
        (list, tuple, np.ndarray, pd.Series)
    ):

        # Some predictors return:
        # (selected_numbers, scores)

        if isinstance(result, tuple) and len(result) > 0:

            first = result[0]

            try:
                values = list(first)

                if len(values) >= 6:
                    result = values

            except Exception:
                pass

        try:

            values = [
                int(x)
                for x in list(result)
            ]

            values = [
                x for x in values
                if 1 <= x <= 49
            ]

            if len(values) >= 6:

                return values[:6]

        except Exception:
            pass


    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if isinstance(result, pd.DataFrame):

        # Common column names
        candidate_columns = [
            "number",
            "selected_number",
            "num",
        ]

        for column in candidate_columns:

            if column in result.columns:

                values = (
                    pd.to_numeric(
                        result[column],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                    .tolist()
                )

                values = [
                    x for x in values
                    if 1 <= x <= 49
                ]

                if len(values) >= 6:
                    return values[:6]

        # Index may contain numbers
        try:

            values = [
                int(x)
                for x in result.index.tolist()
            ]

            values = [
                x for x in values
                if 1 <= x <= 49
            ]

            if len(values) >= 6:
                return values[:6]

        except Exception:
            pass


    raise RuntimeError(
        "Could not convert prediction output "
        "into six lottery numbers."
    )


def find_function(module, candidates):
    """
    Find the first available callable.
    """

    for name in candidates:

        function = getattr(
            module,
            name,
            None,
        )

        if callable(function):
            return function

    return None


# ============================================================
# META-SCORE
# ============================================================

def predict_meta_score(
    module,
    historical_matrix,
):
    """
    Meta-score requires:

        1. train_meta_model()
        2. predict_top6()

    The model is trained exclusively on historical data.
    """

    train_end = len(
        historical_matrix
    )

    train_function = getattr(
        module,
        "train_meta_model",
        None,
    )

    predict_function = getattr(
        module,
        "predict_top6",
        None,
    )

    if not callable(train_function):
        raise RuntimeError(
            "meta_score does not expose "
            "train_meta_model()."
        )

    if not callable(predict_function):
        raise RuntimeError(
            "meta_score does not expose "
            "predict_top6()."
        )

    print(
        "Training Meta-Score model "
        "using historical data only..."
    )

    model = train_function(
        historical_matrix,
        train_end,
    )

    print(
        "Generating frozen Meta-Score prediction..."
    )

    result = predict_function(
        historical_matrix,
        train_end,
        model,
    )

    return normalize_top6(result)


# ============================================================
# GENERIC SCORE-BASED STRATEGY
# ============================================================

def score_based_prediction(
    module,
    historical_df,
):
    """
    Try to use the scoring functions from the original
    strategy module.

    This avoids inventing a new strategy.

    We search for the same calculation functions that the
    walk-forward scripts use and then select the highest
    scoring six numbers.
    """

    number_columns = NUMBER_COLUMNS

    # --------------------------------------------------------
    # Possible score functions
    # --------------------------------------------------------

    candidates = [
        "calculate_scores",
        "calculate_recency_scores",
        "calculate_stability_scores",
        "calculate_diversity_scores",
        "calculate_ensemble_scores",
        "calculate_window_scores",
        "calculate_scores_for_numbers",
        "calculate_number_scores",
    ]

    score_function = find_function(
        module,
        candidates,
    )

    if score_function is None:
        raise RuntimeError(
            "No compatible score function found."
        )

    # --------------------------------------------------------
    # Try common signatures.
    # --------------------------------------------------------

    attempts = [
        (
            historical_df,
            number_columns,
        ),
        (
            historical_df,
        ),
    ]

    last_error = None

    for args in attempts:

        try:

            result = score_function(
                *args
            )

            top6 = normalize_top6(
                result
            )

            return top6

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        "Could not execute strategy score "
        f"function: {last_error}"
    )


# ============================================================
# STRATEGY-SPECIFIC FALLBACKS
# ============================================================

def predict_recency(
    module,
    historical_df,
):
    """
    Recency strategy.

    First try the strategy's own scoring function.
    """

    function = find_function(
        module,
        [
            "calculate_recency_scores",
            "calculate_recency_score",
            "calculate_scores",
        ],
    )

    if function is None:
        raise RuntimeError(
            "No recency scoring function found."
        )

    errors = []

    for args in [
        (historical_df, NUMBER_COLUMNS),
        (historical_df,),
    ]:

        try:

            result = function(*args)

            return normalize_top6(result)

        except Exception as exc:

            errors.append(str(exc))

    raise RuntimeError(
        "Recency prediction failed:\n"
        + "\n".join(errors)
    )


def predict_stability(
    module,
    historical_df,
):
    """
    Stability strategy.
    """

    function = find_function(
        module,
        [
            "calculate_stability_scores",
            "calculate_stability_score",
            "calculate_scores",
        ],
    )

    if function is None:
        raise RuntimeError(
            "No stability scoring function found."
        )

    errors = []

    for args in [
        (historical_df, NUMBER_COLUMNS),
        (historical_df,),
    ]:

        try:

            result = function(*args)

            return normalize_top6(result)

        except Exception as exc:

            errors.append(str(exc))

    raise RuntimeError(
        "Stability prediction failed:\n"
        + "\n".join(errors)
    )


def predict_diversity(
    module,
    historical_df,
):
    """
    Diversity strategy.

    The original diversity implementation produces a
    DataFrame sorted by diversity_score, so we explicitly
    preserve that logic.
    """

    function = getattr(
        module,
        "calculate_diversity_scores",
        None,
    )

    if not callable(function):

        function = getattr(
            module,
            "calculate_scores",
            None,
        )

    if not callable(function):
        raise RuntimeError(
            "No diversity scoring function found."
        )

    errors = []

    for args in [
        (historical_df, NUMBER_COLUMNS),
        (historical_df,),
    ]:

        try:

            result = function(*args)

            return normalize_top6(result)

        except Exception as exc:

            errors.append(str(exc))

    raise RuntimeError(
        "Diversity prediction failed:\n"
        + "\n".join(errors)
    )


def predict_ensemble(
    module,
    historical_df,
):
    """
    Ensemble strategy.
    """

    function = find_function(
        module,
        [
            "calculate_ensemble_scores",
            "calculate_ensemble_score",
            "calculate_scores",
        ],
    )

    if function is None:
        raise RuntimeError(
            "No ensemble scoring function found."
        )

    errors = []

    for args in [
        (historical_df, NUMBER_COLUMNS),
        (historical_df,),
    ]:

        try:

            result = function(*args)

            return normalize_top6(result)

        except Exception as exc:

            errors.append(str(exc))

    raise RuntimeError(
        "Ensemble prediction failed:\n"
        + "\n".join(errors)
    )


# ============================================================
# MAIN STRATEGY DISPATCH
# ============================================================

def predict_strategy(
    strategy,
    module,
    historical_df,
    historical_matrix,
):
    """

    Dispatch prediction to the correct frozen strategy.
    """

    if strategy == "meta_score":

        return predict_meta_score(
            module,
            historical_matrix,
        )

    if strategy == "recency":

        return predict_recency(
            module,
            historical_df,
        )

    if strategy == "stability":

        return predict_stability(
            module,
            historical_df,
        )

    if strategy == "diversity":

        return predict_diversity(
            module,
            historical_df,
        )

    if strategy == "ensemble":

        return predict_ensemble(
            module,
            historical_df,
        )

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATE FROZEN OUT-OF-SAMPLE PREDICTIONS V2")
    print("=" * 70)

    print(
        f"Historical cutoff: "
        f"{CUTOFF_DATE.date()}"
    )

    print()

    print("IMPORTANT:")
    print(
        "Training uses ONLY data <= cutoff."
    )
    print(
        "OOS data is NOT passed to strategies."
    )
    print(
        "No OOS optimization."
    )
    print(
        "No OOS strategy selection."
    )
    print()

    # ========================================================
    # LOAD RAW DATA
    # ========================================================

    if not RAW_DATA.exists():

        raise FileNotFoundError(
            f"Raw data not found: {RAW_DATA}"
        )

    df = pd.read_csv(
        RAW_DATA
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    for column in NUMBER_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=NUMBER_COLUMNS
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SPLIT HISTORICAL / OOS
    # ========================================================

    historical_df = df[
        df["date"] <= CUTOFF_DATE
    ].copy()

    oos_df = df[
        df["date"] > CUTOFF_DATE
    ].copy()

    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Total draws:       {len(df)}"
    )

    print(
        f"Historical draws:  {len(historical_df)}"
    )

    print(
        f"OOS draws:         {len(oos_df)}"
    )

    if not historical_df.empty:

        print(
            f"Historical range:  "
            f"{historical_df['date'].min().date()} "
            f"-> "
            f"{historical_df['date'].max().date()}"
        )

    if not oos_df.empty:

        print(
            f"OOS range:         "
            f"{oos_df['date'].min().date()} "
            f"-> "
            f"{oos_df['date'].max().date()}"
        )

    if oos_df.empty:

        raise RuntimeError(
            "No unseen OOS draws exist after cutoff."
        )

    # ========================================================
    # CREATE HISTORICAL MATRIX
    # ========================================================

    historical_matrix = create_draw_matrix(
        historical_df
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # RUN FIVE FROZEN STRATEGIES
    # ========================================================

    predictions = {}

    print()
    print("=" * 70)
    print("GENERATING FROZEN PREDICTIONS")
    print("=" * 70)

    for strategy in STRATEGIES:

        print()
        print("-" * 70)
        print(
            f"FROZEN STRATEGY: "
            f"{strategy.upper()}"
        )
        print("-" * 70)

        path = STRATEGY_FILES[
            strategy
        ]

        if not path.exists():

            print(
                f"SKIPPED: file not found: "
                f"{path}"
            )

            continue

        try:

            module = load_module(
                path,
                f"frozen_{strategy}",
            )

            selected = predict_strategy(
                strategy,
                module,
                historical_df,
                historical_matrix,
            )

            # Ensure exactly six unique numbers.
            selected = list(
                dict.fromkeys(
                    int(x)
                    for x in selected
                )
            )

            selected = [
                x for x in selected
                if 1 <= x <= 49
            ]

            if len(selected) != 6:

                raise RuntimeError(
                    "Strategy did not produce "
                    "exactly six unique numbers."
                )

            predictions[strategy] = selected

            print()
            print(
                f"Frozen Top-6: {selected}"
            )

        except Exception as exc:

            print()
            print(
                f"FAILED: {strategy}"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    print()
    print("=" * 70)
    print("SAVING FROZEN PREDICTIONS")
    print("=" * 70)

    for strategy, selected in predictions.items():

        output_file = (
            OUTPUT_DIR
            / f"{strategy}_frozen_predictions.csv"
        )

        output_df = pd.DataFrame(
            [
                {
                    "cutoff_date":
                        CUTOFF_DATE.date(),

                    "prediction_date":
                        oos_df["date"].min().date(),

                    "train_draws":
                        len(historical_df),

                    "oos_draws":
                        len(oos_df),

                    "n1":
                        selected[0],

                    "n2":
                        selected[1],

                    "n3":
                        selected[2],

                    "n4":
                        selected[3],

                    "n5":
                        selected[4],

                    "n6":
                        selected[5],
                }
            ]
        )

        output_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"{strategy:<12}: "
            f"{output_file}"
        )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    missing = []

    for strategy in STRATEGIES:

        output_file = (
            OUTPUT_DIR
            / f"{strategy}_frozen_predictions.csv"
        )

        if output_file.exists():

            print(
                f"{strategy:<12}: OK"
            )

        else:

            print(
                f"{strategy:<12}: MISSING"
            )

            missing.append(
                strategy
            )

    print()

    if missing:

        print(
            "FROZEN PREDICTIONS ARE INCOMPLETE."
        )

        print(
            "Missing:"
        )

        for strategy in missing:

            print(
                f"  - {strategy}"
            )

        print()
        print(
            "Do NOT run frozen_oos_validation.py yet."
        )

    else:

        print(
            "ALL FIVE FROZEN PREDICTIONS CREATED."
        )

        print()
        print(
            "Next step:"
        )

        print(
            "python src/frozen_oos_validation.py"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
