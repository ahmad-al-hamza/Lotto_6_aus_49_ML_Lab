"""
CREATE FROZEN OUT-OF-SAMPLE PREDICTIONS

Purpose:
    Generate completely frozen Top-6 predictions using ONLY
    historical data <= HISTORICAL_CUTOFF.

Important:
    - OOS draws are NEVER used for training.
    - No parameters are optimized on OOS.
    - No strategy is selected using OOS performance.
    - Existing strategy implementations are reused.
"""

from pathlib import Path
import importlib.util
import inspect
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "results.csv"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "frozen_predictions"
)

HISTORICAL_CUTOFF = pd.Timestamp("2026-08-19")

NUMBER_COLUMNS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
]

NUMBER_MIN = 1
NUMBER_MAX = 49
TOP_K = 6


STRATEGY_FILES = {
    "meta_score": BASE_DIR / "src" / "meta_score_top6_walk_forward.py",
    "recency": BASE_DIR / "src" / "recency_top6_walk_forward.py",
    "stability": BASE_DIR / "src" / "stability_top6_walk_forward.py",
    "diversity": BASE_DIR / "src" / "diversity_top6_walk_forward.py",
    "ensemble": BASE_DIR / "src" / "ensemble_top6_walk_forward.py",
}


# ============================================================
# HELPERS
# ============================================================

def load_module(path, module_name):
    """Load a Python module directly from a file."""

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def normalize(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if not np.isfinite(minimum):
        return np.zeros_like(values)

    if not np.isfinite(maximum):
        return np.zeros_like(values)

    if maximum - minimum < 1e-12:
        return np.full_like(
            values,
            0.5,
            dtype=float,
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def load_data():

    df = pd.read_csv(RAW_DATA)

    if "date" not in df.columns:
        raise ValueError(
            "Raw dataset does not contain 'date'."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=["date"] + NUMBER_COLUMNS
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


def dataframe_to_matrix(df):

    matrix = np.zeros(
        (len(df), NUMBER_MAX),
        dtype=np.int8,
    )

    for i, row in enumerate(
        df[NUMBER_COLUMNS].itertuples(
            index=False
        )
    ):

        for number in row:

            number = int(number)

            if NUMBER_MIN <= number <= NUMBER_MAX:
                matrix[i, number - 1] = 1

    return matrix


def save_prediction(
    strategy,
    prediction_date,
    numbers,
    scores=None,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    numbers = [
        int(x)
        for x in numbers
    ]

    if len(numbers) != TOP_K:
        raise ValueError(
            f"{strategy}: expected {TOP_K} numbers, "
            f"got {len(numbers)}"
        )

    if len(set(numbers)) != TOP_K:
        raise ValueError(
            f"{strategy}: duplicate numbers detected."
        )

    if not all(
        NUMBER_MIN <= x <= NUMBER_MAX
        for x in numbers
    ):
        raise ValueError(
            f"{strategy}: number outside 1..49."
        )

    result = {
        "date": prediction_date,
        "strategy": strategy,
    }

    for i, number in enumerate(
        sorted(numbers),
        start=1,
    ):
        result[f"n{i}"] = number

    if scores is not None:

        for number, score in scores.items():

            result[
                f"score_{int(number)}"
            ] = safe_float(score)

    output_file = (
        OUTPUT_DIR
        / f"{strategy}_frozen_predictions.csv"
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================
# GENERIC SCORE EXTRACTION
# ============================================================

def extract_numbers_from_result(result):

    """
    Convert common strategy return formats into Top-6.
    """

    if result is None:
        return None, None

    # --------------------------------------------------------
    # Direct list / tuple
    # --------------------------------------------------------

    if isinstance(result, (list, tuple, np.ndarray)):

        if len(result) >= TOP_K:

            values = list(result)

            if all(
                isinstance(x, (int, np.integer))
                and NUMBER_MIN <= int(x) <= NUMBER_MAX
                for x in values[:TOP_K]
            ):
                return (
                    [
                        int(x)
                        for x in values[:TOP_K]
                    ],
                    None,
                )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if isinstance(result, pd.DataFrame):

        number_column_candidates = [
            "number",
            "numbers",
        ]

        number_col = None

        for col in number_column_candidates:

            if col in result.columns:
                number_col = col
                break

        if number_col is not None:

            df = result.copy()

            score_candidates = [
                "score",
                "recency_score",
                "stability_score",
                "diversity_score",
                "ensemble_score",
                "meta_score",
            ]

            score_col = None

            for col in score_candidates:

                if col in df.columns:
                    score_col = col
                    break

            if score_col is not None:

                df = df.sort_values(
                    score_col,
                    ascending=False,
                )

            numbers = (
                pd.to_numeric(
                    df[number_col],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )

            numbers = [
                x
                for x in numbers
                if NUMBER_MIN <= x <= NUMBER_MAX
            ]

            if len(numbers) >= TOP_K:

                scores = None

                if score_col is not None:

                    scores = dict(
                        zip(
                            df[number_col].astype(int),
                            df[score_col].astype(float),
                        )
                    )

                return (
                    numbers[:TOP_K],
                    scores,
                )

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(result, dict):

        for key in [
            "top6",
            "selected",
            "selected_numbers",
            "numbers",
            "prediction",
        ]:

            if key in result:

                numbers = result[key]

                if isinstance(
                    numbers,
                    (list, tuple, np.ndarray),
                ):

                    numbers = [
                        int(x)
                        for x in numbers
                    ]

                    if len(numbers) >= TOP_K:

                        return (
                            numbers[:TOP_K],
                            None,
                        )

    return None, None


# ============================================================
# META-SCORE
# ============================================================

def run_meta_score(
    historical_df,
):

    print("\n" + "=" * 70)
    print("FROZEN STRATEGY: META_SCORE")
    print("=" * 70)

    module = load_module(
        STRATEGY_FILES["meta_score"],
        "frozen_meta_score",
    )

    draw_matrix = dataframe_to_matrix(
        historical_df
    )

    train_end = len(
        historical_df
    )

    print(
        f"Historical training draws: {train_end}"
    )

    if not hasattr(
        module,
        "train_meta_model",
    ):
        raise RuntimeError(
            "meta_score does not expose train_meta_model()."
        )

    if not hasattr(
        module,
        "predict_top6",
    ):
        raise RuntimeError(
            "meta_score does not expose predict_top6()."
        )

    print(
        "Training Meta-Model using historical data only..."
    )

    model = module.train_meta_model(
        draw_matrix,
        train_end,
    )

    print(
        "Generating frozen prediction..."
    )

    result = module.predict_top6(
        draw_matrix,
        train_end,
        model,
    )

    numbers = None
    scores = None

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], (list, tuple, np.ndarray))
    ):

        top6, candidates = result

        numbers = [int(x) for x in top6]

        if (
            isinstance(candidates, pd.DataFrame)
            and {"number", "meta_score"}.issubset(candidates.columns)
        ):
            scores = dict(
                zip(
                    candidates["number"].astype(int),
                    candidates["meta_score"].astype(float),
                )
            )

    else:
        numbers, scores = extract_numbers_from_result(
            result
        )

    if numbers is None:
        raise RuntimeError(
            "Could not extract Top-6 from meta_score."
        )

    print(
        f"Frozen Meta-Score Top-6: {sorted(numbers)}"
    )

    return save_prediction(
        "meta_score",
        "2026-08-22",
        numbers,
        scores,
    )


# ============================================================
# RECENCY
# ============================================================

def run_recency(
    historical_df,
):

    print("\n" + "=" * 70)
    print("FROZEN STRATEGY: RECENCY")
    print("=" * 70)

    module = load_module(
        STRATEGY_FILES["recency"],
        "frozen_recency",
    )

    for name in (
        "create_training_samples",
        "calculate_scores",
        "WINDOW_SIZE",
        "RANDOM_SEED",
    ):
        if not hasattr(module, name):
            raise RuntimeError(
                f"recency module does not expose '{name}'."
            )

    window_size = module.WINDOW_SIZE

    train_df = historical_df.tail(
        window_size
    ).reset_index(drop=True)

    print(
        f"Training window: last {len(train_df)} draws"
    )

    print(
        "Creating training samples..."
    )

    X_train, y_train = module.create_training_samples(
        train_df
    )

    if len(X_train) == 0:
        raise RuntimeError(
            "recency: not enough historical data "
            "to build training samples."
        )

    print(
        "Training Logistic Regression..."
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=module.RANDOM_SEED,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Generating frozen prediction..."
    )

    scores = module.calculate_scores(
        train_df,
        model,
    )

    ranking = np.argsort(scores)[::-1]

    numbers = [
        int(x)
        for x in (ranking[:TOP_K] + 1)
    ]

    score_dict = {
        number: float(scores[number - 1])
        for number in range(NUMBER_MIN, NUMBER_MAX + 1)
    }

    print(
        f"Frozen Recency Top-6: {sorted(numbers)}"
    )

    return save_prediction(
        "recency",
        "2026-08-22",
        numbers,
        score_dict,
    )


# ============================================================
# ENSEMBLE
# ============================================================

def run_ensemble(
    historical_df,
):

    print("\n" + "=" * 70)
    print("FROZEN STRATEGY: ENSEMBLE")
    print("=" * 70)

    module = load_module(
        STRATEGY_FILES["ensemble"],
        "frozen_ensemble",
    )

    if not hasattr(module, "calculate_ensemble"):
        raise RuntimeError(
            "ensemble module does not expose 'calculate_ensemble'."
        )

    train_array = historical_df[NUMBER_COLUMNS].astype(int).values

    print(
        f"Historical training draws: {len(train_array)}"
    )

    print(
        "Calculating ensemble predictions..."
    )

    result = module.calculate_ensemble(train_array)

    selected = result[0]
    ensemble_score = result[6]

    numbers = [int(x) for x in selected]

    score_dict = {
        number: float(ensemble_score[number - 1])
        for number in range(NUMBER_MIN, NUMBER_MAX + 1)
    }

    print(
        f"Frozen Ensemble Top-6: {sorted(numbers)}"
    )

    return save_prediction(
        "ensemble",
        "2026-08-22",
        numbers,
        score_dict,
    )


# ============================================================
# GENERIC SCORING STRATEGY
# ============================================================

def find_scoring_function(
    module,
    strategy,
):

    candidates = []

    for name, obj in inspect.getmembers(
        module,
        inspect.isfunction,
    ):

        name_lower = name.lower()

        if name in {
            "main",
            "load_data",
            "load_dataset",
            "create_folds",
            "random_simulation",
            "evaluate_selection",
        }:
            continue

        if name.startswith("_"):
            continue

        score_keywords = [
            "score",
            "calculate",
            "select",
            "predict",
        ]

        if not any(
            keyword in name_lower
            for keyword in score_keywords
        ):
            continue

        candidates.append(
            (name, obj)
        )

    preferred = []

    for name, func in candidates:

        name_lower = name.lower()

        if strategy in name_lower:
            preferred.append(
                (name, func)
            )

    if preferred:
        return preferred[0]

    if candidates:
        return candidates[0]

    return None


def call_scoring_function(
    func,
    historical_df,
):

    number_cols = NUMBER_COLUMNS

    draw_matrix = dataframe_to_matrix(
        historical_df
    )

    train_end = len(
        historical_df
    )

    attempts = [
        (
            "train_df, number_cols",
            (
                historical_df,
                number_cols,
            ),
        ),
        (
            "historical_df, number_cols",
            (
                historical_df,
                number_cols,
            ),
        ),
        (
            "draw_matrix, train_end",
            (
                draw_matrix,
                train_end,
            ),
        ),
        (
            "draw_matrix",
            (
                draw_matrix,
            ),
        ),
        (
            "historical_df",
            (
                historical_df,
            ),
        ),
    ]

    errors = []

    for description, args in attempts:

        try:

            signature = inspect.signature(
                func
            )

            required = [
                p
                for p in signature.parameters.values()
                if (
                    p.default is inspect.Parameter.empty
                    and p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )
            ]

            if len(required) > len(args):
                continue

            result = func(*args)

            numbers, scores = (
                extract_numbers_from_result(
                    result
                )
            )

            if numbers is not None:
                return numbers, scores

        except Exception as exc:

            errors.append(
                f"{description}: {exc}"
            )

    raise RuntimeError(
        "Could not execute scoring function.\n"
        + "\n".join(errors)
    )


def run_generic_strategy(
    strategy,
):

    print("\n" + "=" * 70)
    print(
        f"FROZEN STRATEGY: "
        f"{strategy.upper()}"
    )
    print("=" * 70)

    module = load_module(
        STRATEGY_FILES[strategy],
        f"frozen_{strategy}",
    )

    found = find_scoring_function(
        module,
        strategy,
    )

    if found is None:

        raise RuntimeError(
            f"No scoring function found in "
            f"{strategy} module."
        )

    function_name, function = found

    print(
        f"Using scoring function: "
        f"{function_name}"
    )

    numbers, scores = call_scoring_function(
        function,
        HISTORICAL_DF,
    )

    print(
        f"Frozen {strategy} Top-6: "
        f"{sorted(numbers)}"
    )

    return save_prediction(
        strategy,
        "2026-08-22",
        numbers,
        scores,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CREATE FROZEN OUT-OF-SAMPLE "
        "PREDICTIONS"
    )
    print("=" * 70)

    print(
        f"Historical cutoff: "
        f"{HISTORICAL_CUTOFF.date()}"
    )

    print("\nIMPORTANT:")
    print(
        "Training uses ONLY data <= cutoff."
    )
    print(
        "OOS data is NOT passed to strategies."
    )
    print(
        "No strategy weights are optimized."
    )
    print(
        "No OOS result is used during prediction."
    )

    print("\n" + "=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    df = load_data()

    historical_df = df[
        df["date"]
        <= HISTORICAL_CUTOFF
    ].copy()

    oos_df = df[
        df["date"]
        > HISTORICAL_CUTOFF
    ].copy()

    print(
        f"Total raw draws: {len(df)}"
    )

    print(
        f"Historical draws: "
        f"{len(historical_df)}"
    )

    print(
        f"OOS draws: "
        f"{len(oos_df)}"
    )

    if len(historical_df) == 0:
        raise RuntimeError(
            "No historical data found."
        )

    if len(oos_df) == 0:

        raise RuntimeError(
            "No unseen OOS draws found."
        )

    print(
        f"Historical range: "
        f"{historical_df['date'].min().date()} "
        f"-> "
        f"{historical_df['date'].max().date()}"
    )

    print(
        f"OOS range: "
        f"{oos_df['date'].min().date()} "
        f"-> "
        f"{oos_df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # CRITICAL SAFETY CHECK
    # --------------------------------------------------------

    if (
        historical_df["date"].max()
        > HISTORICAL_CUTOFF
    ):
        raise RuntimeError(
            "Historical data contains dates "
            "after the cutoff."
        )

    if (
        oos_df["date"].min()
        <= HISTORICAL_CUTOFF
    ):
        raise RuntimeError(
            "OOS data contains dates at or "
            "before the cutoff."
        )

    global HISTORICAL_DF
    HISTORICAL_DF = historical_df

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategies = [
        "meta_score",
        "recency",
        "stability",
        "diversity",
        "ensemble",
    ]

    results = {}

    # --------------------------------------------------------
    # META
    # --------------------------------------------------------

    try:

        output = run_meta_score(
            historical_df
        )

        results["meta_score"] = output

    except Exception as exc:

        print(
            "\nFAILED: meta_score"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # RECENCY (dedicated runner -- see run_recency())
    # --------------------------------------------------------

    try:

        output = run_recency(
            historical_df
        )

        results["recency"] = output

    except Exception:

        print(
            "\nFAILED: recency"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # REMAINING STRATEGIES (safe for the generic runner)
    # --------------------------------------------------------

    for strategy in [
        "stability",
        "diversity",
    ]:

        try:

            output = run_generic_strategy(
                strategy
            )

            results[strategy] = output

        except Exception:

            print(
                f"\nFAILED: {strategy}"
            )

            traceback.print_exc()

    # --------------------------------------------------------
    # ENSEMBLE (dedicated runner -- see run_ensemble())
    # --------------------------------------------------------

    try:

        output = run_ensemble(
            historical_df
        )

        results["ensemble"] = output

    except Exception:

        print(
            "\nFAILED: ensemble"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "VERIFYING FROZEN PREDICTIONS"
    )
    print("=" * 70)

    for strategy in strategies:

        output_file = (
            OUTPUT_DIR
            / f"{strategy}_frozen_predictions.csv"
        )

        if output_file.exists():

            print(
                f"{strategy:<12}: OK"
            )

            try:

                check = pd.read_csv(
                    output_file
                )

                row = check.iloc[0]

                numbers = [
                    int(row[f"n{i}"])
                    for i in range(1, 7)
                ]

                print(
                    f"              "
                    f"{sorted(numbers)}"
                )

            except Exception as exc:

                print(
                    f"{strategy:<12}: "
                    f"INVALID ({exc})"
                )

        else:

            print(
                f"{strategy:<12}: MISSING"
            )

    missing = []

    for strategy in strategies:

        output_file = (
            OUTPUT_DIR
            / f"{strategy}_frozen_predictions.csv"
        )

        if not output_file.exists():
            missing.append(strategy)

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

        print(
            "\nDo NOT run frozen_oos_validation.py yet."
        )

        return

    print(
        "ALL FIVE FROZEN PREDICTIONS "
        "CREATED SUCCESSFULLY."
    )

    print(
        "\nThe next step is:"
    )

    print(
        "python src/frozen_oos_validation.py"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()