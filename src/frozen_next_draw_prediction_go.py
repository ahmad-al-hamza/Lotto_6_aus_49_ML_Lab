"""
FROZEN NEXT DRAW PREDICTION
===========================

Generate a prediction for the next Lotto 6aus49 draw using the
FROZEN strategies.

NO:
- retraining
- parameter optimization
- V11 development
- OOS information
- modification of V10 weights

The script:
1. Loads the latest known historical draw.
2. Determines the next Wednesday/Saturday draw.
3. Loads the frozen V10 weights.
4. Runs the five frozen strategies using the SAME logic each
   strategy file already uses internally (same feature
   engineering, same model, same window sizes) — just applied
   once, at the end of history, instead of inside a walk-forward
   loop.
5. Produces Top-6 for each strategy.
6. Produces a deterministic weighted consensus Top-6.
7. Saves the prediction.

FIX NOTES (compared to the previous version of this file):
-------------------------------------------------------------
The previous version tried to call every strategy file through a
single generic "guess the function signature" wrapper
(`execute_prediction`). That wrapper had two bugs:

  1. It looked for a model-builder function using a short fixed
     list of names (`build_model`, `train_model`, `fit_model`,
     `create_model`, `build_meta_model`). meta_score's actual
     builder is called `train_meta_model`, so it was never found.

  2. It passed the *last draw date* into any parameter named
     `train_end`/`cutoff`. But every strategy file in this project
     uses `train_end` as an **integer row index** into the draw
     matrix (see meta_score_top6_walk_forward.py,
     stability_top6_walk_forward.py), not a date. So even functions
     that were found failed or produced nonsense.

  3. recency / stability / ensemble don't expose a
     `predict_top6(...)`-style function at all — they only expose
     the *building blocks* (`create_training_samples`,
     `calculate_scores`, `calculate_stability_scores`,
     `calculate_ensemble`, ...). The walk-forward "next selection"
     logic lives inline inside each file's `main()` loop.

This version fixes that by giving each of the four strategies we
have source for (meta_score, recency, stability, ensemble) an
explicit adapter that reproduces exactly what that file's own
`main()` loop does for a single fold — just evaluated at the very
end of history, so it scores the *next, still-unknown* draw
instead of a held-out test fold. No parameters, weights, windows,
or model settings are changed from what is already hard-coded in
each strategy file.

`diversity_top6_walk_forward.py` was not provided when this file
was fixed, so it still falls back to a generic best-effort adapter
(now with both bugs above corrected). If it keeps failing, send me
that file and I will give it the same explicit treatment as the
other four. Since the current frozen V10 weight for diversity is
0.0000, this does not affect the final prediction either way.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA = Path("data/raw/results.csv")

WEIGHTS_FILE = Path(
    "data/processed/adaptive_meta_v10_weights.csv"
)

OUTPUT_DIR = Path(
    "data/processed/frozen_predictions"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "next_draw_predictions.csv"
)

NUMBER_COLUMNS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
]

STRATEGIES = {
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


RANDOM_EXPECTED_HITS = 6.0 * 6.0 / 49.0


# ============================================================
# NEXT LOTTO DRAW
# ============================================================

def get_next_draw_date(last_date: pd.Timestamp) -> pd.Timestamp:
    """
    Lotto 6aus49 draws:
        Wednesday
        Saturday

    weekday:
        Monday    = 0
        Tuesday   = 1
        Wednesday = 2
        Thursday  = 3
        Friday    = 4
        Saturday  = 5
        Sunday    = 6
    """

    candidate = (
        pd.Timestamp(last_date).normalize()
        + pd.Timedelta(days=1)
    )

    while candidate.weekday() not in (2, 5):
        candidate += pd.Timedelta(days=1)

    return candidate


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw data file not found:\n{RAW_DATA}"
        )

    df = pd.read_csv(RAW_DATA)

    required_columns = {
        "date",
        *NUMBER_COLUMNS,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns in results.csv: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    for column in NUMBER_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(int)

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    # Ensure numbers are sorted.
    df[NUMBER_COLUMNS] = df[
        NUMBER_COLUMNS
    ].apply(
        lambda row: sorted(row.tolist()),
        axis=1,
        result_type="expand",
    )

    return df


# ============================================================
# LOAD STRATEGY MODULE
# ============================================================

def load_strategy_module(
    path: Path,
    name: str,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Strategy file not found:\n{path}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(name, path)
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    spec.loader.exec_module(module)

    return module


# ============================================================
# NORMALIZE PREDICTION
# ============================================================

def normalize_prediction(result) -> list[int]:
    """
    Convert common prediction formats to six numbers.
    """

    if isinstance(result, dict):

        for key in (
            "prediction",
            "numbers",
            "top6",
            "predicted_numbers",
        ):

            if key in result:
                result = result[key]
                break

    if isinstance(result, pd.DataFrame):

        if all(
            c in result.columns
            for c in NUMBER_COLUMNS
        ):

            result = (
                result.iloc[0]
                [NUMBER_COLUMNS]
                .tolist()
            )

        elif "number" in result.columns:

            score_column = None

            for candidate in (
                "score",
                "meta_score",
                "stability_score",
                "diversity_score",
                "ensemble_score",
                "ml_score",
            ):

                if candidate in result.columns:
                    score_column = candidate
                    break

            if score_column is not None:

                result = (
                    result
                    .sort_values(
                        score_column,
                        ascending=False,
                    )
                    .head(6)
                    ["number"]
                    .tolist()
                )

            else:

                result = (
                    result
                    .head(6)
                    ["number"]
                    .tolist()
                )

        elif len(result) == 1:

            result = (
                result.iloc[0]
                .tolist()
            )

    if isinstance(result, pd.Series):
        result = result.tolist()

    if isinstance(result, np.ndarray):
        result = result.flatten().tolist()

    if isinstance(result, tuple):
        # Some strategy functions return
        # (selected_numbers, extra_details, ...).
        # The first element is the actual prediction.
        result = result[0]

    if not isinstance(
        result,
        (list, tuple),
    ):
        raise ValueError(
            "Unsupported prediction type: "
            f"{type(result).__name__}"
        )

    result = list(result)

    numbers = []

    for value in result:

        try:
            number = int(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if 1 <= number <= 49:
            numbers.append(number)

    numbers = list(
        dict.fromkeys(numbers)
    )

    if len(numbers) != 6:

        raise ValueError(
            "Prediction must contain "
            "exactly 6 unique numbers "
            f"between 1 and 49.\n"
            f"Received: {numbers}"
        )

    return sorted(numbers)


# ============================================================
# EXPLICIT STRATEGY ADAPTERS
#
# Each adapter reproduces exactly what that strategy file's own
# main() loop does for one fold: build the same features, fit the
# same model (if any) with the same settings, then score/select
# Top-6. The only difference from a normal walk-forward fold is
# that "the end of the training window" is now the last known
# real draw, so the output is a prediction for the *next* draw.
# ============================================================

def predict_meta_score(module, historical_df, historical_matrix):

    draw_matrix = module.create_draw_matrix(historical_df)

    # train_end is a row INDEX (exclusive), not a date.
    # Using the full length means "use all available history".
    train_end = len(draw_matrix)

    model = module.train_meta_model(
        draw_matrix,
        train_end,
    )

    top6, _features = module.predict_top6(
        draw_matrix,
        train_end,
        model,
    )

    return top6


def predict_recency(module, historical_df):

    window_size = getattr(module, "WINDOW_SIZE", 1000)
    top_k = getattr(module, "TOP_K", 6)
    random_seed = getattr(module, "RANDOM_SEED", 42)

    if len(historical_df) > window_size:
        train_df = (
            historical_df
            .tail(window_size)
            .reset_index(drop=True)
        )
    else:
        train_df = historical_df.reset_index(drop=True)

    X_train, y_train = module.create_training_samples(
        train_df
    )

    if len(X_train) == 0:
        raise RuntimeError(
            "Not enough historical draws to build "
            "recency training samples."
        )

    model = module.Pipeline([
        (
            "scaler",
            module.StandardScaler(),
        ),
        (
            "classifier",
            module.LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_seed,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    scores = module.calculate_scores(
        train_df,
        model,
    )

    ranking = np.argsort(scores)[::-1]

    selected = (
        ranking[:top_k] + 1
    ).tolist()

    return selected


def predict_stability(module, historical_df):

    top_k = getattr(module, "TOP_K", 6)

    presence = module.create_presence_matrix(
        historical_df
    )

    train_end = len(presence)

    score_table = module.calculate_stability_scores(
        presence,
        train_end,
    )

    selected = (
        score_table
        .head(top_k)
        ["number"]
        .astype(int)
        .tolist()
    )

    return selected


def predict_ensemble(module, historical_matrix):

    result = module.calculate_ensemble(
        historical_matrix
    )

    selected = list(result[0])

    return [int(n) for n in selected]


EXPLICIT_ADAPTERS = {
    "meta_score": predict_meta_score,
    "recency": predict_recency,
    "stability": predict_stability,
    "ensemble": predict_ensemble,
}


# ============================================================
# GENERIC FALLBACK ADAPTER
#
# Used only for strategies with no explicit adapter above
# (currently: diversity, whose source file was not available
# when this script was fixed). This is a best-effort adapter:
# it introspects the module for a plausible "score everything"
# function and a plausible model-builder function.
#
# Compared to the previous version of this file:
#   - the model-builder search is no longer a fixed short list
#     of names; it matches ANY callable whose name contains
#     "model" and starts with train_/build_/fit_/create_.
#   - "train_end"/"cutoff"-style parameters now receive an
#     INTEGER row index (matching every real strategy file in
#     this project), not a date.
# ============================================================

def find_model_builder(module):

    for name, value in vars(module).items():

        if not callable(value):
            continue

        lowered = name.lower()

        if "model" not in lowered:
            continue

        if lowered.startswith((
            "train_",
            "build_",
            "fit_",
            "create_",
        )):
            return value

    return None


def execute_prediction_generic(
    module,
    historical_df: pd.DataFrame,
    historical_matrix: np.ndarray,
):

    predictor_names = [
        "predict_top6",
        "predict",
        "generate_prediction",
        "get_prediction",
        "predict_numbers",
        "calculate_diversity_scores",
        "calculate_stability_scores",
        "calculate_scores",
        "calculate_ensemble",
    ]

    predictors = []

    for name in predictor_names:

        function = getattr(
            module,
            name,
            None,
        )

        if callable(function):
            predictors.append(function)

    if not predictors:

        available = [
            name
            for name, value
            in vars(module).items()
            if callable(value)
        ]

        raise RuntimeError(
            "No supported prediction function "
            "was found.\n"
            f"Available callables: {available}"
        )

    train_end = len(historical_matrix)

    errors = []

    for predictor in predictors:

        signature = inspect.signature(
            predictor
        )

        parameters = list(
            signature.parameters.values()
        )

        kwargs = {}

        failed_model = False

        for parameter in parameters:

            parameter_name = (
                parameter.name.lower()
            )

            if parameter_name in {
                "draw_matrix",
                "matrix",
                "historical_matrix",
                "train_array",
                "presence",
            }:

                kwargs[parameter.name] = (
                    historical_matrix
                )

            elif parameter_name in {
                "historical_df",
                "historical_dataframe",
                "dataframe",
                "df",
                "data",
            }:

                kwargs[parameter.name] = (
                    historical_df
                )

            elif parameter_name in {
                "train_end",
                "training_end",
                "cutoff",
                "historical_cutoff",
                "end_index",
            }:

                # Integer row index, NOT a date.
                kwargs[parameter.name] = (
                    train_end
                )

            elif parameter_name in {
                "model",
                "fitted_model",
                "trained_model",
            }:

                builder = find_model_builder(module)

                if builder is None:

                    errors.append(
                        f"{predictor.__name__}"
                        f"{signature}: "
                        "requires model but "
                        "no model builder "
                        "was found."
                    )

                    failed_model = True
                    break

                try:

                    builder_signature = (
                        inspect.signature(builder)
                    )

                    builder_kwargs = {}

                    for builder_parameter in (
                        builder_signature.parameters.values()
                    ):

                        name = (
                            builder_parameter.name.lower()
                        )

                        if name in {
                            "draw_matrix",
                            "matrix",
                            "historical_matrix",
                            "train_array",
                            "presence",
                        }:

                            builder_kwargs[
                                builder_parameter.name
                            ] = historical_matrix

                        elif name in {
                            "historical_df",
                            "historical_dataframe",
                            "dataframe",
                            "df",
                            "data",
                        }:

                            builder_kwargs[
                                builder_parameter.name
                            ] = historical_df

                        elif name in {
                            "train_end",
                            "training_end",
                            "cutoff",
                            "historical_cutoff",
                            "end_index",
                        }:

                            builder_kwargs[
                                builder_parameter.name
                            ] = train_end

                    kwargs[parameter.name] = (
                        builder(
                            **builder_kwargs
                        )
                    )

                except Exception as exc:

                    errors.append(
                        f"{predictor.__name__}: "
                        "model creation failed: "
                        f"{exc}"
                    )

                    failed_model = True
                    break

        if failed_model:
            continue

        missing_required = []

        for parameter in parameters:

            if (
                parameter.default
                is inspect.Parameter.empty
                and parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and parameter.name
                not in kwargs
            ):

                missing_required.append(
                    parameter.name
                )

        if missing_required:

            errors.append(
                f"{predictor.__name__}"
                f"{signature}: "
                f"missing parameters "
                f"{missing_required}"
            )

            continue

        try:

            result = predictor(
                **kwargs
            )

            return normalize_prediction(
                result
            )

        except Exception as exc:

            errors.append(
                f"{predictor.__name__}"
                f"{signature}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    raise RuntimeError(
        "Could not execute prediction function.\n"
        + "\n".join(errors)
    )


# ============================================================
# DISPATCH
# ============================================================

def run_strategy(
    strategy: str,
    module,
    historical_df: pd.DataFrame,
    historical_matrix: np.ndarray,
) -> list[int]:

    adapter = EXPLICIT_ADAPTERS.get(strategy)

    if adapter is predict_meta_score:
        prediction = predict_meta_score(
            module, historical_df, historical_matrix
        )

    elif adapter is predict_recency:
        prediction = predict_recency(
            module, historical_df
        )

    elif adapter is predict_stability:
        prediction = predict_stability(
            module, historical_df
        )

    elif adapter is predict_ensemble:
        prediction = predict_ensemble(
            module, historical_matrix
        )

    else:
        prediction = execute_prediction_generic(
            module, historical_df, historical_matrix
        )

    return normalize_prediction(prediction)


# ============================================================
# LOAD FROZEN V10 WEIGHTS
# ============================================================

def load_frozen_weights() -> dict[str, float]:

    if not WEIGHTS_FILE.exists():

        raise FileNotFoundError(
            "Frozen V10 weights file not found:\n"
            f"{WEIGHTS_FILE}"
        )

    df = pd.read_csv(
        WEIGHTS_FILE
    )

    missing = (
        set(STRATEGIES)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing strategy columns "
            f"in weights file: {sorted(missing)}"
        )

    # The last row represents the final
    # frozen V10 state produced by the
    # existing validation pipeline.

    row = df.iloc[-1]

    weights = {}

    for strategy in STRATEGIES:

        value = float(
            row[strategy]
        )

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid weight for "
                f"{strategy}: {value}"
            )

        if value < 0:
            raise ValueError(
                f"Negative frozen weight "
                f"for {strategy}: {value}"
            )

        weights[strategy] = value

    total = sum(
        weights.values()
    )

    if total <= 0:

        raise ValueError(
            "Frozen V10 weights sum to zero."
        )

    # Numerical normalization only.
    # This is NOT optimization.

    weights = {
        strategy: value / total
        for strategy, value
        in weights.items()
    }

    return weights


# ============================================================
# WEIGHTED CONSENSUS
# ============================================================

def create_final_prediction(
    predictions: dict[str, list[int]],
    weights: dict[str, float],
) -> list[int]:

    scores = {
        number: 0.0
        for number in range(1, 50)
    }

    for strategy, numbers in (
        predictions.items()
    ):

        weight = weights.get(
            strategy,
            0.0,
        )

        for number in numbers:

            scores[number] += weight

    ranked_numbers = sorted(
        scores,
        key=lambda number: (
            -scores[number],
            number,
        ),
    )

    return sorted(
        ranked_numbers[:6]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FROZEN NEXT DRAW PREDICTION"
    )
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "No model optimization is performed."
    )
    print(
        "No parameters are changed."
    )
    print(
        "No OOS information is used."
    )
    print(
        "V10 weights remain frozen."
    )
    print()

    # ========================================================
    # DATA
    # ========================================================

    print("=" * 70)
    print("LOADING HISTORICAL DATA")
    print("=" * 70)

    df = load_data()

    last_date = df["date"].max()

    next_date = get_next_draw_date(
        last_date
    )

    historical_df = (
        df[
            df["date"] <= last_date
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    historical_matrix = (
        historical_df[
            NUMBER_COLUMNS
        ]
        .to_numpy(dtype=int)
    )

    print(
        f"Total draws:       {len(df)}"
    )

    print(
        f"Last known draw:   "
        f"{last_date.date()}"
    )

    print(
        f"Historical range:  "
        f"{historical_df['date'].min().date()} "
        f"-> {last_date.date()}"
    )

    print(
        f"Next draw:         "
        f"{next_date.date()}"
    )

    print()

    # ========================================================
    # SAFETY
    # ========================================================

    if next_date <= last_date:

        raise RuntimeError(
            "Safety failure: "
            "next draw is not after "
            "last known draw."
        )

    if (
        df["date"] >= next_date
    ).any():

        raise RuntimeError(
            "The next draw already exists "
            "in results.csv. "
            "Prediction cancelled."
        )

    # ========================================================
    # WEIGHTS
    # ========================================================

    print("=" * 70)
    print(
        "LOADING FROZEN V10 WEIGHTS"
    )
    print("=" * 70)

    weights = load_frozen_weights()

    for strategy in STRATEGIES:

        print(
            f"  {strategy:<12}: "
            f"{weights[strategy]:.4f}"
        )

    print()

    # ========================================================
    # STRATEGIES
    # ========================================================

    print("=" * 70)
    print(
        "FROZEN STRATEGY PREDICTIONS"
    )
    print("=" * 70)

    predictions = {}
    failures = {}

    for strategy, path in (
        STRATEGIES.items()
    ):

        print()
        print(
            f"{strategy.upper()}"
        )

        print(
            f"File: {path}"
        )

        if weights[strategy] <= 0.0:

            print(
                "Skipped: frozen V10 weight is 0.0000 "
                "(does not affect the final prediction)."
            )

            continue

        try:

            module = (
                load_strategy_module(
                    path,
                    f"frozen_{strategy}",
                )
            )

            prediction = run_strategy(
                strategy,
                module,
                historical_df,
                historical_matrix,
            )

            predictions[
                strategy
            ] = prediction

            print(
                "Prediction: "
                + " ".join(
                    f"{n:02d}"
                    for n in prediction
                )
            )

        except Exception as exc:

            failures[
                strategy
            ] = str(exc)

            print(
                "FAILED:"
            )

            print(
                str(exc)
            )

    print()

    if not predictions:

        raise RuntimeError(
            "None of the frozen strategies "
            "with a non-zero weight could "
            "produce a prediction."
        )

    # ========================================================
    # FINAL
    # ========================================================

    final_prediction = (
        create_final_prediction(
            predictions,
            weights,
        )
    )

    print("=" * 70)
    print(
        "FROZEN V10 FINAL PREDICTION"
    )
    print("=" * 70)

    print()

    print(
        f"Prediction date: "
        f"{next_date.date()}"
    )

    print(
        "Final Top-6: "
        + " ".join(
            f"{n:02d}"
            for n in final_prediction
        )
    )

    print()

    # ========================================================
    # WARNINGS
    # ========================================================

    if failures:

        print("=" * 70)
        print(
            "WARNING: STRATEGIES WITH ERRORS"
        )
        print("=" * 70)

        for strategy, error in (
            failures.items()
        ):

            print(
                f"- {strategy}: {error}"
            )

        print()

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "prediction_created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "last_known_draw_date":
            last_date.date().isoformat(),

        "next_draw_date":
            next_date.date().isoformat(),

        "final_n1":
            final_prediction[0],

        "final_n2":
            final_prediction[1],

        "final_n3":
            final_prediction[2],

        "final_n4":
            final_prediction[3],

        "final_n5":
            final_prediction[4],

        "final_n6":
            final_prediction[5],
    }

    for strategy in STRATEGIES:

        row[
            f"{strategy}_weight"
        ] = weights[strategy]

        prediction = (
            predictions.get(strategy)
        )

        for index in range(6):

            column = (
                f"{strategy}_n{index + 1}"
            )

            if prediction is None:

                row[column] = np.nan

            else:

                row[column] = (
                    prediction[index]
                )

    result = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # Append while avoiding duplicate prediction date.
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        existing = pd.read_csv(
            OUTPUT_FILE
        )

        if (
            "next_draw_date"
            in existing.columns
        ):

            existing = existing[
                existing[
                    "next_draw_date"
                ].astype(str)
                !=
                next_date.date().isoformat()
            ]

        result = pd.concat(
            [
                existing,
                result,
            ],
            ignore_index=True,
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print("=" * 70)
    print(
        "PREDICTION SAVED"
    )
    print("=" * 70)

    print(
        OUTPUT_FILE
    )

    print()

    print("=" * 70)
    print(
        "DONE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()