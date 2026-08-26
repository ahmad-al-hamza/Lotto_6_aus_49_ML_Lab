"""
META-SCORE TOP-6 WALK-FORWARD TEST

Goal:
    Combine multiple number-level signals into one Meta-Score
    for numbers 1..49 and evaluate it using strict walk-forward testing.

Signals:
    1. Ensemble-like historical score
    2. Recency score
    3. Stability score
    4. Diversity score
    5. Logistic Regression meta-model

Important:
    - No future data is used when creating features.
    - Meta-model is trained only on previous draws.
    - Test draws are used only for final evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/lotto_6aus49_clean.csv")
RESULT_PATH = Path(
    "data/processed/meta_score_top6_walk_forward_results.csv"
)

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]

MIN_NUMBER = 1
MAX_NUMBER = 49

TOP_K = 6

RANDOM_SEED = 42

# Walk-forward:
# First 1008 draws for training.
# Then test 1008 draws.
# Continue until the end.
INITIAL_TRAIN_SIZE = 1008
TEST_SIZE = 1008


# ============================================================
# RANDOM BASELINE
# ============================================================

RANDOM_EXPECTED = TOP_K * TOP_K / MAX_NUMBER


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMBER_COLUMNS)

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    print()
    print("Number columns:")
    print(NUMBER_COLUMNS)

    print()
    print(f"Total draws: {len(df)}")
    print(
        f"Number range: "
        f"{int(df[NUMBER_COLUMNS].min().min())} -> "
        f"{int(df[NUMBER_COLUMNS].max().max())}"
    )

    return df


# ============================================================
# DRAW MATRIX
# ============================================================

def create_draw_matrix(df):

    matrix = np.zeros(
        (len(df), MAX_NUMBER),
        dtype=np.int8
    )

    for i, row in enumerate(
        df[NUMBER_COLUMNS].itertuples(index=False)
    ):

        for number in row:

            number = int(number)

            if MIN_NUMBER <= number <= MAX_NUMBER:
                matrix[i, number - 1] = 1

    return matrix


# ============================================================
# BASIC NUMBER FEATURES
# ============================================================

def calculate_number_features(draw_matrix, end_index):

    """
    Create features using draws [0:end_index].

    end_index is EXCLUSIVE.

    Therefore:
        draw_matrix[:end_index]

    contains only historical information.
    """

    history = draw_matrix[:end_index]

    total_draws = len(history)

    numbers = np.arange(
        MIN_NUMBER,
        MAX_NUMBER + 1
    )

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    frequency = history.sum(axis=0).astype(float)

    frequency_rate = frequency / max(total_draws, 1)

    expected_rate = TOP_K / MAX_NUMBER

    strength = frequency_rate / expected_rate

    # --------------------------------------------------------
    # Recent frequencies
    # --------------------------------------------------------

    def recent_rate(window):

        if total_draws < window:
            recent = history
        else:
            recent = history[-window:]

        return recent.mean(axis=0)

    rate_20 = recent_rate(20)
    rate_50 = recent_rate(50)
    rate_100 = recent_rate(100)

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    gap = np.zeros(MAX_NUMBER)

    for number_index in range(MAX_NUMBER):

        positions = np.where(
            history[:, number_index] == 1
        )[0]

        if len(positions) == 0:

            gap[number_index] = total_draws

        else:

            gap[number_index] = (
                total_draws - 1 - positions[-1]
            )

    # --------------------------------------------------------
    # Recency
    # --------------------------------------------------------

    # Moderate recency signal.
    #
    # We intentionally avoid giving huge importance to
    # extremely large gaps.

    recency_score = (
        0.45 * rate_20
        + 0.35 * rate_50
        + 0.20 * rate_100
    )

    # Normalize.

    recency_score = normalize(recency_score)

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    windows = [
        50,
        100,
        200,
        500,
    ]

    window_rates = []

    for window in windows:

        if total_draws < window:
            continue

        window_data = history[-window:]

        window_rates.append(
            window_data.mean(axis=0)
        )

    if len(window_rates) == 0:

        mean_window_rate = frequency_rate
        std_window_rate = np.zeros(MAX_NUMBER)

    else:

        window_rates = np.array(window_rates)

        mean_window_rate = window_rates.mean(axis=0)

        std_window_rate = window_rates.std(
            axis=0
        )

    stability = 1.0 / (
        1.0 + std_window_rate
    )

    stability = normalize(stability)

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    # Diversity measures whether the number appears across
    # different historical windows rather than being concentrated
    # in one short period.

    diversity = (
        0.30 * normalize(rate_20)
        + 0.30 * normalize(rate_50)
        + 0.25 * normalize(rate_100)
        + 0.15 * normalize(frequency_rate)
    )

    diversity = normalize(diversity)

    # --------------------------------------------------------
    # Ensemble-like score
    # --------------------------------------------------------

    conditional_score = (
        0.40 * normalize(rate_20)
        + 0.30 * normalize(rate_50)
        + 0.30 * normalize(strength)
    )

    pairwise_proxy = (
        0.50 * normalize(frequency_rate)
        + 0.50 * normalize(rate_100)
    )

    regime_score = (
        0.50 * normalize(rate_20)
        + 0.50 * normalize(rate_50)
    )

    ensemble_score = (
        0.35 * conditional_score
        + 0.25 * pairwise_proxy
        + 0.20 * regime_score
        + 0.20 * normalize(strength)
    )

    ensemble_score = normalize(
        ensemble_score
    )

    # --------------------------------------------------------
    # Final feature matrix
    # --------------------------------------------------------

    features = pd.DataFrame({

        "number": numbers,

        "frequency": frequency,

        "frequency_rate": frequency_rate,

        "strength": strength,

        "freq_20": rate_20,

        "freq_50": rate_50,

        "freq_100": rate_100,

        "gap": gap,

        "recency_score": recency_score,

        "stability_score": stability,

        "diversity_score": diversity,

        "ensemble_score": ensemble_score,

    })

    return features


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(values):

    values = np.asarray(values, dtype=float)

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
            dtype=float
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


# ============================================================
# META FEATURES
# ============================================================

META_FEATURE_COLUMNS = [
    "ensemble_score",
    "recency_score",
    "stability_score",
    "diversity_score",
    "strength",
    "gap",
    "freq_20",
    "freq_50",
    "freq_100",
]


def create_meta_features(
    history_matrix,
    end_index
):

    features = calculate_number_features(
        history_matrix,
        end_index
    )

    X = features[META_FEATURE_COLUMNS].copy()

    # Log-transform gap.
    X["gap"] = np.log1p(X["gap"])

    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    X = X.fillna(0)

    return features, X


# ============================================================
# TRAIN META MODEL
# ============================================================

def train_meta_model(
    draw_matrix,
    train_end
):

    print("Training number-level Meta-Model...")

    """
    Build training examples from historical points.

    For each historical draw t:

        features based on draws before t
        target = whether number appeared in draw t

    This creates a proper time-series learning problem.
    """

    X_list = []
    y_list = []

    # Start after enough history exists.

    start = min(
        200,
        train_end - 1
    )

    for t in range(start, train_end):

        features, X = create_meta_features(
            draw_matrix,
            t
        )

        target = draw_matrix[t]

        X_list.append(
            X.values
        )

        y_list.append(
            target
        )

    if not X_list:

        raise RuntimeError(
            "Not enough historical data "
            "to train Meta-Model."
        )

    X_train = np.vstack(X_list)

    y_train = np.concatenate(
        y_list
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            )
        ),
    ])

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# PREDICT TOP 6
# ============================================================

def predict_top6(
    draw_matrix,
    train_end,
    model
):

    features, X = create_meta_features(
        draw_matrix,
        train_end
    )

    # Logistic probability.

    ml_probability = model.predict_proba(
        X
    )[:, 1]

    features["ml_score"] = normalize(
        ml_probability
    )

    # --------------------------------------------------------
    # Meta Score
    # --------------------------------------------------------

    #
    # The four strategy scores are explicit features.
    #
    # Logistic regression provides the learned component.
    #
    # We keep the final combination partly transparent.
    #

    features["meta_score"] = (
        0.20 * features["ensemble_score"]
        + 0.20 * features["recency_score"]
        + 0.20 * features["stability_score"]
        + 0.20 * features["diversity_score"]
        + 0.20 * features["ml_score"]
    )

    features = features.sort_values(
        "meta_score",
        ascending=False
    ).reset_index(drop=True)

    top6 = (
        features
        .head(TOP_K)["number"]
        .astype(int)
        .tolist()
    )

    return top6, features


# ============================================================
# EVALUATION
# ============================================================

def evaluate_selection(
    selected_numbers,
    test_matrix
):

    selected = set(
        selected_numbers
    )

    hits = []

    for row in test_matrix:

        actual = set(
            np.where(row == 1)[0] + 1
        )

        hit_count = len(
            selected.intersection(actual)
        )

        hits.append(hit_count)

    hits = np.array(hits)

    return {
        "average_hits": hits.mean(),
        "total_hits": hits.sum(),
        "maximum_hits": hits.max(),
    }


# ============================================================
# RANDOM SIMULATION
# ============================================================

def random_simulation(
    test_matrix,
    simulations=1000
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    averages = []

    numbers = np.arange(
        MIN_NUMBER,
        MAX_NUMBER + 1
    )

    for _ in range(simulations):

        selected = rng.choice(
            numbers,
            size=TOP_K,
            replace=False
        )

        result = evaluate_selection(
            selected,
            test_matrix
        )

        averages.append(
            result["average_hits"]
        )

    averages = np.array(averages)

    return {
        "mean": averages.mean(),
        "lower": np.percentile(
            averages,
            2.5
        ),
        "upper": np.percentile(
            averages,
            97.5
        ),
    }


# ============================================================
# P-VALUE
# ============================================================

def empirical_p_value(
    observed,
    random_distribution
):

    return (
        np.mean(
            random_distribution
            >= observed
        )
    )


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

def create_folds(
    total_draws
):

    folds = []

    train_end = INITIAL_TRAIN_SIZE

    while train_end < total_draws:

        test_end = min(
            train_end + TEST_SIZE,
            total_draws
        )

        folds.append(
            (
                train_end,
                test_end
            )
        )

        train_end = test_end

    return folds


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "META-SCORE TOP-6 WALK-FORWARD TEST"
    )
    print("=" * 70)

    print(
        f"Random expected hits: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print()

    df = load_dataset()

    draw_matrix = create_draw_matrix(
        df
    )

    folds = create_folds(
        len(df)
    )

    print()
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    results = []

    for fold_number, (
        train_end,
        test_end
    ) in enumerate(
        folds,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"FOLD {fold_number}"
        )
        print("=" * 70)

        train_start_date = df.iloc[0]["date"]
        train_end_date = df.iloc[
            train_end - 1
        ]["date"]

        test_start_date = df.iloc[
            train_end
        ]["date"]

        test_end_date = df.iloc[
            test_end - 1
        ]["date"]

        print(
            f"Training: "
            f"{train_start_date} -> "
            f"{train_end_date}"
        )

        print(
            f"Testing:  "
            f"{test_start_date} -> "
            f"{test_end_date}"
        )

        print(
            f"Training draws: "
            f"{train_end}"
        )

        print(
            f"Testing draws:  "
            f"{test_end - train_end}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model = train_meta_model(
            draw_matrix,
            train_end
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        print()
        print(
            "Calculating Meta-Scores..."
        )

        selected, candidates = predict_top6(
            draw_matrix,
            train_end,
            model
        )

        print()
        print(
            "Meta-Score Top-6 selected:"
        )

        print(selected)

        print()
        print("Top candidates:")

        display_columns = [
            "number",
            "frequency",
            "freq_20",
            "freq_50",
            "freq_100",
            "gap",
            "ensemble_score",
            "recency_score",
            "stability_score",
            "diversity_score",
            "ml_score",
            "meta_score",
        ]

        print(
            candidates[
                display_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_matrix = draw_matrix[
            train_end:test_end
        ]

        print()
        print(
            "Evaluating Meta-Score Top-6..."
        )

        evaluation = evaluate_selection(
            selected,
            test_matrix
        )

        # ----------------------------------------------------
        # RANDOM
        # ----------------------------------------------------

        print()
        print(
            "Running random Top-6 simulation..."
        )

        random_result = random_simulation(
            test_matrix
        )

        observed = evaluation[
            "average_hits"
        ]

        difference = (
            observed
            - RANDOM_EXPECTED
        )

        difference_percent = (
            difference
            / RANDOM_EXPECTED
            * 100
        )

        # Generate distribution for p-value.

        rng = np.random.default_rng(
            RANDOM_SEED + fold_number
        )

        random_averages = []

        numbers = np.arange(
            MIN_NUMBER,
            MAX_NUMBER + 1
        )

        for _ in range(5000):

            random_selection = rng.choice(
                numbers,
                size=TOP_K,
                replace=False
            )

            random_eval = evaluate_selection(
                random_selection,
                test_matrix
            )

            random_averages.append(
                random_eval["average_hits"]
            )

        random_averages = np.array(
            random_averages
        )

        p_value = empirical_p_value(
            observed,
            random_averages
        )

        print()
        print("Results")
        print("-" * 40)

        print(
            f"Average hits:       "
            f"{observed:.6f}"
        )

        print(
            f"Total hits:         "
            f"{evaluation['total_hits']}"
        )

        print(
            f"Maximum hits:       "
            f"{evaluation['maximum_hits']}"
        )

        print(
            f"Random expected:    "
            f"{RANDOM_EXPECTED:.6f}"
        )

        print(
            f"Difference:         "
            f"{difference:+.6f}"
        )

        print(
            f"Difference %:       "
            f"{difference_percent:+.3f}%"
        )

        print(
            f"Random simulation:  "
            f"{random_result['mean']:.6f}"
        )

        print(
            f"Random 95% range:   "
            f"[{random_result['lower']:.6f}, "
            f"{random_result['upper']:.6f}]"
        )

        print(
            f"Empirical p-value:  "
            f"{p_value:.6f}"
        )

        # ----------------------------------------------------
        # SAVE FOLD
        # ----------------------------------------------------

        results.append({

            "fold": fold_number,

            "train_draws": train_end,

            "test_draws": (
                test_end - train_end
            ),

            "train_end_date": (
                train_end_date
            ),

            "test_start_date": (
                test_start_date
            ),

            "test_end_date": (
                test_end_date
            ),

            "selected_numbers": (
                ",".join(
                    map(
                        str,
                        selected
                    )
                )
            ),

            "average_hits": observed,

            "total_hits": (
                evaluation[
                    "total_hits"
                ]
            ),

            "maximum_hits": (
                evaluation[
                    "maximum_hits"
                ]
            ),

            "random_expected": (
                RANDOM_EXPECTED
            ),

            "difference": difference,

            "difference_percent": (
                difference_percent
            ),

            "random_simulation": (
                random_result["mean"]
            ),

            "random_lower": (
                random_result["lower"]
            ),

            "random_upper": (
                random_result["upper"]
            ),

            "empirical_p": p_value,

        })

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    summary_columns = [
        "fold",
        "test_draws",
        "average_hits",
        "random_expected",
        "difference",
        "difference_percent",
        "empirical_p",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # OVERALL
    # ========================================================

    mean_hits = (
        results_df[
            "average_hits"
        ].mean()
    )

    mean_difference = (
        results_df[
            "difference"
        ].mean()
    )

    mean_difference_percent = (
        results_df[
            "difference_percent"
        ].mean()
    )

    mean_p = (
        results_df[
            "empirical_p"
        ].mean()
    )

    above_random = (
        results_df[
            "difference"
        ] > 0
    ).sum()

    below_random = (
        results_df[
            "difference"
        ] < 0
    ).sum()

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Mean Meta-Score hits: "
        f"{mean_hits:.6f}"
    )

    print(
        f"Mean random expected: "
        f"{RANDOM_EXPECTED:.6f}"
    )

    print(
        f"Mean difference:      "
        f"{mean_difference:+.6f}"
    )

    print(
        f"Mean difference %:    "
        f"{mean_difference_percent:+.3f}%"
    )

    print(
        f"Mean empirical p:     "
        f"{mean_p:.6f}"
    )

    print()
    print("Fold consistency:")

    print(
        f"Above random: "
        f"{above_random}"
    )

    print(
        f"Below random: "
        f"{below_random}"
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        mean_difference > 0
        and above_random > below_random
        and mean_p < 0.05
    ):

        print(
            "Potential Meta-Score advantage "
            "over random selection."
        )

    else:

        print(
            "No meaningful Meta-Score advantage "
            "over random selection."
        )

    # ========================================================
    # SAVE
    # ========================================================

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print()
    print(
        "Results saved to:"
    )

    print(RESULT_PATH)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
