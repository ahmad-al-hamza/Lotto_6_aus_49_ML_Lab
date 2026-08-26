import os
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

OUTPUT_PATH = "data/processed/ensemble_top6_walk_forward_results.csv"

N_NUMBERS = 49
TOP_K = 6

# Walk-forward settings
TEST_SIZE = 1008
STEP_SIZE = 1008

# Recent windows
WINDOW_20 = 20
WINDOW_50 = 50

# Ensemble weights
WEIGHT_FREQUENCY = 0.20
WEIGHT_CONDITIONAL = 0.20
WEIGHT_ML = 0.25
WEIGHT_PAIRWISE = 0.15
WEIGHT_REGIME = 0.20

RANDOM_SEED = 42
N_SIMULATIONS = 10000


# ============================================================
# HELPERS
# ============================================================

def minmax_normalize(values):
    values = np.asarray(values, dtype=float)

    vmin = np.min(values)
    vmax = np.max(values)

    if vmax - vmin < 1e-12:
        return np.ones_like(values) * 0.5

    return (values - vmin) / (vmax - vmin)


def zscore_normalize(values):
    values = np.asarray(values, dtype=float)

    mean = np.mean(values)
    std = np.std(values)

    if std < 1e-12:
        return np.zeros_like(values)

    return (values - mean) / std


def calculate_hits(prediction, test_array):
    prediction = set(prediction)

    hits = []

    for row in test_array:
        actual = set(row)
        hits.append(len(prediction.intersection(actual)))

    return np.array(hits)


def random_simulation(test_array, n_simulations=10000, seed=42):

    rng = np.random.default_rng(seed)

    n_test = len(test_array)

    hits = np.empty(n_simulations, dtype=float)

    for i in range(n_simulations):

        random_numbers = rng.choice(
            np.arange(1, N_NUMBERS + 1),
            size=TOP_K,
            replace=False
        )

        random_set = set(random_numbers)

        total = 0

        for row in test_array:
            total += len(random_set.intersection(set(row)))

        hits[i] = total / n_test

    return hits


# ============================================================
# FREQUENCY SCORE
# ============================================================

def frequency_scores(train_array):

    frequencies = np.zeros(N_NUMBERS + 1)

    for row in train_array:
        for number in row:
            frequencies[int(number)] += 1

    scores = frequencies[1:]

    return frequencies[1:], minmax_normalize(scores)


# ============================================================
# CONDITIONAL SCORE
# ============================================================

def conditional_scores(train_array):

    n = len(train_array)

    frequencies = np.zeros(N_NUMBERS + 1)

    for row in train_array:
        for number in row:
            frequencies[int(number)] += 1

    recent_20 = train_array[-WINDOW_20:]
    recent_50 = train_array[-WINDOW_50:]

    freq20 = np.zeros(N_NUMBERS + 1)
    freq50 = np.zeros(N_NUMBERS + 1)

    for row in recent_20:
        for number in row:
            freq20[int(number)] += 1

    for row in recent_50:
        for number in row:
            freq50[int(number)] += 1

    # Gap since last appearance
    gaps = np.full(N_NUMBERS + 1, n, dtype=float)

    for idx in range(n - 1, -1, -1):

        for number in train_array[idx]:

            number = int(number)

            if gaps[number] == n:
                gaps[number] = n - 1 - idx

    freq_score = minmax_normalize(frequencies[1:])
    recent20_score = minmax_normalize(freq20[1:])
    recent50_score = minmax_normalize(freq50[1:])

    # Moderate gap preference.
    # Avoid extremely large gaps dominating the score.
    gap_score = 1.0 / (1.0 + gaps[1:])
    gap_score = minmax_normalize(gap_score)

    score = (
        0.25 * freq_score
        + 0.30 * recent20_score
        + 0.25 * recent50_score
        + 0.20 * gap_score
    )

    return {
        "frequency": frequencies[1:],
        "freq20": freq20[1:],
        "freq50": freq50[1:],
        "gap": gaps[1:],
        "score": minmax_normalize(score)
    }


# ============================================================
# ML SCORE
# ============================================================

def create_ml_dataset(train_array):

    X = []
    y = []

    # Need enough history
    for i in range(WINDOW_50, len(train_array)):

        history = train_array[:i]

        frequencies = np.zeros(N_NUMBERS + 1)

        for row in history:
            for number in row:
                frequencies[int(number)] += 1

        recent20 = np.zeros(N_NUMBERS + 1)
        recent50 = np.zeros(N_NUMBERS + 1)

        for row in history[-WINDOW_20:]:
            for number in row:
                recent20[int(number)] += 1

        for row in history[-WINDOW_50:]:
            for number in row:
                recent50[int(number)] += 1

        current_draw = set(train_array[i])

        for number in range(1, N_NUMBERS + 1):

            total_freq = frequencies[number]

            freq20 = recent20[number]
            freq50 = recent50[number]

            gap = 0

            for j in range(i - 1, -1, -1):

                if number in set(train_array[j]):
                    gap = i - 1 - j
                    break

            X.append([
                total_freq,
                freq20,
                freq50,
                gap
            ])

            y.append(
                1 if number in current_draw else 0
            )

    return np.asarray(X), np.asarray(y)


def train_ml_model(train_array):

    X, y = create_ml_dataset(train_array)

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_SEED
            )
        )
    ])

    model.fit(X, y)

    return model


def ml_prediction_scores(model, history_array):

    frequencies = np.zeros(N_NUMBERS + 1)

    for row in history_array:
        for number in row:
            frequencies[int(number)] += 1

    recent20 = np.zeros(N_NUMBERS + 1)
    recent50 = np.zeros(N_NUMBERS + 1)

    for row in history_array[-WINDOW_20:]:
        for number in row:
            recent20[int(number)] += 1

    for row in history_array[-WINDOW_50:]:
        for number in row:
            recent50[int(number)] += 1

    features = []

    n = len(history_array)

    for number in range(1, N_NUMBERS + 1):

        gap = n

        for j in range(n - 1, -1, -1):

            if number in set(history_array[j]):
                gap = n - 1 - j
                break

        features.append([
            frequencies[number],
            recent20[number],
            recent50[number],
            gap
        ])

    probabilities = model.predict_proba(
        np.asarray(features)
    )[:, 1]

    return minmax_normalize(probabilities)


# ============================================================
# PAIRWISE SCORE
# ============================================================

def pairwise_scores(train_array):

    pair_counts = {}

    number_counts = np.zeros(N_NUMBERS + 1)

    for row in train_array:

        numbers = sorted(set(int(x) for x in row))

        for number in numbers:
            number_counts[number] += 1

        for a, b in itertools.combinations(numbers, 2):

            pair = (a, b)

            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    scores = np.zeros(N_NUMBERS + 1)

    for number in range(1, N_NUMBERS + 1):

        pair_values = []

        for other in range(1, N_NUMBERS + 1):

            if other == number:
                continue

            a = min(number, other)
            b = max(number, other)

            pair_count = pair_counts.get((a, b), 0)

            expected = (
                number_counts[number]
                * number_counts[other]
                / max(len(train_array), 1)
            )

            if expected > 0:
                pair_values.append(
                    pair_count / expected
                )

        if pair_values:
            scores[number] = np.mean(pair_values)

    return minmax_normalize(scores[1:])


# ============================================================
# REGIME SCORE
# ============================================================

def regime_scores(train_array):

    n = len(train_array)

    frequencies = np.zeros(N_NUMBERS + 1)

    for row in train_array:
        for number in row:
            frequencies[int(number)] += 1

    recent20 = np.zeros(N_NUMBERS + 1)
    recent50 = np.zeros(N_NUMBERS + 1)

    for row in train_array[-WINDOW_20:]:
        for number in row:
            recent20[int(number)] += 1

    for row in train_array[-WINDOW_50:]:
        for number in row:
            recent50[int(number)] += 1

    # Historical expected frequency per draw
    historical_rate = frequencies / max(n, 1)

    recent20_rate = recent20 / WINDOW_20
    recent50_rate = recent50 / WINDOW_50

    # Recent regime change
    rate_change = recent20_rate - recent50_rate

    # Standardized recent activity
    mean = np.mean(rate_change[1:])
    std = np.std(rate_change[1:])

    if std > 0:
        z_score = (rate_change[1:] - mean) / std
    else:
        z_score = np.zeros(N_NUMBERS)

    positive_z = np.maximum(z_score, 0)

    regime_strength = np.maximum(rate_change[1:], 0)

    score = (
        0.60 * minmax_normalize(positive_z)
        + 0.40 * minmax_normalize(regime_strength)
    )

    return minmax_normalize(score)


# ============================================================
# ENSEMBLE
# ============================================================

def calculate_ensemble(
    train_array
):

    print("Training Logistic Regression...")

    ml_model = train_ml_model(train_array)

    print("Calculating ensemble components...")

    # 1. Frequency
    frequency_raw, frequency_score = frequency_scores(
        train_array
    )

    # 2. Conditional
    conditional = conditional_scores(
        train_array
    )

    conditional_score = conditional["score"]

    # 3. ML
    ml_score = ml_prediction_scores(
        ml_model,
        train_array
    )

    # 4. Pairwise
    pairwise_score = pairwise_scores(
        train_array
    )

    # 5. Regime
    regime_score = regime_scores(
        train_array
    )

    # Final weighted score
    ensemble_score = (
        WEIGHT_FREQUENCY * frequency_score
        + WEIGHT_CONDITIONAL * conditional_score
        + WEIGHT_ML * ml_score
        + WEIGHT_PAIRWISE * pairwise_score
        + WEIGHT_REGIME * regime_score
    )

    ensemble_score = minmax_normalize(
        ensemble_score
    )

    selected = (
        np.argsort(ensemble_score)[::-1][:TOP_K]
        + 1
    )

    return (
        selected.tolist(),
        frequency_raw,
        conditional,
        ml_score,
        pairwise_score,
        regime_score,
        ensemble_score
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ENSEMBLE TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df = df.sort_values(
            "date"
        ).reset_index(drop=True)

    number_columns = [
        c for c in df.columns
        if c.lower() in [
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6"
        ]
    ]

    if len(number_columns) != 6:

        raise ValueError(
            f"Could not find exactly 6 number columns. "
            f"Found: {number_columns}"
        )

    print(
        f"Dataset shape: {df.shape}"
    )

    if "date" in df.columns:

        print(
            f"Date range: "
            f"{df['date'].min()} -> "
            f"{df['date'].max()}"
        )

    print(
        "\nNumber columns:"
    )

    print(number_columns)

    data = (
        df[number_columns]
        .astype(int)
        .values
    )

    print(
        f"\nTotal draws: {len(data)}"
    )

    print(
        f"Number range: "
        f"{data.min()} -> {data.max()}"
    )

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    results = []

    fold = 1
    train_end = TEST_SIZE

    while train_end < len(data):

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            len(data)
        )

        train_array = data[:train_end]

        test_array = data[
            test_start:test_end
        ]

        if len(test_array) == 0:
            break

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        if "date" in df.columns:

            print(
                f"Training: "
                f"{df['date'].iloc[0]} -> "
                f"{df['date'].iloc[train_end - 1]}"
            )

            print(
                f"Testing:  "
                f"{df['date'].iloc[test_start]} -> "
                f"{df['date'].iloc[test_end - 1]}"
            )

        print(
            f"Training draws: {len(train_array)}"
        )

        print(
            f"Testing draws:  {len(test_array)}"
        )

        # ----------------------------------------------------
        # ENSEMBLE
        # ----------------------------------------------------

        print(
            "\nCalculating ensemble scores..."
        )

        (
            selected,
            frequency_raw,
            conditional,
            ml_score,
            pairwise_score,
            regime_score,
            ensemble_score
        ) = calculate_ensemble(
            train_array
        )

        print(
            "\nEnsemble Top-6 selected:"
        )

        print(selected)

        # ----------------------------------------------------
        # SCORE TABLE
        # ----------------------------------------------------

        score_table = pd.DataFrame({

            "number":
                np.arange(1, N_NUMBERS + 1),

            "frequency":
                frequency_raw,

            "freq_20":
                conditional["freq20"],

            "freq_50":
                conditional["freq50"],

            "gap":
                conditional["gap"],

            "conditional_score":
                conditional["score"],

            "ml_score":
                ml_score,

            "pairwise_score":
                pairwise_score,

            "regime_score":
                regime_score,

            "ensemble_score":
                ensemble_score
        })

        score_table = (
            score_table
            .sort_values(
                "ensemble_score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print(
            "\nTop candidates:"
        )

        print(
            score_table.head(10).to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        print(
            "\nEvaluating ensemble Top-6..."
        )

        hits = calculate_hits(
            selected,
            test_array
        )

        average_hits = float(
            np.mean(hits)
        )

        total_hits = int(
            np.sum(hits)
        )

        maximum_hits = int(
            np.max(hits)
        )

        random_expected = (
            TOP_K * 6 / N_NUMBERS
        )

        difference = (
            average_hits
            - random_expected
        )

        difference_percent = (
            difference
            / random_expected
            * 100
        )

        # ----------------------------------------------------
        # RANDOM SIMULATION
        # ----------------------------------------------------

        print(
            "\nRunning random Top-6 simulation..."
        )

        random_results = random_simulation(
            test_array,
            N_SIMULATIONS,
            RANDOM_SEED + fold
        )

        random_mean = float(
            np.mean(random_results)
        )

        random_low = float(
            np.percentile(
                random_results,
                2.5
            )
        )

        random_high = float(
            np.percentile(
                random_results,
                97.5
            )
        )

        empirical_p = float(
            np.mean(
                random_results >= average_hits
            )
        )

        print("\nResults")
        print("-" * 40)

        print(
            f"Average hits:       {average_hits:.6f}"
        )

        print(
            f"Total hits:         {total_hits}"
        )

        print(
            f"Maximum hits:       {maximum_hits}"
        )

        print(
            f"Random expected:    {random_expected:.6f}"
        )

        print(
            f"Difference:         {difference:+.6f}"
        )

        print(
            f"Difference %:       {difference_percent:+.3f}%"
        )

        print(
            f"Random simulation:  {random_mean:.6f}"
        )

        print(
            f"Random 95% range:   "
            f"[{random_low:.6f}, {random_high:.6f}]"
        )

        print(
            f"Empirical p-value:  {empirical_p:.6f}"
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "fold": fold,

            "train_draws": len(train_array),

            "test_draws": len(test_array),

            "selected_numbers":
                ",".join(
                    map(str, selected)
                ),

            "average_hits":
                average_hits,

            "total_hits":
                total_hits,

            "maximum_hits":
                maximum_hits,

            "random_expected":
                random_expected,

            "difference":
                difference,

            "difference_percent":
                difference_percent,

            "random_simulation":
                random_mean,

            "random_low":
                random_low,

            "random_high":
                random_high,

            "empirical_p":
                empirical_p
        })

        # ----------------------------------------------------
        # NEXT FOLD
        # ----------------------------------------------------

        train_end += STEP_SIZE
        fold += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "fold",
                "test_draws",
                "average_hits",
                "random_expected",
                "difference",
                "difference_percent",
                "empirical_p"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # OVERALL
    # ========================================================

    mean_model_hits = float(
        results_df[
            "average_hits"
        ].mean()
    )

    mean_random_expected = float(
        results_df[
            "random_expected"
        ].mean()
    )

    mean_difference = (
        mean_model_hits
        - mean_random_expected
    )

    mean_difference_percent = (
        mean_difference
        / mean_random_expected
        * 100
    )

    mean_empirical_p = float(
        results_df[
            "empirical_p"
        ].mean()
    )

    above_random = int(
        np.sum(
            results_df["difference"] > 0
        )
    )

    below_random = int(
        np.sum(
            results_df["difference"] < 0
        )
    )

    print("\n")
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Mean ensemble hits:       "
        f"{mean_model_hits:.6f}"
    )

    print(
        f"Mean random expected:      "
        f"{mean_random_expected:.6f}"
    )

    print(
        f"Mean difference:           "
        f"{mean_difference:+.6f}"
    )

    print(
        f"Mean difference %:         "
        f"{mean_difference_percent:+.3f}%"
    )

    print(
        f"Mean empirical p:          "
        f"{mean_empirical_p:.6f}"
    )

    print("\nFold consistency:")

    print(
        f"Above random: {above_random}"
    )

    print(
        f"Below random: {below_random}"
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        mean_difference > 0
        and mean_empirical_p < 0.05
        and above_random > below_random
    ):

        print(
            "\nPotential ensemble signal detected."
        )

    else:

        print(
            "\nNo meaningful ensemble advantage "
            "over random selection."
        )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()