import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
RESULTS_PATH = "data/processed/recency_top6_walk_forward_results.csv"

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

N_NUMBERS = 49
TOP_K = 6

# Number of recent draws used for training
WINDOW_SIZE = 1000

# Test block size
TEST_SIZE = 1008

# Random simulation
N_SIMULATIONS = 10000

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("RECENCY TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])

    else:
        raise ValueError("Could not find date column.")

    df = df.sort_values("date").reset_index(drop=True)

    missing = [c for c in NUMBER_COLS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing number columns: {missing}"
        )

    print(f"Dataset shape: {df.shape}")

    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    print("\nNumber columns:")
    print(NUMBER_COLS)

    print(f"\nTotal draws: {len(df)}")

    print(
        f"Number range: "
        f"{df[NUMBER_COLS].min().min()} -> "
        f"{df[NUMBER_COLS].max().max()}"
    )

    return df


# ============================================================
# CREATE TRAINING DATA
# ============================================================

def create_training_samples(history):

    X = []
    y = []

    numbers = history[NUMBER_COLS].values

    if len(numbers) < 101:
        return np.empty((0, 9)), np.empty((0,))

    # For every draw after the first 100 draws
    for i in range(100, len(numbers)):

        previous = numbers[:i]

        features = []

        for number in range(1, N_NUMBERS + 1):

            mask = previous == number

            total_frequency = mask.sum()

            freq_20 = mask[-20:].sum()

            freq_50 = mask[-50:].sum()

            freq_100 = mask[-100:].sum()

            # Draws since last appearance
            positions = np.where(mask)[0]

            if len(positions) > 0:
                gap = i - positions[-1]
            else:
                gap = i

            # Recent frequency ratios
            rate_20 = freq_20 / 20.0
            rate_50 = freq_50 / 50.0

            # Long-term rate
            rate_total = total_frequency / float(i)

            # Recent momentum
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
                momentum
            ])

        # Target for next draw
        target_draw = numbers[i]

        for number in range(1, N_NUMBERS + 1):

            X.append(features[number - 1])

            y.append(
                1 if number in target_draw else 0
            )

    return np.asarray(X), np.asarray(y)


# ============================================================
# SCORE CURRENT NUMBERS
# ============================================================

def calculate_scores(history, model):

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
            momentum
        ])

    X = np.asarray(features)

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities


# ============================================================
# RANDOM BASELINE
# ============================================================

def calculate_hits(selected, draw):

    return len(set(selected) & set(draw))


def random_simulation(test_draws, n_simulations=N_SIMULATIONS):

    rng = np.random.default_rng(RANDOM_SEED)

    hits = []

    for _ in range(n_simulations):

        total_hits = 0

        for draw in test_draws:

            random_numbers = rng.choice(
                np.arange(1, N_NUMBERS + 1),
                size=TOP_K,
                replace=False
            )

            total_hits += calculate_hits(
                random_numbers,
                draw
            )

        hits.append(
            total_hits / len(test_draws)
        )

    hits = np.asarray(hits)

    return (
        hits.mean(),
        np.percentile(hits, 2.5),
        np.percentile(hits, 97.5)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    print("\n")
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    results = []

    total_draws = len(df)

    fold = 1

    test_start = WINDOW_SIZE

    while test_start < total_draws:

        test_end = min(
            test_start + TEST_SIZE,
            total_draws
        )

        train_start = max(
            0,
            test_start - WINDOW_SIZE
        )

        train_end = test_start

        train_df = df.iloc[
            train_start:train_end
        ].copy()

        test_df = df.iloc[
            test_start:test_end
        ].copy()

        if len(train_df) < 200:
            break

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        print(
            f"Training: "
            f"{train_df['date'].iloc[0]} -> "
            f"{train_df['date'].iloc[-1]}"
        )

        print(
            f"Testing:  "
            f"{test_df['date'].iloc[0]} -> "
            f"{test_df['date'].iloc[-1]}"
        )

        print(
            f"Training draws: {len(train_df)}"
        )

        print(
            f"Testing draws:  {len(test_df)}"
        )

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        print("\nCreating training samples...")

        X_train, y_train = create_training_samples(
            train_df
        )

        print(
            f"Training feature shape: {X_train.shape}"
        )

        if len(X_train) == 0:
            print("Skipping fold.")
            test_start = test_end
            fold += 1
            continue

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print("\nTraining Logistic Regression...")

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        print("\nCalculating recent-history scores...")

        scores = calculate_scores(
            train_df,
            model
        )

        ranking = np.argsort(
            scores
        )[::-1]

        selected = (
            ranking[:TOP_K] + 1
        ).tolist()

        print("\nRecency Top-6 selected:")
        print(selected)

        # ----------------------------------------------------
        # SCORE TABLE
        # ----------------------------------------------------

        numbers = train_df[NUMBER_COLS].values

        table = []

        for number in range(
            1,
            N_NUMBERS + 1
        ):

            mask = numbers == number

            positions = np.where(mask)[0]

            if len(positions) > 0:
                gap = (
                    len(numbers)
                    - positions[-1]
                )
            else:
                gap = len(numbers)

            table.append({
                "number": number,
                "frequency": mask.sum(),
                "freq_20": mask[-20:].sum(),
                "freq_50": mask[-50:].sum(),
                "freq_100": mask[-100:].sum(),
                "gap": gap,
                "ml_score": scores[number - 1]
            })

        score_table = pd.DataFrame(table)

        score_table = score_table.sort_values(
            "ml_score",
            ascending=False
        )

        print("\nTop candidates:")
        print(
            score_table.head(10).to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        print("\nEvaluating Recency Top-6...")

        test_draws = (
            test_df[NUMBER_COLS]
            .values
        )

        hit_values = []

        for draw in test_draws:

            hits = calculate_hits(
                selected,
                draw
            )

            hit_values.append(hits)

        hit_values = np.asarray(
            hit_values
        )

        average_hits = hit_values.mean()

        total_hits = hit_values.sum()

        maximum_hits = hit_values.max()

        # ----------------------------------------------------
        # RANDOM BASELINE
        # ----------------------------------------------------

        random_expected = (
            TOP_K * TOP_K
        ) / N_NUMBERS

        print(
            "\nRunning random Top-6 simulation..."
        )

        (
            random_sim,
            random_low,
            random_high
        ) = random_simulation(
            test_draws
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

        # Empirical probability that random
        # simulation performs at least as well
        empirical_p = (
            np.mean(
                np.array([
                    random_sim
                ]) >= average_hits
            )
        )

        # Better empirical p-value:
        # run direct random test distribution again
        rng = np.random.default_rng(
            RANDOM_SEED + fold
        )

        random_fold_scores = []

        for _ in range(
            5000
        ):

            total = 0

            for draw in test_draws:

                random_numbers = rng.choice(
                    np.arange(
                        1,
                        N_NUMBERS + 1
                    ),
                    size=TOP_K,
                    replace=False
                )

                total += calculate_hits(
                    random_numbers,
                    draw
                )

            random_fold_scores.append(
                total / len(test_draws)
            )

        random_fold_scores = np.asarray(
            random_fold_scores
        )

        empirical_p = (
            np.mean(
                random_fold_scores
                >= average_hits
            )
        )

        print("\nResults")
        print("-" * 40)

        print(
            f"Average hits:       "
            f"{average_hits:.6f}"
        )

        print(
            f"Total hits:         "
            f"{total_hits}"
        )

        print(
            f"Maximum hits:       "
            f"{maximum_hits}"
        )

        print(
            f"Random expected:    "
            f"{random_expected:.6f}"
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
            f"{random_sim:.6f}"
        )

        print(
            f"Random 95% range:   "
            f"[{random_low:.6f}, "
            f"{random_high:.6f}]"
        )

        print(
            f"Empirical p-value:  "
            f"{empirical_p:.6f}"
        )

        results.append({
            "fold": fold,
            "train_draws": len(train_df),
            "test_draws": len(test_df),
            "average_hits": average_hits,
            "random_expected": random_expected,
            "difference": difference,
            "difference_percent": difference_percent,
            "empirical_p": empirical_p,
            "selected_numbers": ",".join(
                map(str, selected)
            )
        })

        test_start = test_end
        fold += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    if len(results_df) > 0:

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

        mean_model = (
            results_df[
                "average_hits"
            ].mean()
        )

        mean_random = (
            results_df[
                "random_expected"
            ].mean()
        )

        mean_difference = (
            results_df[
                "difference"
            ].mean()
        )

        mean_difference_percent = (
            mean_difference
            / mean_random
            * 100
        )

        mean_p = (
            results_df[
                "empirical_p"
            ].mean()
        )

        above = (
            results_df[
                "difference"
            ] > 0
        ).sum()

        below = (
            results_df[
                "difference"
            ] < 0
        ).sum()

        print("\n")
        print("=" * 70)
        print("OVERALL")
        print("=" * 70)

        print(
            f"Mean recency hits:      "
            f"{mean_model:.6f}"
        )

        print(
            f"Mean random expected:   "
            f"{mean_random:.6f}"
        )

        print(
            f"Mean difference:        "
            f"{mean_difference:+.6f}"
        )

        print(
            f"Mean difference %:      "
            f"{mean_difference_percent:+.3f}%"
        )

        print(
            f"Mean empirical p:       "
            f"{mean_p:.6f}"
        )

        print("\nFold consistency:")
        print(
            f"Above random: {above}"
        )
        print(
            f"Below random: {below}"
        )

        print("\n")
        print("=" * 70)
        print("FINAL CONCLUSION")
        print("=" * 70)

        if (
            mean_difference > 0
            and mean_p < 0.05
            and above > below
        ):

            print(
                "\nPotential recency advantage detected."
            )

        else:

            print(
                "\nNo meaningful recency advantage "
                "over random selection."
            )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n")
    print(
        "Results saved to:"
    )

    print(RESULTS_PATH)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()