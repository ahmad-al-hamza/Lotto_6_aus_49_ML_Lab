# Window Optimization Top-6 Walk-Forward

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

WINDOWS = [250, 500, 750, 1000, 1500, 2000]
TOP_K = 6
MAX_NUMBER = 49


def create_features(df, number):
    values = df[[f"n{i}" for i in range(1, 7)]].values
    target = np.array([
        1 if number in row else 0
        for row in values
    ])

    features = []

    for i in range(len(df)):
        if i < 100:
            features.append([0] * 9)
            continue

        history = target[:i]

        freq_20 = history[-20:].sum()
        freq_50 = history[-50:].sum()
        freq_100 = history[-100:].sum()

        gap = 0
        for j in range(len(history) - 1, -1, -1):
            if history[j] == 1:
                break
            gap += 1

        recent_rate_20 = freq_20 / 20
        recent_rate_50 = freq_50 / 50
        recent_rate_100 = freq_100 / 100

        total_rate = history.mean()

        features.append([
            freq_20,
            freq_50,
            freq_100,
            gap,
            recent_rate_20,
            recent_rate_50,
            recent_rate_100,
            total_rate,
            i / len(df)
        ])

    return np.array(features), target


def calculate_window_scores(train_df, window):
    scores = {}

    for number in range(1, MAX_NUMBER + 1):

        X, y = create_features(train_df, number)

        valid = np.arange(100, len(train_df))

        X_train = X[valid]
        y_train = y[valid]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        start = max(100, len(train_df) - window)

        X_recent = X[start:]
        y_recent = y[start:]

        if len(X_recent) == 0:
            continue

        X_recent_scaled = scaler.transform(X_recent)

        probabilities = model.predict_proba(X_recent_scaled)[:, 1]

        recent_score = probabilities.mean()

        frequency = y_recent.sum()
        frequency_rate = frequency / len(y_recent)

        scores[number] = (
            0.7 * recent_score +
            0.3 * frequency_rate
        )

    return scores


def evaluate(test_df, selected_numbers):
    hits = []

    for _, row in test_df.iterrows():
        actual = set(row[[f"n{i}" for i in range(1, 7)]])
        hit_count = len(actual.intersection(selected_numbers))
        hits.append(hit_count)

    return np.mean(hits), np.sum(hits), max(hits)


def random_expected():
    return TOP_K * 6 / MAX_NUMBER


def main():

    print("=" * 70)
    print("WINDOW OPTIMIZATION TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    number_cols = [f"n{i}" for i in range(1, 7)]

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    # ---------------------------------------------------------
    # WALK-FORWARD FOLDS
    # ---------------------------------------------------------

    fold_size = 1008

    folds = []

    train_end = fold_size

    while train_end < len(df):

        test_end = min(
            train_end + fold_size,
            len(df)
        )

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if len(test_df) == 0:
            break

        folds.append((train_df, test_df))

        train_end = test_end

    print("\nNumber of folds:", len(folds))

    all_results = []

    # ---------------------------------------------------------
    # TEST EACH WINDOW
    # ---------------------------------------------------------

    for window in WINDOWS:

        print("\n" + "=" * 70)
        print(f"WINDOW = {window}")
        print("=" * 70)

        fold_results = []

        for fold_id, (train_df, test_df) in enumerate(
            folds, start=1
        ):

            print(f"\nFOLD {fold_id}")

            print(
                f"Training: "
                f"{train_df['date'].min()} -> "
                f"{train_df['date'].max()}"
            )

            print(
                f"Testing:  "
                f"{test_df['date'].min()} -> "
                f"{test_df['date'].max()}"
            )

            print(
                f"Training draws: {len(train_df)}"
            )

            print(
                f"Testing draws: {len(test_df)}"
            )

            print(
                f"\nCalculating window-{window} scores..."
            )

            scores = calculate_window_scores(
                train_df,
                window
            )

            ranked = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            selected = [
                number
                for number, score in ranked[:TOP_K]
            ]

            print("\nSelected Top-6:")
            print(selected)

            print("\nTop candidates:")

            for number, score in ranked[:10]:
                print(
                    f"{number:2d} -> "
                    f"{score:.6f}"
                )

            avg_hits, total_hits, max_hits = evaluate(
                test_df,
                selected
            )

            expected = random_expected()

            difference = avg_hits - expected

            difference_percent = (
                difference / expected * 100
            )

            print("\nResults")
            print("-" * 40)
            print(
                f"Average hits:       {avg_hits:.6f}"
            )
            print(
                f"Total hits:         {total_hits}"
            )
            print(
                f"Maximum hits:       {max_hits}"
            )
            print(
                f"Random expected:    {expected:.6f}"
            )
            print(
                f"Difference:         {difference:+.6f}"
            )
            print(
                f"Difference %:       "
                f"{difference_percent:+.3f}%"
            )

            fold_results.append({
                "fold": fold_id,
                "window": window,
                "test_draws": len(test_df),
                "average_hits": avg_hits,
                "random_expected": expected,
                "difference": difference,
                "difference_percent": difference_percent
            })

        all_results.extend(fold_results)

        results_df = pd.DataFrame(fold_results)

        print("\nWINDOW SUMMARY")
        print(results_df.to_string(index=False))

        print("\nWindow mean:")
        print(
            f"Average hits: "
            f"{results_df['average_hits'].mean():.6f}"
        )

        print(
            f"Difference: "
            f"{results_df['difference'].mean():+.6f}"
        )

        print(
            f"Difference %: "
            f"{results_df['difference_percent'].mean():+.3f}%"
        )

    # ---------------------------------------------------------
    # FINAL COMPARISON
    # ---------------------------------------------------------

    final_df = pd.DataFrame(all_results)

    summary = (
        final_df
        .groupby("window")
        .agg(
            mean_hits=("average_hits", "mean"),
            mean_difference=("difference", "mean"),
            mean_difference_percent=(
                "difference_percent",
                "mean"
            ),
            above_random=(
                "difference",
                lambda x: (x > 0).sum()
            ),
            below_random=(
                "difference",
                lambda x: (x < 0).sum()
            )
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "mean_difference",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FINAL WINDOW COMPARISON")
    print("=" * 70)

    print(
        summary.to_string(index=False)
    )

    best_window = summary.iloc[0]

    print("\n" + "=" * 70)
    print("BEST WINDOW")
    print("=" * 70)

    print(
        f"Window: {int(best_window['window'])}"
    )

    print(
        f"Mean hits: "
        f"{best_window['mean_hits']:.6f}"
    )

    print(
        f"Mean difference: "
        f"{best_window['mean_difference']:+.6f}"
    )

    print(
        f"Mean difference %: "
        f"{best_window['mean_difference_percent']:+.3f}%"
    )

    print(
        f"Above random: "
        f"{int(best_window['above_random'])}"
    )

    print(
        f"Below random: "
        f"{int(best_window['below_random'])}"
    )

    output_path = (
        "data/processed/"
        "window_optimization_top6_walk_forward_results.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
