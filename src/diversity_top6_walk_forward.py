import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
RESULT_PATH = "data/processed/diversity_top6_walk_forward_results.csv"

NUMBER_MIN = 1
NUMBER_MAX = 49
TOP_K = 6

TRAIN_SIZE = 1008
TEST_SIZE = 1008

RANDOM_SIMULATIONS = 5000
RANDOM_SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    number_cols = [f"n{i}" for i in range(1, 7)]

    return df, number_cols


def create_folds(df):
    folds = []

    train_end = TRAIN_SIZE

    while train_end < len(df):
        test_start = train_end
        test_end = min(test_start + TEST_SIZE, len(df))

        if test_end - test_start < 100:
            break

        folds.append(
            (
                df.iloc[:train_end].copy(),
                df.iloc[test_start:test_end].copy()
            )
        )

        train_end += TEST_SIZE

    return folds


def number_frequency(train_df, number_cols):
    counts = np.zeros(NUMBER_MAX + 1)

    for col in number_cols:
        values = train_df[col].astype(int).values

        for value in values:
            if NUMBER_MIN <= value <= NUMBER_MAX:
                counts[value] += 1

    return counts


def window_frequency(train_df, number_cols, window):
    recent = train_df.tail(window)

    counts = np.zeros(NUMBER_MAX + 1)

    for col in number_cols:
        values = recent[col].astype(int).values

        for value in values:
            if NUMBER_MIN <= value <= NUMBER_MAX:
                counts[value] += 1

    return counts


def calculate_gap(train_df, number_cols):
    last_seen = {}

    for number in range(NUMBER_MIN, NUMBER_MAX + 1):
        last_seen[number] = None

    for idx, row in train_df.iterrows():
        for col in number_cols:
            number = int(row[col])

            if NUMBER_MIN <= number <= NUMBER_MAX:
                last_seen[number] = idx

    last_index = train_df.index[-1]

    gaps = np.zeros(NUMBER_MAX + 1)

    for number in range(NUMBER_MIN, NUMBER_MAX + 1):
        if last_seen[number] is None:
            gaps[number] = len(train_df)
        else:
            gaps[number] = last_index - last_seen[number]

    return gaps


def calculate_pairwise_diversity(train_df, number_cols):
    """
    Measures how different each number's historical behavior is
    from the other numbers.

    Higher value = more unique/diverse behavior.
    """

    features = []

    windows = [20, 50, 100, 200]

    for window in windows:
        counts = window_frequency(train_df, number_cols, window)

        total = window * len(number_cols)

        rates = counts / total

        features.append(rates[1:])

    feature_matrix = np.column_stack(features)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(feature_matrix)

    diversity = np.zeros(NUMBER_MAX + 1)

    for i in range(NUMBER_MAX):
        distances = np.linalg.norm(
            normalized - normalized[i],
            axis=1
        )

        distances[i] = np.nan

        diversity[i + 1] = np.nanmean(distances)

    return diversity


def calculate_coverage_score(train_df, number_cols):
    """
    Rewards numbers that contribute to coverage across
    different historical windows.
    """

    windows = [20, 50, 100, 200]

    rates = []

    for window in windows:
        counts = window_frequency(train_df, number_cols, window)

        total = window * len(number_cols)

        rates.append(counts / total)

    rates = np.array(rates)

    mean_rate = np.mean(rates, axis=0)
    std_rate = np.std(rates, axis=0)

    # Stable presence across windows
    consistency = 1.0 / (1.0 + std_rate)

    # Moderate frequency preference
    frequency_component = mean_rate

    score = (
        0.60 * frequency_component +
        0.40 * consistency
    )

    return score


def calculate_diversity_scores(train_df, number_cols):

    frequency = number_frequency(train_df, number_cols)

    freq_20 = window_frequency(train_df, number_cols, 20)
    freq_50 = window_frequency(train_df, number_cols, 50)
    freq_100 = window_frequency(train_df, number_cols, 100)

    gaps = calculate_gap(train_df, number_cols)

    diversity = calculate_pairwise_diversity(
        train_df,
        number_cols
    )

    coverage = calculate_coverage_score(
        train_df,
        number_cols
    )

    # Normalize components
    def normalize(values):

        values = values.astype(float)

        min_v = np.min(values[1:])
        max_v = np.max(values[1:])

        if max_v - min_v == 0:
            return np.ones_like(values)

        result = np.zeros_like(values)

        result[1:] = (
            (values[1:] - min_v)
            / (max_v - min_v)
        )

        return result

    frequency_score = normalize(frequency)

    freq20_score = normalize(freq_20)

    freq50_score = normalize(freq_50)

    freq100_score = normalize(freq_100)

    diversity_score = normalize(diversity)

    coverage_score = normalize(coverage)

    # Avoid extremely overdue numbers dominating
    gap_score = normalize(np.minimum(gaps, 30))

    final_score = (
        0.20 * frequency_score +
        0.15 * freq20_score +
        0.15 * freq50_score +
        0.10 * freq100_score +
        0.20 * diversity_score +
        0.15 * coverage_score +
        0.05 * gap_score
    )

    result = pd.DataFrame({
        "number": np.arange(NUMBER_MIN, NUMBER_MAX + 1),
        "frequency": frequency[1:],
        "freq_20": freq_20[1:],
        "freq_50": freq_50[1:],
        "freq_100": freq_100[1:],
        "gap": gaps[1:],
        "diversity": diversity[1:],
        "coverage": coverage[1:],
        "diversity_score": final_score[1:]
    })

    result = result.sort_values(
        "diversity_score",
        ascending=False
    ).reset_index(drop=True)

    return result


def evaluate_selection(selected_numbers, test_df, number_cols):

    hits = []

    selected = set(selected_numbers)

    for _, row in test_df.iterrows():

        actual = {
            int(row[col])
            for col in number_cols
        }

        hit_count = len(selected.intersection(actual))

        hits.append(hit_count)

    return np.mean(hits), np.sum(hits), np.max(hits)


def random_simulation(test_df, number_cols):

    rng = np.random.default_rng(RANDOM_SEED)

    averages = []

    for _ in range(RANDOM_SIMULATIONS):

        selected = rng.choice(
            np.arange(NUMBER_MIN, NUMBER_MAX + 1),
            size=TOP_K,
            replace=False
        )

        avg_hits, _, _ = evaluate_selection(
            selected,
            test_df,
            number_cols
        )

        averages.append(avg_hits)

    return np.array(averages)


def main():

    print("=" * 70)
    print("DIVERSITY TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print("\nLoading dataset...")

    df, number_cols = load_data()

    print(f"Dataset shape: {df.shape}")

    if "date" in df.columns:
        print(
            f"Date range: "
            f"{df['date'].min()} -> {df['date'].max()}"
        )

    print("\nNumber columns:")
    print(number_cols)

    print(f"\nTotal draws: {len(df)}")
    print(f"Number range: {NUMBER_MIN} -> {NUMBER_MAX}")

    print("\n")
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    folds = create_folds(df)

    results = []

    for fold_number, (train_df, test_df) in enumerate(
        folds,
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold_number}")
        print("=" * 70)

        if "date" in df.columns:

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

        print(f"Training draws: {len(train_df)}")
        print(f"Testing draws:  {len(test_df)}")

        print("\nCalculating diversity scores...")

        scores = calculate_diversity_scores(
            train_df,
            number_cols
        )

        selected = scores.head(TOP_K)["number"].tolist()

        print("\nDiversity Top-6 selected:")
        print(selected)

        print("\nTop candidates:")
        print(scores.head(10).to_string(index=False))

        print("\nEvaluating Diversity Top-6...")

        avg_hits, total_hits, max_hits = evaluate_selection(
            selected,
            test_df,
            number_cols
        )

        print("\nRunning random Top-6 simulation...")

        random_results = random_simulation(
            test_df,
            number_cols
        )

        random_expected = 6 * 6 / 49

        difference = avg_hits - random_expected

        difference_percent = (
            difference / random_expected
        ) * 100

        empirical_p = (
            np.sum(random_results >= avg_hits)
            / len(random_results)
        )

        lower = np.percentile(random_results, 2.5)
        upper = np.percentile(random_results, 97.5)

        print("\nResults")
        print("-" * 40)
        print(f"Average hits:       {avg_hits:.6f}")
        print(f"Total hits:         {total_hits}")
        print(f"Maximum hits:       {max_hits}")
        print(f"Random expected:    {random_expected:.6f}")
        print(f"Difference:         {difference:+.6f}")
        print(f"Difference %:       {difference_percent:+.3f}%")
        print(f"Random simulation:  {np.mean(random_results):.6f}")
        print(
            f"Random 95% range:   "
            f"[{lower:.6f}, {upper:.6f}]"
        )
        print(f"Empirical p-value:  {empirical_p:.6f}")

        results.append({
            "fold": fold_number,
            "test_draws": len(test_df),
            "average_hits": avg_hits,
            "random_expected": random_expected,
            "difference": difference,
            "difference_percent": difference_percent,
            "empirical_p": empirical_p
        })

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    print(results_df.to_string(index=False))

    mean_hits = results_df["average_hits"].mean()
    mean_random = results_df["random_expected"].mean()
    mean_difference = results_df["difference"].mean()
    mean_difference_percent = results_df[
        "difference_percent"
    ].mean()

    mean_p = results_df["empirical_p"].mean()

    above_random = np.sum(
        results_df["difference"] > 0
    )

    below_random = np.sum(
        results_df["difference"] < 0
    )

    print("\n")
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Mean diversity hits:   {mean_hits:.6f}"
    )

    print(
        f"Mean random expected:   {mean_random:.6f}"
    )

    print(
        f"Mean difference:        {mean_difference:+.6f}"
    )

    print(
        f"Mean difference %:      "
        f"{mean_difference_percent:+.3f}%"
    )

    print(
        f"Mean empirical p:       {mean_p:.6f}"
    )

    print("\nFold consistency:")
    print(f"Above random: {above_random}")
    print(f"Below random: {below_random}")

    print("\n")
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if (
        mean_difference > 0
        and above_random > below_random
        and mean_p < 0.05
    ):
        print(
            "\nPotential diversity advantage detected."
        )
    else:
        print(
            "\nNo meaningful diversity advantage "
            "over random selection."
        )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\nResults saved to:")
    print(RESULT_PATH)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()