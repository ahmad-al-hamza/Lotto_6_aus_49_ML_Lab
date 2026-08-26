from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/processed/lotto_6aus49_clean.csv")
OUTPUT_PATH = Path("data/processed/conditional_top6_results.csv")

NUMBER_MIN = 1
NUMBER_MAX = 49
TOP_K = 6

N_FOLDS = 5
TEST_DRAWS = 1008

RANDOM_SIMULATIONS = 10000
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CONDITIONAL TOP-6 WALK-FORWARD TEST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

number_columns = [
    c for c in ["n1", "n2", "n3", "n4", "n5", "n6"]
    if c in df.columns
]

if len(number_columns) != 6:
    raise ValueError(
        f"Expected n1..n6 columns. Found: {number_columns}"
    )

print(f"Dataset shape: {df.shape}")

if "date" in df.columns:
    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

print(f"\nNumber columns:")
print(number_columns)

print(f"\nTotal draws: {len(df)}")

all_numbers = df[number_columns].to_numpy(dtype=int)

print(
    f"Number range: "
    f"{all_numbers.min()} -> {all_numbers.max()}"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_number_frequency(draws):
    """
    Count how many times each number appeared.
    """
    counts = np.zeros(NUMBER_MAX + 1, dtype=int)

    for row in draws:
        for number in row:
            counts[number] += 1

    return counts


def calculate_gap(draws, number):
    """
    Number of draws since the number last appeared.

    If the number appeared in the latest training draw:
        gap = 1

    If it appeared two draws ago:
        gap = 2

    If never appeared:
        gap = len(draws)
    """

    for i in range(len(draws) - 1, -1, -1):
        if number in draws[i]:
            return len(draws) - i

    return len(draws)


def get_frequency_window(draws, number, window):
    """
    Frequency of a number inside the last `window` draws.
    """

    recent = draws[-window:]

    return sum(number in row for row in recent)


def minmax_normalize(values):
    """
    Normalize values to [0, 1].
    """

    values = np.asarray(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.ones_like(values)

    return (values - minimum) / (maximum - minimum)


def build_number_scores(training_draws):
    """
    Build a conditional score for every number.

    Components:

    1. Long-term frequency
    2. Recent frequency
    3. Gap
    4. Medium-term frequency
    """

    frequencies = get_number_frequency(training_draws)

    freq_20 = np.array([
        get_frequency_window(training_draws, n, 20)
        for n in range(NUMBER_MIN, NUMBER_MAX + 1)
    ])

    freq_50 = np.array([
        get_frequency_window(training_draws, n, 50)
        for n in range(NUMBER_MIN, NUMBER_MAX + 1)
    ])

    gaps = np.array([
        calculate_gap(training_draws, n)
        for n in range(NUMBER_MIN, NUMBER_MAX + 1)
    ])

    long_freq = frequencies[
        NUMBER_MIN:NUMBER_MAX + 1
    ]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    frequency_score = minmax_normalize(long_freq)

    recent_score = minmax_normalize(freq_20)

    medium_score = minmax_normalize(freq_50)

    # --------------------------------------------------------
    # Gap score
    #
    # We deliberately give gap a WEAK weight.
    # The previous recency analysis showed almost no
    # predictive effect.
    # --------------------------------------------------------

    gap_score = minmax_normalize(gaps)

    # --------------------------------------------------------
    # Combined score
    #
    # Frequency is dominant.
    # Recency and gap are secondary.
    # --------------------------------------------------------

    score = (
        0.50 * frequency_score
        + 0.20 * recent_score
        + 0.20 * medium_score
        + 0.10 * gap_score
    )

    result = pd.DataFrame({
        "number": np.arange(
            NUMBER_MIN,
            NUMBER_MAX + 1
        ),
        "frequency": long_freq,
        "freq_20": freq_20,
        "freq_50": freq_50,
        "gap": gaps,
        "score": score
    })

    result = result.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    return result


def calculate_hits(selected_numbers, test_draws):
    """
    Calculate hits per draw.
    """

    selected = set(selected_numbers)

    hits = []

    for row in test_draws:
        hit_count = len(
            selected.intersection(set(row))
        )

        hits.append(hit_count)

    return np.array(hits)


def random_top6_simulation(
    test_draws,
    simulations=10000,
    seed=42
):
    """
    Monte Carlo random Top-6 baseline.
    """

    rng = np.random.default_rng(seed)

    results = []

    for _ in range(simulations):

        random_numbers = rng.choice(
            np.arange(
                NUMBER_MIN,
                NUMBER_MAX + 1
            ),
            size=TOP_K,
            replace=False
        )

        hits = calculate_hits(
            random_numbers,
            test_draws
        )

        results.append(hits.mean())

    return np.array(results)


# ============================================================
# CREATE WALK-FORWARD FOLDS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING WALK-FORWARD FOLDS")
print("=" * 70)

total_draws = len(df)

results = []

rng = np.random.default_rng(RANDOM_SEED)


for fold in range(1, N_FOLDS + 1):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    # --------------------------------------------------------
    # Expanding training window
    # --------------------------------------------------------

    test_end = total_draws - (
        N_FOLDS - fold
    ) * TEST_DRAWS

    test_start = test_end - TEST_DRAWS

    if test_start <= 0:
        print("Skipping fold because there is no training data.")
        continue

    train_draws = all_numbers[:test_start]
    test_draws = all_numbers[test_start:test_end]

    print(
        f"Training: "
        f"{df.iloc[0]['date'] if 'date' in df.columns else 0}"
        f" -> "
        f"{df.iloc[test_start - 1]['date'] if 'date' in df.columns else test_start - 1}"
    )

    print(
        f"Testing:  "
        f"{df.iloc[test_start]['date'] if 'date' in df.columns else test_start}"
        f" -> "
        f"{df.iloc[test_end - 1]['date'] if 'date' in df.columns else test_end - 1}"
    )

    print(f"Training draws: {len(train_draws)}")
    print(f"Testing draws:  {len(test_draws)}")

    # ========================================================
    # BUILD SCORES
    # ========================================================

    scores = build_number_scores(train_draws)

    top6 = scores.head(TOP_K)["number"].tolist()

    print("\nConditional Top-6 selected:")
    print(top6)

    print("\nScores:")

    display_columns = [
        "number",
        "frequency",
        "freq_20",
        "freq_50",
        "gap",
        "score"
    ]

    print(
        scores.head(TOP_K)[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # MODEL HITS
    # ========================================================

    model_hits = calculate_hits(
        top6,
        test_draws
    )

    model_mean = model_hits.mean()

    model_total = model_hits.sum()

    model_max = model_hits.max()

    # ========================================================
    # RANDOM BASELINE
    # ========================================================

    print("\nRunning random Top-6 simulation...")

    random_results = random_top6_simulation(
        test_draws,
        simulations=RANDOM_SIMULATIONS,
        seed=RANDOM_SEED + fold
    )

    random_mean = random_results.mean()

    random_low = np.percentile(
        random_results,
        2.5
    )

    random_high = np.percentile(
        random_results,
        97.5
    )

    # ========================================================
    # THEORETICAL EXPECTATION
    # ========================================================

    random_expected = TOP_K * TOP_K / NUMBER_MAX

    difference = model_mean - random_expected

    difference_percent = (
        difference / random_expected
    ) * 100

    # ========================================================
    # EMPIRICAL P-VALUE
    # ========================================================

    p_value = (
        np.sum(random_results >= model_mean)
        + 1
    ) / (
        len(random_results) + 1
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nResults")
    print("-" * 40)

    print(
        f"Average hits:       {model_mean:.6f}"
    )

    print(
        f"Total hits:         {model_total}"
    )

    print(
        f"Maximum hits:       {model_max}"
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
        f"Empirical p-value:  {p_value:.6f}"
    )

    # ========================================================
    # SAVE FOLD RESULT
    # ========================================================

    results.append({
        "fold": fold,
        "train_draws": len(train_draws),
        "test_draws": len(test_draws),

        "top6": ",".join(
            map(str, top6)
        ),

        "average_hits": model_mean,
        "total_hits": model_total,
        "maximum_hits": model_max,

        "random_expected": random_expected,

        "difference": difference,
        "difference_percent": difference_percent,

        "random_simulation_mean": random_mean,
        "random_ci_low": random_low,
        "random_ci_high": random_high,

        "empirical_p": p_value
    })


# ============================================================
# WALK-FORWARD SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

if len(results_df) == 0:
    raise RuntimeError(
        "No valid folds were generated."
    )

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
    ].to_string(index=False)
)


# ============================================================
# OVERALL
# ============================================================

mean_model = results_df[
    "average_hits"
].mean()

mean_random = results_df[
    "random_expected"
].mean()

mean_difference = (
    mean_model - mean_random
)

mean_difference_percent = (
    mean_difference / mean_random
) * 100

mean_p = results_df[
    "empirical_p"
].mean()

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
    f"Mean model hits:       {mean_model:.6f}"
)

print(
    f"Mean random expected:  {mean_random:.6f}"
)

print(
    f"Mean difference:       {mean_difference:+.6f}"
)

print(
    f"Mean difference %:     "
    f"{mean_difference_percent:+.3f}%"
)

print(
    f"Mean empirical p:      {mean_p:.6f}"
)

print("\nFold consistency:")

print(
    f"Above random: {above_random}"
)

print(
    f"Below random: {below_random}"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

if (
    mean_difference > 0
    and mean_p < 0.05
    and above_random > below_random
):

    conclusion = (
        "Potential advantage detected. "
        "Further validation is required."
    )

elif abs(mean_difference_percent) < 1:

    conclusion = (
        "No meaningful advantage over random selection."
    )

elif mean_difference < 0:

    conclusion = (
        "Conditional Top-6 performs worse than random selection."
    )

else:

    conclusion = (
        "Some advantage appears, "
        "but statistical evidence is insufficient."
    )

print("\n" + conclusion)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("Results saved to:")
print(OUTPUT_PATH)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)