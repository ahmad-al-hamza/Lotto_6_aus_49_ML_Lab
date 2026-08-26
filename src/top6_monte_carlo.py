import pandas as pd
import numpy as np


# =========================
# Load Lotto data
# =========================

lotto = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

lotto["date"] = pd.to_datetime(
    lotto["date"]
)


# =========================
# Load model results
# =========================

results = pd.read_csv(
    "data/processed/top6_results.csv"
)

results["date"] = pd.to_datetime(
    results["date"]
)


# =========================
# Test period
# =========================

test_dates = results["date"]

test_lotto = lotto[
    lotto["date"].isin(test_dates)
].copy()


number_columns = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6"
]


# =========================
# Prepare actual draws
# =========================

actual_draws = (
    test_lotto[number_columns]
    .astype(int)
    .values
)


print(
    "Test draws:",
    len(actual_draws)
)


# =========================
# Monte Carlo
# =========================

SIMULATIONS = 10000

rng = np.random.default_rng(42)

simulation_means = []


for simulation in range(SIMULATIONS):

    total_hits = 0

    for actual in actual_draws:

        random_numbers = rng.choice(
            np.arange(1, 50),
            size=6,
            replace=False
        )

        hits = len(
            set(random_numbers)
            &
            set(actual)
        )

        total_hits += hits

    average_hits = (
        total_hits /
        len(actual_draws)
    )

    simulation_means.append(
        average_hits
    )


simulation_means = np.array(
    simulation_means
)


# =========================
# Statistics
# =========================

real_model_hits = results["hits"].mean()


print("\nMonte Carlo Results")
print("-" * 30)

print(
    "Simulations:",
    SIMULATIONS
)

print(
    "Mean:",
    simulation_means.mean()
)

print(
    "Standard deviation:",
    simulation_means.std()
)

print(
    "5th percentile:",
    np.percentile(
        simulation_means,
        5
    )
)

print(
    "50th percentile:",
    np.percentile(
        simulation_means,
        50
    )
)

print(
    "95th percentile:",
    np.percentile(
        simulation_means,
        95
    )
)

print(
    "99th percentile:",
    np.percentile(
        simulation_means,
        99
    )
)

print(
    "\nRandom expected:",
    36 / 49
)

print(
    "Random Forest:",
    real_model_hits
)


# =========================
# Empirical p-value
# =========================

count = np.sum(
    simulation_means >= real_model_hits
)

p_value = (
    (count + 1) /
    (SIMULATIONS + 1)
)


print(
    "\nRandom simulations >= model:",
    count
)

print(
    "Empirical p-value:",
    p_value
)