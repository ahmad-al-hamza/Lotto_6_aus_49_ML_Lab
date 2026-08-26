import pandas as pd
import numpy as np


# =========================
# Load XGBoost results
# =========================

results = pd.read_csv(
    "data/processed/xgboost_top6_results.csv"
)

model_mean = results["hits"].mean()

test_draws = len(results)

print("Test draws:", test_draws)
print("XGBoost Average Hits:", model_mean)


# =========================
# Monte Carlo
# =========================

SIMULATIONS = 10000

rng = np.random.default_rng(42)

simulation_means = []


for _ in range(SIMULATIONS):

    total_hits = 0

    for _ in range(test_draws):

        # Random 6 numbers from 1-49
        random_numbers = rng.choice(
            np.arange(1, 50),
            size=6,
            replace=False
        )

        # Actual Lotto draw
        # We only need the probability distribution,
        # so generate another random Lotto draw.

        actual_numbers = rng.choice(
            np.arange(1, 50),
            size=6,
            replace=False
        )

        hits = len(
            set(random_numbers)
            &
            set(actual_numbers)
        )

        total_hits += hits

    simulation_means.append(
        total_hits / test_draws
    )


simulation_means = np.array(
    simulation_means
)


# =========================
# Statistics
# =========================

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


# =========================
# Comparison
# =========================

count = np.sum(
    simulation_means >= model_mean
)

p_value = (
    (count + 1) /
    (SIMULATIONS + 1)
)


print("\nXGBoost:", model_mean)

print(
    "Random expected:",
    36 / 49
)

print(
    "Random simulations >= XGBoost:",
    count
)

print(
    "Percentage:",
    count / SIMULATIONS * 100,
    "%"
)

print(
    "Empirical p-value:",
    p_value
)