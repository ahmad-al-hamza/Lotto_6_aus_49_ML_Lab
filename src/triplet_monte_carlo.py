import pandas as pd
import numpy as np
from itertools import combinations

# ==============================
# Configuration
# ==============================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

SIMULATIONS = 10000
N_DRAWS = 5040
N_NUMBERS = 49
NUMBERS_PER_DRAW = 6

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

# Fixed position-triples for choosing 3 of 6 drawn numbers (always the same 20 combos)
pos_arr = np.array(list(combinations(range(NUMBERS_PER_DRAW), 3)))  # shape (20, 3)
ENC_BASE = N_NUMBERS  # encode with 0-indexed numbers (0..48)


def triplet_max_count(draws_sorted: np.ndarray) -> int:
    """
    draws_sorted: shape (n_draws, 6), each row sorted ascending, values 1..49
    Returns the max frequency among all (a,b,c) triplets across draws.
    """
    zero_idx = draws_sorted - 1  # 0..48
    trip = zero_idx[:, pos_arr]  # shape (n_draws, 20, 3)
    a, b, c = trip[..., 0], trip[..., 1], trip[..., 2]
    encoded = (a * ENC_BASE + b) * ENC_BASE + c
    counts = np.bincount(encoded.ravel(), minlength=ENC_BASE ** 3)
    return counts.max()


def sample_draws(n_draws: int) -> np.ndarray:
    """Vectorized sampling of n_draws draws of 6 numbers from 1..49 without replacement."""
    rand = np.random.rand(n_draws, N_NUMBERS)
    idx = np.argpartition(rand, NUMBERS_PER_DRAW, axis=1)[:, :NUMBERS_PER_DRAW]
    numbers = idx + 1
    numbers.sort(axis=1)
    return numbers


# ==============================
# Load real data & compute real max (vectorized)
# ==============================

df = pd.read_csv(DATA_PATH)
real_arr = df[number_columns].astype(int).to_numpy()
real_arr.sort(axis=1)
real_maximum = triplet_max_count(real_arr)

# ==============================
# Monte Carlo
# ==============================

print(f"Monte Carlo simulations: {SIMULATIONS}")

maximum_frequencies = np.empty(SIMULATIONS, dtype=np.int64)

for simulation in range(SIMULATIONS):
    draws = sample_draws(N_DRAWS)
    maximum_frequencies[simulation] = triplet_max_count(draws)

# ==============================
# Results
# ==============================

print("\nMonte Carlo Results")
print("------------------------------")
print("Mean maximum triplet frequency:")
print(maximum_frequencies.mean())
print("\nStandard deviation:")
print(maximum_frequencies.std())
print("\n50th percentile:")
print(np.percentile(maximum_frequencies, 50))
print("\n95th percentile:")
print(np.percentile(maximum_frequencies, 95))
print("\n99th percentile:")
print(np.percentile(maximum_frequencies, 99))
print("\nMaximum observed in simulations:")
print(maximum_frequencies.max())

# ==============================
# Empirical p-value
# ==============================

greater_or_equal = np.sum(maximum_frequencies >= real_maximum)
p_value = greater_or_equal / SIMULATIONS

print("\nReal Lotto maximum:")
print(real_maximum)
print("\nRandom simulations >= real:")
print(greater_or_equal)
print("\nPercentage:")
print(greater_or_equal / SIMULATIONS * 100)
print("\nEmpirical p-value:")
print(p_value)