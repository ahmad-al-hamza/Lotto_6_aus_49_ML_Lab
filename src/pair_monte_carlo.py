import numpy as np

# -------------------------
# Parameters
# -------------------------

number_of_draws = 5038
numbers_per_draw = 6
number_of_numbers = 49
simulations = 1000

rng = np.random.default_rng(42)

max_pair_frequencies = []

# -------------------------
# Monte Carlo
# -------------------------

for simulation in range(simulations):

    pair_counts = {}

    for _ in range(number_of_draws):

        draw = rng.choice(
            np.arange(1, number_of_numbers + 1),
            size=numbers_per_draw,
            replace=False
        )

        # Generate 15 pairs
        for i in range(numbers_per_draw):
            for j in range(i + 1, numbers_per_draw):

                pair = (draw[i], draw[j])

                # Make pair order independent
                pair = tuple(sorted(pair))

                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Maximum pair frequency in this simulation
    max_frequency = max(pair_counts.values())

    max_pair_frequencies.append(max_frequency)

# -------------------------
# Results
# -------------------------

max_pair_frequencies = np.array(max_pair_frequencies)

print("Monte Carlo simulations:", simulations)

print("\nMean maximum pair frequency:")
print(max_pair_frequencies.mean())

print("\nStandard deviation:")
print(max_pair_frequencies.std())

print("\n50th percentile:")
print(np.percentile(max_pair_frequencies, 50))

print("\n95th percentile:")
print(np.percentile(max_pair_frequencies, 95))

print("\n99th percentile:")
print(np.percentile(max_pair_frequencies, 99))

print("\nMaximum observed in simulations:")
print(max_pair_frequencies.max())

# Real data
real_max = 94

count = np.sum(max_pair_frequencies >= real_max)

print("\nReal Lotto maximum:", real_max)

print("Random simulations >= real:")
print(count)

print("Percentage:")
print(count / simulations * 100)