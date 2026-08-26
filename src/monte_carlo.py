import numpy as np

# -------------------------
# Simulation parameters
# -------------------------

number_of_draws = 5038
numbers_per_draw = 6
number_of_numbers = 49

simulations = 10_000

# Expected probability for one number
p = numbers_per_draw / number_of_numbers

# Standard deviation
std = np.sqrt(
    number_of_draws * p * (1 - p)
)

# Expected frequency
expected = number_of_draws * p

# Store maximum absolute Z-score
max_z_scores = []

# -------------------------
# Monte Carlo simulation
# -------------------------

for _ in range(simulations):

    # Generate random lottery draws
    draws = np.array([
        np.random.choice(
            np.arange(1, number_of_numbers + 1),
            size=numbers_per_draw,
            replace=False
        )
        for _ in range(number_of_draws)
    ])

    # Count appearances of each number
    counts = np.bincount(
        draws.flatten(),
        minlength=number_of_numbers + 1
    )[1:]

    # Calculate Z-scores
    z_scores = (counts - expected) / std

    # Largest absolute deviation
    max_z = np.max(np.abs(z_scores))

    max_z_scores.append(max_z)

# -------------------------
# Results
# -------------------------

max_z_scores = np.array(max_z_scores)

print("Monte Carlo simulations:", simulations)

print("\nMean maximum |Z|:")
print(max_z_scores.mean())

print("\n95th percentile:")
print(np.percentile(max_z_scores, 95))

print("\n99th percentile:")
print(np.percentile(max_z_scores, 99))