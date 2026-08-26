import numpy as np


# -------------------------
# Parameters
# -------------------------

number_of_draws = 5038
number_of_numbers = 49
numbers_per_draw = 6

simulations = 10_000

rng = np.random.default_rng(42)


# -------------------------
# Helper function
# -------------------------

def consecutive_features(numbers):

    numbers = np.sort(numbers)

    differences = np.diff(numbers)

    consecutive_pairs = np.sum(differences == 1)

    # Find longest consecutive streak
    max_streak = 1
    current_streak = 1

    for difference in differences:

        if difference == 1:
            current_streak += 1
            max_streak = max(
                max_streak,
                current_streak
            )
        else:
            current_streak = 1

    return consecutive_pairs, max_streak


# -------------------------
# Monte Carlo
# -------------------------

average_pairs = []
percentage_with_pairs = []
maximum_streaks = []


for simulation in range(simulations):

    total_pairs = 0
    draws_with_pairs = 0
    max_streak = 1

    for _ in range(number_of_draws):

        draw = rng.choice(
            np.arange(1, number_of_numbers + 1),
            size=numbers_per_draw,
            replace=False
        )

        pairs, streak = consecutive_features(draw)

        total_pairs += pairs

        if pairs > 0:
            draws_with_pairs += 1

        max_streak = max(
            max_streak,
            streak
        )

    average_pairs.append(
        total_pairs / number_of_draws
    )

    percentage_with_pairs.append(
        draws_with_pairs / number_of_draws * 100
    )

    maximum_streaks.append(max_streak)


# -------------------------
# Results
# -------------------------

average_pairs = np.array(average_pairs)
percentage_with_pairs = np.array(percentage_with_pairs)
maximum_streaks = np.array(maximum_streaks)


print("Monte Carlo simulations:", simulations)

print("\nAverage consecutive pairs:")
print(average_pairs.mean())

print("\n5th percentile:")
print(np.percentile(average_pairs, 5))

print("\n95th percentile:")
print(np.percentile(average_pairs, 95))


print("\nPercentage of draws with consecutive numbers:")
print(percentage_with_pairs.mean())

print("\n5th percentile:")
print(np.percentile(percentage_with_pairs, 5))

print("\n95th percentile:")
print(np.percentile(percentage_with_pairs, 95))


print("\nMaximum streak:")
print("Mean:", maximum_streaks.mean())

print("50th percentile:",
      np.percentile(maximum_streaks, 50))

print("95th percentile:",
      np.percentile(maximum_streaks, 95))

print("99th percentile:",
      np.percentile(maximum_streaks, 99))

print("Maximum observed:",
      maximum_streaks.max())
# Real Lotto result
real_average_pairs = 0.6319968241365621

count = np.sum(
    average_pairs >= real_average_pairs
)

p_value = count / simulations

print("\nReal Lotto average pairs:")
print(real_average_pairs)

print("\nRandom simulations >= real:")
print(count)

print("\nEmpirical p-value:")
print(p_value)