import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/processed/lotto_6aus49_clean.csv")

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

window_size = 100
top_n = 10
simulations = 1000

# Actual future draws used for testing
test_data = df.iloc[window_size:]

actual_numbers = test_data[number_columns].to_numpy()

# Random generator
rng = np.random.default_rng(42)

average_hits = []

for simulation in range(simulations):

    # Generate random numbers for ALL test draws at once
    random_numbers = np.array([
        rng.choice(
            np.arange(1, 50),
            size=top_n,
            replace=False
        )
        for _ in range(len(test_data))
    ])

    # Compare random selections with actual draws
    hits = np.sum(
        np.any(
            random_numbers[:, :, None] == actual_numbers[:, None, :],
            axis=2
        )
    )

    average = hits / len(test_data)

    average_hits.append(average)

average_hits = np.array(average_hits)

print("Random simulations:", simulations)

print("\nMean:")
print(average_hits.mean())

print("\nStandard deviation:")
print(average_hits.std())

print("\n5th percentile:")
print(np.percentile(average_hits, 5))

print("\n50th percentile:")
print(np.percentile(average_hits, 50))

print("\n95th percentile:")
print(np.percentile(average_hits, 95))

# Hot strategy
hot_result = 1.2266099635479952

count = np.sum(average_hits >= hot_result)

percentile = (count / simulations) * 100

print("\nHot strategy:", hot_result)

print("Random simulations >= Hot:", count)

print("Percentage:", percentile, "%")