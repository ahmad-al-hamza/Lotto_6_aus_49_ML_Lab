import pandas as pd
from collections import Counter

# Load data
df = pd.read_csv("data/processed/lotto_6aus49_clean.csv")

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

window_size = 100
top_n = 10

hits = 0
total_predictions = 0

for i in range(window_size, len(df)):

    # Previous 100 draws
    previous_draws = df.iloc[i - window_size:i]

    # Numbers from previous draws
    previous_numbers = previous_draws[number_columns].values.flatten()

    # Frequency
    frequency = Counter(previous_numbers)

    # Top 10 hot numbers
    hot_numbers = {
        number
        for number, count in frequency.most_common(top_n)
    }

    # Actual next draw
    actual_numbers = set(
        df.iloc[i][number_columns].values
    )

    # How many hot numbers appeared?
    hit_count = len(hot_numbers & actual_numbers)

    hits += hit_count
    total_predictions += 1

# Average number of hot numbers appearing
average_hits = hits / total_predictions

print("Window size:", window_size)
print("Top hot numbers:", top_n)
print("Test draws:", total_predictions)
print("Total hits:", hits)
print("Average hot numbers appearing:", average_hits)