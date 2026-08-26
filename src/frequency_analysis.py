import pandas as pd
import math

# Load cleaned dataset
df = pd.read_csv("data/processed/lotto_6aus49_clean.csv")

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

# Combine all six number columns
all_numbers = df[number_columns].stack()

# Observed frequency
frequency = all_numbers.value_counts().sort_index()

# Total number of draws
number_of_draws = len(df)

# Total number appearances
total_appearances = number_of_draws * 6

# Expected frequency for each number
expected_frequency = total_appearances / 49

# Create results table
results = pd.DataFrame({
    "observed": frequency,
    "expected": expected_frequency
})

# Difference between observed and expected
results["difference"] = results["observed"] - results["expected"]

# Percentage difference
results["difference_percent"] = (
    results["difference"] / results["expected"] * 100
)
# Probability of a specific number appearing in one draw
p = 6 / 49

# Standard deviation under the binomial model
std = math.sqrt(number_of_draws * p * (1 - p))

# Z-score
results["z_score"] = (
    results["difference"] / std
)

print("\nExpected standard deviation:", std)

print("\nFrequency analysis with Z-score:")
print(results.round(2))