import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/lotto_6aus49_clean.csv")

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

# Combine all lottery numbers
all_numbers = df[number_columns].stack()

# Count frequency
frequency = all_numbers.value_counts().sort_values(ascending=False)

print("🔥 Top 10 Hot Numbers:")
print(frequency.head(10))

print("\n❄️ Bottom 10 Cold Numbers:")
print(frequency.tail(10).sort_values())