import pandas as pd

# Read the local CSV file
DATA_PATH = "data/processed/lotto_6aus49_clean.csv"

df = pd.read_csv(DATA_PATH)

# Columns containing the six lottery numbers
number_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]

# Convert the date column
df["date"] = pd.to_datetime(df["date"])

# Extract the year
df["year"] = df["date"].dt.year

# Create a normalized combination.
# Sorting ensures that the same six numbers are treated
# as the same combination regardless of their order.
df["combination"] = df[number_cols].apply(
    lambda row: tuple(sorted(row.astype(int))),
    axis=1
)

# Find combinations that appeared more than once
duplicates = df[
    df.duplicated("combination", keep=False)
].copy()

# Sort the results
duplicates = duplicates.sort_values(
    ["combination", "date"]
)

if duplicates.empty:

    print("No repeated lottery combinations were found.")

else:

    print("\nRepeated lottery combinations:")
    print("=" * 70)

    for combination, group in duplicates.groupby("combination"):

        years = sorted(group["year"].unique())

        print("-" * 70)
        print("Numbers:", combination)
        print("Number of appearances:", len(group))
        print("Years:", years)

        for _, row in group.iterrows():

            print(
                f"  Date: {row['date'].strftime('%Y-%m-%d')} "
                f"| Year: {row['year']}"
            )


# Save the repeated combinations to a new CSV file
duplicates.to_csv(
    "repeated_results.csv",
    index=False
)

print("\nResults saved to: repeated_results.csv")