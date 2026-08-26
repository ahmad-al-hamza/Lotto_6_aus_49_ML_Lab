import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/lotto_6aus49_clean.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

print("\nDate information:")
print("First draw:", df["date"].min())
print("Last draw:", df["date"].max())

# Check duplicate dates
print("\nDuplicate dates:")
print(df["date"].duplicated().sum())

# Check invalid lottery numbers
number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

print("\nInvalid numbers:")
for column in number_columns:
    invalid = ((df[column] < 1) | (df[column] > 49)).sum()
    print(column, ":", invalid)

# Check whether numbers are sorted inside each draw
sorted_rows = df[number_columns].apply(
    lambda row: list(row) == sorted(row),
    axis=1
)

print("\nRows with unsorted numbers:")
print((~sorted_rows).sum())
print("\nUnsorted draw:")
print(df.loc[~sorted_rows, ["date"] + number_columns])