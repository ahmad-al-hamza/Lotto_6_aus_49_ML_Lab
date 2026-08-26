import pandas as pd
from pathlib import Path

# -------------------------
# 1. Load raw data
# https://github.com/daowa89/lottery-archive/blob/main/de/lotto_6aus49/results.csv
# -------------------------
input_file = Path("data/raw/results.csv")
output_file = Path("data/processed/lotto_6aus49_clean.csv")

df = pd.read_csv(input_file)

# -------------------------
# 2. Convert date
# -------------------------
df["date"] = pd.to_datetime(df["date"])

# -------------------------
# 3. Sort lottery numbers
# -------------------------
number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

df[number_columns] = df[number_columns].apply(
    lambda row: sorted(row),
    axis=1,
    result_type="expand"
)

# -------------------------
# 4. Create processed folder
# -------------------------
output_file.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# 5. Save cleaned data
# -------------------------
df.to_csv(output_file, index=False)

print("Clean dataset created successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_file}")