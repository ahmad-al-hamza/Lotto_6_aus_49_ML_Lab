import pandas as pd


df = pd.read_csv(
    "data/processed/lotto_6aus49_clean.csv"
)

number_columns = [
    "n1", "n2", "n3",
    "n4", "n5", "n6"
]


def consecutive_pairs(numbers):

    numbers = sorted(numbers)

    count = 0

    for i in range(1, len(numbers)):

        if numbers[i] == numbers[i - 1] + 1:
            count += 1

    return count


# Calculate feature
df["consecutive_pairs"] = df[
    number_columns
].apply(
    lambda row: consecutive_pairs(row.tolist()),
    axis=1
)


# Split dataset into two equal periods
midpoint = len(df) // 2

period_1 = df.iloc[:midpoint]
period_2 = df.iloc[midpoint:]


def analyze_period(data, name):

    average = data["consecutive_pairs"].mean()

    percentage = (
        (data["consecutive_pairs"] > 0).mean()
        * 100
    )

    print(f"\n{name}")
    print("-" * 30)

    print("Draws:", len(data))

    print(
        "Average consecutive pairs:",
        average
    )

    print(
        "Draws with consecutive numbers:",
        percentage,
        "%"
    )


analyze_period(
    period_1,
    "Period 1"
)

analyze_period(
    period_2,
    "Period 2"
)