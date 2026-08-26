import os
import numpy as np
import pandas as pd


# ============================================================
# TOP-6 STATISTICAL SIGNIFICANCE TEST
# ============================================================

RESULTS_PATH = "data/processed/xgboost_v3_top6_results.csv"

OUTPUT_PATH = (
    "data/processed/top6_significance_results.csv"
)

FOLD_OUTPUT_PATH = (
    "data/processed/top6_significance_folds.csv"
)

N_BOOTSTRAP = 10000
N_MONTE_CARLO = 10000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TOP-6 STATISTICAL SIGNIFICANCE TEST")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Top-6 results...")

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(
        f"Results file not found:\n{RESULTS_PATH}"
    )

df = pd.read_csv(RESULTS_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "date",
    "hits"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found."
        )


# ============================================================
# CLEAN DATA
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df["hits"] = pd.to_numeric(
    df["hits"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "hits"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# BASIC STATISTICS
# ============================================================

hits = df["hits"].to_numpy(dtype=float)

n_draws = len(hits)

model_mean = np.mean(hits)

model_std = np.std(
    hits,
    ddof=1
)

model_se = (
    model_std / np.sqrt(n_draws)
)


# ------------------------------------------------------------
# Random expected Top-6 hits
#
# Probability that one predicted number is in the
# actual six numbers:
#
# 6 / 49
#
# Six predictions:
#
# 6 * 6 / 49
# ------------------------------------------------------------

random_expected = (
    36 / 49
)

difference = (
    model_mean - random_expected
)

difference_percent = (
    difference
    / random_expected
    * 100
)


print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

print(f"Test draws:          {n_draws}")
print(f"Model mean hits:     {model_mean:.6f}")
print(f"Model std:           {model_std:.6f}")
print(f"Random expected:     {random_expected:.6f}")
print(f"Difference:          {difference:+.6f}")
print(
    f"Difference %:        "
    f"{difference_percent:+.4f}%"
)


# ============================================================
# NORMAL 95% CONFIDENCE INTERVAL
# ============================================================

z = 1.96

normal_lower = (
    model_mean
    - z * model_se
)

normal_upper = (
    model_mean
    + z * model_se
)


print("\n" + "=" * 70)
print("95% NORMAL CONFIDENCE INTERVAL")
print("=" * 70)

print(f"Mean:       {model_mean:.6f}")
print(f"Lower:      {normal_lower:.6f}")
print(f"Upper:      {normal_upper:.6f}")
print(f"Random:     {random_expected:.6f}")

if (
    normal_lower
    <= random_expected
    <= normal_upper
):

    print(
        "\nRandom expectation IS inside "
        "the 95% CI."
    )

else:

    print(
        "\nRandom expectation is OUTSIDE "
        "the 95% CI."
    )


# ============================================================
# BOOTSTRAP
# ============================================================

print("\n" + "=" * 70)
print("BOOTSTRAP TEST")
print("=" * 70)

print(
    f"Bootstrap simulations: "
    f"{N_BOOTSTRAP}"
)


bootstrap_means = np.empty(
    N_BOOTSTRAP
)


for i in range(N_BOOTSTRAP):

    sample = rng.choice(
        hits,
        size=n_draws,
        replace=True
    )

    bootstrap_means[i] = np.mean(
        sample
    )


bootstrap_lower = np.percentile(
    bootstrap_means,
    2.5
)

bootstrap_upper = np.percentile(
    bootstrap_means,
    97.5
)

bootstrap_mean = np.mean(
    bootstrap_means
)


print(
    f"Bootstrap mean: "
    f"{bootstrap_mean:.6f}"
)

print(
    f"2.5 percentile: "
    f"{bootstrap_lower:.6f}"
)

print(
    f"97.5 percentile: "
    f"{bootstrap_upper:.6f}"
)

if (
    bootstrap_lower
    <= random_expected
    <= bootstrap_upper
):

    print(
        "\nRandom expectation IS inside "
        "the bootstrap 95% CI."
    )

else:

    print(
        "\nRandom expectation is OUTSIDE "
        "the bootstrap 95% CI."
    )


# ============================================================
# MONTE CARLO RANDOM TOP-6
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO RANDOM TOP-6 TEST")
print("=" * 70)

print(
    f"Monte Carlo simulations: "
    f"{N_MONTE_CARLO}"
)


# ------------------------------------------------------------
# Hypergeometric distribution
#
# 49 total numbers
# 6 actual winning numbers
# 6 predicted numbers
#
# Result = number of hits
# ------------------------------------------------------------

random_hits = rng.hypergeometric(
    ngood=6,
    nbad=43,
    nsample=6,
    size=(N_MONTE_CARLO, n_draws)
)


random_mean_per_simulation = (
    random_hits.mean(axis=1)
)


mc_mean = np.mean(
    random_mean_per_simulation
)

mc_std = np.std(
    random_mean_per_simulation,
    ddof=1
)

mc_lower = np.percentile(
    random_mean_per_simulation,
    2.5
)

mc_upper = np.percentile(
    random_mean_per_simulation,
    97.5
)


print(
    f"Random simulation mean: "
    f"{mc_mean:.6f}"
)

print(
    f"Random simulation std:  "
    f"{mc_std:.6f}"
)

print(
    f"2.5 percentile:         "
    f"{mc_lower:.6f}"
)

print(
    f"97.5 percentile:        "
    f"{mc_upper:.6f}"
)


# ============================================================
# ONE-SIDED EMPIRICAL P-VALUE
# ============================================================

print("\n" + "=" * 70)
print("EMPIRICAL P-VALUE")
print("=" * 70)


random_ge_model = np.sum(
    random_mean_per_simulation
    >= model_mean
)


p_value = (
    random_ge_model + 1
) / (
    N_MONTE_CARLO + 1
)


print(
    "Random simulations >= model: "
    f"{random_ge_model}"
)

print(
    f"Empirical p-value: "
    f"{p_value:.6f}"
)


# ============================================================
# TWO-SIDED P-VALUE
# ============================================================

distance = np.abs(
    random_mean_per_simulation
    - random_expected
)

model_distance = abs(
    model_mean
    - random_expected
)


two_sided_count = np.sum(
    distance >= model_distance
)


two_sided_p = (
    two_sided_count + 1
) / (
    N_MONTE_CARLO + 1
)


print(
    "\nTwo-sided empirical p-value:"
)

print(
    f"{two_sided_p:.6f}"
)


# ============================================================
# WALK-FORWARD PERIOD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WALK-FORWARD PERIOD ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# IMPORTANT:
#
# We use pandas.iloc here instead of np.array_split.
# This guarantees that every fold remains a DataFrame.
# ------------------------------------------------------------

n_folds = 5

fold_size = (
    len(df) // n_folds
)


fold_results = []


for fold_number in range(
    n_folds
):

    start = (
        fold_number
        * fold_size
    )

    if fold_number == n_folds - 1:

        end = len(df)

    else:

        end = (
            fold_number + 1
        ) * fold_size


    fold = df.iloc[
        start:end
    ].copy()


    fold_hits = fold[
        "hits"
    ].to_numpy(
        dtype=float
    )


    fold_mean = np.mean(
        fold_hits
    )


    fold_difference = (
        fold_mean
        - random_expected
    )


    fold_difference_percent = (
        fold_difference
        / random_expected
        * 100
    )


    fold_results.append({

        "fold": fold_number + 1,

        "test_draws": len(fold),

        "start_date": fold[
            "date"
        ].min(),

        "end_date": fold[
            "date"
        ].max(),

        "average_hits": fold_mean,

        "random_expected":
            random_expected,

        "difference":
            fold_difference,

        "difference_percent":
            fold_difference_percent
    })


    print(
        f"Fold {fold_number + 1}: "
        f"{fold_mean:.6f} "
        f"("
        f"{fold_difference_percent:+.3f}%"
        ")"
    )


fold_df = pd.DataFrame(
    fold_results
)


folds_above_random = int(
    np.sum(
        fold_df[
            "average_hits"
        ]
        > random_expected
    )
)


folds_below_random = int(
    np.sum(
        fold_df[
            "average_hits"
        ]
        < random_expected
    )
)


print(
    f"\nFolds above random: "
    f"{folds_above_random}"
)

print(
    f"Folds below random: "
    f"{folds_below_random}"
)


# ============================================================
# EFFECT SIZE
# ============================================================

print("\n" + "=" * 70)
print("EFFECT SIZE")
print("=" * 70)


if model_std > 0:

    cohens_d = (
        difference
        / model_std
    )

else:

    cohens_d = 0


print(
    f"Cohen's d: "
    f"{cohens_d:.6f}"
)


if abs(cohens_d) < 0.2:

    effect_description = (
        "Very small"
    )

elif abs(cohens_d) < 0.5:

    effect_description = (
        "Small"
    )

elif abs(cohens_d) < 0.8:

    effect_description = (
        "Medium"
    )

else:

    effect_description = (
        "Large"
    )


print(
    f"Effect size: "
    f"{effect_description}"
)


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("FINAL STATISTICAL CONCLUSION")
print("=" * 70)


alpha = 0.05


if (
    p_value < alpha
    and model_mean > random_expected
):

    conclusion = (
        "SIGNIFICANTLY ABOVE RANDOM"
    )


elif (
    p_value >= alpha
    and model_mean > random_expected
):

    conclusion = (
        "ABOVE RANDOM BUT NOT SIGNIFICANT"
    )


elif (
    p_value < alpha
    and model_mean < random_expected
):

    conclusion = (
        "SIGNIFICANTLY BELOW RANDOM"
    )


else:

    conclusion = (
        "NOT SIGNIFICANT"
    )


print(
    f"\nConclusion: "
    f"{conclusion}"
)


print("\nInterpretation:")


if conclusion == (
    "SIGNIFICANTLY ABOVE RANDOM"
):

    print(
        "The Top-6 model shows statistically "
        "significant evidence of outperforming "
        "random selection."
    )


elif conclusion == (
    "ABOVE RANDOM BUT NOT SIGNIFICANT"
):

    print(
        "The model is numerically above random, "
        "but the difference is not statistically "
        "significant."
    )


elif conclusion == (
    "SIGNIFICANTLY BELOW RANDOM"
):

    print(
        "The model performs significantly worse "
        "than random."
    )


else:

    print(
        "There is insufficient statistical evidence "
        "that the model differs from random selection."
    )


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

results = {

    "test_draws":
        n_draws,

    "model_mean_hits":
        model_mean,

    "model_std":
        model_std,

    "random_expected":
        random_expected,

    "difference":
        difference,

    "difference_percent":
        difference_percent,

    "normal_ci_lower":
        normal_lower,

    "normal_ci_upper":
        normal_upper,

    "bootstrap_ci_lower":
        bootstrap_lower,

    "bootstrap_ci_upper":
        bootstrap_upper,

    "monte_carlo_mean":
        mc_mean,

    "monte_carlo_std":
        mc_std,

    "monte_carlo_ci_lower":
        mc_lower,

    "monte_carlo_ci_upper":
        mc_upper,

    "one_sided_p_value":
        p_value,

    "two_sided_p_value":
        two_sided_p,

    "folds_above_random":
        folds_above_random,

    "folds_below_random":
        folds_below_random,

    "cohens_d":
        cohens_d,

    "conclusion":
        conclusion
}


results_df = pd.DataFrame(
    [results]
)


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_df.to_csv(
    FOLD_OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Model mean:        "
    f"{model_mean:.6f}"
)

print(
    f"Random expected:   "
    f"{random_expected:.6f}"
)

print(
    f"Difference:        "
    f"{difference:+.6f}"
)

print(
    f"Difference %:      "
    f"{difference_percent:+.4f}%"
)

print(
    f"95% Bootstrap CI:  "
    f"[{bootstrap_lower:.6f}, "
    f"{bootstrap_upper:.6f}]"
)

print(
    f"One-sided p-value: "
    f"{p_value:.6f}"
)

print(
    f"Two-sided p-value: "
    f"{two_sided_p:.6f}"
)

print(
    f"Cohen's d:         "
    f"{cohens_d:.6f}"
)

print(
    f"Conclusion:        "
    f"{conclusion}"
)


print("\nResults saved to:")
print(OUTPUT_PATH)
print(FOLD_OUTPUT_PATH)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)