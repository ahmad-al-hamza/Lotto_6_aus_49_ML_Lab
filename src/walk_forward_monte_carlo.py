import pandas as pd
import numpy as np


# ==============================
# Configuration
# ==============================

SIMULATIONS = 10000
N_NUMBERS = 49
PICK = 6

# نتائج XGBoost من walk-forward
MODEL_RESULTS = [
    0.7494824016563147,
    0.7370600414078675,
    0.7494824016563147,
    0.6832298136645962,
    0.7453416149068323
]

TEST_DRAWS = 483


# ==============================
# Random simulation
# ==============================

def simulate_random_hits(test_draws, simulations):

    results = []

    for _ in range(simulations):

        # في كل سحب:
        # النموذج يختار 6 أرقام عشوائية
        # والسحب الحقيقي يحتوي 6 أرقام
        #
        # احتمال عدد الإصابات يتبع توزيع Hypergeometric

        random_hits = np.random.hypergeometric(
            ngood=PICK,
            nbad=N_NUMBERS - PICK,
            nsample=PICK,
            size=test_draws
        )

        average_hits = random_hits.mean()

        results.append(average_hits)

    return np.array(results)


# ==============================
# Main
# ==============================

print("Walk-Forward Monte Carlo")
print("=" * 50)

all_fold_results = []

for fold, model_score in enumerate(MODEL_RESULTS, start=1):

    random_results = simulate_random_hits(
        TEST_DRAWS,
        SIMULATIONS
    )

    p_value = np.mean(random_results >= model_score)

    mean_random = random_results.mean()
    std_random = random_results.std()

    percentile_5 = np.percentile(random_results, 5)
    percentile_95 = np.percentile(random_results, 95)

    print(f"\nFold {fold}")
    print("-" * 30)

    print(f"Model average hits: {model_score:.6f}")
    print(f"Random mean:        {mean_random:.6f}")
    print(f"Random std:         {std_random:.6f}")

    print(f"5th percentile:     {percentile_5:.6f}")
    print(f"95th percentile:    {percentile_95:.6f}")

    print(f"Empirical p-value:  {p_value:.4f}")

    all_fold_results.append({
        "fold": fold,
        "model": model_score,
        "random_mean": mean_random,
        "p_value": p_value
    })


# ==============================
# Overall result
# ==============================

df = pd.DataFrame(all_fold_results)

model_mean = df["model"].mean()
random_mean = df["random_mean"].mean()

print("\n")
print("=" * 50)
print("OVERALL WALK-FORWARD MONTE CARLO")
print("=" * 50)

print(f"Model mean:  {model_mean:.6f}")
print(f"Random mean: {random_mean:.6f}")

print("\nFold results:")
print(df.to_string(index=False))