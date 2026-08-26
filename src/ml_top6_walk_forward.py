import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
OUTPUT_PATH = "data/processed/ml_top6_walk_forward_results.csv"

N_NUMBERS = 49
TOP_K = 6
FOLD_SIZE = 1008
N_RANDOM_SIMULATIONS = 5000
RANDOM_SEED = 42
START_HISTORY = 200

FEATURE_COLUMNS = [
    "frequency", "freq_5", "freq_10", "freq_20",
    "freq_50", "freq_100", "freq_200", "gap", "gap_ratio",
]

np.random.seed(RANDOM_SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ML TOP-6 WALK-FORWARD TEST")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

number_columns = ["n1", "n2", "n3", "n4", "n5", "n6"]

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
print(f"\nTotal draws: {len(df)}")

for col in number_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

if df[number_columns].isnull().any().any():
    raise ValueError("Number columns contain NaN values.")

if not df[number_columns].apply(lambda x: x.between(1, 49).all()).all():
    raise ValueError("Number values outside 1-49 detected.")

n_draws = len(df)

# ============================================================
# VECTORIZED FEATURE ENGINEERING (computed once for whole dataset)
# ============================================================
#
# A[d, k] = 1 if number (k+1) appeared in draw d
#
# For "history length" h (draws[0:h]), predicting draw h:
#   frequency      = Cext[h] / h
#   freq_window(w) = (Cext[h] - Cext[max(h-w,0)]) / min(h,w)
#   gap            = h - 1 - last_seen_before_h   (or h if never seen)

print("\nPrecomputing features for the entire dataset (vectorized)...")

number_values = df[number_columns].to_numpy(dtype=int)  # (n_draws, 6)

A = np.zeros((n_draws, N_NUMBERS), dtype=np.int32)
rows = np.repeat(np.arange(n_draws), 6)
cols = number_values.ravel() - 1
A[rows, cols] = 1

Cext = np.vstack([np.zeros((1, N_NUMBERS)), np.cumsum(A, axis=0)])  # (n_draws+1, 49)

occ_idx = np.where(A == 1, np.arange(n_draws)[:, None], -1)
last_seen_inclusive = np.maximum.accumulate(occ_idx, axis=0)         # (n_draws, 49)
last_seen_ext = np.vstack([-np.ones((1, N_NUMBERS)), last_seen_inclusive])  # (n_draws+1, 49)

h_arr = np.arange(START_HISTORY, n_draws)  # all history lengths we'll ever need

frequency = Cext[h_arr] / h_arr[:, None]

def windowed_freq(w):
    idx = np.maximum(h_arr - w, 0)
    numerator = Cext[h_arr] - Cext[idx]
    denom = np.minimum(h_arr, w).astype(float)
    return numerator / denom[:, None]

freq_5 = windowed_freq(5)
freq_10 = windowed_freq(10)
freq_20 = windowed_freq(20)
freq_50 = windowed_freq(50)
freq_100 = windowed_freq(100)
freq_200 = windowed_freq(200)

last_seen_h = last_seen_ext[h_arr]  # (len(h_arr), 49)
gap = np.where(last_seen_h == -1, h_arr[:, None], h_arr[:, None] - 1 - last_seen_h)
gap_ratio = gap / h_arr[:, None]

# feature_tensor[j, num, feat] where j indexes h_arr (h = START_HISTORY + j), num = 0..48
feature_tensor = np.stack(
    [frequency, freq_5, freq_10, freq_20, freq_50, freq_100, freq_200, gap, gap_ratio],
    axis=-1,
)  # shape (len(h_arr), 49, 9)

targets_all = A  # targets_all[h] == create_targets(df.iloc[h])

def features_for_h(h):
    """Return the (49, 9) feature matrix used to predict draw index h."""
    return feature_tensor[h - START_HISTORY]

print(f"Feature tensor shape: {feature_tensor.shape}")

# ============================================================
# RANDOM BASELINE (exact hypergeometric, no simulation loop needed)
# ============================================================
#
# Picking TOP_K numbers uniformly at random out of N_NUMBERS and
# comparing against a fixed set of 6 winning numbers is exactly
# Hypergeometric(N_NUMBERS, 6, TOP_K) regardless of which numbers
# were actually drawn. So we can sample hit-counts directly.

def random_baseline_stats(test_draws):
    samples = np.random.hypergeometric(
        ngood=6, nbad=N_NUMBERS - 6, nsample=TOP_K,
        size=(N_RANDOM_SIMULATIONS, test_draws),
    )
    return samples.mean(axis=1)  # one mean per simulation

# ============================================================
# WALK-FORWARD
# ============================================================

print("\n" + "=" * 70)
print("CREATING WALK-FORWARD FOLDS")
print("=" * 70)

results = []
fold = 1
test_start = FOLD_SIZE

while test_start < len(df):

    test_end = min(test_start + FOLD_SIZE, len(df))
    train_df = df.iloc[:test_start]
    test_df = df.iloc[test_start:test_end]

    print("\n" + "=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)
    print(f"Training: {train_df['date'].min()} -> {train_df['date'].max()}")
    print(f"Testing:  {test_df['date'].min()} -> {test_df['date'].max()}")
    print(f"Training draws: {len(train_df)}")
    print(f"Testing draws:  {len(test_df)}")

    # --------------------------------------------------------
    # Slice precomputed features/targets (no recomputation!)
    # --------------------------------------------------------

    train_h = np.arange(START_HISTORY, test_start)
    X_train = feature_tensor[train_h - START_HISTORY]        # (n_train, 49, 9)
    y_train = targets_all[train_h]                           # (n_train, 49)

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    y_train_flat = y_train.reshape(-1)

    print(f"Training feature shape: {X_train.shape}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        )),
    ])
    model.fit(X_train_flat, y_train_flat)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nRunning model on test period...")

    test_h = np.arange(test_start, test_end)
    X_test_all = feature_tensor[test_h - START_HISTORY]  # (n_test, 49, 9)

    n_test = len(test_h)
    probs = model.predict_proba(
        X_test_all.reshape(-1, X_test_all.shape[-1])
    )[:, 1].reshape(n_test, N_NUMBERS)

    top6_idx = np.argsort(-probs, axis=1)[:, :TOP_K]  # 0-based number indices
    actual_matrix = A[test_h]  # (n_test, 49), 1 if number drawn

    hit_mask = np.take_along_axis(actual_matrix, top6_idx, axis=1)
    model_hits = hit_mask.sum(axis=1)

    # --------------------------------------------------------
    # Model statistics
    # --------------------------------------------------------

    average_hits = np.mean(model_hits)
    total_hits = np.sum(model_hits)
    max_hits = np.max(model_hits)
    random_expected = TOP_K * 6 / N_NUMBERS
    difference = average_hits - random_expected
    difference_percent = difference / random_expected * 100

    # --------------------------------------------------------
    # Random simulation (vectorized, exact hypergeometric)
    # --------------------------------------------------------

    print("\nRunning random Top-6 simulation...")
    random_means = random_baseline_stats(len(test_df))
    random_mean = np.mean(random_means)
    random_lower = np.percentile(random_means, 2.5)
    random_upper = np.percentile(random_means, 97.5)

    empirical_p = (np.sum(random_means >= average_hits) + 1) / (len(random_means) + 1)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\nResults")
    print("-" * 40)
    print(f"Average hits:       {average_hits:.6f}")
    print(f"Total hits:         {total_hits}")
    print(f"Maximum hits:       {max_hits}")
    print(f"Random expected:    {random_expected:.6f}")
    print(f"Difference:         {difference:+.6f}")
    print(f"Difference %:       {difference_percent:+.3f}%")
    print(f"Random simulation:  {random_mean:.6f}")
    print(f"Random 95% range:   [{random_lower:.6f}, {random_upper:.6f}]")
    print(f"Empirical p-value:  {empirical_p:.6f}")

    results.append({
        "fold": fold,
        "train_draws": len(train_df),
        "test_draws": len(test_df),
        "average_hits": average_hits,
        "total_hits": total_hits,
        "max_hits": max_hits,
        "random_expected": random_expected,
        "difference": difference,
        "difference_percent": difference_percent,
        "random_mean": random_mean,
        "random_lower": random_lower,
        "random_upper": random_upper,
        "empirical_p": empirical_p,
    })

    fold += 1
    test_start = test_end

# ============================================================
# SUMMARY / OVERALL / CONCLUSION / SAVE  (unchanged)
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)
print(results_df[[
    "fold", "test_draws", "average_hits", "random_expected",
    "difference", "difference_percent", "empirical_p",
]].to_string(index=False))

print("\n" + "=" * 70)
print("OVERALL")
print("=" * 70)

mean_model_hits = results_df["average_hits"].mean()
mean_random = results_df["random_expected"].mean()
mean_difference = results_df["difference"].mean()
mean_difference_percent = mean_difference / mean_random * 100
mean_empirical_p = results_df["empirical_p"].mean()
above_random = np.sum(results_df["difference"] > 0)
below_random = np.sum(results_df["difference"] < 0)

print(f"Mean model hits:       {mean_model_hits:.6f}")
print(f"Mean random expected:  {mean_random:.6f}")
print(f"Mean difference:        {mean_difference:+.6f}")
print(f"Mean difference %:      {mean_difference_percent:+.3f}%")
print(f"Mean empirical p:       {mean_empirical_p:.6f}")
print("\nFold consistency:")
print(f"Above random: {above_random}")
print(f"Below random: {below_random}")

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

if mean_difference > 0 and mean_empirical_p < 0.05:
    print("\nPotential statistically significant ML advantage over random selection.")
elif mean_difference > 0:
    print("\nModel is above random on average, but the advantage is not statistically significant.")
else:
    print("\nNo meaningful ML advantage over random selection.")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:")
print(OUTPUT_PATH)
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)