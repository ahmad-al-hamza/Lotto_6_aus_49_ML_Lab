# src/hybrid_top6_walk_forward.py

import os
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
OUTPUT_PATH = "data/processed/hybrid_top6_walk_forward_results.csv"

N_NUMBERS = 49
PICK_COUNT = 6

WINDOWS = [5, 10, 20, 50, 100]

# Walk-forward configuration
N_FOLDS = 5
TEST_SIZE = 1008

# ML
ML_C = 0.5
ML_MAX_ITER = 1000

# Hybrid weights
W_FREQUENCY = 0.20
W_RECENT = 0.20
W_GAP = 0.10
W_ML = 0.30
W_PAIR = 0.20

RANDOM_SEED = 42

# Populated once in main() by precompute_feature_arrays().
# Every "history" used anywhere in this script is always the
# full prefix df.iloc[:h] for some h, so a global lookup table
# indexed by h works everywhere instead of recomputing per call.
FEATURES = None


# ============================================================
# HELPERS
# ============================================================

def normalize_series(s):
    """
    Min-max normalization.
    If all values are identical, return 0.5 for all.
    """
    s = pd.Series(s, dtype=float)

    min_v = s.min()
    max_v = s.max()

    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(0.5, index=s.index)

    if max_v == min_v:
        return pd.Series(0.5, index=s.index)

    return (s - min_v) / (max_v - min_v)


def get_number_columns(df):
    """
    Find n1...n6.
    """
    expected = [f"n{i}" for i in range(1, 7)]

    if all(c in df.columns for c in expected):
        return expected

    numeric_candidates = []

    for col in df.columns:
        try:
            values = pd.to_numeric(df[col], errors="coerce")
            valid = values.dropna()

            if len(valid) == len(df):
                if valid.min() >= 1 and valid.max() <= 49:
                    numeric_candidates.append(col)
        except Exception:
            pass

    if len(numeric_candidates) == 6:
        return numeric_candidates

    raise ValueError(
        f"Could not find exactly 6 number columns. "
        f"Found: {numeric_candidates}"
    )


def validate_data(df, number_columns):
    """
    Validate lottery data.
    """

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Invalid dates found in dataset.")

    for col in number_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[number_columns].isna().any().any():
        raise ValueError("Missing number values found.")

    for col in number_columns:
        if not df[col].between(1, 49).all():
            raise ValueError(f"Values outside 1-49 found in column {col}")

    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# VECTORIZED FEATURE PRECOMPUTATION (the main speedup)
# ============================================================
#
# Instead of recomputing 49-number features from scratch (with a
# Python loop scanning full history for each number) every single
# time build_number_feature_vector() used to be called -- which
# happened once per training example per fold (O(T^2 * 49) total,
# by far the slowest part of the original script -- we compute a
# lookup table ONCE for every possible history length h = 0..T.
#
# Every "history" array used anywhere in this file is always
# df.iloc[:h] for some h (a plain prefix), so indexing by h is
# equivalent to recomputing from that history, just O(1) instead
# of O(h * 49).

def precompute_feature_arrays(df, number_columns):

    n_draws = len(df)
    values = df[number_columns].to_numpy(dtype=int)  # (n_draws, 6)

    # One-hot: A[d, k] = 1 if number (k+1) appeared in draw d
    A = np.zeros((n_draws, N_NUMBERS), dtype=np.int64)
    rows = np.repeat(np.arange(n_draws), values.shape[1])
    cols = values.ravel() - 1
    A[rows, cols] = 1

    # Cext[h] = counts of each number among draws[0:h]  (h = 0..n_draws)
    Cext = np.vstack(
        [np.zeros((1, N_NUMBERS), dtype=np.int64), np.cumsum(A, axis=0)]
    )

    # last_seen_ext[h] = index of last occurrence within draws[0:h], or -1
    occ_idx = np.where(A == 1, np.arange(n_draws)[:, None], -1)
    last_seen_incl = np.maximum.accumulate(occ_idx, axis=0)
    last_seen_ext = np.vstack(
        [-np.ones((1, N_NUMBERS), dtype=np.int64), last_seen_incl]
    )

    h_all = np.arange(0, n_draws + 1)

    frequency = Cext.astype(float)

    def windowed(w):
        idx = np.maximum(h_all - w, 0)
        return (Cext - Cext[idx]).astype(float)

    freq_5 = windowed(5)
    freq_10 = windowed(10)
    freq_20 = windowed(20)
    freq_50 = windowed(50)
    freq_100 = windowed(100)

    gap = np.where(
        last_seen_ext == -1,
        999,
        h_all[:, None] - 1 - last_seen_ext,
    ).astype(float)

    def row_normalize(mat):
        mn = mat.min(axis=1, keepdims=True)
        mx = mat.max(axis=1, keepdims=True)
        out = np.full_like(mat, 0.5)
        mask = (mx > mn).ravel()
        out[mask] = (mat[mask] - mn[mask]) / (mx[mask] - mn[mask])
        return out

    frequency_norm = row_normalize(frequency)

    recent_raw = 0.50 * freq_20 + 0.30 * freq_50 + 0.20 * freq_100
    recent_norm = row_normalize(recent_raw)

    gap_score = np.exp(-np.abs(gap - 8.0) / 12.0)
    gap_score[gap >= 999] = 0.5

    return {
        "A": A,
        "frequency": frequency,
        "freq_5": freq_5,
        "freq_10": freq_10,
        "freq_20": freq_20,
        "freq_50": freq_50,
        "freq_100": freq_100,
        "gap": gap,
        "frequency_norm": frequency_norm,
        "recent_norm": recent_norm,
        "gap_score": gap_score,
    }


def get_feature_df(h):
    """
    O(1) replacement for build_number_feature_vector(history),
    where h == len(history).
    """
    f = FEATURES
    data = {
        "frequency": f["frequency"][h],
        "freq_5": f["freq_5"][h],
        "freq_10": f["freq_10"][h],
        "freq_20": f["freq_20"][h],
        "freq_50": f["freq_50"][h],
        "freq_100": f["freq_100"][h],
        "gap": f["gap"][h],
        "frequency_norm": f["frequency_norm"][h],
        "recent_norm": f["recent_norm"][h],
        "gap_score": f["gap_score"][h],
    }
    return pd.DataFrame(data, index=range(1, N_NUMBERS + 1))


# ============================================================
# ML DATASET (fully vectorized, no per-row Python loop)
# ============================================================

def build_ml_dataset(start_idx, end_idx):
    """
    Build supervised learning data.

    Each sample represents:
        history before draw t  ->  whether number appeared in draw t

    Equivalent to the original nested loop, but built in one shot
    from the precomputed feature arrays.
    """

    ts = np.arange(max(start_idx, 20), end_idx)  # skip t with len(history) < 20

    if len(ts) == 0:
        return np.empty((0, 9)), np.empty((0,), dtype=int)

    f = FEATURES
    gap_clipped = np.minimum(f["gap"][ts], 100)

    X_per_t = np.stack(
        [
            f["frequency"][ts],
            f["freq_5"][ts],
            f["freq_10"][ts],
            f["freq_20"][ts],
            f["freq_50"][ts],
            f["freq_100"][ts],
            gap_clipped,
            f["frequency_norm"][ts],
            f["recent_norm"][ts],
        ],
        axis=-1,
    )  # (len(ts), 49, 9)

    y_per_t = f["A"][ts]  # (len(ts), 49)

    X = X_per_t.reshape(-1, 9)
    y = y_per_t.reshape(-1)

    return X, y


# ============================================================
# ML MODEL
# ============================================================

def train_ml_model(train_start, train_end):

    X, y = build_ml_dataset(train_start, train_end)

    if len(X) == 0:
        raise ValueError("Not enough data to train ML model.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=ML_C,
                    max_iter=ML_MAX_ITER,
                    class_weight=None,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    model.fit(X, y)

    return model


def ml_predict_scores(model, h):

    f = FEATURES

    X = np.stack(
        [
            f["frequency"][h],
            f["freq_5"][h],
            f["freq_10"][h],
            f["freq_20"][h],
            f["freq_50"][h],
            f["freq_100"][h],
            np.minimum(f["gap"][h], 100),
            f["frequency_norm"][h],
            f["recent_norm"][h],
        ],
        axis=-1,
    )

    probabilities = model.predict_proba(X)[:, 1]

    return pd.Series(probabilities, index=range(1, N_NUMBERS + 1))


# ============================================================
# PAIRWISE INTERACTION
# ============================================================
# Not a performance bottleneck (called once per fold on <=5000
# draws), left logically identical to the original.

def calculate_pair_scores(history):

    pair_counts = {}
    number_counts = {n: 0 for n in range(1, N_NUMBERS + 1)}

    draws = len(history)

    if draws == 0:
        return pair_counts

    for draw in history:

        unique_numbers = sorted(set(int(x) for x in draw))

        for n in unique_numbers:
            number_counts[n] += 1

        for a, b in itertools.combinations(unique_numbers, 2):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    scores = {}

    for a in range(1, N_NUMBERS + 1):
        for b in range(a + 1, N_NUMBERS + 1):

            observed = pair_counts.get((a, b), 0)

            pa = number_counts[a] / draws
            pb = number_counts[b] / draws

            expected = draws * pa * pb

            score = 1.0 if expected <= 0 else observed / expected
            score = np.clip(score, 0.25, 2.5)

            scores[(a, b)] = score

    return scores


# ============================================================
# HYBRID SCORING
# ============================================================

def calculate_hybrid_scores(history, ml_model):
    """
    Calculate hybrid score for all 49 numbers.
    """

    h = len(history)
    feature_df = get_feature_df(h)

    frequency_score = normalize_series(feature_df["frequency"])
    recent_score = feature_df["recent_norm"]
    gap_score = feature_df["gap_score"]

    ml_scores = ml_predict_scores(ml_model, h)
    ml_score = normalize_series(ml_scores)

    base_score = (
        W_FREQUENCY * frequency_score
        + W_RECENT * recent_score
        + W_GAP * gap_score
        + W_ML * ml_score
    )

    # Candidate pool: pairwise interactions are only evaluated
    # for the top 15 base-score candidates.
    candidate_numbers = (
        base_score.sort_values(ascending=False).head(15).index.tolist()
    )

    pair_scores = calculate_pair_scores(history)

    pair_values = {}

    for number in candidate_numbers:

        values = []

        for other in candidate_numbers:
            if number == other:
                continue
            pair = tuple(sorted((number, other)))
            values.append(pair_scores.get(pair, 1.0))

        pair_values[number] = np.mean(values) if values else 1.0

    pair_series = pd.Series(pair_values)
    pair_norm = normalize_series(pair_series)

    final_scores = base_score.copy()

    for number in candidate_numbers:
        final_scores.loc[number] = (
            base_score.loc[number] + W_PAIR * pair_norm.loc[number]
        )

    return (
        final_scores.sort_values(ascending=False),
        feature_df,
        ml_scores,
        pair_series,
    )


# ============================================================
# RANDOM BASELINE
# ============================================================
#
# Picking a fixed set of PICK_COUNT numbers uniformly at random
# out of N_NUMBERS, and counting matches against ANY fixed
# 6-number draw, is exactly Hypergeometric(N_NUMBERS, PICK_COUNT,
# PICK_COUNT) -- regardless of which numbers make up that draw.
# So instead of looping over 5000 simulations x test_draws with
# Python sets, we sample directly from the exact distribution.

def random_baseline(test_draws, n_simulations=5000, seed=RANDOM_SEED):

    rng = np.random.default_rng(seed)

    n_test = len(test_draws)

    samples = rng.hypergeometric(
        ngood=PICK_COUNT,
        nbad=N_NUMBERS - PICK_COUNT,
        nsample=PICK_COUNT,
        size=(n_simulations, n_test),
    )

    random_hits = samples.mean(axis=1)

    return (
        random_hits.mean(),
        np.percentile(random_hits, 2.5),
        np.percentile(random_hits, 97.5),
        random_hits,
    )


# ============================================================
# MODEL TEST
# ============================================================

def evaluate_selection(selected_numbers, test_draws):

    selected_mask = np.zeros(N_NUMBERS + 1, dtype=bool)
    selected_mask[list(selected_numbers)] = True

    hits = selected_mask[test_draws].sum(axis=1)

    return hits.mean(), hits.sum(), hits.max(), hits


# ============================================================
# EMPIRICAL P-VALUE
# ============================================================

def empirical_p_value(observed, random_distribution):
    return np.mean(random_distribution >= observed)


# ============================================================
# MAIN
# ============================================================

def main():

    global FEATURES

    print("=" * 70)
    print("HYBRID TOP-6 WALK-FORWARD TEST")
    print("=" * 70)

    print()
    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    number_columns = get_number_columns(df)

    df = validate_data(df, number_columns)

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")

    print()
    print("Number columns:")
    print(number_columns)

    print()
    print(f"Total draws: {len(df)}")

    print()
    print("Precomputing features for the entire dataset (vectorized)...")
    FEATURES = precompute_feature_arrays(df, number_columns)

    print()
    print("=" * 70)
    print("CREATING WALK-FORWARD FOLDS")
    print("=" * 70)

    total_draws = len(df)

    possible_folds = []

    for fold in range(N_FOLDS):

        train_end = TEST_SIZE * (fold + 1)
        test_start = train_end
        test_end = min(test_start + TEST_SIZE, total_draws)

        if test_start >= total_draws:
            break

        possible_folds.append((fold + 1, 0, train_end, test_start, test_end))

    results = []

    for (
        fold_number,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in possible_folds:

        print()
        print("=" * 70)
        print(f"FOLD {fold_number}")
        print("=" * 70)

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        if len(test_df) == 0:
            continue

        print(f"Training: {train_df['date'].iloc[0]} -> {train_df['date'].iloc[-1]}")
        print(f"Testing:  {test_df['date'].iloc[0]} -> {test_df['date'].iloc[-1]}")
        print(f"Training draws: {len(train_df)}")
        print(f"Testing draws:  {len(test_df)}")

        history = train_df[number_columns].values
        test_draws = test_df[number_columns].values

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        print()
        print("Training Logistic Regression...")

        ml_model = train_ml_model(train_start + 20, train_end)

        # ----------------------------------------------------
        # HYBRID SELECTION
        # ----------------------------------------------------

        print()
        print("Calculating hybrid scores...")

        (
            hybrid_scores,
            feature_df,
            ml_scores,
            pair_scores,
        ) = calculate_hybrid_scores(history, ml_model)

        selected_numbers = (
            hybrid_scores.head(PICK_COUNT).index.astype(int).tolist()
        )

        print()
        print("Hybrid Top-6 selected:")
        print(selected_numbers)

        # ----------------------------------------------------
        # SCORE TABLE
        # ----------------------------------------------------

        score_table = feature_df.copy()
        score_table["number"] = np.arange(1, N_NUMBERS + 1)
        score_table["ml_score"] = ml_scores.reindex(range(1, N_NUMBERS + 1)).values
        score_table["hybrid_score"] = hybrid_scores.reindex(
            range(1, N_NUMBERS + 1)
        ).values

        score_table = score_table[
            ["number", "frequency", "freq_20", "freq_50", "gap", "ml_score", "hybrid_score"]
        ]

        score_table = score_table.sort_values("hybrid_score", ascending=False)

        print()
        print("Top candidates:")
        print(score_table.head(10).to_string(index=False))

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        print()
        print("Evaluating hybrid Top-6...")

        average_hits, total_hits, maximum_hits, hits = evaluate_selection(
            selected_numbers, test_draws
        )

        # ----------------------------------------------------
        # RANDOM BASELINE
        # ----------------------------------------------------

        print()
        print("Running random Top-6 simulation...")

        random_mean, random_low, random_high, random_distribution = random_baseline(
            test_draws, n_simulations=5000, seed=RANDOM_SEED + fold_number
        )

        random_expected = PICK_COUNT * PICK_COUNT / N_NUMBERS

        difference = average_hits - random_expected
        difference_percent = difference / random_expected * 100

        p_value = empirical_p_value(average_hits, random_distribution)

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print()
        print("Results")
        print("-" * 40)
        print(f"Average hits:       {average_hits:.6f}")
        print(f"Total hits:         {total_hits}")
        print(f"Maximum hits:       {maximum_hits}")
        print(f"Random expected:    {random_expected:.6f}")
        print(f"Difference:         {difference:+.6f}")
        print(f"Difference %:       {difference_percent:+.3f}%")
        print(f"Random simulation:  {random_mean:.6f}")
        print(f"Random 95% range:   [{random_low:.6f}, {random_high:.6f}]")
        print(f"Empirical p-value:  {p_value:.6f}")

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "fold": fold_number,
                "train_start": train_df["date"].iloc[0],
                "train_end": train_df["date"].iloc[-1],
                "test_start": test_df["date"].iloc[0],
                "test_end": test_df["date"].iloc[-1],
                "train_draws": len(train_df),
                "test_draws": len(test_df),
                "selected_numbers": ",".join(map(str, selected_numbers)),
                "average_hits": average_hits,
                "total_hits": total_hits,
                "maximum_hits": maximum_hits,
                "random_expected": random_expected,
                "random_simulation": random_mean,
                "random_low": random_low,
                "random_high": random_high,
                "difference": difference,
                "difference_percent": difference_percent,
                "empirical_p": p_value,
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    if len(results_df) > 0:
        summary_columns = [
            "fold",
            "test_draws",
            "average_hits",
            "random_expected",
            "difference",
            "difference_percent",
            "empirical_p",
        ]
        print(results_df[summary_columns].to_string(index=False))

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    if len(results_df) > 0:

        mean_model_hits = results_df["average_hits"].mean()
        mean_random = results_df["random_expected"].mean()
        mean_difference = results_df["difference"].mean()
        mean_difference_percent = results_df["difference_percent"].mean()
        mean_p = results_df["empirical_p"].mean()

        above_random = np.sum(results_df["difference"] > 0)
        below_random = np.sum(results_df["difference"] < 0)

        print(f"Mean model hits:       {mean_model_hits:.6f}")
        print(f"Mean random expected:   {mean_random:.6f}")
        print(f"Mean difference:        {mean_difference:+.6f}")
        print(f"Mean difference %:      {mean_difference_percent:+.3f}%")
        print(f"Mean empirical p:       {mean_p:.6f}")

        print()
        print("Fold consistency:")
        print(f"Above random: {above_random}")
        print(f"Below random: {below_random}")

        print()
        print("=" * 70)
        print("FINAL CONCLUSION")
        print("=" * 70)

        if (
            mean_difference > 0
            and mean_difference_percent >= 3
            and above_random >= 4
            and mean_p < 0.05
        ):
            print("Potential hybrid advantage detected.")
        elif mean_difference > 0 and above_random > below_random:
            print("Weak positive signal, but not statistically strong.")
        else:
            print("No meaningful hybrid advantage over random selection.")

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()