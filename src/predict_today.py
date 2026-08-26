import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import date

DATA_PATH = "data/processed/lotto_6aus49_clean.csv"
WINDOW = 1000
TOP_K = 6
MAX_NUMBER = 49
NUM_SETS = 5          # how many main-number sets to generate
POOL_SIZE = 15        # draw sets from the top N scored numbers (for variety)

SUPERZAHL_MIN = 0
SUPERZAHL_MAX = 9
SUPERZAHL_WINDOW = 500   # separate window since superzahl history is shorter


def create_features(history_array, i, n, cum):
    history = history_array[:i]
    freq_20 = history[-20:].sum()
    freq_50 = history[-50:].sum()
    freq_100 = history[-100:].sum()
    nz = np.nonzero(history)[0]
    gap = i if len(nz) == 0 else (i - 1 - nz[-1])
    total_rate = cum[i - 1] / i
    return [
        freq_20, freq_50, freq_100, gap,
        freq_20 / 20, freq_50 / 50, freq_100 / 100,
        total_rate, i / n
    ]


def build_features_for_target(target):
    n = len(target)
    features = np.zeros((n, 9))
    cum = np.cumsum(target)
    for i in range(100, n):
        features[i] = create_features(target, i, n, cum)
    return features


def calculate_main_number_scores(df, window):
    values = df[[f"n{i}" for i in range(1, 7)]].values
    scores = {}
    for number in range(1, MAX_NUMBER + 1):
        target = np.array([1 if number in row else 0 for row in values])
        X = build_features_for_target(target)
        y = target
        valid = np.arange(100, len(df))
        X_train, y_train = X[valid], y[valid]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        start = max(100, len(df) - window)
        X_recent, y_recent = X[start:], y[start:]
        if len(X_recent) == 0:
            continue
        X_recent_scaled = scaler.transform(X_recent)
        probabilities = model.predict_proba(X_recent_scaled)[:, 1]
        recent_score = probabilities.mean()
        frequency_rate = y_recent.sum() / len(y_recent)
        scores[number] = 0.7 * recent_score + 0.3 * frequency_rate
    return scores


def calculate_superzahl_scores(df, window):
    sz_df = df.dropna(subset=["superzahl"]).reset_index(drop=True)
    sz_df["superzahl"] = sz_df["superzahl"].astype(int)
    values = sz_df["superzahl"].values

    scores = {}
    for digit in range(SUPERZAHL_MIN, SUPERZAHL_MAX + 1):
        target = (values == digit).astype(int)
        n = len(target)
        if n < 150:
            continue
        X = build_features_for_target(target)
        y = target
        valid = np.arange(100, n)
        X_train, y_train = X[valid], y[valid]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        start = max(100, n - window)
        X_recent, y_recent = X[start:], y[start:]
        if len(X_recent) == 0:
            continue
        X_recent_scaled = scaler.transform(X_recent)
        probabilities = model.predict_proba(X_recent_scaled)[:, 1]
        recent_score = probabilities.mean()
        frequency_rate = y_recent.sum() / len(y_recent)
        scores[digit] = 0.7 * recent_score + 0.3 * frequency_rate

    return scores, len(sz_df)


def main():
    print("=" * 70)
    print("LOTTO 6/49 + SUPERZAHL - TODAY'S PICKER (FOR FUN & LEARNING ONLY)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Last known draw date: {df['date'].max().date()}")

    # ---- Main numbers ----
    print(f"\nMain-number scoring window: last {WINDOW} draws")
    scores = calculate_main_number_scores(df, WINDOW)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {POOL_SIZE} main-number candidates (score):")
    for number, score in ranked[:POOL_SIZE]:
        print(f"  {number:2d} -> {score:.6f}")

    top6 = sorted([n for n, s in ranked[:TOP_K]])

    pool_numbers = [n for n, s in ranked[:POOL_SIZE]]
    pool_weights = np.array([s for n, s in ranked[:POOL_SIZE]])
    pool_weights = pool_weights / pool_weights.sum()

    rng = np.random.default_rng(42)
    extra_sets = []
    seen = {tuple(top6)}
    attempts = 0
    while len(extra_sets) < NUM_SETS - 1 and attempts < 200:
        attempts += 1
        pick = rng.choice(pool_numbers, size=TOP_K, replace=False, p=pool_weights)
        pick = tuple(sorted(int(x) for x in pick))
        if pick not in seen:
            seen.add(pick)
            extra_sets.append(pick)
    all_sets = [tuple(top6)] + extra_sets

    # ---- Superzahl ----
    print(f"\nSuperzahl scoring window: last {SUPERZAHL_WINDOW} draws")
    sz_scores, sz_n = calculate_superzahl_scores(df, SUPERZAHL_WINDOW)
    sz_ranked = sorted(sz_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"Superzahl history available: {sz_n} draws")
    print("\nAll Superzahl candidates (score):")
    for digit, score in sz_ranked:
        print(f"  {digit} -> {score:.6f}")

    top_sz = [d for d, s in sz_ranked[:3]]  # top 3 superzahl guesses

    # ---- Final report ----
    today = date.today().isoformat()
    print("\n" + "=" * 70)
    print(f"PICKED SETS FOR TODAY ({today})")
    print("=" * 70)
    for idx, s in enumerate(all_sets, start=1):
        label = "  (highest score)" if idx == 1 else ""
        print(f"  Option {idx}: " + "  ".join(f"{n:02d}" for n in s) + label)

    print("\nSuperzahl guesses, ranked:")
    for i, d in enumerate(top_sz, start=1):
        print(f"  Rank {i}: {d}")
    print("=" * 70)


if __name__ == "__main__":
    main()