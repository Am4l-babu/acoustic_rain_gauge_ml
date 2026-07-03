"""
Feature Engineering & Preprocessing (Stage 3)
==============================================
Turns the combined Stage 2 dataset (data/processed/combined_dataset.csv,
built by notebooks/eda.ipynb) into train/test splits ready for modeling.

Per-clip acoustic feature computation (PAR, spectral stats, MFCCs, ...)
already happens during Stage 1 in data_cleaning.py / data_cleaning_gpu.py.
This script operates one level up, on the combined dataset:

  1. Drop the duration-confounded short clips (<5s / 5-10s). EDA showed
     these are 97% recorded in different acoustic conditions (mean
     spectral_centroid ~1160Hz vs ~478Hz) and almost never rainy (2.0%
     vs 15.0%), so keeping them risks the model learning "clip length"
     instead of "rain".
  2. Split chronologically (earliest ~80% -> train, latest ~20% -> test).
     No shuffling: the target use case is predicting on future audio,
     so validation has to happen on future audio too.
  3. Compute class weights for the rain/no-rain imbalance from the
     training split only.
  4. Fit a StandardScaler on the training split only, to avoid leaking
     test-set statistics into the features.

Run:
    python src/feature_extraction.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT     = Path(__file__).resolve().parent.parent
COMBINED_DATA = REPO_ROOT / "data" / "processed" / "combined_dataset.csv"
OUTPUT_DIR    = REPO_ROOT / "data" / "processed"
MODELS_DIR    = REPO_ROOT / "models"

FEATURE_COLS = [
    "rms", "peak", "par",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "zero_crossing_rate", "energy_variance",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
]
KEEP_DURATION_CATEGORY = "10-15s"
TEST_FRACTION = 0.2

# Legitimate readings in this dataset top out around 21.6mm (a real heavy
# downpour in May 2024). Anything past this is treated as a sensor
# artifact rather than real rainfall -- most visibly a repeating ~655.2-
# 655.33mm value that shows up identically across unrelated recording
# campaigns (May 2025 and Feb-Mar 2026), which is a signature of a
# corrupted/overflowed counter, not weather.
RAINFALL_MM_CAP = 50.0


def load_combined() -> pd.DataFrame:
    if not COMBINED_DATA.exists():
        raise FileNotFoundError(
            f"{COMBINED_DATA} not found. Run notebooks/eda.ipynb first "
            "(section 9) to build the combined dataset."
        )
    df = pd.read_csv(COMBINED_DATA, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def drop_confounded_clips(df: pd.DataFrame) -> pd.DataFrame:
    kept = df[df["duration_category"] == KEEP_DURATION_CATEGORY].reset_index(drop=True)
    dropped = len(df) - len(kept)
    print(f"  Dropped {dropped:,} non-{KEEP_DURATION_CATEGORY} clips "
          f"({dropped / len(df):.1%} of dataset)")
    return kept


def drop_sensor_artifact_readings(df: pd.DataFrame) -> pd.DataFrame:
    kept = df[df["rainfall_mm"] <= RAINFALL_MM_CAP].reset_index(drop=True)
    dropped = len(df) - len(kept)
    print(f"  Dropped {dropped:,} rows with rainfall_mm > {RAINFALL_MM_CAP} "
          f"(sensor artifacts, e.g. repeating ~655mm readings)")
    return kept


def chronological_split(df: pd.DataFrame, test_fraction: float):
    """
    Split chronologically *within each month_folder* rather than on one
    global cutoff. The 19 folders are separate recording campaigns spread
    unevenly across ~2.5 years (e.g. 3 days in Aug 2025, weeks elsewhere),
    each with its own rainy-rate. A single global time cutoff would push
    entire high-rain campaigns (e.g. Feb-Mar 2026 at ~93% rainy) wholesale
    into test while train stays dry, which is what happened on the first
    attempt (train 8.9% rainy vs test 39.4% rainy). Splitting per-folder
    keeps every campaign represented in both splits while still holding
    out each campaign's most recent recordings for testing.
    """
    train_parts, test_parts = [], []
    for _, group in df.groupby("month_folder", sort=False):
        group = group.sort_values("timestamp")
        split_idx = int(len(group) * (1 - test_fraction))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train = pd.concat(train_parts).sort_values("timestamp").reset_index(drop=True)
    test = pd.concat(test_parts).sort_values("timestamp").reset_index(drop=True)
    return train, test


def main():
    print("=" * 68)
    print("  STAGE 3 - FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 68)

    print("\n[1] Loading combined dataset...")
    df = load_combined()
    rows_before_filter = len(df)
    print(f"  Rows: {rows_before_filter:,}")

    print("\n[2] Dropping duration-confounded clips...")
    df = drop_confounded_clips(df)
    rows_after_duration_filter = len(df)

    print("\n[3] Dropping sensor-artifact rainfall readings...")
    df = drop_sensor_artifact_readings(df)
    rows_after_artifact_filter = len(df)
    df["is_rainy"] = (df["rainfall_mm"] > 0).astype(int)

    print("\n[4] Chronological train/test split (per month_folder)...")
    train_df, test_df = chronological_split(df, TEST_FRACTION)
    print(f"  Train      : {len(train_df):,} rows "
          f"({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"  Test       : {len(test_df):,} rows "
          f"({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")
    print(f"  Train rainy: {train_df['is_rainy'].mean():.1%}")
    print(f"  Test rainy : {test_df['is_rainy'].mean():.1%}")

    print("\n[5] Computing class weights (from training split only)...")
    classes = sorted(train_df["is_rainy"].unique())
    weights = compute_class_weight(class_weight="balanced", classes=classes,
                                    y=train_df["is_rainy"])
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Class weights: {class_weights}")

    print("\n[6] Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
    print(f"  Saved: {MODELS_DIR / 'feature_scaler.pkl'}")

    print("\n[7] Saving train/test splits...")
    keep_cols = ["audio_filename", "audio_full_path", "month_folder", "timestamp",
                 "duration_sec", *FEATURE_COLS, "rainfall_mm", "is_rainy"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df[keep_cols].to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df[keep_cols].to_csv(OUTPUT_DIR / "test.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'train.csv'}")
    print(f"  Saved: {OUTPUT_DIR / 'test.csv'}")

    metadata = {
        "rows_before_duration_filter": rows_before_filter,
        "rows_after_duration_filter": rows_after_duration_filter,
        "dropped_duration_confounded_rows": rows_before_filter - rows_after_duration_filter,
        "rows_after_artifact_filter": rows_after_artifact_filter,
        "dropped_sensor_artifact_rows": rows_after_duration_filter - rows_after_artifact_filter,
        "rainfall_mm_cap": RAINFALL_MM_CAP,
        "kept_duration_category": KEEP_DURATION_CATEGORY,
        "feature_columns": FEATURE_COLS,
        "test_fraction": TEST_FRACTION,
        "split_strategy": "per-month_folder chronological (last test_fraction of each campaign)",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_rainy_fraction": float(train_df["is_rainy"].mean()),
        "test_rainy_fraction": float(test_df["is_rainy"].mean()),
        "class_weights": class_weights,
        "scaler_file": "models/feature_scaler.pkl",
    }
    with open(OUTPUT_DIR / "feature_engineering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'feature_engineering_metadata.json'}")

    print("\n" + "=" * 68)
    print("  DONE - ready for Stage 4 (src/train_model.py)")
    print("=" * 68)


if __name__ == "__main__":
    main()
