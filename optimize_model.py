"""
optimize_model.py — Lightweight Model Retrainer for HF Spaces Deployment
==========================================================================

Retrains the traffic prediction model using HistGradientBoostingRegressor
instead of RandomForestRegressor to shrink the serialized .pkl from
~727 MB down to <50 MB.

WHY THIS WORKS:
  - RandomForest stores every split threshold for every tree node across
    100 full-depth trees → huge serialized size.
  - HistGradientBoosting uses histogram-based binning (max 255 bins per
    feature), stores only bin indices + compact leaf values → 90-95%
    smaller. On large tabular datasets it matches or beats RF accuracy.

COMPATIBILITY:
  - Outputs the EXACT same Pipeline([preprocessor, regressor]) structure
    that app.py expects (ColumnTransformer → StandardScaler → regressor).
  - app.py's feature-detection logic (inspecting pipeline.named_steps
    ['preprocessor'].transformers) works unchanged.

USAGE:
    python optimize_model.py

  The script reads all CSVs from RAW/, trains the model, evaluates it,
  and overwrites models/best_traffic_model.pkl.
"""

import glob
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "Raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PKL = os.path.join(MODELS_DIR, "best_traffic_model.pkl")

# ── Peak-hour definitions (same as train_models.py) ─────────────────────
MORNING_PEAK = (8, 11)
EVENING_PEAK = (17, 20)

DOWNSAMPLE_LIMIT = 100_000  # max rows for training (RAM-friendly)


def _is_peak(hour: float) -> int:
    return int(
        (MORNING_PEAK[0] <= hour <= MORNING_PEAK[1])
        or (EVENING_PEAK[0] <= hour <= EVENING_PEAK[1])
    )


# =====================================================================
# 1. DATA INGESTION (identical to train_models.py)
# =====================================================================

def load_and_clean() -> pd.DataFrame:
    csv_pattern = os.path.join(RAW_DIR, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"❌ No CSV files found in '{RAW_DIR}'.")
        print("   Place your Uber Movement CSV(s) in the Raw/ folder.")
        sys.exit(1)

    print(f"📂 Found {len(csv_files)} CSV file(s) in Raw/:")
    for f in csv_files:
        print(f"   • {os.path.basename(f)}")

    # Read & concatenate
    frames = []
    for f in csv_files:
        chunk = pd.read_csv(f)
        print(f"   ↳ {os.path.basename(f):50s} → {chunk.shape[0]:>8,} rows")
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n   Combined raw shape: {df.shape}")

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # Target column
    target_col = "mean_travel_time"
    if target_col not in df.columns:
        print(f"❌ Expected column '{target_col}' not found.")
        print(f"   Available: {df.columns.tolist()}")
        sys.exit(1)

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    df = df.fillna(0)

    # Remove outliers (negative / infinite)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = df[np.isfinite(df[col]) & (df[col] >= 0)]

    # Downsample for memory
    if len(df) > DOWNSAMPLE_LIMIT:
        print(f"   ⚠️  Downsampling from {len(df):,} to {DOWNSAMPLE_LIMIT:,} rows")
        df = df.sample(n=DOWNSAMPLE_LIMIT, random_state=42).reset_index(drop=True)

    # Feature engineering (same as train_models.py)
    if "hod" in df.columns:
        df["hour_of_day"] = df["hod"].astype(float)
    else:
        df["hour_of_day"] = 0.0

    df["is_peak_hour"] = df["hour_of_day"].apply(_is_peak)
    df["day_of_week"] = 0  # not in Uber Movement data

    # Target: seconds → minutes
    df["travel_time_min"] = df[target_col] / 60.0
    df = df[(df["travel_time_min"] > 0) & (df["travel_time_min"] < 120)]

    print(f"   Final shape: {df.shape}")
    return df


# =====================================================================
# 2. BUILD PIPELINE & TRAIN
# =====================================================================

def train_optimized(df: pd.DataFrame):
    """
    Train a HistGradientBoostingRegressor with the same Pipeline structure
    that app.py expects: Pipeline([preprocessor, regressor]).
    """
    # Feature columns — MUST match what train_models.py uses
    num_cols = ["hour_of_day", "day_of_week", "is_peak_hour"]
    for col in ["sourceid", "dstid",
                 "standard_deviation_travel_time",
                 "geometric_mean_travel_time",
                 "geometric_standard_deviation_travel_time"]:
        if col in df.columns:
            num_cols.append(col)

    print(f"\n⚙️  Features: {num_cols}")

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="drop",
    )

    X = df[num_cols].copy()
    y = df["travel_time_min"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── HistGradientBoosting: histogram-based = tiny model file ──────
    # Tuned for accuracy + small size:
    #   - max_iter=300: enough boosting rounds for convergence
    #   - max_depth=8: prevents overfitting while capturing complexity
    #   - max_bins=255: max histogram resolution (default)
    #   - min_samples_leaf=20: regularization
    #   - learning_rate=0.1: standard learning rate
    regressor = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=8,
        min_samples_leaf=20,
        learning_rate=0.1,
        max_bins=255,
        random_state=42,
        verbose=1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    print(f"\n🔧 Training HistGradientBoostingRegressor on {len(X_train):,} rows …\n")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  HistGradientBoosting Results")
    print(f"  RMSE = {rmse:.4f} min")
    print(f"  R²   = {r2:.4f}")
    print(f"{'='*50}")

    return pipeline, rmse, r2


# =====================================================================
# 3. SAVE & VERIFY SIZE
# =====================================================================

def main():
    print("🚀 Model Optimization for HF Spaces Deployment\n")
    print("   Goal: Shrink best_traffic_model.pkl from ~727 MB to <50 MB")
    print("   Method: HistGradientBoostingRegressor (histogram-based binning)\n")

    # Check RAW/ exists
    if not os.path.isdir(RAW_DIR):
        print(f"❌ Raw folder not found: {RAW_DIR}")
        sys.exit(1)

    # Check for existing model (to show size comparison)
    old_size_mb = None
    if os.path.exists(OUTPUT_PKL):
        old_size_mb = os.path.getsize(OUTPUT_PKL) / (1024 * 1024)
        print(f"📦 Current model size: {old_size_mb:.1f} MB")

    # Load data
    df = load_and_clean()

    # Train
    pipeline, rmse, r2 = train_optimized(df)

    # Save with compression (compress=3 gives ~30-50% additional shrinkage)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PKL, compress=3)

    new_size_mb = os.path.getsize(OUTPUT_PKL) / (1024 * 1024)

    print(f"\n💾 Model saved → {OUTPUT_PKL}")
    print(f"   New size: {new_size_mb:.1f} MB")
    if old_size_mb:
        reduction = ((old_size_mb - new_size_mb) / old_size_mb) * 100
        print(f"   Old size: {old_size_mb:.1f} MB")
        print(f"   Reduction: {reduction:.1f}%")

    if new_size_mb < 50:
        print(f"\n✅ SUCCESS — Model is under 50 MB ({new_size_mb:.1f} MB)")
    else:
        print(f"\n⚠️  Model is {new_size_mb:.1f} MB — still over 50 MB target.")
        print("   Consider reducing max_iter or max_depth to shrink further.")

    print("\n✅ Ready for Hugging Face deployment. Run your Docker build next.\n")


if __name__ == "__main__":
    main()
