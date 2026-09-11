"""
train_models.py — Multi-Model ML Pipeline for Bengaluru Traffic Prediction
===========================================================================

Dynamically ingests **all CSV files** from a local `RAW/` folder (located in
the same directory as this script), concatenates them into a single DataFrame,
engineers temporal + categorical features, trains three regressors
(Random Forest, HistGradientBoosting, SVR), evaluates on RMSE / R²,
and persists the best model + preprocessing pipeline as
`models/best_traffic_model.pkl`.

Adapted for the **Uber Movement aggregated** dataset schema:
  sourceid, dstid, hod, mean_travel_time, standard_deviation_travel_time,
  geometric_mean_travel_time, geometric_standard_deviation_travel_time, …
"""

import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "RAW")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PKL = os.path.join(MODELS_DIR, "best_traffic_model.pkl")

# ── Peak-hour definitions ────────────────────────────────────────────────
MORNING_PEAK = (8, 11)   # 08:00 – 11:00
EVENING_PEAK = (17, 20)  # 17:00 – 20:00

SVR_SAMPLE_LIMIT = 20_000   # max rows fed to SVR to avoid CPU bottleneck
DOWNSAMPLE_LIMIT = 100_000  # max rows to keep for training (RAM-friendly)


def _is_peak(hour: float) -> int:
    """Return 1 if hour falls within morning or evening peak windows."""
    return int(
        (MORNING_PEAK[0] <= hour <= MORNING_PEAK[1])
        or (EVENING_PEAK[0] <= hour <= EVENING_PEAK[1])
    )


# =====================================================================
# 1. DATA INGESTION & PREPROCESSING
# =====================================================================

def load_and_clean() -> pd.DataFrame:
    """
    Discover every CSV in ``RAW/``, concatenate them into one DataFrame,
    clean selectively, and engineer features for the Uber Movement schema.
    """
    # ── Discover CSV files ────────────────────────────────────────────
    csv_pattern = os.path.join(RAW_DIR, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{RAW_DIR}'.\n"
            "Place your dataset CSV(s) inside the RAW/ folder and re-run."
        )

    print(f"📂 Found {len(csv_files)} CSV file(s) in RAW/:")
    for f in csv_files:
        print(f"   • {os.path.basename(f)}")

    # ── Read & concatenate ────────────────────────────────────────────
    frames = []
    for f in csv_files:
        chunk = pd.read_csv(f)
        print(f"   ↳ {os.path.basename(f):40s} → {chunk.shape[0]:>8,} rows, {chunk.shape[1]:>3} cols")
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n   Combined raw shape: {df.shape}")

    # Normalise column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # ── Print actual columns so we can verify the schema ─────────────
    print(f"   Columns: {df.columns.tolist()}")

    # ── Drop rows only where the TARGET is missing ───────────────────
    target_col = "mean_travel_time"
    if target_col not in df.columns:
        raise KeyError(
            f"Expected target column '{target_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    if dropped:
        print(f"   Dropped {dropped:,} rows where '{target_col}' was NaN.")

    # Fill remaining NaNs with 0 so we don't destroy the dataset
    nan_counts = df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        print(f"   Filling NaNs with 0 in {len(cols_with_nans)} column(s): "
              f"{cols_with_nans.index.tolist()}")
    df = df.fillna(0)

    print(f"   After cleaning: {df.shape}")

    # ── Remove mathematical outliers (negative / infinite values) ────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = np.isfinite(df[col]) & (df[col] >= 0)
        df = df[mask]
    print(f"   After outlier removal: {df.shape}")

    # ── Downsample for memory ────────────────────────────────────────
    if len(df) > DOWNSAMPLE_LIMIT:
        print(f"\n   ⚠️  Dataset has {len(df):,} rows — downsampling to "
              f"{DOWNSAMPLE_LIMIT:,} for training speed & RAM.")
        df = df.sample(n=DOWNSAMPLE_LIMIT, random_state=42)
        df = df.reset_index(drop=True)

    # ── Feature Engineering (Uber Movement schema) ───────────────────
    # 'hod' = Hour of Day (integer 0–23), no date column exists.
    if "hod" in df.columns:
        df["hour_of_day"] = df["hod"].astype(float)
    else:
        # Fallback: if somehow an 'hour' column exists under another name
        print("   ⚠️  'hod' column not found — setting hour_of_day to 0.")
        df["hour_of_day"] = 0.0

    df["is_peak_hour"] = df["hour_of_day"].apply(_is_peak)

    # Day-of-week is not available in Uber Movement aggregated data;
    # set a placeholder so the pipeline shape stays consistent.
    df["day_of_week"] = 0

    # ── Target variable ──────────────────────────────────────────────
    # Convert mean_travel_time (seconds) → minutes for readability.
    df["travel_time_min"] = df[target_col] / 60.0

    # Remove unrealistic targets
    df = df[df["travel_time_min"] > 0]
    df = df[df["travel_time_min"] < 120]  # cap at 2 hours

    print(f"   Final shape: {df.shape}")
    return df


# =====================================================================
# 2. BUILD PREPROCESSING PIPELINE
# =====================================================================

def build_preprocessor(df: pd.DataFrame):
    """
    Build a ColumnTransformer that scales numeric features.

    Returns (preprocessor, X, y).
    """
    num_cols = ["hour_of_day", "day_of_week", "is_peak_hour"]

    # Uber Movement numeric features (add if present)
    for col in ["sourceid", "dstid",
                 "standard_deviation_travel_time",
                 "geometric_mean_travel_time",
                 "geometric_standard_deviation_travel_time"]:
        if col in df.columns:
            num_cols.append(col)

    # Also pick up any other numeric columns from the original Bengaluru
    # dataset if they happen to be present (backwards-compatible).
    for col in ["average speed", "traffic volume", "congestion level",
                 "road capacity utilization"]:
        if col in df.columns:
            num_cols.append(col)

    # Categorical features (only if present — Uber data has none by default)
    cat_cols = []
    for col in ["area name", "road/intersection name", "weather conditions"]:
        if col in df.columns:
            cat_cols.append(col)

    print(f"   Numeric features:     {num_cols}")
    if cat_cols:
        print(f"   Categorical features: {cat_cols}")

    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    y = df["travel_time_min"].copy()

    return preprocessor, X, y


# =====================================================================
# 3. MODEL TRAINING & COMPARISON
# =====================================================================

def train_and_compare(preprocessor, X: pd.DataFrame, y: pd.Series):
    """
    Train RF, HistGBR, SVR.  Evaluate on RMSE and R².
    Return (best_pipeline, best_name, results_dict).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressors = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=42
        ),
        "SVR (RBF)": SVR(kernel="rbf"),
    }

    results = {}
    best_score = -np.inf
    best_name = None
    best_pipeline = None

    for name, reg in regressors.items():
        print(f"\n🔧 Training {name} …")

        # For SVR, sample if dataset is large
        if name == "SVR (RBF)" and len(X_train) > SVR_SAMPLE_LIMIT:
            sample_idx = np.random.RandomState(42).choice(
                len(X_train), SVR_SAMPLE_LIMIT, replace=False
            )
            X_tr = X_train.iloc[sample_idx]
            y_tr = y_train.iloc[sample_idx]
            print(f"   (Sampled {SVR_SAMPLE_LIMIT:,} rows for SVR)")
        else:
            X_tr = X_train
            y_tr = y_train

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg),
        ])
        pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"rmse": rmse, "r2": r2}
        print(f"   RMSE = {rmse:.4f}  |  R² = {r2:.4f}")

        if r2 > best_score:
            best_score = r2
            best_name = name
            # Re-fit on full training set for SVR (it was sampled)
            if name == "SVR (RBF)" and len(X_train) > SVR_SAMPLE_LIMIT:
                best_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("regressor", SVR(kernel="rbf")),
                ])
                best_pipeline.fit(X_tr, y_tr)  # keep sampled fit for consistency
            else:
                best_pipeline = pipe

    return best_pipeline, best_name, results


def print_comparison_table(results: dict, best_name: str):
    """Print a clean terminal comparison table."""
    print("\n" + "=" * 58)
    print(f"  {'Model':<28} {'RMSE':>10} {'R²':>10}  ")
    print("-" * 58)
    for name, metrics in results.items():
        marker = " 🏆" if name == best_name else ""
        print(
            f"  {name:<28} {metrics['rmse']:>10.4f} {metrics['r2']:>10.4f}{marker}"
        )
    print("=" * 58)


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("🚀 Bengaluru Traffic — Multi-Model ML Pipeline\n")

    # ── Verify RAW/ folder exists ─────────────────────────────────────
    if not os.path.isdir(RAW_DIR):
        print(f"❌ RAW folder not found: {RAW_DIR}")
        print("   Create a 'RAW/' directory next to this script and place")
        print("   your CSV dataset file(s) inside it.")
        return

    # 1. Load & clean (reads all CSVs from RAW/)
    df = load_and_clean()

    # 2. Build preprocessor
    print("\n⚙️  Building feature pipeline …")
    preprocessor, X, y = build_preprocessor(df)

    # 3. Train & compare
    print("\n📊 Training three regressors (80/20 split) …")
    best_pipeline, best_name, results = train_and_compare(preprocessor, X, y)

    # 4. Print comparison
    print_comparison_table(results, best_name)

    # 5. Persist best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, OUTPUT_PKL)
    print(f"\n💾 Best model saved → {OUTPUT_PKL}")
    print(f"   Model: {best_name}")
    print(f"   R² = {results[best_name]['r2']:.4f}")
    print("\n✅ Pipeline complete. The Flask app will use this model.\n")


if __name__ == "__main__":
    main()
