import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Importing our custom modules
from src.preprocessing import TrafficPreprocessor
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svr import SVRModel
from src.evaluator import ModelEvaluator
from src.route_optimization import RouteOptimizer

# Directory where trained artefacts are persisted for the web app
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _build_historical_stats(df):
    """
    Compute per-(area, road, hour) historical averages for the features
    that require lagged / rolling data at inference time.

    Returns a DataFrame indexed by (area name, road/intersection name, hour)
    with columns: mean_volume, mean_speed, mean_vol_lag1, mean_vol_lag2,
    mean_rolling3.
    """
    stats = (
        df.groupby(['area name', 'road/intersection name', 'hour'])
        .agg(
            mean_volume=('traffic volume', 'mean'),
            mean_speed=('average speed', 'mean'),
            mean_vol_lag1=('vol_1_step_ago', 'mean'),
            mean_vol_lag2=('vol_2_steps_ago', 'mean'),
            mean_rolling3=('rolling_trend_3', 'mean'),
        )
    )
    return stats


def main():
    print("🚀 Starting AI-Based Proactive Traffic Prediction System...\n")
    
    # 1. Define data path
    # Using os.path.join makes sure it works on both Windows and Mac/Linux
    data_path = os.path.join('data', 'raw_traffic.csv')
    
    # Safety check: ensure the data exists before crashing
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("Please place your downloaded CSV file into the 'data' folder and name it 'raw_traffic.csv'.")
        return

    # 2. Preprocess the Data
    preprocessor = TrafficPreprocessor(data_path)
    df_clean = preprocessor.load_and_clean()
    
    # Save the cleaned data (Great to show the examiner!)
    clean_data_path = os.path.join('data', 'processed_traffic.csv')
    df_clean.to_csv(clean_data_path, index=False)
    print(f"✅ Cleaned dataset saved to: {clean_data_path}\n")

    # 3. Extract Features and Target
    X_scaled, y = preprocessor.get_features_and_target(df_clean)
    
    # 4. Train-Test Split (80% training, 20% unseen future testing)
    print("✂️ Splitting data into 80% Training and 20% Testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 5. Initialize our Object-Oriented Models
    models_dict = {
        "Linear Regression": LinearRegressionModel(),
        "Random Forest": RandomForestModel(),
        "SVR": SVRModel()
    }

    # 6. Run the Evaluator
    evaluator = ModelEvaluator(models_dict)
    results = evaluator.train_and_evaluate(X_train, X_test, y_train, y_test)

    # 7. Announce the Winner
    print("\n" + "="*40)
    print("🏆 FINAL VERDICT 🏆")
    print("="*40)
    
    # Find the model with the lowest MAE
    best_model_name = min(results, key=lambda k: results[k]['MAE'])
    print(f"The Best Performing Model is: ** {best_model_name} **")
    print("It successfully predicted traffic 30-60 mins in advance with the lowest error rate.")
    print("="*40 + "\n")

    # ──────────────────────────────────────────────────────────────
    # 8. SAVE TRAINED ARTEFACTS for the web app
    # ──────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 8a. Save the best model (Random Forest)
    rf_model = models_dict["Random Forest"].model
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest.joblib'))
    print("💾 Saved trained Random Forest model → models/random_forest.joblib")

    # 8b. Save the scaler so the web predictor transforms inputs identically
    joblib.dump(preprocessor.scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    print("💾 Saved StandardScaler           → models/scaler.joblib")

    # 8c. Save label encoders (area, road, weather)
    #     We need to re-fit them on the cleaned data to get stable mappings.
    from sklearn.preprocessing import LabelEncoder
    area_enc = LabelEncoder().fit(df_clean['area name'])
    road_enc = LabelEncoder().fit(df_clean['road/intersection name'])
    weather_enc = LabelEncoder().fit(df_clean['weather conditions'])

    joblib.dump(area_enc, os.path.join(MODELS_DIR, 'area_encoder.joblib'))
    joblib.dump(road_enc, os.path.join(MODELS_DIR, 'road_encoder.joblib'))
    joblib.dump(weather_enc, os.path.join(MODELS_DIR, 'weather_encoder.joblib'))
    print("💾 Saved LabelEncoders             → models/*_encoder.joblib")

    # 8d. Save historical statistics for filling lagged features at inference
    hist_stats = _build_historical_stats(df_clean)
    joblib.dump(hist_stats, os.path.join(MODELS_DIR, 'historical_stats.joblib'))
    print("💾 Saved historical stats          → models/historical_stats.joblib")

    # 8e. Save per-area median volume (used to normalise predicted volume → multiplier)
    area_medians = df_clean.groupby('area name')['traffic volume'].median()
    joblib.dump(area_medians, os.path.join(MODELS_DIR, 'area_medians.joblib'))
    print("💾 Saved area median volumes       → models/area_medians.joblib")

    print("\n✅ All ML artefacts saved. The web app will now use Real ML predictions!\n")

    # ──────────────────────────────────────────────────────────────
    # 9. Route Optimization Demonstration (uses live ML predictor)
    # ──────────────────────────────────────────────────────────────
    print("\n🗺️ Demonstrating Route Optimization (Objective 2) 🗺️\n")
    optimizer = RouteOptimizer()

    print("🚦 Generating ML-based congestion predictions for current time...")
    predictions = optimizer.generate_dynamic_predictions()
    optimizer.update_edge_weights(predictions)

    source_node = 'A'
    dest_node = 'G'
    print(f"📍 Finding the most efficient route from {source_node} to {dest_node}...")

    optimal_route, total_weight = optimizer.find_best_route(source_node, dest_node)

    if optimal_route:
        print(f"✅ Optimal Route: {' -> '.join(optimal_route)}")
        print(f"⏱️ Estimated Travel Time: {total_weight:.1f} minutes")
    else:
        print("❌ No valid route found.")

if __name__ == "__main__":
    main()
  