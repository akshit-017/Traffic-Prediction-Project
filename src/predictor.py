"""
TrafficPredictor — Bridge between the trained ML model and the route optimizer.

Loads the saved Random Forest model, scaler, encoders, and historical
statistics from the ``models/`` directory and exposes a single method
``predict_volume()`` that returns a predicted traffic volume for a given
(area, road, weather) at the current IST time.
"""

import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))

# Resolve the models/ directory relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')


class TrafficPredictor:
    """
    Loads a pre-trained Random Forest model and generates real-time
    traffic-volume predictions that the RouteOptimizer can consume.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.area_enc = None
        self.road_enc = None
        self.weather_enc = None
        self.hist_stats = None
        self.area_medians = None
        self.is_loaded = False
        self._load_artefacts()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_artefacts(self):
        """Try to load all saved artefacts.  Sets ``self.is_loaded = True``
        only if every file is present and loads successfully."""
        try:
            import joblib

            model_path = os.path.join(MODELS_DIR, 'random_forest.joblib')
            if not os.path.exists(model_path):
                print("[WARNING] ML model not found -- run `python main.py` first. "
                      "Using time-based heuristic fallback.")
                return

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
            self.area_enc = joblib.load(os.path.join(MODELS_DIR, 'area_encoder.joblib'))
            self.road_enc = joblib.load(os.path.join(MODELS_DIR, 'road_encoder.joblib'))
            self.weather_enc = joblib.load(os.path.join(MODELS_DIR, 'weather_encoder.joblib'))
            self.hist_stats = joblib.load(os.path.join(MODELS_DIR, 'historical_stats.joblib'))
            self.area_medians = joblib.load(os.path.join(MODELS_DIR, 'area_medians.joblib'))
            self.is_loaded = True
            print("[OK] ML model loaded -- predictions powered by Random Forest.")
        except Exception as exc:
            print(f"[WARNING] Failed to load ML artefacts: {exc}")
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_volume(self, area_name: str, road_name: str,
                       weather: str = 'Clear') -> float:
        """
        Predict traffic volume for the given area / road at the current
        IST time.

        The feature vector matches the 14 columns used during training
        (see ``preprocessing.py``):
            day_of_week, month, hour_sin, hour_cos,
            is_weekend, is_peak_hour,
            area_encoded, road_encoded, weather_encoded, average_speed,
            traffic_volume, vol_1_step_ago, vol_2_steps_ago, rolling_trend_3

        Lagged / rolling features are filled from historical averages.

        Returns the predicted **next-step traffic volume** (float).
        """
        if not self.is_loaded:
            return self._heuristic_volume(area_name)

        now = datetime.now(IST)
        hour = now.hour
        day_of_week = now.weekday()
        month = now.month

        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if hour in (8, 9, 10, 17, 18, 19, 20) else 0

        # --- Encode categorical features ---
        area_encoded = self._safe_encode(self.area_enc, area_name)
        road_encoded = self._safe_encode(self.road_enc, road_name)
        weather_encoded = self._safe_encode(self.weather_enc, weather)

        # --- Historical averages for lagged features ---
        avg_speed, cur_vol, lag1, lag2, roll3 = self._lookup_historical(
            area_name, road_name, hour
        )

        # Build the 14-feature row in the EXACT order used during training
        features = np.array([[
            day_of_week, month, hour_sin, hour_cos,
            is_weekend, is_peak_hour,
            area_encoded, road_encoded, weather_encoded, avg_speed,
            cur_vol, lag1, lag2, roll3
        ]])

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predicted = self.model.predict(features_scaled)[0]

        # Clamp to a sensible range (volumes can't be negative)
        return max(0.0, float(predicted))

    # ------------------------------------------------------------------
    # Volume → Congestion Multiplier
    # ------------------------------------------------------------------
    def volume_to_multiplier(self, predicted_volume: float,
                             area_name: str) -> float:
        """
        Convert a predicted traffic volume into a congestion multiplier
        (1.0 = free-flow, up to 2.5 = heavy jam).

        The multiplier is ``predicted_volume / median_volume`` for the
        area, clamped to [1.0, 2.5].
        """
        if self.area_medians is not None and area_name in self.area_medians.index:
            median = self.area_medians[area_name]
        else:
            median = 29000.0  # dataset-wide approximate median

        if median <= 0:
            median = 29000.0

        multiplier = predicted_volume / median
        return max(1.0, min(2.5, multiplier))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_encode(encoder, value: str) -> int:
        """Encode a categorical value; fall back to 0 if unseen."""
        try:
            return int(encoder.transform([value])[0])
        except (ValueError, KeyError):
            return 0

    def _lookup_historical(self, area: str, road: str, hour: int):
        """
        Return (avg_speed, cur_volume, lag1, lag2, rolling3) from the
        saved historical stats table.  Falls back to dataset-wide
        averages if the specific combination isn't found.
        """
        defaults = (35.0, 29000.0, 29000.0, 29000.0, 29000.0)
        if self.hist_stats is None:
            return defaults
        try:
            row = self.hist_stats.loc[(area, road, hour)]
            return (
                float(row['mean_speed']),
                float(row['mean_volume']),
                float(row['mean_vol_lag1']),
                float(row['mean_vol_lag2']),
                float(row['mean_rolling3']),
            )
        except KeyError:
            # Try the area-level average (any road, same hour)
            try:
                subset = self.hist_stats.loc[area]
                row = subset.xs(hour, level='hour').mean()
                return (
                    float(row['mean_speed']),
                    float(row['mean_volume']),
                    float(row['mean_vol_lag1']),
                    float(row['mean_vol_lag2']),
                    float(row['mean_rolling3']),
                )
            except KeyError:
                return defaults

    @staticmethod
    def _heuristic_volume(area_name: str) -> float:
        """
        Simple time-based heuristic used when the ML model isn't trained
        yet.  Returns an approximate traffic volume so the app still works.
        """
        now = datetime.now(IST)
        hour = now.hour
        is_peak = hour in (8, 9, 10, 17, 18, 19, 20)
        is_weekend = now.weekday() >= 5

        base = 29000.0          # dataset-wide mean
        if is_peak:
            base *= 1.4
        if is_weekend:
            base *= 0.7

        return base
