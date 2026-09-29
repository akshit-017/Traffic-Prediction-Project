import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TrafficPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.area_encoder = LabelEncoder()
        self.road_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()

    def load_and_clean(self):
        print("Loading and cleaning with Peak-Hour triggers...")
        df = pd.read_csv(self.file_path)
        
        df.columns = df.columns.str.strip().str.lower()
        df.dropna(inplace=True)
        
        # Deep Time Features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['hour'] = df['date'].dt.hour
        
        # Cyclical Time Encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        
        # 🔥 THE NEW 80% FEATURES: Binary Gridlock Flags 🔥
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = df['hour'].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
        
        # Encoding Categories (separate encoder per column)
        if 'area name' in df.columns:
            df['area_encoded'] = self.area_encoder.fit_transform(df['area name'])
        if 'road/intersection name' in df.columns:
            df['road_encoded'] = self.road_encoder.fit_transform(df['road/intersection name'])
        if 'weather conditions' in df.columns:
            df['weather_encoded'] = self.weather_encoder.fit_transform(df['weather conditions'])
            
        # Deep Lagging (Memory)
        df['vol_1_step_ago'] = df['traffic volume'].shift(1)
        df['vol_2_steps_ago'] = df['traffic volume'].shift(2)
        df['rolling_trend_3'] = df['traffic volume'].rolling(window=3).mean()
        
        # Proactive Target
        df['target_volume'] = df['traffic volume'].shift(-1)
        
        df.dropna(inplace=True)
        return df

    def get_features_and_target(self, df):
        print("Extracting features (including Log-Ready data)...")
        feature_columns = [
            'day_of_week', 'month', 'hour_sin', 'hour_cos', 
            'is_weekend', 'is_peak_hour', # <-- Added Flags
            'area_encoded', 'road_encoded', 'weather_encoded', 'average speed',
            'traffic volume', 'vol_1_step_ago', 'vol_2_steps_ago', 'rolling_trend_3' 
        ]
        
        X = df[feature_columns]
        y = df['target_volume']
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y