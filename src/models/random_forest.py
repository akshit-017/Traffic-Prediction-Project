import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        # The core engine
        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # 🔥 THE 80% CHEAT CODE: Log Transformation 🔥
        # This forces the AI to predict the logarithm of the traffic, ignoring massive outliers
        self.model = TransformedTargetRegressor(
            regressor=rf,
            func=np.log1p,          # Log transformation before training
            inverse_func=np.expm1   # Reverses log back to normal numbers for the final output
        )

    def train(self, X_train, y_train):
        print("Training Log-Transformed Random Forest Engine...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)