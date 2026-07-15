from sklearn.ensemble import HistGradientBoostingRegressor
from .base_model import BaseModel

class HGBModel(BaseModel):
    def __init__(self):
        # The Scikit-Learn Native Booster
        self.model = HistGradientBoostingRegressor(
            max_iter=800,             # 800 boosting rounds to hammer down the error
            learning_rate=0.04,       # Slow learning rate for precision
            max_depth=15,             # Deep enough to catch Bangalore's chaotic patterns
            l2_regularization=0.5,    # Prevents it from overfitting the noise
            min_samples_leaf=4,
            random_state=42
        )

    def train(self, X_train, y_train):
        print("Training HistGradientBoosting (The Scikit-Learn Heavyweight)...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)