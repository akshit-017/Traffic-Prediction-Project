from sklearn.svm import SVR
from .base_model import BaseModel

class SVRModel(BaseModel):
    def __init__(self):
        # Hyper-tuned parameters for Scikit-Learn SVR
        self.model = SVR(
            kernel='rbf', 
            C=10000,          # Massive penalty for errors (forces higher accuracy)
            gamma='auto',     # Automatically adjusts to the new feature count
            epsilon=0.01      # Incredibly tight margin of error allowed
        )

    def train(self, X_train, y_train):
        print("Training Hyper-Tuned SVR...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)