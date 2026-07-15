from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        # Initialize the Scikit-learn model
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        print("Training Linear Regression...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)