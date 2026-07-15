import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluator:
    def __init__(self, models_dict):
        # This takes in a dictionary of our 3 models
        self.models = models_dict
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("\n--- Starting Model Training and Evaluation ---")
        
        for name, model in self.models.items():
            # 1. Train the model using the base_model structure
            model.train(X_train, y_train)
            
            # 2. Make future predictions on the test set
            predictions = model.predict(X_test)
            
            # 3. Calculate performance metrics
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            # 4. Store and print the results
            self.results[name] = {'MAE': mae, 'RMSE': rmse}
            print(f"✅ {name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")
            
        return self.results