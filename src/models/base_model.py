from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for all traffic prediction models.
    Enforces a standard structure for training and predicting.
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the machine learning model.
        Must be implemented by all child classes.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Makes predictions using the trained model.
        Must be implemented by all child classes.
        """
        pass