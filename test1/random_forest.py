from sklearn.ensemble import RandomForestClassifier
from test1.classifier_interface import MnistClassifierInterface
import numpy as np

class RandomForestMnist(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test) 
