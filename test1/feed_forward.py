from sklearn.neural_network import MLPClassifier
from test1.classifier_interface import MnistClassifierInterface
import numpy as np

class FeedForwardMnist(MnistClassifierInterface):
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=100):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
