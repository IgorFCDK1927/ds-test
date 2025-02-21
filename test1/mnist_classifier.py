import sys
import os
sys.path.append(os.getcwd())  # Добавляем текущую директорию
sys.path.append(os.path.abspath(".."))  # Добавляем родительскую папку (ds-test)
from test1.random_forest import RandomForestMnist
from test1.feed_forward import FeedForwardMnist
from test1.cnn import CNNMnist

class MnistClassifier:
    def __init__(self, algorithm="rf"):
        """
        Инициализация модели.

        :param algorithm: "rf" для RandomForest, "nn" для Feed-Forward, "cnn" для Convolutional NN
        """
        if algorithm == "rf":
            self.model = RandomForestMnist()
        elif algorithm == "nn":
            self.model = FeedForwardMnist()
        elif algorithm == "cnn":
            self.model = CNNMnist()
        else:
            raise ValueError("Invalid algorithm. Choose from 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train, **kwargs):
        """Тренировка модели."""
        if isinstance(self.model, CNNMnist):
            self.model.train(X_train, y_train, **kwargs)  # CNN требует epochs
        else:
            self.model.train(X_train, y_train)

    def predict(self, X_test):
        """Предсказание меток."""
        return self.model.predict(X_test)
 
