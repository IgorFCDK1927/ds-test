from .classifier_interface import MnistClassifierInterface
from .cnn import CNNClassifier
from .feed_forward import FeedForwardClassifier
from .random_forest import RandomForestMnist 


class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm="cnn"):

        self.algorithm = algorithm.lower()

        if self.algorithm == "cnn":
            self.model = CNNClassifier()
        elif self.algorithm == "nn":
            self.model = FeedForwardClassifier()
        elif self.algorithm == "rf":
            self.model = RandomForestMnist()
        else:
            raise ValueError("Invalid algorithm")

    def train(self, X_train, y_train, **kwargs):
        # Training of selected model
        return self.model.train(X_train, y_train, **kwargs)

    def predict(self, X_test):

        return self.model.predict(X_test)
