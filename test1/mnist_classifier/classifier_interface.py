from abc import ABC, abstractmethod

# Abstract class for Mnist prediction
class MnistClassifierInterface(ABC):

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):

        pass

    @abstractmethod
    def predict(self, X_test):

        pass
