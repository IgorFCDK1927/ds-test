from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Абстрактный класс для всех моделей MNIST.
    Все модели (CNN, FFNN, Random Forest) должны реализовать `train()` и `predict()`.
    """

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Метод обучения модели.
        :param X_train: Данные для обучения
        :param y_train: Метки классов
        :param kwargs: Дополнительные параметры (например, количество эпох)
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Метод предсказания классов.
        :param X_test: Данные для предсказания
        :return: numpy-массив с предсказанными значениями
        """
        pass
