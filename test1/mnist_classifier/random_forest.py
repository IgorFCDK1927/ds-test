import joblib
from sklearn.ensemble import RandomForestClassifier
from mnist_classifier.classifier_interface import MnistClassifierInterface

class RandomForestMnist(MnistClassifierInterface):  # new train class RF
    def __init__(self, n_estimators=100):

        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def train(self, X_train, y_train):

        print("Training Random Forest")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):

        print("Predicting Random Forest")
        return self.model.predict(X_test)
