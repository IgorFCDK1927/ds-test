import joblib
import torch
from mnist_classifier import MnistClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣ Загружаем данные MNIST
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)

# 2️⃣ Разбиваем на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Масштабируем данные (для RF и NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Обучаем и сохраняем модели
models = ["rf", "nn", "cnn"]

for model_type in models:
    print(f"Training model: {model_type.upper()}")

    clf = MnistClassifier(algorithm=model_type)

    if model_type == "cnn":
        # Подготовка данных для CNN
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).view(-1, 1, 28, 28)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

        clf.train(X_train_tensor, y_train_tensor, epochs=5)

        # Сохраняем обученную модель
        torch.save(clf.model.state_dict(), f"{model_type}_model.pth")
    else:
        clf.train(X_train_scaled, y_train)

        # Сохраняем обученную модель
        joblib.dump(clf.model, f"{model_type}_model.pkl")

    print(f"Model {model_type.upper()} saved!\n")

# 5️⃣ Сохраняем scaler для нормализации (для RF и NN)
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved!")
