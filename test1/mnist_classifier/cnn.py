import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mnist_classifier.classifier_interface import MnistClassifierInterface

class CNN(nn.Module):
    # MNIST CNN, choosing size, image and activation
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Without softmax
        return x

class CNNClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()  # Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=5, batch_size=64):
        self.model.train()

        # Transform y_train
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        # Transform X_train
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # calculate and print epochs
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} of {epochs}, Loss: {total_loss:.4f}")

    def predict(self, X_test):
        self.model.eval()

        # Transform X_test tensor
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_test)
            probabilities = torch.softmax(outputs, dim=1)  # Softmax
            predictions = torch.argmax(probabilities, dim=1)
        return predictions.numpy()
