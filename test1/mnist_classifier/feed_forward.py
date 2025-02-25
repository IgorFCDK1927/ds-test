import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mnist_classifier.classifier_interface import MnistClassifierInterface

class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # The first  layer 
        self.fc2 = nn.Linear(128, 64)  # Second connected layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (digits 0-9)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Layer + ReLU
        x = self.relu(self.fc2(x))  # Second FC layer + ReLU
        x = self.fc3(x)  # Output layer (logits, without Softmax)
        return x

class FeedForwardClassifier(MnistClassifierInterface):
    def __init__(self):
        print("Initializing Feed-Forward Neural Network")
        self.model = FFNN()  # Creating FFNN model
        self.criterion = nn.CrossEntropyLoss()  # Loss function (CrossEntropyLoss classification)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # optimizer by Adam
        print("FFNN initialized")

    def train(self, X_train, y_train, epochs=10, batch_size=64):
        # Training
        self.model.train()  # Set model

        # Converting y_train to torch
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        # Converting X_train to torch
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)

        # Checking training size
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Creating PyTorch Dataset and DataLoader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()  # Reset gradients
                outputs = self.model(batch_X)  # Forward pass
                loss = self.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  #
                self.optimizer.step()  # Update weights
                total_loss += loss.item()
            print(f"Epoch {epoch+1} of {epochs}, Loss: {total_loss:.4f}")  # Printing epochs

    def predict(self, X_test):

        self.model.eval()  # Set model

        # Converting X_test to torch
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        # Running inference
        with torch.no_grad():
            outputs = self.model(X_test) 
            probabilities = torch.softmax(outputs, dim=1)  # Softmax
            predictions = torch.argmax(probabilities, dim=1)  # Get predicted class

        return predictions.numpy()  # Converting to NumPy
