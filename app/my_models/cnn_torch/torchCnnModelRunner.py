import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from loguru import logger

from app.my_models.cnn_torch.torchCnnModel import SimpleTorchCNNModel

class SimpleTorchCNNModelRunner:
    def __init__(self, X, y):
        n, nchans, ntimes = X.shape
        logger.info(f"X.shape: {n}, {nchans}, {ntimes}")
        X = torch.tensor(X).to(torch.float32).reshape(-1, 1, nchans, ntimes)  # Ensure 4D shape
        y = torch.tensor(y.astype(bool)).to(torch.float32).reshape(-1, 1)

        self.X = X
        self.y = y

    def train(self, epochs=10, batch_size=32, learning_rate=0.001):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=33)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, loss function, and optimizer
        model = SimpleTorchCNNModel()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

        # Evaluation
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                y_pred.extend(outputs.numpy())
                y_true.extend(labels.numpy())

        y_pred = (np.array(y_pred) >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

# Example usage
# Assuming you have your data in variables X and y
# runner = SimpleTorchCNNModelRunner(X, y)
# runner.train(epochs=10, batch_size=32, learning_rate=0.001)