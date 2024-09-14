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
        self.nchans = nchans
        self.ntimes = ntimes
        logger.info(f"X.shape: {n}, {nchans}, {ntimes}")
        logger.debug(f"before reshape, X shape: {X.shape}, y shape: {y.shape}")
        # before reshape, 
        # X.shape = (#event, #channel, # timepoint)
        # y.shape = (#event, )
        
        X = torch.tensor(X).to(torch.float32).reshape(-1, 1, nchans, ntimes)  # Ensure 4D shape       
        y = torch.tensor(y.astype(bool)).to(torch.float32).reshape(-1, 1)
        logger.debug(f"after reshape, X shape: {X.shape}, y shape: {y.shape}")
        # after reshape and to tensor
        # X.shape=torch.Size([#event, 1, #channel, #timepoint]), y shape=torch.Size([#event, 1])

        """
        notes: what should be shape of tensor? -> (batch_size, num_channels, height, width)
        batch_size = Number of data samples -> should = #event
        num_channels-> 1 (not the channel in MEG) why 1?->with reference EEGNet and typical usage
        height-> for time serise, often correspond to time domain
        width ->for time serise, often correspond to spatial domain (MEG channel)

        supposingly, flipping time and MEG channel should work too, (but look like cnn will focus more on height (not sure))

        """

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
        model = SimpleTorchCNNModel(self.nchans, self.ntimes)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()   # empty gradient
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