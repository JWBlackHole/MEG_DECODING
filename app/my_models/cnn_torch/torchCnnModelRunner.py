import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import numpy as np
from loguru import logger

from app.my_models.cnn_torch.torchCnnModel import SimpleTorchCNNModel

class SimpleTorchCNNModelRunner:
    def __init__(self, megData):
        self.megData = megData



    def train(self, epochs=10, batch_size=32, learning_rate=0.001, train_test_ratio=0.8):
        # Split the data into training and testing sets
        train_size = int(train_test_ratio * len(self.megData))
        test_size = len(self.megData) - train_size
        rand_generator = torch.Generator().manual_seed(33)      # use for fix random seede
        train_dataset, test_dataset = random_split(self.megData, 
                                                    lengths=[train_size, test_size], 
                                                    generator=rand_generator)


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
                inputs = inputs.unsqueeze(1)  # Add the channel dimension [batch_size, 1, nchans, ntimes]
                # not sure if above is needed
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