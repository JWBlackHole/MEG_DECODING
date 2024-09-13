import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger

class SimpleTorchCNNModel(nn.Module):
    def __init__(self):
        super(SimpleTorchCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 204 * 37, 128)  # last number = output feature
        self.fc2 = nn.Linear(128, 1)  # Binary classification

    def forward(self, x):
        """
        notes: forward is automatically called on outputs = model(inputs)
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # print(f"Shape after conv2: {x.shape}")
        # shape should be [batch_size, 64, 204, 37]
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

