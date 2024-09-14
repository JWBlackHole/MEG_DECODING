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
        logger.info("SimpleTorchCNNModelRunner is inited")
        self.megData = megData
        self.check_gpu()

    
    def check_gpu(self):
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available")
        else:
            logger.error("no GPU, program exit")
            raise Exception



    def train(self, epochs=10, batch_size=32, learning_rate=0.001, train_test_ratio=0.1):
        # Split the data into training and testing sets

        # for testing
        ratio = (0.2, 0.1, 0.7)
        
        train_size = int(ratio[0] * len(self.megData))
        test_size = int(ratio[1] * len(self.megData))
        not_used = len(self.megData) - train_size - test_size  # remaining data for test

        #train_size = int(train_test_ratio * len(self.megData))
        #test_size = len(self.megData) - train_size
        rand_generator = torch.Generator().manual_seed(33)      # use for fix random seede
        train_dataset, test_dataset, not_used_dataset = random_split(self.megData, 
                                                    lengths=[train_size, test_size, not_used], 
                                                    generator=rand_generator)


        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            assert device.type == "cuda"
        except Exception as e:
            logger.error(f"device is: {device}")
            logger.error(f"device.type: {device.type},  is not cuda! program exit!")
            raise AssertionError

        model = SimpleTorchCNNModel().to(device)
        criterion = nn.BCELoss()    
        # notes:
        # using BCELoss expect X, y both in type float32
        
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)  # Add the channel dimension 
                # expexted dimension of inputs: [batch_size, 1, nchans, ntimes]
                # not sure if above is needed
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

        # Evaluation
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        y_pred = (np.array(y_pred) >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

# Example usage
# Assuming you have your data in variables X and y
# runner = SimpleTorchCNNModelRunner(X, y)
# runner.train(epochs=10, batch_size=32, learning_rate=0.001)