import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from loguru import logger

import pandas as pd


# cutom import
from app.my_models.nn.nnModel import MyNNModel
from app.common.commonSetting import TargetLabel

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def pca_d(X: Tensor , dimension: int) -> Tensor:
    X = X.numpy().reshape(X.shape[0], -1)
    print(X.shape)
    
    pca = PCA(n_components = dimension)
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)
        
    return torch.from_numpy(X_pca)
    
def balance_label(X: Tensor, y: Tensor):
    print("Balancing Label...")
    X_true,  y_true  = Tensor(), Tensor()
    X_false, y_false = Tensor(), Tensor()
    
        
    len_train_data = len(X)
    for i in range(len_train_data):
        if(y[i] == True):
            X_true = torch.cat((X_true, torch.unsqueeze(X[i], 0)), 0) if len(X_true) else torch.unsqueeze(X[i], 0)
            y_true = torch.cat((y_true, torch.unsqueeze(y[i], 0)), 0) if len(y_true) else torch.unsqueeze(y[i], 0)
        else:
            X_false = torch.cat((X_false, torch.unsqueeze(X[i], 0)), 0) if len(X_false) else torch.unsqueeze(X[i], 0)
            y_false = torch.cat((y_false, torch.unsqueeze(y[i], 0)), 0) if len(y_false) else torch.unsqueeze(y[i], 0)
    
    min_len = min(len(X_true), len(X_false))
    X_true,  y_true  = X_true[:min_len],  y_true[:min_len]
    X_false, y_false = X_false[:min_len], y_false[:min_len]
    
    print("X length", len(X_true), len(X_false))
    print("Y length", len(y_true), len(y_false))
    
    X, y = torch.cat((X_true, X_false), 0), torch.cat((y_true, y_false), 0)

    return X, y

class NNModelRunner():
    BATCH_SIZE = 2048
    def __init__(self, megData, target_label) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.megData = megData
        self.target_label =  target_label
        
        
    def train(self, epochs, batch_size, lr):
        print("Start Training...")
        
        if self.target_label != TargetLabel.VOICED_PHONEME:
            logger.error("preprocessing for setting other than \"voiced\" is not implemented. program exit")
            exit(0)
        
        # Our "model", "loss function" and "optimizer"
        model_0   = MyNNModel().to(self.device)
        loss_fn   = torch.nn.BCELoss().to(self.device)
        optimizer = torch.optim.SGD(params = model_0.parameters(), lr = lr, momentum=0.9, weight_decay = 1e-4)
        
        for i, task in enumerate(self.megData):
            print(f"In task {i}")
            print("-------------------------")
            
            X, y = task
            X, y = balance_label(X, y)
            # X    = pca_d(X, 10)
            X, y = X.to(self.device), y.to(self.device)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2,   # 20% test, 80% train
                                                            random_state=25) # shffuling, making the random split reproducible
            
            dataset    = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset = dataset, batch_size = batch_size)
            dataset_size = len(dataloader.dataset)
            
            for epoch in range(epochs):
                if(epoch % 100 == 0):
                    print(f"Epoch {epoch}")
                    print("-----------------------")
                
                for id_batch, (X_batch, y_batch) in enumerate(dataloader):
                    # set to training mode
                    model_0.train()
                    
                    # Binary classification, just using sigmoid to predict output
                    # Round: <0.5 class 1, >0.5 class2
                    y_logits = model_0(X_batch).squeeze()
                        
                    # Calculate loss
                    loss = loss_fn(y_logits, y_batch)
                    
                    # Reset Optimizer zero grad
                    optimizer.zero_grad()
                    # Loss backwards
                    loss.backward()
                    # Optimizer step
                    optimizer.step()
                    
                    if(epoch % 100 == 0):
                        y_pred = torch.round(y_logits)
                        train_acc = accuracy_fn(y_true = y_batch, y_pred = y_pred)
                            
                        loss_item, current = loss.item(), (id_batch + 1) * batch_size
                        print(f"Loss: {loss_item:>7f}  [{current:>5d}/{dataset_size:>5d}], Accuracy: {train_acc:>7f}%")
                        
                if epoch % 100 == 0:
                    test_logits = model_0(X_test).squeeze() 
                    test_pred = torch.round(test_logits)

                    test_loss = loss_fn(test_logits, y_test)
                    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
                    print(f"Test Loss: {test_loss:>7f}, Test Accuracy: {test_acc:>7f}%\n")  
        return
