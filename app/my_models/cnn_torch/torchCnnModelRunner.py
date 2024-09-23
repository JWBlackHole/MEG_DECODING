import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from loguru import logger
import random


from app.my_models.cnn_torch.torchCnnModel import SimpleTorchCNNModel
from app.signal.newTorchMegLoader import MegDataIterator
from app.utils import my_utils as util
from app.my_models.nn.nnModelRunner import accuracy_fn, balance_label




class SimpleTorchCNNModelRunner:
    def __init__(self, megData, nchans, ntimes, p_drop_true=0.572):
        """
        Parameters:
        -----------

        p_drop_true : float
            The drop probability for `True` labels when balancing the classes. set to 0 to not apply balancing

        
        
        """
        logger.info("SimpleTorchCNNModelRunner is inited")
        self.check_gpu()
        self.megData = megData      # instance of TorchMegLoader
        self.nchans: int = nchans
        self.ntimes: int = ntimes
        self.p_drop_true: float = p_drop_true     # Drop probability for `True` labels


    
    def check_gpu(self):
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available")
        else:
            logger.error("no GPU, program exit")
            raise Exception
        
    def process_one_task(self, i, task):
        pass


    def train(self, epochs=10, batch_size=512, learning_rate=0.001, train_test_ratio=0.8, to_save_csv=True,
               graph_interval: int=1, print_interval: int=100):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            assert device.type == "cuda"
        except Exception as e:
            logger.error(f"device is: {device}")
            logger.error(f"device.type: {device.type},  is not cuda! program exit!")
            raise AssertionError


        #  --------- setup model spec   ---------#

        model = SimpleTorchCNNModel(self.nchans, self.ntimes).to(device)
        loss_fn = nn.BCELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        total_epoch = 0
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        final_test_pred = None
        final_y_test = None
        
        for i, task in enumerate(self.megData): # loop each task in megData -------

            self.process_one_task(i, task)
            print(f"In task {i}")
            print("-------------------------")
            X, y = task
            X, y = balance_label(X, y)
            X, y = X.to(device), y.to(device)

            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=(1-train_test_ratio),   # 20% test, 80% train
                                                            random_state=25) # shffuling, making the random split reproducible

   
            train_dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            dataset_size = len(dataloader.dataset)
            
            
            for epoch in range(epochs):   # loop each epoch for each task in megData ------
                loss_item = train_acc = None

                model.train()   # set model to training mode
                for batch_num, (X_batch, y_batch) in enumerate(dataloader): # loop thru each `batch_size` events ---
                    

                    # debugging
                    if (i==0) and (epoch==0):
                        logger.debug(f"shape of X_batch: {X_batch.shape}")
                        logger.debug(f"shape of y_batch: {y_batch.shape}")


                    #   ----   training    ----  #                     
                    outputs = model(X_batch)    # forward pass
                    loss = loss_fn(outputs, y_batch)

                    optimizer.zero_grad()   # empty gradient
                    loss.backward()
                    optimizer.step()

                
                
                # ---    calculate loss and accuracy up to the point  --- #
                    
                
                if(epoch % graph_interval== 0):
                    model.eval()   # set model to evaluation mode
                    y_pred = torch.round(outputs)
                    train_acc = accuracy_fn(y_true = y_batch, y_pred = y_pred)
                            
                    loss_item, current = loss.item(), (batch_num + 1) * batch_size

                    if(epoch % print_interval== 0):
                        print(f"Loss: {loss_item:>7f}  [{current:>5d}/{dataset_size:>5d}], Accuracy: {train_acc:>7f}%")


                    test_logits = model(X_test)
                    test_pred = torch.round(test_logits)

                    test_loss = loss_fn(test_logits, y_test)
                    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
                    if(epoch % print_interval== 0):
                        print(f"Test Loss: {test_loss:>7f}, Test Accuracy: {test_acc:>7f}%\n")

                    # record loss and accuracy for plot graph
                    train_losses.append(loss_item)
                    train_accuracies.append(train_acc)
                    test_losses.append(test_loss.item())        # .item will convert the var to float (CPU-bound)
                    test_accuracies.append(test_acc)
                
                total_epoch += 1



                        