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
from app.my_models.nn_pca.nnPCAModel import NNPCAModel
from app.common.commonSetting import TargetLabel
import app.utils.my_utils as util

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def pca_d(X: Tensor , dimension: int) -> Tensor:
    print("PCA ing...")
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

class NNPCAModelRunner():
    def __init__(self, megData, target_label, nchans:int, ntimes: int) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.megData = megData
        self.target_label =  target_label
        self.nchans = nchans
        self.ntimes = ntimes
        
        
    def train(self, epochs, batch_size, lr, graph_interval: int=1, print_interval: int=100)->dict:
        """
        parameters
        -----------
        graph_interval: how often to plot a point on loss and accuracy vs epoch graph 
                        (and hence need to calculate loss and accuracy up to that point)

        print_interval: how often to print the loss and accuracy on console log
                        need to be a multiple of graph_interval
                         (e.g. graph_interval=10, print_interval=100)
        return
        ------------
        metircs: dict of metrics of training result
        
        """
        print("Start Training...")
        
        if self.target_label != TargetLabel.VOICED_PHONEME:
            logger.error("preprocessing for setting other than \"voiced\" is not implemented. program exit")
            exit(0)
        
        # Our "model", "loss function" and "optimizer"
        model_0   = NNPCAModel(self.nchans, self.ntimes).to(self.device)
        # loss_fn   = torch.nn.BCELoss().to(self.device)
        loss_fn   = torch.nn.MSELoss().to(self.device)
        # optimizer = torch.optim.SGD(params = model_0.parameters(), lr = lr, momentum=0.9, weight_decay = 1e-4)
        # optimizer = torch.optim.SGD(params = model_0.parameters(), lr = lr, momentum=0.9)
        optimizer = torch.optim.Adagrad(params = model_0.parameters(), lr = lr)
        
        total_epoch = 0
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        final_test_pred = None
        final_y_test = None

        for i, task in enumerate(self.megData):
            print(f"In task {i}")
            print("-------------------------")
            
            X, y = task
            X = pca_d(X, 100).to(self.device)
            X, y = X.to(self.device), y.to(self.device)
            X, y = balance_label(X, y)
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2,   # 20% test, 80% train
                                                            random_state=25) # shffulinug, making the random split reproducible

            dataset    = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset = dataset, batch_size = batch_size)
            dataset_size = len(dataloader.dataset)
            
            
            for epoch in range(epochs + 1):
                if(epoch % print_interval == 0):
                    print(f"Epoch {epoch}")
                    print("-----------------------")
                
                loss_item = train_acc = None
                
                tmp_train_losses: list = [] # added
                tmp_train_accys : list = [] # added
                for id_batch, (X_batch, y_batch) in enumerate(dataloader):
                    # set to training mode
                    model_0.train() 

                    y_logits = model_0(X_batch).squeeze()
                    # y_logits = torch.round(y_logits)
                        
                    # Calculate loss
                    loss = loss_fn(y_logits, y_batch)
                    
                    # Reset Optimizer zero grad
                    optimizer.zero_grad()
                    # Loss backwards
                    loss.backward()
                    # Optimizer step
                    optimizer.step()
                    
                    # --------- Accuracy --------- #
                    y_pred = torch.round(y_logits)
                    train_acc = accuracy_fn(y_true = y_batch, y_pred = y_pred)
                            
                    loss_item, current = loss.item(), (id_batch + 1) * batch_size
                        
                    tmp_train_losses.append(loss_item) # added
                    tmp_train_accys.append(train_acc)  # added
                    # --------- Accuracy --------- #
                    
                    if(epoch % print_interval == 0):
                        print(f"Loss: {loss_item:>7f}  [{current:>5d}/{dataset_size:>5d}], Accuracy: {train_acc:>7f}%")
                        
                if epoch % graph_interval == 0:
                    test_logits = model_0(X_test).squeeze() 
                    test_pred = torch.round(test_logits)

                    test_loss = loss_fn(test_logits, y_test)
                    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
                if(epoch % print_interval == 0):
                    print(f"Test Loss: {test_loss:>7f}, Test Accuracy: {test_acc:>7f}%\n")
                
                # record loss and accuracy for plot graph
                # train_losses.append(loss_item)
                # train_accuracies.append(train_acc)
                train_losses.append(sum(tmp_train_losses) / len(tmp_train_losses))   # added
                train_accuracies.append(sum(tmp_train_accys) / len(tmp_train_accys)) # added
                test_losses.append(test_loss.item())        # .item will convert the var to float (CPU-bound)
                test_accuracies.append(test_acc)
                
                # save the pred and y in last epoch of last task for later use
                if (epoch == (epochs -1)) and ( i== (len(self.megData)-1) ):
                    final_test_pred = test_pred.cpu().detach()  # .detach() explicitly free gpu memory
                    final_y_test = y_test.cpu().detach()

                total_epoch += 1

        
        metrics = self.save_result(final_test_pred, final_y_test, train_losses, train_accuracies, test_losses, test_accuracies, total_epoch)
        return metrics
    
    def save_result(self, test_pred, y_test, train_losses, train_accuracies, test_losses, test_accuracies, total_epoch)->dict:

        graph_save_path = util.get_unique_file_name("NN_PCA_loss_accuracy_graph.png", "./results/nn_pca/graph")

        util.plot_loss_accu_across_epoch(train_losses, train_accuracies, test_losses, test_accuracies, total_epoch, graph_save_path)
        # Save the result metrics
        prediction_df = pd.DataFrame({
            'prediction': test_pred.cpu().numpy().flatten(),
            'ground_truth': y_test.cpu().numpy().flatten()
        })

        # Calculate metrics
        prediction_df = util.add_comparison_column(prediction_df)
        dstr = "Description of the training configuration"
        metrics = util.get_eval_metrics(prediction_df, 
                            file_name="voiced_metrics_cnn", save_path="./results", 
                            description_str=dstr)
        
        return metrics