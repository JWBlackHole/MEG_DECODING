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

def plot(X: Tensor, y: Tensor):
    
    x_dict    = dict()
    for i in range(len(X)):
        x_dict[f"variable{i + 1}"] = X[i].numpy()
        
    df = pd.DataFrame(x_dict)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    df = pd.DataFrame(data_scaled, columns=df.columns)
    
    # pca = PCA()
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(df)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_results[:, 0], # position on the first principal component of the observations
                pca_results[:, 1], alpha=0.7) # position on the second principal component of the observations

    # Add title and axis label
    plt.title('Scatter Plot of Observations in 2D PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    
    explained_variance = pca.explained_variance_ratio_

    # Set figsize
    plt.figure(figsize=(10, 6))

    # Create a scree plot to visualize the explained variance
    plt.plot(range(1, len(explained_variance) + 1), # x-axis
            explained_variance*100, # convert explained variance in percentage
            marker='o', # add a marker at each value
            )
    
    # Add title and axis label
    plt.title('Scree Plot of Explained Variance for Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (in %)')

    # Add label to x-axis
    plt.xticks(range(1, len(explained_variance) + 1))

    # Add grid in the background
    plt.grid(True)
    
    plt.savefig('scree_plot.png')
    
def pca(X: Tensor, y: Tensor):
    X, y = X.numpy(), y.numpy()
    
    scaler = StandardScaler()
    scaler.fit(X) 
    X_scaled = scaler.transform(X)
    
    pca = PCA(n_components = 3)
    pca.fit(X_scaled) 
    X_pca = pca.transform(X_scaled) 
    
    Xax = X_pca[:,0]
    Yax = X_pca[:,1]
    Zax = X_pca[:,2]

    cdict  = {0:'red',1:'green'}
    labl   = {0:'False',1:'True'}
    marker = {0:'*',1:'o'}
    alpha  = {0:.3, 1:.5}
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)

    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix = np.where(y==l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
                label=labl[l], marker=marker[l], alpha=alpha[l])
        # ax.scatter(Xax[ix], Yax[ix], c=cdict[l], s=40,
        #         label=labl[l], marker=marker[l], alpha=alpha[l])
        # for loop ends
        ax.set_xlabel("First Principal Component",  fontsize=14)
        ax.set_ylabel("Second Principal Component", fontsize=14)
        ax.set_zlabel("Third Principal Component",  fontsize=14)

    ax.legend()
    plt.savefig('pca_plot_3d.png')
    
    return torch.from_numpy(X_pca), torch.from_numpy(y)

def pca_d(X: Tensor, y: Tensor, dimension: int):
    X, y = X.numpy(), y.numpy()
    
    scaler = StandardScaler()
    scaler.fit(X) 
    X_scaled = scaler.transform(X)
    
    pca = PCA(n_components = dimension)
    pca.fit(X_scaled) 
    X_pca = pca.transform(X_scaled) 
    
    return torch.from_numpy(X_pca), torch.from_numpy(y)
    
def balance_label(X: Tensor, y: Tensor):
    X_true,  y_true  = Tensor(), Tensor()
    X_false, y_false = Tensor(), Tensor()
    
    # print(X, y)
        
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
    # plot(X, X_true, X_false)
    
    return X, y
    

class NNModelRunner():
    BATCH_SIZE = 2048
    def __init__(self, X, y, target_label) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        X = torch.tensor(X).to(torch.float32).reshape(-1, 208*41)   # reshape(-1, 208*81) collapse 208chans * 81 time points to
        y = torch.tensor(y.astype(bool)).to(torch.float32)
        '''
        e.g. original:
        [
            [ [chan0t0, chan0t1, ..., chan0t80],
            [chan1t0, chan1t1, ..., chan1t80],
            ...
            [chan207t0, chan207t1, ..., chan207t80] ],
            
            [...], ...
        ]
        now:
        [   [chan0t0, chan0t1, ..., chan0t80, chan1t0, chan1t1, ,..., chan1t80, ...],
            [...], ...
        ]
        this means collapse channels and timepoints in 1 window into 1D and treat them all as a feature,
        and predcit one probability for each window

        '''

        self.X = X
        self.y = y
        self.target_label =  target_label
        
    def train(self):
        print("Start Training")
        if self.target_label != TargetLabel.VOICED_PHONEME:
            logger.error("preprocessing for setting other than \"voiced\" is not implemented. program exit")
            exit(0)
            
        X, y = balance_label(self.X, self.y)
        X, y = pca_d(X, y, 10)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2,   # 20% test, 80% train
                                                            random_state=25) # shffuling, making the random split reproducible
        
        
        dataset    = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset = dataset, batch_size = NNModelRunner.BATCH_SIZE,
                                collate_fn=lambda x: tuple(x_.to(self.device) for x_ in default_collate(x)))
        dataset_size = len(dataloader.dataset)
        
        # ------ Start Training ------ #
        # Put data to target device
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_test, y_test   = X_test.to(self.device),  y_test.to(self.device)
        
        # Our "model", "loss function" and "optimizer"
        model_0   = MyNNModel().to(self.device)
        loss_fn   = torch.nn.BCELoss().to(self.device)
        optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-4)
            
        n_epochs = 10000
        for epoch in range(n_epochs + 1):
            if(epoch % 100 == 0):
                print(f"Epoch {epoch}")
                print(f"----------------------------------------")
           
            for id_batch, (X_batch, y_batch) in enumerate(dataloader):
                # set to training mode
                model_0.train()
                
                # Binary classification, just using sigmoid to predict output
                # Round: <0.5 class 1, >0.5 class2
                y_logits = model_0(X_batch).squeeze()
                # y_pred = torch.round(torch.sigmoid(y_logits))
                    
                # Calculate loss
                loss = loss_fn(y_logits, y_batch)
                
                # Reset Optimizer zero grad
                optimizer.zero_grad()

                # Loss backwards
                loss.backward()

                # Optimizer step
                optimizer.step()
                
                # ------ Testing ------ #
                if(epoch % 100 == 0):
                    with torch.inference_mode():
                        y_pred = torch.round(y_logits)
                        train_acc = accuracy_fn(y_true = y_batch, y_pred = y_pred)
                        
                        loss_item, current = loss.item(), (id_batch + 1) * NNModelRunner.BATCH_SIZE
                        print(f"Loss: {loss_item:>7f}  [{current:>5d}/{dataset_size:>5d}], Accuracy: {train_acc:>7f}%")
    
            if epoch % 100 == 0:
                test_logits = model_0(X_test).squeeze() 
                test_pred = torch.round(test_logits)

                test_loss = loss_fn(test_logits, y_test)
                test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
                print(f"Test Loss: {test_loss:>7f}, Test Accuracy: {test_acc:>7f}%\n")  
