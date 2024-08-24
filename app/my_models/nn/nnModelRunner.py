import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from loguru import logger

# cutom import
from app.my_models.nn.nnModel import MyNNModel
from app.common.commonSetting import TargetLabel

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

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
    
    return torch.cat((X_true, X_false), 0), torch.cat((y_true, y_false), 0)
    

class NNModelRunner():
    BATCH_SIZE = 64
    
    def __init__(self, X, y, target_label) -> None:

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(X)
        # print(y)
        
        X = torch.tensor(X).to(torch.float32).reshape(-1, 208*81)   # reshape(-1, 208*81) collapse 208chans * 81 time points to
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
        if self.target_label != TargetLabel.VOICED_PHONEME:
            logger.error("preprocessing for setting other than \"voiced\" is not implemented. program exit")
            exit(0)
            
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            test_size=0.2,   # 20% test, 80% train
                                                            random_state=25) # shffuling, making the random split reproducible
        
        X_train, y_train = balance_label(X_train, y_train)
        # print(X_train)
        # print(y_train)
        
        dataset    = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset = dataset, batch_size = NNModelRunner.BATCH_SIZE)
        batch_number = len(dataloader)
        
        # ------ Start Training ------ #
        # Put data to target device
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_test, y_test   = X_test.to(self.device),  y_test.to(self.device)
        
        # Our "model", "loss function" and "optimizer"
        model_0   = MyNNModel(2).to(self.device)
        loss_fn   = torch.nn.BCEWithLogitsLoss()
        # loss_fn   = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01, momentum=0.9)
            
        n_epochs = 1000
        for epoch in range(n_epochs + 1):
            train_loss = 0
            train_acc  = 0
            for id_batch, (X_batch, y_batch) in enumerate(dataloader):
                # set to training mode
                model_0.train()
                
                # Binary classification, just using sigmoid to predict output
                # Round: <0.5 class 1, >0.5 class2
                y_logits = model_0(X_batch).squeeze()
                y_pred = torch.round(y_logits)
                    
                # Calculate loss and accuracy
                loss = loss_fn(y_logits, y_batch) 
                
                train_loss += loss
                train_acc  += accuracy_fn(y_true = y_batch, y_pred = y_pred) 

                # Loss backwards
                loss.backward()

                # Optimizer step
                optimizer.step()
                
                # Reset Optimizer zero grad
                optimizer.zero_grad()
                
            # ------ Testing ------ #
            train_acc  /= batch_number
            train_loss /= batch_number
            model_0.eval()
            with torch.inference_mode():
                # 1. Forward pass
                test_logits = model_0(X_test).squeeze() 
                test_pred = torch.round(torch.sigmoid(test_logits))
                # 2. Caculate loss/accuracy
                test_loss = loss_fn(test_logits,
                                    y_test)
                test_acc = accuracy_fn(y_true=y_test,
                                    y_pred=test_pred)
            if epoch % 1 == 0:
                print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")  
