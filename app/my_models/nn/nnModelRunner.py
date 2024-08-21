import torch 
from torch import nn
from sklearn.model_selection import train_test_split

from loguru import logger

# cutom import
from app.my_models.nn.nnModel import MyNNModel
from app.common.commonSetting import TargetLabel

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

class NNModelRunner():
    def __init__(self, X, y, target_label) -> None:

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        X = torch.tensor(X).to(torch.float32).reshape(-1, 208*81)   # reshape(-1, 208*81) collapse 208chans * 81 time points to 
    
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
        y = torch.tensor(y.astype(bool)).to(torch.float32)

        self.X = X
        self.y = y
        self.target_label =  target_label
    
    def train(self, total_epoch: int, log_interval: int|None = None):
        """
        total_epoch
            - how many epochs to train in total
        log_interval
            - how often to print log message

        """

        if log_interval is None:
            log_interval = total_epoch / 10

        if self.target_label == TargetLabel.VOICED_PHONEME:
        
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                                test_size=0.2,   # 20% test, 80% train
                                                                random_state=25) # shffuling, making the random split reproducible
            
            # ------ Start Training ------ #

            # Put data to target device
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            X_test, y_test   = X_test.to(self.device),  y_test.to(self.device)
            
            # Our "model", "loss function" and "optimizer"
            model_0 = MyNNModel(num_classes=2).to(self.device)
            # loss_fn = torch.nn.BCELoss()
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer     = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)
            
            n_epochs = total_epoch

            for epoch in range(n_epochs + 1):
                # set to training mode
                model_0.train()
                
                y_logits = model_0(X_train).squeeze()
                # Binary classification, just using sigmoid to predict output
                # Round: <0.5 class 1, >0.5 class2
                y_pred = torch.round(torch.sigmoid(y_logits))
                
                # Calculate loss and accuracy
                loss = loss_fn(y_logits, y_train) 
                acc = accuracy_fn(y_true = y_train, y_pred = y_pred) 
                
                # 3. Reset Optimizer zero grad
                optimizer.zero_grad()

                # 4. Loss backwards
                loss.backward()

                # 5. Optimizer step
                optimizer.step()
                
                ### Testing
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

                

                if epoch % log_interval == 0:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        else:
            logger.error("preprocessing for setting other than \"voiced\" is not implemented. program exit")
            exit(0)
            