import torch 
from sklearn.model_selection import train_test_split

import pandas as pd

import mne_bids

from MEGSignal import MEGSignal
from MyModel import MyModel


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def get_data():
    ph_info:pd.DataFrame = pd.read_csv("./phoneme_info.csv")

    # Specify a path to a epoch
    bids_path = mne_bids.BIDSPath(
        subject = '01',
        session = '0',
        task = '0',
        datatype = "meg",
        root = './data'
    )
    
    meg_signal = MEGSignal(bids_path)
    meg_signal.load_meta(info = ph_info)
    meg_signal.load_epochs()
    
    phonemes = meg_signal.epochs["not is_word"]
    X = phonemes.get_data()
    y = phonemes.metadata["voiced"].values
    
    return X, y

if __name__ == "__main__":
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------ Data Getting and Preprocessing ------ #
    X, y = get_data()
    # print(X.shape, y.shape)
    
    # print(X, y)
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,   # 20% test, 80% train
                                                        random_state=25) # shffuling, making the random split reproducible
    
    # ------ Start Training ------ #

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test   = X_test.to(device),  y_test.to(device)
    
    # Our "model", "loss function" and "optimizer"
    model_0 = MyModel(2).to(device)
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer     = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)
    
    n_epochs = 100
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

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        