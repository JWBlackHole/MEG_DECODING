import torch 
from torch import nn

import pandas as pd

import mne_bids

from MEGSignal import MEGSignal
from MyModel import MyModel

# def accuracy_fn(y_true, y_pred):
#     correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
#     acc = (correct / len(y_pred)) * 100 
#     return acc

if __name__ == "__main__":
    
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
    
    
    X = torch.tensor(X)
    y = torch.tensor(y)
    print(type(X), type(y))
    
    # print(len(X), len(y))
    # print(len(X[0]), len(X[1]))
    # print(len(X[0][0]), len(X[1][0]))
    # print(X[0][0])
    # print(X, y)
    
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test,  y_test  = X[train_split:], y[train_split:]

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_0 = MyModel(2).to(device)
    
    loss_function = torch.nn.BCELoss
    optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)
    
    # for epoch in range(100):
        
        