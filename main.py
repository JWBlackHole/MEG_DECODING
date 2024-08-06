import torch 
from torch import nn

import pandas as pd

import mne_bids
import json
from pathlib import Path

from MEGSignal import MEGSignal
from MyModel import MyModel

# def accuracy_fn(y_true, y_pred):
#     correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
#     acc = (correct / len(y_pred)) * 100 
#     return acc

if __name__ == "__main__":
    # Load the JSON configuration file
    config_path = Path('train_config.json')
    try:
        with config_path.open('r') as file:
            config = json.load(file)
        training_config = config['training']
        subject = str(training_config['until_subject'])
        session = str(training_config['until_session'])
        task = str(training_config['until_task'])
        raw_data_path = training_config['raw_data_path']
        low_pass_filter = training_config['preprocess_low_pass']
        high_pass_filter = training_config['preprocess_high_pass']
        training_flow = training_config['flow']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Use hardcoded values if the config file is not found or is invalid
        subject = '01'
        session = '0'
        task = '0'
        raw_data_path = './data'
        low_pass_filter = high_pass_filter = training_flow = None
    
    
    ph_info:pd.DataFrame = pd.read_csv("./phoneme_info.csv")

    # Specify a path to a epoch
    bids_path = mne_bids.BIDSPath(
        subject = subject,
        session = session,
        task = task,
        datatype = "meg",
        root = raw_data_path
    )

    if(training_flow == "nn"):
    
        meg_signal = MEGSignal(bids_path, low_pass = low_pass_filter, high_pass = high_pass_filter)
        meg_signal.load_meta(meta_data_src = ph_info)
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
        model_0 = MyModel(num_classes = 2).to(device)
        
        loss_function = torch.nn.BCELoss
        optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)
        
        # for epoch in range(100):
    elif(training_flow == "lda"):
        # example code for running differnt training flow
        # to be implemented
        pass
        
        