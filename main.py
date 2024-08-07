
import pandas as pd

import mne_bids
import json
from pathlib import Path

from MEGSignal import MEGSignal
from MyModel import MyModel
from NNModelRunner import NNModelRunner
from LDAModel import MyLDA

# def accuracy_fn(y_true, y_pred):
#     correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
#     acc = (correct / len(y_pred)) * 100 
#     return acc

if __name__ == "__main__":

    # ---   config load in --- #

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



    # --- signal processing --- #
    
    meg_signal = MEGSignal(bids_path, low_pass = low_pass_filter, high_pass = high_pass_filter)
    meg_signal.load_meta(meta_data_src = ph_info)
    meg_signal.load_epochs()
    
    phonemes = meg_signal.epochs["not is_word"]
    X = phonemes.get_data()
    y = phonemes.metadata["voiced"].values


    # ---- train model ---- #

    if(training_flow == "nn"):
        nnRunner = NNModelRunner(X, y)
        
        
        
       
    elif(training_flow == "lda"):
        # example code for running differnt training flow
        # to be implemented
        metadata = meg_signal.get_metadata()
        lda_model = MyLDA()
        predictions = lda_model.decode_binary(X, y, metadata)
        print(type(predictions))
        print(predictions)

        
        