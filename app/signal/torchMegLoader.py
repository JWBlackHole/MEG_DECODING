"""
for MEG loader for torch
competible with torch.utils.data.Dataset so that data can be loaded by batch during training in cnn
"""


import pandas as pd
import numpy as np

import mne_bids
from mne import Epochs, concatenate_epochs, EpochsArray
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
 
from loguru import logger

# custom import
from app.signal.megSignal import MEGSignal
from app.common.commonSetting import TargetLabel
import app.utils.my_utils as util

class TorchMegLoader(Dataset):
    def __init__(self, subject, until_session, until_task, raw_data_path, target_label, 
                low_pass_filter, high_pass_filter, to_print_interim_csv=False):
        self.subjcet = subject
        self.until_session = until_session
        self.until_task = until_task
        self.raw_data_path = raw_data_path
        self.concated_epochs: EpochsArray | None = None
        self.to_print_interim_csv = to_print_interim_csv
        
        # ---   target label checking ------ #
        if type(target_label) is str:   # convert to TargetLabel class if it is str
            if target_label  == "voiced":
                preprocess_setting = TargetLabel.VOICED_PHONEME
                logger.info("target label to predicted got: \"voiced\"")
            elif target_label  == "word_freq":
                preprocess_setting = TargetLabel.WORD_FREQ
            elif target_label  == "word_onset":
                preprocess_setting = TargetLabel.WORD_ONSET
            elif target_label == "is_sound":
                preprocess_setting = TargetLabel.IS_SOUND
            elif target_label == "plot_word":
                preprocess_setting = TargetLabel.PLOT_WORD_ONSET

            #     preprocess_setting = TargetLabel.DEFAULT
            else:
                raise NotImplementedError
                
        elif type(target_label) is TargetLabel:
            preprocess_setting = target_label

        
        # set default setting
        if preprocess_setting == TargetLabel.DEFAULT:
            preprocess_setting = TargetLabel.VOICED_PHONEME   # assume voiced is default
            logger.info("use default target label to predicted: \"voiced\"")

        
        # -----  prepare concatenated epoch
        concated_epoch = self.load_all_epochs(subject, until_session, until_task, raw_data_path, preprocess_setting, low_pass_filter, high_pass_filter)

    def __len__(self):
        """
        this method is required by Torch Dataset
        this define the len of data
        """
        try:
            return len(self.concated_epochs)  # Total number of epochs
        except Exception as err:
            logger.error(err)
            logger.error(f"type of self.concated_epochs: {type(self.concated_epochs)}")

    
    def __getitem__(self, idx):
        """
        Load a single epoch (and corresponding label) into memory for training.
        this method is required by Torch Dataset
        this define what is each data sample
        """
        # Load the specific epoch and its corresponding metadata
        epoch = self.concated_epochs[idx]
        X = epoch.get_data()  # Get the data as a 3D array (n_channels, n_times)
        y = epoch.events[:, -1]  # Example: label from events array

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).squeeze(0)  # Remove batch dimension
        y_tensor = torch.tensor(y, dtype=torch.long)  # Assuming classification task

        return X_tensor, y_tensor
    
    def load_all_epochs(self, subject, until_session, until_task, raw_data_path, setting,
                         low_pass_filter, high_pass_filter):
        """
        load epochs of all sessions, all tasks for one subject to self.concated_epochs
        """

        epochs_list = []    # list for save epochs of all sesions, all tasks

        logger.debug(f"raw data path: {raw_data_path}")

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir


        
        # load epochs for each session each task
        for session in range(until_session + 1):
            for task in range(until_task + 1):
                cur_epochs = self.load_epochs_one_task(subject, session, task, raw_data_path, setting, low_pass_filter, high_pass_filter)
                
                # make columns task and session to record the current handling task and session
                cur_epochs.metadata["task"] = task
                cur_epochs.metadata["session"] = session
                epochs_list.append(cur_epochs)
        
        if not len(epochs_list):
            logger.error("error in loading all epochs, program exit.")
            raise ValueError

        self.concated_epochs = concatenate_epochs(epochs_list)
        return self.concated_epochs
    

    def load_epochs_one_task(self, subject, session, task, raw_data_path, setting: TargetLabel,
                low_pass_filter, high_pass_filter) -> None:
        """
        prepare mne epoch of one task of one session one subject
        """



        logger.debug(f"raw data path: {raw_data_path}")

        

        # --- signal processing --- #
        
        signal_handler = MEGSignal(setting, low_pass=low_pass_filter, high_pass=high_pass_filter, to_print_interim_csv=self.to_print_interim_csv,
                                   preload=False)   # must set preload=False, this mean only load data when accessed [MUST !!!] 

        # set mne epoch for each session, each task
        # Specify a path to a epoch
        bids_path = mne_bids.BIDSPath(
            subject = f"{subject:02}",     # subject need to be 2-digit str (e.g. "01" to align folder name sub-01)  
            session = str(session),
            task = str(task),
            datatype = "meg",
            root = raw_data_path
        )

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir
            return signal_handler.prepare_one_epochs(bids_path, supplementary_meta = ph_info)

        elif setting in [TargetLabel.PLOT_WORD_ONSET,  TargetLabel.WORD_FREQ] :
            return signal_handler.prepare_one_epochs(bids_path, None)
        
        return
    
    def get_concated_epochs(self)-> Epochs:
        if self.concated_epochs is not None:
            return self.concated_epochs
        else:
            raise ValueError