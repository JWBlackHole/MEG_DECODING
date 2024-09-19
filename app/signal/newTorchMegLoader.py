"""
new torch meg loader
create epoch only when iterated to
"""

import pandas as pd
import numpy as np
from typing import Tuple

import mne_bids
from mne import Epochs, EpochsArray
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
 
from loguru import logger

# custom import
from app.signal.megSignal import MEGSignal
from app.common.commonSetting import TargetLabel

class MegDataIterator(Dataset):
    def __init__(self, until_subject, until_session, until_task, raw_data_path, target_label,
                  to_print_interim_csv=False, 
                  meg_param:dict={"tmin":None, "tmax":None, "decim":None, "low_pass": None, "high_pass":None},
        ):
        """
        subject start from 01
        session start from 0
        task start from 0
        """
        logger.info("MegDataIterator is inited")
        self.until_subjcet:int = until_subject
        self.until_session:int = until_session
        self.until_task:int = until_task
        self.raw_data_path = raw_data_path
        self.to_print_interim_csv = to_print_interim_csv
        self.meg_param: dict = meg_param
        self.nchans:int = None
        self.ntimes:int = None
        self.voiced_phoneme_epoch = None
        self.target_label = target_label
        self.cur_idx: int = 0
        self.totaltask: int = (until_subject-1) * 8 + (until_session+1) * (until_task +1)

        logger.info(f"train until sub: {self.until_subjcet}, ses: {self.until_session}, task: {self.until_task}")
        logger.info(f"total no. of task: {self.totaltask}")

    def __len__(self):
        return self.totaltask
    
    def __getitem__(self, idx):

        if idx > self.totaltask: 
            raise IndexError    # define stop point of iterator
        
        
        verbose = True

        sub, ses, task = self.idx_to_bids_path_num(idx)
        epoch = self.get_meg_epoch(sub, ses, task)
        if verbose:
                logger.debug(f"idx= {idx}, epoch: {epoch}, type: {type(epoch)}")
        if epoch is None:
            return None

        self.cur_idx += 1

        epoch.apply_baseline(verbose="WARNING")  # Apply baseline correction
        X = epoch.get_data(copy=True)   # Get the data as a 3D array (n_channels, n_times)
        y = epoch.metadata["voiced"].values

        # clip to 95 percentile for twice
        th = np.percentile(np.abs(X), 95)
        X = np.clip(X, -th, th)
        th = np.percentile(np.abs(X), 95)
        X = np.clip(X, -th, th)

        if(verbose):
            logger.debug(f"type of X: {type(X)}")   # numpy.ndarray
            try:
                logger.debug(f"X.shape: {X.shape}") # (1, 208, 41)
            except Exception as err:
                logger.error(err)
            logger.debug(f"type of y: {type(y)}")   # numpy.ndarray
            try:
                logger.debug(f"y.shape: {y.shape}") # (1,)
            except Exception as err:
                logger.error(err)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        # X_tensor = X_tensor.unsqueeze(1)
        # if verbose :
        #         logger.debug(f"after unsqeeze(1), X_tensor shape: {X_tensor.shape}")
        
        y_tensor = torch.tensor(y.astype(int), dtype=torch.float32) 
        # BCE loss expect dtype float32
        # if change to other loss function, may need to look for what dtype expected
        
        self.getitem_debug_printed = True
        return X_tensor, y_tensor


    def idx_to_bids_path_num(self, idx: int)->Tuple[int, int, int]:
        sub = (idx // 8) + 1
        ses = (idx // 4)%2
        task = idx %4
        return sub, ses, task
    
    def get_meg_epoch(self, subject, session, task):

        logger.info("try to meg epoch from MegSignal class")

        signal_handler = MEGSignal(             # must set preload=False, this means only load data when accessed [MUST !!!] 
            self.target_label, 
            low_pass=self.meg_param["low_pass"] if self.meg_param["low_pass"] else None, 
            high_pass=self.meg_param["high_pass"] if self.meg_param["high_pass"] else None,
            to_print_interim_csv=self.to_print_interim_csv if self.to_print_interim_csv else None,
            preload=True, 
            tmin=self.meg_param["tmin"] if self.meg_param["tmin"] else None,
            tmax=self.meg_param["tmax"] if self.meg_param["tmax"] else None,
            decim=self.meg_param["decim"] if self.meg_param["decim"] else None
        ) 
        if self.nchans is None or self.ntimes is None:
            self.nchans, self.ntimes = signal_handler.get_nchans_ntimes()

        bids_path = mne_bids.BIDSPath(
            subject = f"{subject:02}",     # subject need to be 2-digit str (e.g. "01" to align folder name sub-01)  
            session = str(session),
            task = str(task),
            datatype = "meg",
            root = self.raw_data_path
        )

        if self.target_label == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir
            return signal_handler.prepare_one_epochs(bids_path, supplementary_meta = ph_info)

        elif self.target_label in [TargetLabel.PLOT_WORD_ONSET,  TargetLabel.WORD_FREQ] :
            return signal_handler.prepare_one_epochs(bids_path, None)
        
        else:
            raise NotImplementedError(f"target label: {self.TargetLabel}, not implemented!")
        return
    
    def get_signal_dim(self):
        if isinstance(self.nchans, int) and isinstance(self.ntimes, int):
            return self.nchans, self.ntimes
        else:
            logger.error("nchans, ntimes is not of int!")
            logger.error(f"nchans: {self.nchans}, ntimes: {self.ntimes}")
            raise ValueError
    
    def cal_ntimes(self):
        try:
            return int((self.meg_param["tmax"] - self.meg_param["tmin"])*(1000/self.meg_param["decim"])) +1
        except Exception as e:
            logger.error(e)
            return None

        


