import os
from pathlib import Path

import mne
import mne_bids
from mne    import Epochs
from mne.io import Raw

from wordfreq import zipf_frequency

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, scale

import sys
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# import torch
# import torch.nn as nn
# import torch.functional as F

# cutom import
import my_utils as util


wk_dir = Path(os.getcwd())

class MEGSignal():
    def __init__(self, bids_path, low_pass:float = 30.0, high_pass:float = 0.5, n_jobs:int = 1):
       self.raw:  Raw|None          = None
       self.meta: pd.DataFrame|None = None
       
       # Epoches
       self.epochs: Epochs|None     = None      #mne.Epochs object
       
       self.load_raw(bids_path, low_pass, high_pass, n_jobs)
       
    
    def load_raw(self, bids_path: mne_bids.BIDSPath, low_pass:float = 30.0, high_pass:float = 0.5, n_jobs:int = 1,
                 to_save_csv:bool = False):
        """
        Load Raw MEG signal
        remark: 
        - low_pass: frequency above this value will be filtered out
        - high_pass: frequency below this value will be filtered out
        
        """
        # Reading associated event.tsv and channels.tsv
        self.raw = mne_bids.read_raw_bids(bids_path)
        # Specify the type of recording we want
        self.raw = self.raw.pick_types(
            meg=True, misc=False, eeg=False, eog=False, ecg=False
        )
        # Load raw data and filter by low and high pass
        self.raw.load_data().filter(high_pass, low_pass, n_jobs=n_jobs)
        
    def load_meta(self, meta_data_src: pd.DataFrame, to_save_csv:bool = False):
        """Load meta data
        set slef.meta fom information in meta_data_src
        meta_data_src: source of meta data
        """
        # Read the annotations, in this experiment are phonemes or sound, like "eh_I", "r_I", "t_B"
        # And append items to it.
        # count = 0 # debug
        meta_list = list()
        for annot in self.raw.annotations:
            d = eval(annot.pop("description"))
            # print(annot)
            for k, v in annot.items():
                assert k not in d.keys()
                d[k] = v
            # print(d.keys())
            meta_list.append(d)

            # # Debug
            # count = count + 1
            # if(count >= 10):
            #     break
        
        # --- Convert meatdata to form of DataFrame --- #
        self.meta = pd.DataFrame(meta_list)
        self.meta["intercept"] = 1.0
        
        # Computing if voicing
        # Replace voiced to True or False
        phonemes = self.meta.query('kind=="phoneme"')
        for ph, d in phonemes.groupby("phoneme"):
            # print(ph, ":\n", d)
            ph = ph.split("_")[0]
            match = meta_data_src.query(f"phoneme==\"{ph}\"")
            # print(match)
            assert len(match) == 1
            self.meta.loc[d.index, "voiced"] = (match.iloc[0].phonation == "v") # True or False
            
            
        # Compute word frequency
        self.meta["is_word"] = False
        words = self.meta.query('kind=="word"').copy()
        self.meta.loc[words.index + 1, "is_word"] = True
        
        # Merge "word frequency" with "phoneme"
        # apply a funcion to calculate the word frequency
        wfreq = lambda x: zipf_frequency(x, "en")  # noqa
        self.meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values
        

        self.meta = self.meta.query('kind=="phoneme"')
        if(to_save_csv):
            self.meta.to_csv(util.get_unique_file_name(file_name="test_wfreq.csv", dir="./test"))

       
    def load_epochs(self, to_save_csv: bool = False):
        """Get epochs by assemble "meatadata" and "raw". """
        # Create event that mne need
        # including time info
        events = np.c_[
            self.meta.onset * self.raw.info["sfreq"], np.ones((len(self.meta), 2))
        ].astype(int)
        logger.debug("SFREQ", self.raw.info["sfreq"])

        self.epochs = mne.Epochs(
            self.raw,
            events,
            tmin=-0.200,
            tmax=0.6,
            decim=10,
            baseline=(-0.2, 0.0),
            metadata=self.meta,
            preload=True,
            event_repeated="drop",
        )

        # threshold
        th = np.percentile(np.abs(self.epochs._data), 95)
        self.epochs._data[:] = np.clip(self.epochs._data, -th, th)
        self.epochs.apply_baseline()
        th = np.percentile(np.abs(self.epochs._data), 95)
        self.epochs._data[:] = np.clip(self.epochs._data, -th, th)
        self.epochs.apply_baseline()
        
        # logger.debug(meta.wordfreq)
        if(to_save_csv):
            self.meta.to_csv(util.get_unique_file_name(file_name="test_wfreq.csv", dir="./"))

    
    