import os
from pathlib import Path

import mne
import mne_bids
from mne    import Epochs
from mne.io import Raw

from wordfreq import zipf_frequency
from sklearn.preprocessing import StandardScaler, scale

import pandas as pd
import numpy as np
from loguru import logger


# import torch
# import torch.nn as nn
# import torch.functional as F

# custom import
import app.utils.my_utils as util
from app.common.commonSetting import TargetLabel


wk_dir = Path(os.getcwd())

class MEGSignal():
    """
    expected flow to prepare MEG signal with metadata:
        init -> load_raw -> load-meta -> load_epochs
    
    """
    def __init__(self, setting: TargetLabel, low_pass:float = 30.0, high_pass:float = 0.5, n_jobs:int = 1):
        self.raw:  Raw|None          = None
        self.meta: pd.DataFrame|None = None
        
        # Epoches
        self.epochs: Epochs|None     = None      #mne.Epochs object
        # self.all_epochs: list = []               # list of mne.Epochs??
        self.setting: TargetLabel | None = setting
        self.low_pass: float = low_pass
        self.high_pass: float = high_pass
        # print("Low Pass",  self.low_pass)
        # print("High Pass", self.high_pass)
        # exit()
        self.n_jobs: int = n_jobs
       
       
       
    def prepare_one_epochs(self, bids_path, supplementary_meta: pd.DataFrame = None):
        """
        bids_path is path to one task of one sesion of one subject
        """
        raw = self.load_raw(bids_path)
        meta = self._load_meta(raw, supplementary_meta, False)
        epochs = self.load_epochs(raw, meta, False)
        return epochs


    def load_raw(self, bids_path) -> Raw:
        """
        Load Raw MEG signal
        Return
        -------
        raw
        remark: 
        - low_pass: frequency above this value will be filtered out
        - high_pass: frequency below this value will be filtered out
        
        """ 
        # Reading associated event.tsv and channels.tsv
        raw: Raw = mne_bids.read_raw_bids(bids_path)
        # Specify the type of recording we want
        raw = raw.pick_types(
            meg=True, misc=False, eeg=False, eog=False, ecg=False
        )
        # Load raw data and filter by low and high pass
        # print(self.low_pass, self.high_pass)
        raw.load_data().filter(l_freq = self.low_pass, h_freq = self.high_pass, n_jobs=self.n_jobs)
        return raw
        
    def _load_meta(self, raw: mne.io.Raw, supplementary_meta: pd.DataFrame, to_save_csv:bool = False)->pd.DataFrame:
        """Load meta data
        set slef.meta fom information in meta_data_src
        supplementary_meta: supplementary info for meta data

        Return
        ------------
        meta: pd.DataFrame
            - meta data of the considered raw
        """
        # Read the annotations, in this experiment are phonemes or sound, like "eh_I", "r_I", "t_B"
        # And append items to it.
        # count = 0 # debug

        if self.setting == TargetLabel.VOICED_PHONEME:

            meta_list = list()
            for annot in raw.annotations:
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
            meta = pd.DataFrame(meta_list)
            meta["intercept"] = 1.0
            
            # Computing if voicing
            # Replace voiced to True or False
            phonemes = meta.query('kind=="phoneme"')
            for ph, d in phonemes.groupby("phoneme"):
                # print(ph, ":\n", d)
                ph = ph.split("_")[0]
                match = supplementary_meta.query(f"phoneme==\"{ph}\"")
                # print(match)
                assert len(match) == 1
                meta.loc[d.index, "voiced"] = (match.iloc[0].phonation == "v") # True or False
                
                
            # Compute word frequency
            meta["is_word"] = False
            words = meta.query('kind=="word"').copy()
            meta.loc[words.index + 1, "is_word"] = True
            
            # Merge "word frequency" with "phoneme"
            # apply a funcion to calculate the word frequency
            wfreq = lambda x: zipf_frequency(x, "en")  # noqa
            meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values
            

            meta = meta.query('kind=="phoneme"')
            if(to_save_csv):
                meta.to_csv(util.get_unique_file_name(file_name="test_wfreq.csv", dir="./test"))

        else:
            logger.error("preprocessing for setting other than \"voiced\" is not implemented. program exit")
            exit(0)

        return meta

       
    def load_epochs(self, raw: Raw, meta: pd.DataFrame, to_save_csv: bool = False, tmin: float = None, tmax: float = None)->mne.Epochs:
        """Get epochs by assemble "meatadata" and "raw". 
        will load epochs of the given raw and meta 
        meta and raw should correspond to the same subject same session same task

        Return
        --------
        epochs: mne.Epochs
        
        """
        # Create event that mne need
        # including time info
        events = np.c_[
            meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
        ].astype(int)
        logger.debug(f"SFREQ: {raw.info["sfreq"]}")

        # epochs = mne.Epochs(
        #     raw,
        #     events,
        #     tmin=-0.200,
        #     tmax=0.6,
        #     decim=10,
        #     baseline=(-0.2, 0.0),
        #     metadata=meta,
        #     preload=True,
        #     event_repeated="drop",
        # )
        epochs = mne.Epochs(
            raw,
            events,
            tmin    = -0.1,
            tmax    = 0.3,
            decim   = 10,
            baseline=(-0.1, 0.0),
            metadata=meta,
            preload =True,
            event_repeated="drop",
        )

        # threshold
        th = np.percentile(np.abs(epochs._data), 95)
        epochs._data[:] = np.clip(epochs._data, -th, th)
        epochs.apply_baseline()
        th = np.percentile(np.abs(epochs._data), 95)
        epochs._data[:] = np.clip(epochs._data, -th, th)
        epochs.apply_baseline()
        
        # logger.debug(meta.wordfreq)
        if(to_save_csv):
            meta.to_csv(util.get_unique_file_name(file_name="test_wfreq.csv", dir="./"))

        return epochs
    
    def get_metadata(self)->pd.DataFrame:
        """
        getter of self.meta (metadata)
        """
        if self.meta is not None:
            return self.meta
        else:
            logger.warning("metadata is not set, cannot get metadata.")
            return None

    
    