
import pandas as pd
import numpy as np
import mne_bids

from loguru import logger

# custom import
from app.my_signal.megSignal import MEGSignal



class Preprocessor:
    def __init__(self):
        pass
    def get_data(self, subject, session, task, raw_data_path, 
                low_pass_filter, high_pass_filter):
        
        logger.debug(f"raw data path: {raw_data_path}")
        ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir

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
        meg_signal.load_meta(meta_data_src = ph_info, to_save_csv=False)
        meg_signal.load_epochs()
        
        self.phonemes = meg_signal.epochs["not is_word"]
        self.X = self.phonemes.get_data()
        self.y = self.phonemes.metadata["voiced"].values

        return self.X, self.y
    
    def get_phonemes(self):
        return self.phonemes

