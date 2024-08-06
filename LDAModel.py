import mne
import mne_bids
from mne    import Epochs
from mne.io import Raw
import pandas as pd
import numpy as np
import os
import sys
from loguru import logger


# cutom import
import my_utils as util

class LDA():
    def __init__(self, bids_path):
       self.raw:  Raw|None          = None
       self.meta: pd.DataFrame|None = None
       
       # Epoches
       self.epochs: Epochs|None     = None      #mne.Epochs object
       
       self.load_raw(bids_path = bids_path)