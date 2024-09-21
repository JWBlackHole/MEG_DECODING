import os
from pathlib import Path
import sys
import copy
import mne
import mne_bids
from mne    import Epochs
from mne.io import Raw

from wordfreq import zipf_frequency
from sklearn.preprocessing import StandardScaler, scale
import matplotlib.pyplot as plt

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
    def __init__(self, setting: TargetLabel, low_pass:float = 0.5, high_pass:float = 180, n_jobs:int = 1, to_print_interim_csv=False, preload=True,
                 tmin: float=-0.1, tmax: float=0.3, decim:int=1, clip_percentile: float=None):
        self.raw:  Raw|None          = None
        self.meta: pd.DataFrame|None = None

        # Epoches
        self.epochs: Epochs|None     = None      #mne.Epochs object
        # self.all_epochs: list = []               # list of mne.Epochs??
        self.target_label: TargetLabel | None = setting
        self.low_pass: float = low_pass
        self.high_pass: float = high_pass
        self.n_jobs: int = n_jobs
        self.to_print_interim_csv: bool = to_print_interim_csv
        self.preload: bool = preload
        self.nchans:int = 208
        self.tmin:float = tmin
        self.tmax:float = tmax
        self.decim:int = decim
        self.ntimes: int = int((tmax-tmin)*(1000/decim)) +1  # this is required for preload false, as data is not loaded, cannot infer ntimes directly from data, hence ntimes must be calculated
                                                            # 1000 is sfreq (frequency for sampling meg)
        self.clip_percentile = clip_percentile
        
        logger.debug(f"ntimes= {self.ntimes}")
       
    def get_nchans_ntimes(self)->tuple[int,int]:
        if (self.nchans and self.ntimes):
            return self.nchans, self.ntimes
        else:
            logger.error(f"self.nchans: {self.nchans}, self.ntimes: {self.ntimes}, at least one is not properly set!")
            raise ValueError
       

    def prepare_one_epochs(self, bids_path, supplementary_meta: pd.DataFrame = None):
        """
        bids_path is path to one task of one sesion of one subject
        """
        res = self.load_raw(bids_path)
        if res is None:
            return None
        meta = self._load_meta(self.raw, supplementary_meta, to_save_csv=self.to_print_interim_csv)
        epochs = self.load_epochs(self.raw, meta, to_save_csv=self.to_print_interim_csv
                                  )
        return epochs

    """
    setter of self.raw
    """
    def load_raw(self, bids_path)->mne.io.Raw:
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
        try:
            raw: Raw = mne_bids.read_raw_bids(bids_path)
            # Specify the type of recording we want
            raw = raw.pick_types(
                meg=True, misc=False, eeg=False, eog=False, ecg=False
            )
            # Load raw data and filter by low and high pass
            self.raw = raw
            # print(self.low_pass, self.high_pass)
            if self.preload:
                raw.load_data().filter(l_freq = self.low_pass, h_freq = self.high_pass, n_jobs=self.n_jobs, verbose='INFO')

        except FileNotFoundError as err:
            logger.warning(err)
            logger.warning("bids path: does not exist! will skip this path.")
            return None
        return raw
    
    def _get_last_el(self, lst):
        if len(lst) == 0:
            return None
        elif len(lst) == 1:
            return lst[0]
        else:
            return lst[-1]

    def _get_2nd_last_el(self, lst):
        if len(lst) == 0:
            return None
        elif len(lst) == 1:
            return lst[0]
        else:
            return lst[-2]
    
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

        logger.info(f"target label got in megSignal: {self.target_label}")
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
        #meta["intercept"] = 1.0
        

        # displace all onset by -0.05
        # Computing if voicing
        # Replace voiced to True or False

        if self.target_label == TargetLabel.VOICED_PHONEME:
            phonemes = meta.query('kind=="phoneme"')
            for ph, d in phonemes.groupby("phoneme"):
                # print(ph, ":\n", d)
                ph = ph.split("_")[0]
                match = supplementary_meta.query(f"phoneme==\"{ph}\"")
                # print(match)
                assert len(match) == 1
                meta.loc[d.index, "voiced"] = (match.iloc[0].phonation == "v") # True or False            

            meta = meta[meta['kind'] == 'phoneme']

            if(to_save_csv):
                meta.to_csv(util.get_unique_file_name(file_name="meta.csv", dir="./test"))
        elif self.target_label == TargetLabel.WORD_FREQ:


            thres = 4.0
            meta = meta[meta['kind'] == 'word']
            wfreq = lambda x: zipf_frequency(x, "en")   # lambda func for cal word frequency

            # Create the 'wordfreq' column and set its values
            meta['wordfreq'] = meta['word'].apply(wfreq)
            meta['word_freq_above_thres'] = meta['wordfreq'] > thres

            median_wordfreq = meta['wordfreq'].median()
            logger.info(f"median of word frequencies is: {median_wordfreq}")
            
            # [notes] for word frequency
            # > 6.0 ->mostly function words like a, the, with, it, is,...
            # > 5.8 -> common word but may not all be like a, the, with, it, is,...
            # < 4.0 may be a good threshold to filter less common word

        elif self.target_label in [TargetLabel.PLOT_WORD_ONSET]:
            # create colmn is_word in meta
            # if column "kind"=="word", is_word
            # else false
            # meta["word_onset"] = False
            # for index, row in meta.iterrows():
            #     if row['kind'] == 'word':
            #         meta.loc[index, "word_onset"] = True
            # # add

            meta = meta[meta['kind'] == 'word']
            meta["is_word"] = True

        elif self.target_label == TargetLabel.WORD_ONSET:


            # -------------- non randomo fashion  ----------------
            # non onset time 設成word onset 往上數兩row

            non_onset_idx = []
            onset_idx = []

            for idx in range(len(meta)):
                if meta.iloc[idx]['kind'] == 'word':
                    if idx - 2 >= 0:  # Ensure the index is valid
                        cur_non_onset = meta.iloc[idx-2]['onset']

                        #  avoid duplicate onset time
                        last_onset_idx = self._get_last_el(onset_idx)
                        if last_onset_idx is not None:
                            if cur_non_onset != meta.iloc[last_onset_idx]['onset']:
                                non_onset_idx.append(idx - 2)                          

                    onset_idx.append(idx)

            meta['is_word_onset'] = False

            meta.loc[onset_idx, 'is_word_onset'] = True

    
            valid_indices = list(set(non_onset_idx + onset_idx))
            valid_indices.sort()  # Optional: Sort the indices to maintain order
            meta = meta.iloc[valid_indices].reset_index(drop=True)




            # -----------   random fashion --------------
            # goal:
            # generate random non word_onset time sameple,
            # for the model to predict is /is not word onset

            # meta = meta[meta['kind'] == 'word']
            # sfreq = raw.info['sfreq']  # Sampling frequency

            # # Step 1: Create a new column 'is_word_onset' and set it to True for all rows
            # meta['is_word_onset'] = True

            # # Step 2: Generate random time points for non-word onset events
            # word_onset_samples = (meta['onset'].values * sfreq).astype(int)  # Convert onset times to sample points
            # num_non_word_onset = len(word_onset_samples)  # Number of non-word onset events to generate
            # non_word_onset_samples = np.random.randint(word_onset_samples[0], word_onset_samples[-1], num_non_word_onset)

            # # Step 3: Create new rows for non-word onset events
            # non_word_onset_df = pd.DataFrame({
            #     'onset': non_word_onset_samples / sfreq,  # Convert sample points back to time
            #     'is_word_onset': False
            # })

            # # Combine the original meta DataFrame with the new non-word onset events
            # meta = pd.concat([meta, non_word_onset_df], ignore_index=True)
                        

        if(to_save_csv):
            meta.to_csv(util.get_unique_file_name(file_name="meta_from_megsignal.csv", dir="./results"))

        return meta

       
    def load_epochs(self, raw: Raw, meta: pd.DataFrame, to_save_csv: bool = False)->mne.Epochs:
        """Get epochs by assemble "meatadata" and "raw". 
        will load epochs of the given raw and meta 
        meta and raw should correspond to the same subject same session same task

        Return
        --------
        epochs: mne.Epochs
        
        """
        # Create event that mne need
        # including time info
        logger.debug(f"in meg signal handler, self.setting: {self.target_label}")
        
        events = None

        # ----------   set up events  --------


        events = np.c_[
                meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
            ].astype(int)
        



        #  ---------  create mne epoch   ---------

        # logger.debug(f"SFREQ: {raw.info["sfreq"]}")

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


        if self.preload:
            logger.warning("preload is true. will load data of this epoch to memory immediately.")
        epochs = mne.Epochs(
            raw,
            events,
            tmin    = self.tmin,      #note: must use tmin, tmax, decim in self, because ntimes is only calculated in init according to value in init
            tmax    = self.tmax,
            decim   = self.decim,
            baseline=None,
            metadata=meta,
            preload =self.preload,
            event_repeated="drop",
        )


        logger.info(f"size of epoch: ")
        print(sys.getsizeof(epochs))

        

        #events
        # 1st col: onset time

        if self.preload:    # this cannot be done if preload==False, as data is not loaded

            # threshold

            if isinstance(self.clip_percentile, (int, float)):
                th = np.percentile(np.abs(epochs._data), self.clip_percentile)
                epochs._data[:] = np.clip(epochs._data, -th, th)
                epochs.apply_baseline()
               
            
        # logger.debug(meta.wordfreq)
        if(to_save_csv):
            meta.to_csv(util.get_unique_file_name(file_name="epochs.csv", dir="./"))

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

    
    def plot_sensor_topology(self):
        """
        Plot the sensor topology of the loaded raw MEG data.
        """
        logger.info("megSignal.plot_sensor_topology is running")
        if self.raw is None:
            raise ValueError("Raw data is not loaded. Please load the raw data first.")
        logger.debug(f"type of self.raw: {type(self.raw)}")
        

        # plot 3D graph (need install modules)
        # fig = mne.viz.plot_alignment(self.raw.info, meg=('helmet', 'sensors'), coord_frame='meg')
        # mne.viz.set_3d_title(figure=fig, title="Sensor Topology")

        # plot 2D graph
        fig = mne.viz.plot_sensors(self.raw.info, show_names=True)# Plot the 2D sensor positions
        fig.suptitle("2D Sensor Topology")
        plt.show(block=True)