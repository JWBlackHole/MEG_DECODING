
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

class Preprocessor():
    def __init__(self,meg_param:dict={"tmin":None, "tmax":None, "decim":None, "low_pass": None, "high_pass":None}):
        self.concated_epochs: Epochs | None = None  # concatenated Epochs of all sessions all tasks
        self.X = None
        self.y = None
        self.meg_param = meg_param
        self.nchans = None
        self.ntimes = None

    def get_signal_dim(self):
        if isinstance(self.nchans, int) and isinstance(self.ntimes, int):
            return self.nchans, self.ntimes
        else:
            logger.error("nchans, ntimes is not of int!")
            logger.error(f"nchans: {self.nchans}, ntimes: {self.ntimes}")
            raise ValueError



    def plot_sensor_topo(self, raw_data_path):
        bids_path = mne_bids.BIDSPath(
            subject = "01",     # subject need to be 2-digit str (e.g. "01" to align folder name sub-01)  
            session = "0",
            task = "0",
            datatype = "meg",
            root = raw_data_path
        )
        signal_handler = MEGSignal(None, low_pass=None, high_pass=None, to_print_interim_csv=None)
        signal_handler.load_raw(bids_path)
        signal_handler.plot_sensor_topology()

        

    def prepare_X_y(self, subject, until_session, until_task, raw_data_path, target_label, 
                low_pass_filter, high_pass_filter, to_print_interim_csv):
        
        self.to_print_interim_csv = to_print_interim_csv
        preprocess_setting = None


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
        


        # in the future, may use preprocess_setting to control the preprocess of MEG signal
        # set self.concated_epochs

        concated_epoch = self.load_all_epochs(subject, until_session, until_task, raw_data_path, preprocess_setting, low_pass_filter, high_pass_filter)
        

        
        # after self.concated_epochs setted in self.load_all_epochs, it is ready to use

        
        # extract target label from self.concated_epochs 
        # remark: at this point, self.concated_epochs.metadata contain the metadata df of all considered tasks, sessions
        
        # example of extracting relevant epochs
        # example_epoch = self.concated_epochs["exampl_col_name"]


        
        if preprocess_setting == TargetLabel.VOICED_PHONEME:
            phonemes = self.concated_epochs      # for now not is_word means phoneme, in the future may change to more intuitive way
            self.X = phonemes.get_data(copy=True)   # use copy=True to avoid changing the original data
            self.y = phonemes.metadata["voiced"].values
        
        elif preprocess_setting == TargetLabel.PLOT_WORD_ONSET:

            logger.info("plot is word is doing")
            self.X = self.concated_epochs.get_data(copy=True)
            meta = self.concated_epochs.metadata
            if self.to_print_interim_csv:
                meta.to_csv(util.get_unique_file_name("meta_from_preprocessor.csv", "./results"))
            self.y = None   # not important

            self.is_word = self.concated_epochs

        elif preprocess_setting == TargetLabel.WORD_FREQ:

            self.X = self.concated_epochs.get_data(copy=True)
            meta = self.concated_epochs.metadata
            self.y = meta['word_freq_thres'].values
            if self.to_print_interim_csv:
                meta.to_csv(util.get_unique_file_name("meta_from_preprocessor.csv", "./results"))
            
        
        elif preprocess_setting == TargetLabel.WORD_ONSET:
            self.X = self.concated_epochs.get_data(copy=True)
            self.y = self.concated_epochs.metadata["is_word_onset"].values
            if self.to_print_interim_csv:
                self.concated_epochs.metadata.to_csv(util.get_unique_file_name("meta_from_preprocessor.csv", "./results"))

        else:
            raise NotImplementedError



        return self.X, self.y
    
    def get_X_y(self):
        if (self.X is not None) and (self.y is not None):
            return self.X, self.y
    
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
        
        signal_handler = MEGSignal(
            setting, 
            low_pass=self.meg_param["low_pass"] if self.meg_param["low_pass"] else None, 
            high_pass=self.meg_param["high_pass"] if self.meg_param["high_pass"] else None,
            to_print_interim_csv=self.to_print_interim_csv if self.to_print_interim_csv else None,
            preload=True, 
            tmin=self.meg_param["tmin"] if self.meg_param["tmin"] else None,
            tmax=self.meg_param["tmax"] if self.meg_param["tmax"] else None,
            decim=self.meg_param["decim"] if self.meg_param["decim"] else None,
            clip_percentile=self.meg_param["clip_percentile"] if self.meg_param["clip_percentile"] else None,
            onset_offset=self.meg_param["onset_offset"] if self.meg_param["onset_offset"] else None,
            baseline=self.meg_param["baseline"] if self.meg_param["baseline"]  else None
        ) 
        # set mne epoch for each session, each task
        # Specify a path to a epoch
        bids_path = mne_bids.BIDSPath(
            subject = f"{subject:02}",     # subject need to be 2-digit str (e.g. "01" to align folder name sub-01)  
            session = str(session),
            task = str(task),
            datatype = "meg",
            root = raw_data_path
        )
        if self.nchans is None or self.ntimes is None:
            self.nchans, self.ntimes = signal_handler.get_nchans_ntimes()

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir
            return signal_handler.prepare_one_epochs(bids_path, supplementary_meta = ph_info)

        elif setting in [TargetLabel.PLOT_WORD_ONSET,  TargetLabel.WORD_FREQ, TargetLabel.WORD_ONSET] :
            return signal_handler.prepare_one_epochs(bids_path, None)
        
        return




    def get_meg_signal(self)->MEGSignal:
        """ to be implement if needed"""
        pass

        #the below looks like is not correct
        #        
        # if self.meg_signal is not None:
        #     return self.meg_signal
        # else:
        #     logger.error("self.meg_signal is not prepared! need to call prepare_meg_signal first!")
        #     return None
    
    def get_metadata(self, target: str="phonemes"):
        if target == "phonemes":
            return self.concated_epochs
        else:
            logger.error("for now only \"phonemes\" is supported! returning None")
            return None

    def get_concated_metadata(self) -> pd.DataFrame:
        """
        get the meta data of the concatenated epochs
        """
        if (self.concated_epochs is not None) and (isinstance(self.concated_epochs.metadata, pd.DataFrame)):
            return self.concated_epochs.metadata
        else:
            logger.error(f"cannot access self.concated_epochs.metadata or it is of wrong type, type of self.concated_epochs.metadata: \
                         {type(self.concated_epochs.metadata)}, returning None")
            
    def get_concated_epochs(self)-> Epochs:
        if self.concated_epochs is not None:
            return self.concated_epochs
        else:
            raise ValueError
            
    def plot_evoked_response(self, target: str):
        """
        notes

        'MEG 001'  is channel

        
        target: "is_word"
        """
        if(target == "is_word"):
            if self.is_word and (isinstance(self.is_word, Epochs) or isinstance(self.is_word, EpochsArray)):

                fig1 = self.is_word.plot()  # plot each channel
                evo = self.is_word.average()    # mne.evoked.EvokedArray of 668 events

                fig_evo = evo.plot(spatial_colors=True, show=True) # type: matplotlib.figure.Figure
                #print(type(fig_evo))

                fig_evo.savefig(util.get_unique_file_name("evoked_response.png", "./results"))
                

            else:
                raise ValueError("self.is_word is not prepared or of wrong type")
        return
            
    def plot_n_events_evo(self, target, n: int,  show_last_fig = False, plot_only_one = False):
        """
        target: "is_word"
        n: how many rows to plot
        """
        if plot_only_one:
            i = n
        else:
            i=1
        if(target == "is_word"):
            if self.is_word and  isinstance(self.is_word, EpochsArray):
                logger.debug(f"type of self.is_word: {type(self.is_word)}")
                for evo in self.is_word.iter_evoked():
                    if(i>n):
                        break
                    
                    fig_evo = evo.plot(spatial_colors = True, show = False)

                    fig_evo.set_size_inches(36,24)
                    fig_evo.savefig(util.get_unique_file_name(f"evoked_event{i}.png", "./graphs"),
                                    dpi=200)
                    plt.close(fig_evo)  # for closing the file before writing next graph

                    #if show_last_fig and i == n:
                    if True:
                        evo.plot(spatial_colors=True, show=True)
                    
                    i=i+1

            else:
                raise ValueError("self.is_word is not type mne.EpochsArray")
        return