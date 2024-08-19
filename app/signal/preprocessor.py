
import pandas as pd
import numpy as np
import mne_bids
from mne import Epochs, concatenate_epochs
 
from loguru import logger

# custom import
from app.signal.megSignal import MEGSignal
from app.common.commonSetting import TargetLabel



class Preprocessor:
    def __init__(self):
        self.concated_epochs: Epochs | None = None  # concatenated Epochs of all sessions all tasks

    def get_data(self, subject, until_session, until_task, raw_data_path, target_label, 
                low_pass_filter, high_pass_filter):
        
        preprocess_setting = None


        # ---   target label checking ------ #
        if type(target_label) is str:   # convert to TargetLabel class if it is str
            if target_label  == "voiced":
                preprocess_setting = TargetLabel.VOICED_PHONEME
                logger.info("target label to predicted got: \"voiced\"")
            elif target_label  == "is_word":
                preprocess_setting = TargetLabel.WORD_FREQ
            elif target_label == "is_sound":
                preprocess_setting = TargetLabel.IS_SOUND

            # elif target_label in ["is voiced", "is_voiced", "phoneme", "phonemes", "voice", "is voice", "is_voice"]:
            #     logger.error(f"target_label \"{target_label}\" is not supported, do you mean \"voiced\"?")
            #     logger.warning("assume default flow")
            #     preprocess_setting = TargetLabel.DEFAULT
            # elif target_label in ["is word", "word"]:
            #     logger.error(f"target_label \"{target_label}\" is not supported, do you mean \"is_word\"?")
            #     logger.warning("assume default flow")
            #     preprocess_setting = TargetLabel.DEFAULT
            # elif target_label in ["is sound", "sound", "have sound", "have_sound"]:
            #     logger.error(f"target_label \"{target_label}\" is not supported, do you mean \"is_sound\"?")
            #     logger.warning("assume default flow")
            #     preprocess_setting = TargetLabel.DEFAULT
            else:
                logger.error(f"target_label \"{target_label}\" not recognised, assume default flow")
                logger.warning("assume default flow")
                preprocess_setting = TargetLabel.DEFAULT
                
        elif type(target_label) is TargetLabel:
            preprocess_setting = target_label


        # set default setting
        if preprocess_setting == TargetLabel.DEFAULT:
            preprocess_setting = TargetLabel.VOICED_PHONEME   # assume voiced is default
            logger.info("use default target label to predicted: \"voiced\"")
        


        # in the future, may use preprocess_setting to control the preprocess of MEG signal
        # set self.concated_epochs

        if preprocess_setting == TargetLabel.VOICED_PHONEME:
            self.load_all_epochs(subject, until_session, until_task, raw_data_path, preprocess_setting, low_pass_filter, high_pass_filter)
        

        elif preprocess_setting == TargetLabel.WORD_FREQ:
            self.load_all_epochs(subject, until_session, until_task, raw_data_path, preprocess_setting, low_pass_filter, high_pass_filter)
        
        else:
            logger.error(f"for now only \"voiced\" is supported. program exit.")
            exit(0)
        
        # after self.concated_epochs setted in self.load_all_epochs, it is ready to use

        
        # extract target label from self.concated_epochs 
        # remark: at this point, self.concated_epochs.metadata contain the metadata df of all considered tasks, sessions
        
        # example of extracting relevant epochs
        # example_epoch = self.concated_epochs["exampl_col_name"]

        # if preprocess_setting == TargetLabel.WORD_FREQ:
            # to be tested
            # words = self.concated_epochs["is_word"]
            # self.X = words.get_data(copy=True)
            # self.y = words.metadata["wordfreq"].values
        
        if preprocess_setting == TargetLabel.VOICED_PHONEME:
            phonemes = self.concated_epochs["not is_word"]      # for now not is_word means phoneme, in the future may change to more intuitive way
            self.X = phonemes.get_data(copy=True)   # use copy=True to avoid changing the original data
            self.y = phonemes.metadata["voiced"].values

        else:
            logger.error(f"for now only \"voiced\" is supported. program exit.")
            exit(0)


        return self.X, self.y
    
    def load_all_epochs(self, subject, until_session, until_task, raw_data_path, setting,
                         low_pass_filter, high_pass_filter):
        """
        load epochs of all sessions, all tasks for one subject to self.concated_epochs
        """
        if setting == TargetLabel.WORD_FREQ:
            logger.error("target label \"is_word\" is not supported yet. program exit.")
            exit(0)

        if setting == TargetLabel.IS_SOUND:
            logger.error("target label \"is_sound\" is not supported yet. program exit.")
            exit(0)

        epochs_list = []    # list for save epochs of all sesions, all tasks

        logger.debug(f"raw data path: {raw_data_path}")

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir

        else:
            logger.error(f"target label \"{setting}\" is not supported yet. program exit.")
            exit(0)

        
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
            exit(0)

        self.concated_epochs = concatenate_epochs(epochs_list)
        return self.concated_epochs

        
        

    def load_epochs_one_task(self, subject, session, task, raw_data_path, setting: TargetLabel,
                low_pass_filter, high_pass_filter) -> None:
        """
        prepare mne epoch of one task of one session one subject
        """

        
        if setting == TargetLabel.WORD_FREQ:
            logger.error("target label \"is_word\" is not supported yet. program exit.")
            exit(0)

        if setting == TargetLabel.IS_SOUND:
            logger.error("target label \"is_sound\" is not supported yet. program exit.")
            exit(0)

        logger.debug(f"raw data path: {raw_data_path}")

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir

        else:
            logger.error(f"target label \"{setting}\" is not supported yet. program exit.")
            exit(0)

        

        # --- signal processing --- #
        
        signal_handler = MEGSignal(setting, low_pass = low_pass_filter, high_pass = high_pass_filter)

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

        else:
            logger.error(f"target label \"{setting}\" is not supported yet. program exit.")
            exit(0)



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
            return self.concated_epochs["not is_word"]
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