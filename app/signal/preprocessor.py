
import pandas as pd
import numpy as np
import mne_bids

from loguru import logger

# custom import
from app.signal.megSignal import MEGSignal
from app.common.commonSetting import TargetLabel



class Preprocessor:
    def __init__(self):
        self.meg_signal = None
    def get_data(self, subject, session, task, raw_data_path, target_label: str, 
                low_pass_filter, high_pass_filter):
        
        preprocess_setting = None
        if target_label  == "voiced":
            preprocess_setting = TargetLabel.VOICED
        elif target_label  == "is_word":
            preprocess_setting = TargetLabel.IS_WORD
        elif target_label == "is_sound":
            preprocess_setting = TargetLabel.IS_SOUND

        elif target_label in ["is voiced", "is_voiced", "phoneme", "phonemes", "voice", "is voice", "is_voice"]:
            logger.error(f"target_label \"{target_label}\" is not supported, do you mean \"voiced\"?")
            logger.warning("assume default flow")
            preprocess_setting = TargetLabel.DEFAULT
        elif target_label in ["is word", "word"]:
            logger.error(f"target_label \"{target_label}\" is not supported, do you mean \"is_word\"?")
            logger.warning("assume default flow")
            preprocess_setting = TargetLabel.DEFAULT
        elif target_label in ["is sound", "sound", "have sound", "have_sound"]:
            logger.error(f"target_label \"{target_label}\" is not supported, do you mean \"is_sound\"?")
            logger.warning("assume default flow")
            preprocess_setting = TargetLabel.DEFAULT
        else:
            logger.error(f"target_label \"{target_label}\" not recognised, assume default flow")
            logger.warning("assume default flow")
            preprocess_setting = TargetLabel.DEFAULT


        # set default setting
        if preprocess_setting == TargetLabel.DEFAULT:
            preprocess_setting = TargetLabel.VOICED   # assume voiced or is_word is default
        

        # for now target_label_flag does not affect anything, 
        # in the future, may use this to control the preprocess of MEG signal
        self.load_meg_signal(subject, session, task, raw_data_path, preprocess_setting, low_pass_filter, high_pass_filter)
        
        # get X, y from loaded meg signal
        if preprocess_setting == TargetLabel.VOICED:
            self.X = self.phonemes.get_data()
            self.y = self.phonemes.metadata["voiced"].values

        else:
            logger.error(f"for now only \"voiced\" is supported. program exit.")
            exit(0)


        return self.X, self.y
    
    def load_meg_signal(self, subject, session, task, raw_data_path, setting: TargetLabel,
                low_pass_filter, high_pass_filter) -> None:
        """
        prepare mne epoch and save to self.meg_signal
        setter of self.meg_signal
        """

        
        if setting == TargetLabel.IS_WORD:
            logger.error("target label \"is_word\" is not supported yet. program exit.")
            exit(0)

        if setting == TargetLabel.IS_SOUND:
            logger.error("target label \"is_sound\" is not supported yet. program exit.")
            exit(0)

        logger.debug(f"raw data path: {raw_data_path}")
        

        # Specify a path to a epoch
        bids_path = mne_bids.BIDSPath(
            subject = subject,
            session = session,
            task = task,
            datatype = "meg",
            root = raw_data_path
        )

        # --- signal processing --- #
        
        self.meg_signal = MEGSignal(bids_path, setting, low_pass = low_pass_filter, high_pass = high_pass_filter)

        if setting == TargetLabel.VOICED:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir
            self.meg_signal.load_meta(meta_data_src = ph_info, to_save_csv=False)
            self.meg_signal.load_epochs()
            
            self.phonemes = self.meg_signal.epochs["not is_word"]

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
            return self.phonemes
        else:
            logger.error("for now only \"phonemes\" is supported! returning None")
            return None

