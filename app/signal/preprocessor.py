
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
    def get_data(self, subject, session, task, raw_data_path, target_label, 
                low_pass_filter, high_pass_filter):
        
        preprocess_setting = None

        if type(target_label) is str:   # convert to TargetLabel class if it is str
            if target_label  == "voiced":
                preprocess_setting = TargetLabel.VOICED
                logger.info("target label to predicted got: \"voiced\"")
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
                
        elif type(target_label) is TargetLabel:
            preprocess_setting = target_label


        # set default setting
        if preprocess_setting == TargetLabel.DEFAULT:
            preprocess_setting = TargetLabel.VOICED   # assume voiced is default
            logger.info("use default target label to predicted: \"voiced\"")
        


        # in the future, may use preprocess_setting to control the preprocess of MEG signal
        self.load_meg_signal(subject, session, task, raw_data_path, preprocess_setting, low_pass_filter, high_pass_filter)
        
        # get X, y from loaded meg signal
        if preprocess_setting == TargetLabel.VOICED:
            self.X = self.phonemes.get_data()
            self.y = self.phonemes.metadata["voiced"].values

        else:
            logger.error(f"for now only \"voiced\" is supported. program exit.")
            exit(0)


        return self.X, self.y
    
    def load_meg_signal(self, subject, until_session, until_task, raw_data_path, setting: TargetLabel,
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

        if setting == TargetLabel.VOICED:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir

        else:
            logger.error(f"target label \"{setting}\" is not supported yet. program exit.")
            exit(0)

        


        # set mne epoch for each session, each task
        # Specify a path to a epoch
        bids_path = mne_bids.BIDSPath(
            subject = f"{subject:02}",     # subject need to be 2-digit str (e.g. "01" to align folder name sub-01)  
            session = str(until_session),
            task = str(until_task),
            datatype = "meg",
            root = raw_data_path
        )

        # --- signal processing --- #
        
        self.meg_signal = MEGSignal(bids_path, setting, low_pass = low_pass_filter, high_pass = high_pass_filter)

        if setting == TargetLabel.VOICED:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir
            self.meg_signal.load_meta(supplementary_meta = ph_info, to_save_csv=False)
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

