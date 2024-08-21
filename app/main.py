
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from loguru import logger


# custom import
from app.signal.preprocessor import Preprocessor
from app.my_models.nn.nnModelRunner import NNModelRunner
from app.my_models.lda.ldaModelRunner import LdaModelRunner
import app.utils.my_utils as util
from app.common.commonSetting import TargetLabel




if __name__ == "__main__":

   
    # ---  load config --- #


    config_path = Path('./app/config/config_mh.json')
    # config_path = Path("./app/config/train_config.json")
    # config_path = Path('./app/config/my_own_config.json')      # put your own config file here cuz setting of everyone may be different
    
    try:
        with config_path.open('r') as file:
            config = json.load(file)
        training_config = config.get('training', {})
        subject = training_config.get('until_subject', 1)       # subject start from 1
        until_session = training_config.get('until_session',0)  # session start from 0
        until_task = training_config.get('until_task', 0)       # task start from 0
        low_pass_filter = training_config.get('preprocess_low_pass', None)
        high_pass_filter = training_config.get('preprocess_high_pass', None)
        training_flow = training_config.get('flow', None)
        target_label = training_config.get('target_label', None)
        dont_kfold_in_lda = training_config.get('dont_kfold_in_lda', None)

        house_keeping_config = config.get('house_keeping', {})
        raw_data_path = house_keeping_config.get('raw_data_path', "DEBUG")
        log_level = house_keeping_config.get('log_level', None)
        result_metrics_save_path = house_keeping_config.get('result_metrics_save_path', None)
        to_print_interim_csv = house_keeping_config.get('to_print_interim_csv', False)
        logger.info(f"Execution start according to config: {config_path}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        logger.error(f"config file: {os.path.abspath(config_path)} not exist, use fall back values")
        # Use hardcoded values if the config file is not found or is invalid
        subject = 1
        until_session = 0
        until_task = 0
        raw_data_path = './data'
        low_pass_filter = high_pass_filter = training_flow = log_level = result_metrics_save_path = dont_kfold_in_lda = None
        target_label = None
        to_print_interim_csv = None
    
    # ----- Set logger ----- #
    util.MyLogger(logger, log_level=log_level, output="console")   # logger comes from loguru logger
    
    # --- wish to redirect error message to loguru logger, but to be developped....
    #sys.stdout = util.StreamToLogger(log_level="INFO", output="console")
    #sys.stderr = util.StreamToLogger(log_level="ERROR", output="console")


    # ------  target label checking  ----  #

    # target label is the label to predict in the training
    # this should affect the preprocessing and the training and predcition process


    if type(target_label) is not str:
        logger.error("target_label is not valid, program exit.")
        exit(0)
    
    logger.warning(f"currently only training for one subject is supported. Will train for subject {subject:02}")

    logger.info(f"target label to predicted got: \"{target_label}\"")
    if target_label  == "voiced":
        target_label = TargetLabel.VOICED_PHONEME
        
    elif target_label  == "is_word":
        target_label = TargetLabel.WORD_FREQ
    elif target_label == "is_sound":
        target_label = TargetLabel.IS_SOUND

    elif target_label in ["is voiced", "is_voiced", "phoneme", "phonemes", "voice", "is voice", "is_voice"]:
        logger.error(f"target_label is not supported, do you mean \"voiced\"?")
        logger.warning("assume default flow")
        target_label = TargetLabel.DEFAULT
    elif target_label in ["is word", "word"]:
        logger.error(f"target_label is not supported, do you mean \"is_word\"?")
        logger.warning("assume default flow")
        target_label = TargetLabel.DEFAULT
    elif target_label in ["is sound", "sound", "have sound", "have_sound"]:
        logger.error(f"target_label is not supported, do you mean \"is_sound\"?")
        logger.warning("assume default flow")
        target_label = TargetLabel.DEFAULT
    else:
        logger.error(f"target_label not recognised, assume default flow")
        logger.warning("assume default flow")
        target_label = TargetLabel.DEFAULT
    

    # ------ Data Getting and Preprocessing ------ #
    
    logger.info("start to preprocess data....")
    preprocessor = Preprocessor()


    X, y = preprocessor.get_data(subject, until_session, until_task, raw_data_path, target_label, low_pass_filter, high_pass_filter)
    # X, y is for the subject for all sessions to `until_session` and all tasks to `until_task`
    # X is the "features" 
    # y is the label 
    phonemes_epochs = preprocessor.get_metadata("phonemes")     # get the epochs only considering is phoneme voiced / not voiced

    if to_print_interim_csv:
        whole_meta_table = preprocessor.get_concated_metadata() # get the df of metadata of all sessions, all tasks
        whole_meta_table.to_csv(util.get_unique_file_name("whole_meta.csv", "./results"))

    # ---- train model ---- #
    logger.info("start to train...")

    if(training_flow == "nn"):
        logger.info("start to train with model: NN")
        nnRunner = NNModelRunner(X, y, target_label)
        nnRunner.train(100)
        
       
    elif(training_flow == "lda"):
        logger.info("start to train with model: LDA")
        ldaRunner = LdaModelRunner(X, y, phonemes_epochs.metadata, target_label, dont_kfold=dont_kfold_in_lda, to_save_csv=True)

    else:
        logger.error("undefined training_flow!")

    logger.info("training finished.")

