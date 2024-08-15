
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




if __name__ == "__main__":

   
    # ---  load config --- #


    config_path = Path('./app/config/config_mh.json')
    # config_path = Path("./app/config/train_config.json")
    # config_path = Path('./app/config/my_own_config.json')      # put your own config file here cuz setting of everyone may be different
    
    try:
        with config_path.open('r') as file:
            config = json.load(file)
        training_config = config.get('training', {})
        subject = str(training_config.get('until_subject', "01"))
        session = str(training_config.get('until_session', "0"))
        task = str(training_config.get('until_task', "0"))
        low_pass_filter = training_config.get('preprocess_low_pass', None)
        high_pass_filter = training_config.get('preprocess_high_pass', None)
        training_flow = training_config.get('flow', None)
        target_label = training_config.get('target_label', None)
        dont_kfold_in_lda = training_config.get('dont_kfold_in_lda', None)

        house_keeping_config = config.get('house_keeping', {})
        raw_data_path = house_keeping_config.get('raw_data_path', "DEBUG")
        log_level = house_keeping_config.get('log_level', None)
        result_metrics_save_path = house_keeping_config.get('result_metrics_save_path', None)
        logger.info(f"Execution start according to config: {config_path}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        logger.error(f"config file: {os.path.abspath(config_path)} not exist, use fall back values")
        # Use hardcoded values if the config file is not found or is invalid
        subject = '01'
        session = '0'
        task = '0'
        raw_data_path = './data'
        low_pass_filter = high_pass_filter = training_flow = log_level = result_metrics_save_path = dont_kfold_in_lda = None
        target_label = None
    
    # ----- Set logger ----- #
    util.MyLogger(logger, log_level=log_level, output="console")   # logger comes from loguru logger
    
    # --- wish to redirect error message to loguru logger, but to be developped....
    #sys.stdout = util.StreamToLogger(log_level="INFO", output="console")
    #sys.stderr = util.StreamToLogger(log_level="ERROR", output="console")

    
    
    # ------ Data Getting and Preprocessing ------ #
    if type(target_label) is not str:
        logger.error("target_label is not valid, program exit.")
        exit(0)
    logger.info("start to preprocess data....")
    preprocessor = Preprocessor()
    X, y = preprocessor.get_data(subject, session, task, raw_data_path, target_label, low_pass_filter, high_pass_filter)
    phonemes = preprocessor.get_metadata("phonemes")

    # ---- train model ---- #
    logger.info("start to train...")

    if(training_flow == "nn"):
        logger.info("start to train with model: NN")
        nnRunner = NNModelRunner(X, y)
        nnRunner.train()
        
       
    elif(training_flow == "lda"):
        logger.info("start to train with model: LDA")
        ldaRunner = LdaModelRunner(X, y, phonemes.metadata, dont_kfold=dont_kfold_in_lda)

    else:
        logger.error("undefined training_flow!")

    logger.info("training finished.")

