
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from loguru import logger


# custom import
from app.signal.preprocessor import Preprocessor
from app.signal.torchMegLoader import TorchMegLoader
from app.signal.newTorchMegLoader import MegDataIterator
from app.my_models.nn.nnModelRunner import NNModelRunner
from app.my_models.lda.ldaModelRunner import LdaModelRunner
from app.my_models.svm.svmModel import svmModel # new added
# from app.my_models.cnn.cnnModel import cnnModel # new added
import app.utils.my_utils as util
from app.common.commonSetting import TargetLabel
from app.signal.sensorTools import plot_sensor
from app.my_models.cnn_torch.torchCnnModelRunner import SimpleTorchCNNModelRunner



if __name__ == "__main__":
    # ---  load config --- #


    config_path = Path('./app/config/config_jw.json')
    # config_path = Path('./app/config/config_mh.json')
    # config_path = Path("./app/config/train_config.json")
    # config_path = Path('./app/config/my_own_config.json') # put your own config file here cuz setting of everyone may be different
    
    try:
        with config_path.open('r') as file:
            config = json.load(file)
        training_config   = config.get('training', {})
        subject           = training_config.get('until_subject', 1)    # subject start from 1
        until_session     = training_config.get('until_session',0)     # session start from 0
        until_task        = training_config.get('until_task', 0)       # task start from 0
        meg_tmin          = training_config.get('meg_tmin', None) 
        meg_tmax          = training_config.get('meg_tmax', None) 
        meg_decim          = training_config.get('meg_decim', None) 
        load_batch_size    = training_config.get('load_batch_size', None) 
        low_pass_filter   = training_config.get('preprocess_low_pass', None)
        high_pass_filter  = training_config.get('preprocess_high_pass', None)
        training_flow     = training_config.get('flow', None)
        target_label      = training_config.get('target_label', None)
        dont_kfold_in_lda = training_config.get('dont_kfold_in_lda', None)
        nn_total_epoch    = training_config.get('nn_total_epoch', None)

        house_keeping_config    = config.get('house_keeping', {})
        raw_data_path           = house_keeping_config.get('raw_data_path', None)
        log_level               = house_keeping_config.get('log_level', "DEBUG")
        result_metrics_save_path = house_keeping_config.get('result_metrics_save_path', None)
        to_print_interim_csv     = house_keeping_config.get('to_print_interim_csv', False)
        num_event_to_plot         =house_keeping_config.get('num_event_to_plot', 1)

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
        nn_total_epoch = meg_tmin = meg_tmax = meg_decim = num_event_to_plot = None

    meg_param={
        "tmin": meg_tmin,
        "tmax": meg_tmax,
        "decim": meg_decim,
        "low_pass": low_pass_filter,
        "high_pass": high_pass_filter
    }

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
    
    # logger.warning(f"currently only training for one subject is supported. Will train for subject {subject:02}")

    logger.info(f"target label to predicted got: \"{target_label}\"")
    if target_label  == "voiced":
        target_label = TargetLabel.VOICED_PHONEME
        
    elif target_label  == "word_freq":
        target_label = TargetLabel.WORD_FREQ
    elif target_label == "is_sound":
        target_label = TargetLabel.IS_SOUND
    elif target_label == "plot_word":
        target_label  = TargetLabel.PLOT_WORD_ONSET
    elif target_label == "word_onset":
        target_label = TargetLabel.WORD_ONSET

    elif target_label in ["is voiced", "is_voiced", "phoneme", "phonemes", "voice", "is voice", "is_voice"]:
        logger.error(f"target_label is not supported, do you mean \"voiced\"?")
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
    
    if (training_flow== "cnn_batch"):
        # new code
        megDataIter = MegDataIterator(subject, until_session, until_task, raw_data_path, target_label, to_print_interim_csv, meg_param)
        ntimes = megDataIter.cal_ntimes()
        torch_cnn_model = SimpleTorchCNNModelRunner(megDataIter, 208, ntimes, p_drop_true=0.572)

        # old code
        # logger.info("start to train with model: CNN (load data by batch)")
        # megData = TorchMegLoader(subject, until_session, until_task, raw_data_path, target_label, 
        #                           to_print_interim_csv, meg_param, load_batch_size)
        
        # nchans, ntimes = megData.get_signal_dim()
        
        # torch_cnn_model = SimpleTorchCNNModelRunner(megData, nchans, ntimes, p_drop_true=0.572)
        torch_cnn_model.train(epochs=2, batch_size=1, learning_rate=0.001)
        logger.info("cnn training finished.")
        exit(0)

    
    preprocessor = Preprocessor(meg_param)

    if training_flow == "plot_sensor":
        logger.info("plotting sensor, not proceeding for training...")
        preprocessor.plot_sensor_topo(raw_data_path)
        logger.info("finish plotting, program exit")
        exit(0)
    
    
        

    X, y = preprocessor.prepare_X_y(subject, until_session, until_task, raw_data_path, target_label, 
                                 low_pass_filter, high_pass_filter, to_print_interim_csv)
   
    # X=X[:100]
    # y=y[:100]
    # X, y is for the subject for all sessions to `until_session` and all tasks to `until_task`
    # X is the "features" 
    # y is the label 
    if(training_flow=="debug"):
        epochs = preprocessor.get_concated_epochs()
        logger.info(f"type of get_concated_epochs: {type(epochs)}") # <class 'mne.epochs.EpochsArray'>
        try:
            logger.info(f"len of get_concated_epochs: {len(epochs)}")   # len of all events
            ep = epochs[0]
            logger.info(f"epochs[0] type: {type(ep)}")      # epochs[0] type: <class 'mne.epochs.EpochsArray'>
            try:
                logger.info(f"len of epochs[0]: {len(ep)}") # len of epochs[0]: 1
            except Exception as err:
                logger.error(err)
            exit(0)
        except Exception as err:
            logger.error(err)
        exit(0)
            
    if(target_label == TargetLabel.VOICED_PHONEME):
        phonemes_epochs = preprocessor.get_metadata("phonemes")     # get the epochs only considering is phoneme voiced / not voiced
        phoneme_meta = phonemes_epochs.metadata
    else:
        phoneme_meta = None

    if to_print_interim_csv:
        whole_meta_table = preprocessor.get_concated_metadata() # get the df of metadata of all sessions, all tasks
        whole_meta_table.to_csv(util.get_unique_file_name("whole_meta.csv", "./results"))

    # ---- train model ---- #
    logger.info("start to train...")

    if(training_flow == "nn"):
        logger.info("start to train with model: NN")
        
        megDataIter = MegDataIterator(subject, until_session, until_task, raw_data_path, target_label, to_print_interim_csv, meg_param)
        ntimes = megDataIter.cal_ntimes()
        
        nnRunner = NNModelRunner(megDataIter, target_label)
        nnRunner.train(epochs = 1000, batch_size = 512, lr = 0.001)
        
       
    elif(training_flow == "lda"):
        logger.info("start to train with model: LDA")

        ldaRunner = LdaModelRunner(X, y, phoneme_meta, target_label, dont_kfold=dont_kfold_in_lda, to_save_csv=True)

    elif(training_flow == "svm"):
        logger.info("start to train with model: SVM")
        svmRunner = svmModel(X, y, target_label)
        
        # import cProfile
        # import pstats

        # profiler = cProfile.Profile()
        # profiler.enable()

        # try:
        #     # 將結果保存到檔案
        #     cProfile.run('svmRunner.train()', 'profile_results.prof')
            
        #     # 讀取分析結果
        #     p = pstats.Stats('profile_results.prof')

        #     # 排序和查看結果
        #     p.sort_stats('cumulative').print_stats(10)  # 顯示前10個最耗時的函數
        # finally:
        #     profiler.disable()
        #     profiler.print_stats(sort='cumulative')

        svmRunner.train()

    # elif(training_flow == "cnn"):
    #     logger.info("start to train with model: CNN")
    #     cnnRunner = cnnModel(X, y)
    #     cnnRunner.train()

    elif (training_flow == "cnn"):
        torch_cnn_model = SimpleTorchCNNModelRunner(X, y)
        torch_cnn_model.train(epochs=2, batch_size=1, learning_rate=0.001)

    elif (training_flow == "plot_word_evo"):
        
        # plot each event
        preprocessor.plot_n_events_evo("is_word", num_event_to_plot, True)

        # plot average of all event
        #preprocessor.plot_evoked_response("is_word")




    else:
        raise NotImplementedError
    

    logger.info("training finished.")
