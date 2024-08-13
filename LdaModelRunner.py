
import pandas as pd
import numpy as np

import json
from pathlib import Path
import sys
from loguru import logger


from LDAModel import MyLDA
import my_utils as util


class LdaModelRunner():
    def __init__(self, X, y, meta, dont_kfold = False) -> None:
        
        logger.info("start to train with model: LDA")


        lda_model = MyLDA()
        
        prediction_df , scores = lda_model.decode_binary(X, y, meta, dont_kfold=dont_kfold)
        logger.debug(f"type of predictions (returned from model): {type(prediction_df)}")
        logger.debug(f"type of scores  (returned from model): {type(scores)}")
        try:
            prediction_df.to_csv(util.get_unique_file_name("voiced_prediction_t=1.csv", "./result"))
        except Exception as err:
            logger.error(err)
            logger.error("fail to output csv, skipping output csv")
        print("scores (returned from model):")
        print(scores)

        # calculate metrics
        prediction_df = util.add_comparison_column(prediction_df)
        util.get_eval_metrics(prediction_df, 
                              file_name="metrics_LDA", save_path="./result", 
                              description_str="LDA sub 1 task 1 ses 1")