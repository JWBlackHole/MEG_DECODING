
import pandas as pd
import numpy as np
from loguru import logger

# cutom import
from app.my_models.lda.ldaModel import MyLDA
import app.utils.my_utils as util


class LdaModelRunner():
    def __init__(self, X: np.ndarray, y: np.ndarray, train_test_ratio:float=0.8, to_save_csv = False, 
                 option: dict = {},
                 balance_train_data_lda=False, 
                    balance_test_data_lda=False,
                    res_path: str=None) -> None:
        
        logger.info("start to train with model: LDA")

        if isinstance(option, dict) and "result_description" in option:
            self.result_description = option["result_description"]
        else:
            self.result_description = "LDA"

        lda_model = MyLDA()
        
        prediction_df = lda_model.train(X, y, train_test_ratio, balance_train_data_lda, balance_test_data_lda)

        logger.debug(f"type of predictions (returned from lda model): {type(prediction_df)}")


        # calculate result metrics
        prediction_df = util.add_comparison_column(prediction_df)
        if to_save_csv:
            try:
                prediction_df.to_csv(util.get_unique_file_name("voiced_prediction_lda.csv", "./results/lda/csv"))
            except Exception as err:
                logger.error(err)
                logger.error("fail to output csv, skipping output csv")

        util.get_eval_metrics(prediction_df, 
                              file_name="metrics_LDA", save_path=res_path, 
                              description_str=self.result_description)
        
        logger.info("LDA model runner finished.")
        return