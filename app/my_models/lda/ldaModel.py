
import pandas as pd
import numpy as np
from typing import Tuple
from loguru import logger


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score


# custom import
import app.utils.my_utils as util
from app.common.commonSetting import TargetLabel

class MyLDA():
    def __init__(self):
        pass

    def train(self, X: np.ndarray, y: np.ndarray, train_test_ratio)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        y is expected to be binary with value True or False

        Parameters
        -----------
        X (np.ndarray)
            shape (n_window, n_chans, n_times)
        y (np.ndarray)
            shape (n_window,)
        meta (pd.DataFrame)
        to_print_csv (bool): 
            - if True: save prediction resultsas a csv file
        prediction_mode (str): Mode of prediction. Available options are:
            - "collapse": Predict a single probability for each window
            - "each_timepoint": Predict a probability for each time point in each window

        """
        self.train_test_ratio = train_test_ratio
        
        try:
            assert len(X) == len(y)
        except AssertionError:
            logger.error(f"X, y meta is not of same length, len(X)={len(X)}, len(y)={len(y)}")
            return None
        logger.debug(f"in decod, len(x)=len(y)={len(X)}")
        logger.debug(f"X.shape: {X.shape}")
        logger.debug(f"y.shape: {y.shape}")

            

        # convert y to correct dtype before LDA
        y = np.array(y, dtype = bool)   # LDA expects y to be np arr with dtype numeric or bool


        pred_df = None

        # call function for LDA according to prediction_mode
        pred_df = self.predict_each_window(X, y)
      
        return pred_df
        

    
    def predict_each_window(self, X: np.ndarray, y: np.ndarray):
        """
        predict a probability for each window (collapse all timepoints in each window)

        Parameters
        -------------
        X (np.ndarray)
            shape (n_window, n_chans, n_times)
        y (np.ndarray)
        """
        # Initialize the LDA model
        lda = LinearDiscriminantAnalysis()

        # Initialize return values
        pred_df = None
        # Check if X is a 3D array
        if len(X.shape) != 3:
            logger.error(f"X is not a 3D array, X.shape={X.shape}, not capable for predict_each_window")
            return pred_df

        # Check if y is a numpy array of dtype bool or numeric, as this is expected by LDA
        if not (y.dtype == bool or np.issubdtype(y.dtype, np.number)):
            logger.error(f"LDA expects y in dtype bool or numeric only, y.dtype={y.dtype}")
            return pred_df

        n, nchans, ntimes = X.shape
        X = X.reshape(-1, nchans * ntimes)     # collapse nchans and ntimes to 1D
        # should be same as X.reshape(n, -1) [TBC]


        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-self.train_test_ratio), random_state=0)
        scores = []        
        
        
        preds = np.zeros(n)
        logger.debug(f"shape of X_train: {X_train.shape}")
        logger.debug(f"shape of y_train: {y_train.shape}")

        logger.info("start to fit LDA...")
        lda.fit(X_train, y_train)

        logger.debug(f"shape of X_test: {X_test.shape}")
        logger.info(f"prediction is expected to have {X_test.shape[0]} rows")
        logger.info("start to predict...")
        preds = lda.predict(X_test)

        # construct df from prediction result
        pred_df = pd.DataFrame(preds, columns=["prediction"])
        pred_df["ground_truth"] = y_test.tolist()

        
        # logger.info("start to run kfold")
        # try:

        #     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats = 3, random_state=1) # remark: 5 split * 3 repeats = 15 Folds
        #     # n_split = no. of folds
        #     score = cross_val_score(lda, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
        #     scores.append(score)
        #     ret_socres = scores
        #     try:
        #         scores_df = pd.DataFrame(scores, columns=[f"fold_{i}" for i in range(len(scores[0]))])
        #         ret_socres = scores_df
        #     except Exception as err:
        #         logger.error(err)
        #         logger.error("error occur in converting scores to df, returning raw scores")
        #         return pred_df, ret_socres
                
        # except Exception as err:
        #     logger.error(err)
        #     logger.error("error occur in get kfold score, returning None for scores")
        #     return pred_df, None
        
        return pred_df
    
