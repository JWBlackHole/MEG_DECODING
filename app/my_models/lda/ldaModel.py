
import pandas as pd
import numpy as np
from typing import Tuple
from loguru import logger


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score


# custom import
import app.utils.my_utils as util

class MyLDA():
    def __init__(self):
        pass

    def decode_binary(self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, times= None, 
                      to_print_csv = False,
                      prediction_mode = "collapse", 
                      dont_kfold = False)-> Tuple[pd.DataFrame, pd.DataFrame]:
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
        logger.info(f"decod_binary is running with prediction_mode: {prediction_mode}")
        try:
            assert len(X) == len(y) == len(meta)
        except AssertionError:
            logger.error(f"X, y meta is not of same length, len(X)={len(X)}, len(y)={len(y)}, len(meta)={len(meta)}")
            return None
        logger.debug(f"in decod, len(x)=len(y)=len(meta)={len(X)}")
        logger.debug(f"X.shape: {X.shape}")
        logger.debug(f"y.shape: {y.shape}")
        logger.debug(f"meta.shape: {meta.shape}")
        meta = meta.reset_index()

        # convert y to correct dtype before LDA
        y = np.array(y, dtype = bool)   # LDA expects y to be np arr with dtype numeric or bool


        pred_df = scores = None

        # call function for LDA according to prediction_mode
        if (prediction_mode == "collapse"):
            pred_df, scores = self.predict_each_window(X, y, meta, dont_kfold)
        elif (prediction_mode == "each_timepoint"):
            self.predict_each_timepoint(X, y, meta, dont_kfold)
        else:
            logger.error("undefined prediction_mode!")
            return pred_df, scores
        
        if to_print_csv:
            pass # TBC

        return pred_df, scores

    def predict_each_timepoint(self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, dont_kfold=False):
        """
        predict a probability for each time point in each window
        Parameters
        -----------
        X (np.ndarray)
            shape (n_window, n_chans, n_times)
        y (np.ndarray)
            shape (n_window,)
        meta (pd.DataFrame)
        """
        # Initialize the LDA model
        lda = LinearDiscriminantAnalysis()

        # Initialize return values
        pred_df = None
        ret_scores = None

        if len(X.shape) != 3:    # assert X is 3D array
            logger.error(f"X is not 3D array, X.shape={X.shape}, non 3D X is not suported yet")
            return pred_df, ret_scores
        if not (y.dtype == bool or np.issubdtype(y.dtype, np.number)):
            logger.error(f"LDA expects y in dtype bool or numeric only, y.dtype={y.dtype}")
            return pred_df, ret_scores
        # scaler = StandardScaler()         

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        scores = []
        
        n, nchans, ntimes = X.shape
        preds = np.zeros((n, ntimes))


        # predict by my self
        for t in range(2):
            # Scale the data (seems no need)
            # X_train_scaled = scaler.fit_transform(X_train[:, :, t])
            # X_test_scaled = scaler.transform(X_test[:, :, t])

            
            if(t==0):   # only print debug message in 1st interation
                logger.debug(f"shape of X_train[:, :, t]:  {X_train[:, :, t].shape}")
                logger.debug(f"shape of y_train: {y_train.shape}")

            # Fit the LDA model
            lda.fit(X_train[:, :, t], y_train)

            if(t==0):   # only print debug message in 1st interation
                logger.info(f'n_components: {lda.n_components}')           
                logger.debug(f"shpae of X_test: {X_test[:, :, t].shape}")

            # Predict probabilities by projected X form the fitted LD
            preds[:, t] = lda.predict(X_test[:, :, t])

        pred_df = pd.DataFrame(preds, columns=[f"timepoint_{t}" for t in range(ntimes)])
        pred_df["ground_truth"] = y_test.tolist()   # there is only one column of y because all timepoints correspond to same event
            
        
        # below still need debug
            
        # evaluate by k fold

        ret_socres = None
        if dont_kfold:
            return pred_df, ret_socres

        try:
            for t in range(2):
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats = 3, random_state=1)
                # n_split = no. of folds
                score = cross_val_score(lda, X[:, :, t], y, scoring="accuracy", cv=cv, n_jobs=-1)
                scores.append(score)
                ret_socres = scores
                try:
                    scores_df = pd.DataFrame(scores, columns=[f"fold_{i}" for i in range(len(scores))])
                    ret_socres = scores_df
                except Exception:
                    logger.error("error occur in converting scores to df")
                    
        except Exception:
            logger.error("error occur in get kfold score, returning None for scores")
            return pred_df, None
        return pred_df, ret_socres
    
    def predict_each_window(self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, dont_kfold = False):
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
        ret_socres = None

        # Check if X is a 3D array
        if len(X.shape) != 3:
            logger.error(f"X is not a 3D array, X.shape={X.shape}, not capable for predict_each_window")
            return pred_df, ret_socres

        # Check if y is a numpy array of dtype bool or numeric, as this is expected by LDA
        if not (y.dtype == bool or np.issubdtype(y.dtype, np.number)):
            logger.error(f"LDA expects y in dtype bool or numeric only, y.dtype={y.dtype}")
            return pred_df, ret_socres

        n, nchans, ntimes = X.shape
        X = X.reshape(-1, nchans * ntimes)     # collapse nchans and ntimes to 1D
        # should be same as X.reshape(n, -1) [TBC]


        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        scores = []        
        
        
        preds = np.zeros(n)
        logger.debug(f"shape of X_train: {X_train.shape}")
        logger.debug(f"shape of y_train: {y_train.shape}")

        lda.fit(X_train, y_train)

        logger.debug(f"shape of X_test: {X_test.shape}")
        preds = lda.predict(X_test)

        # construct df from prediction result
        pred_df = pd.DataFrame(preds, columns=["prediction"])
        pred_df["ground_truth"] = y_test.tolist()

        if dont_kfold:
            return pred_df, ret_socres
        
        logger.debug("start to run kfold")
        try:

            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats = 3, random_state=1) # remark: 5 split * 3 repeats = 15 Folds
            # n_split = no. of folds
            score = cross_val_score(lda, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
            scores.append(score)
            ret_socres = scores
            try:
                scores_df = pd.DataFrame(scores, columns=[f"fold_{i}" for i in range(len(scores[0]))])
                ret_socres = scores_df
            except Exception as err:
                logger.error(err)
                logger.error("error occur in converting scores to df, returning raw scores")
                return pred_df, ret_socres
                
        except Exception as err:
            logger.error(err)
            logger.error("error occur in get kfold score, returning None for scores")
            return pred_df, None
        
        return pred_df, ret_socres
    
