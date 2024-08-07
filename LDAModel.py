import mne
import mne_bids
from mne    import Epochs
from mne.io import Raw
import pandas as pd
import numpy as np
import os
import sys
from loguru import logger

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


# cutom import
import my_utils as util

class MyLDA():
    def __init__(self, X, y):
        pass

    def decode_binary(X, y, meta, times):
        """
        y is expected to be binary with value True or False
        """
        to_print_csv  =  True
        logger.info("decod_binary is running")
        assert len(X) == len(y) == len(meta)
        logger.debug(f"in decod, len(x)=len(y)=len(meta)={len(X)}")
        meta = meta.reset_index()

        # Initialize the scaler and the LDA model
        scaler = StandardScaler()
        lda = LinearDiscriminantAnalysis()

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        
        n, nchans, ntimes = X.shape
        preds = np.zeros((len(X_test), ntimes, 2))  # Adjusted to store probabilities for both classes

        for t in range(ntimes):
            # Scale the data
            X_train_scaled = scaler.fit_transform(X_train[:, :, t])
            X_test_scaled = scaler.transform(X_test[:, :, t])

            # Fit the LDA model
            lda.fit(X_train_scaled, y_train)
            logger.info(f'n_components: {lda.n_components}')
            exit(0)


            # Predict probabilities by projected X form the fitted LD
            preds[:, t, :] = lda.predict_proba(X_test_scaled)

        return preds