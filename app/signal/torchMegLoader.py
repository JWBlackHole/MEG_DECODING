"""
for MEG loader for torch
competible with torch.utils.data.Dataset so that data can be loaded by batch during training in cnn
"""


import pandas as pd
import numpy as np

import mne_bids
from mne import Epochs, concatenate_epochs, EpochsArray
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
 
from loguru import logger

# custom import
from app.signal.megSignal import MEGSignal
from app.common.commonSetting import TargetLabel
import app.utils.my_utils as util

class TorchMegLoader(Dataset):
    def __init__(self, subject, until_session, until_task, raw_data_path, target_label,
                  to_print_interim_csv=False, 
                  meg_param:dict={"tmin":None, "tmax":None, "decim":None, "low_pass": None, "high_pass":None}):
        logger.info("TorchMegLoader is inited")
        self.subjcet:int = subject
        self.until_session:int = until_session
        self.until_task:int = until_task
        self.raw_data_path = raw_data_path
        self.concated_epochs: EpochsArray | None = None
        self.to_print_interim_csv = to_print_interim_csv
        self.meg_param: dict = meg_param
        self.nchans:int = None
        self.ntimes:int = None
        self.getitem_debug_printed=False
        self.num_batch: int = None
        self.voiced_phoneme_epoch = None

        required_keys = ["tmin", "tmax", "decim", "low_pass", "high_pass"]
        for key in required_keys:
            if key not in self.meg_param:
                for k, v in self.meg_param.items():
                    print(f"{k}: {v}")
                raise ValueError(f"meg_param is missing required key: {key}")


        
        # ---   target label checking ------ #
        if type(target_label) is str:   # convert to TargetLabel class if it is str
            if target_label  == "voiced":
                target_label = TargetLabel.VOICED_PHONEME
                logger.info("target label to predicted got: \"voiced\"")
            elif target_label  == "word_freq":
                target_label = TargetLabel.WORD_FREQ
            elif target_label  == "word_onset":
                target_label = TargetLabel.WORD_ONSET
            elif target_label == "is_sound":
                target_label = TargetLabel.IS_SOUND
            elif target_label == "plot_word":
                target_label = TargetLabel.PLOT_WORD_ONSET

            #     preprocess_setting = TargetLabel.DEFAULT
            else:
                raise NotImplementedError
                
        elif type(target_label) is TargetLabel:
            target_label = target_label

        
        # set default setting
        if target_label == TargetLabel.DEFAULT:
            target_label = TargetLabel.VOICED_PHONEME   # assume voiced is default
            logger.info("use default target label to predicted: \"voiced\"")

        self.target_label = target_label
        
        # -----  prepare concatenated epoch
        self.load_all_epochs(subject, until_session, until_task, raw_data_path, target_label)

        if self.target_label == TargetLabel.VOICED_PHONEME:
            self.voiced_phoneme_epoch = self.create_batch_id(self.concated_epochs["not is_word"], batch_size=100)


    def get_voiced_phoneme_epoch(self):
        if self.voiced_phoneme_epoch is not None:
            return self.voiced_phoneme_epoch
        else:
            raise ValueError("self.voiced_phoneme_epoch is not properly set!")
        
    def __len__(self):
        """
        this method is required by Torch Dataset
        this define the len of data
        """
        logger.debug("__len__ of loader ran")
        # notes
        # len(self.concated_epochs) = total #event in all tasks
        # len(epoch[0]) also is  total #event in all tasks
        # try:
        #     epoch = self.get_voiced_phoneme_epoch()
        #     logger.debug(f"epoch : {epoch}")
        #     logger.debug(f"len(epoch): {len(epoch)}")
        #     try:
        #         logger.debug(f"epoch[0]: {epoch[0]}")
        #         try:
        #             logger.debug(f"len(epoch[0])= {len(epoch[0])}")
        #         except Exception as e:
        #             logger.error(e)
        #     except Exception as e:
        #         logger.error(e)

        #     return len(epoch)  # Total number of epochs
        # except Exception as err:
        #     logger.error(err)
        #     logger.error(f"type of self.concated_epochs: {type(self.concated_epochs)}")

        if self.target_label == TargetLabel.VOICED_PHONEME:
            assert self.voiced_phoneme_epoch is not None

            if (not ("batch_id" in self.voiced_phoneme_epoch.metadata.columns.values.tolist())):
                logger.error("cols:")
                for col in self.concated_epochs.metadata.columns.values.tolist():
                    print(col)
                
                logger.error("no batch_id ! (which shld have been set in create_batch_id called by __init__)")
                raise ValueError
            elif not self.num_batch:
                logger.error(f"num_batch: {self.num_batch}")
                logger.error("num_batch not set! (which shld have been set in create_batch_id called by __init__)")
                raise ValueError
            else:
                return self.num_batch


        # if isinstance(self.subjcet, int) and isinstance(self.until_session, int) and isinstance(self.until_task, int):
        #     num = self.subjcet * (self.until_session+1) * (self.until_task+1)
        #     logger.info(f"len of data: {num}")
        #     return num
        # else:
        #     logger.error(f"type subject: {type(self.subjcet)},type until session: {type(self.until_session)}, type until task: {type(self.until_task)}")
        #     logger.error("not all int!")
        #     raise ValueError

    
    def __getitem__(self, idx):
        """
        Load a single epoch (and corresponding label) into memory.
        here is where the data *actually* loaded

        this method is required by Torch Dataset
        this define what is each data sample

        everything applied to data should be run here
        """
        #if not self.getitem_debug_printed:
        logger.debug("__getitem__ of loader is called")
        verbose = True
        # Load the specific epoch and its corresponding metadata
        if self.target_label == TargetLabel.VOICED_PHONEME:

            epoch = self.get_voiced_phoneme_epoch()

            epoch = self.extract_epochs_by_id(epoch, idx)

            if verbose and not self.getitem_debug_printed:
                logger.debug(f"idx= {idx}, epoch: {epoch}, type: {type(epoch)}")

            epoch.apply_baseline(verbose="WARNING")  # Apply baseline correction
            X = epoch.get_data(copy=True)   # Get the data as a 3D array (n_channels, n_times)
            y = epoch.metadata["voiced"].values

            # clip to 95 percentile for twice
            th = np.percentile(np.abs(X), 95)
            X = np.clip(X, -th, th)
            th = np.percentile(np.abs(X), 95)
            X = np.clip(X, -th, th)
            
            if(verbose and not self.getitem_debug_printed):
                logger.debug(f"type of X: {type(X)}")   # numpy.ndarray
                try:
                    logger.debug(f"X.shape: {X.shape}") # (1, 208, 41)
                except Exception as err:
                    logger.error(err)
                logger.debug(f"type of y: {type(y)}")   # numpy.ndarray
                try:
                    logger.debug(f"y.shape: {y.shape}") # (1,)
                except Exception as err:
                    logger.error(err)
         

            # Convert to PyTorch tensors

            # investigation
            X_tensor = torch.tensor(X, dtype=torch.float32)  # torch.Size([1, 208, 41])
            if verbose and not self.getitem_debug_printed:
                logger.debug(f"not squeezed, X_tensor shape: {X_tensor.shape}")
            X_tensor = X_tensor.squeeze(0)  #torch.Size([208, 41])
            if verbose and not self.getitem_debug_printed:
                logger.debug(f"squeeze(0), X_tensor shape: {X_tensor.shape}")   # 
            X_tensor = X_tensor.unsqueeze(0)    # torch.Size([1, 208, 41])
            if verbose and not self.getitem_debug_printed:
                logger.debug(f"unsqueeze(0) 1 time, X_tensor shape: {X_tensor.shape}")
            X_tensor = X_tensor.unsqueeze(0)    #torch.Size([1, 1, 208, 41])
            if verbose  and not self.getitem_debug_printed:
                logger.debug(f"unsqueeze(0) 2 times, X_tensor shape: {X_tensor.shape}")



            '''
            original: X.shape: (#event, 208, 41)
            x_tensor need to be:(#event, 1, 208, 41)
            '''

            X_tensor = torch.tensor(X, dtype=torch.float32)
            if verbose and not self.getitem_debug_printed:
                logger.debug(f"X_tensor shape: {X_tensor.shape}")
            
            X_tensor = X_tensor.unsqueeze(1)
            if verbose and not self.getitem_debug_printed:
                logger.debug(f"after unsqeeze(1), X_tensor shape: {X_tensor.shape}")


            y_tensor = torch.tensor(y.astype(int), dtype=torch.float32) 
            # BCE loss expect dtype float32
            # if change to other loss function, may need to look for what dtype expected
            
            self.getitem_debug_printed = True
            return X_tensor, y_tensor
        else:
            raise NotImplementedError
        
    def get_signal_dim(self):
        if isinstance(self.nchans, int) and isinstance(self.ntimes, int):
            return self.nchans, self.ntimes
        else:
            logger.error("nchans, ntimes is not of int!")
            logger.error(f"nchans: {self.nchans}, ntimes: {self.ntimes}")
            raise ValueError

    
    def load_all_epochs(self, subject, until_session, until_task, raw_data_path, setting):
        """
        load epochs of all sessions, all tasks for one subject to self.concated_epochs
        """

        epochs_list = []    # list for save epochs of all sesions, all tasks

        logger.debug(f"raw data path: {raw_data_path}")

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir


        
        # load epochs for each session each task
        glob_id = 0
        for session in range(until_session + 1):
            for task in range(until_task + 1):
                cur_epochs = self.load_epochs_one_task(subject, session, task, raw_data_path, setting)
                
                # make columns task and session to record the current handling task and session
                cur_epochs.metadata["task"] = task
                cur_epochs.metadata["session"] = session
                cur_epochs.metadata["subject"] = subject
                #cur_epochs.metadata["global_id"] = glob_id  # to identify each task
                epochs_list.append(cur_epochs)
                glob_id +=1
        
        if not len(epochs_list):
            logger.error("error in loading all epochs, program exit.")
            raise ValueError

        self.concated_epochs = concatenate_epochs(epochs_list)
        logger.debug(f"self.concated_epochs is set, type: {type(self.concated_epochs)}")
        return self.concated_epochs
    

    def load_epochs_one_task(self, subject, session, task, raw_data_path, setting: TargetLabel) -> None:
        """
        prepare mne epoch of one task of one session one subject
        """



        logger.debug(f"raw data path: {raw_data_path}")

        

        # --- signal processing --- #

        signal_handler = MEGSignal(             # must set preload=False, this means only load data when accessed [MUST !!!] 
            setting, 
            low_pass=self.meg_param["low_pass"] if self.meg_param["low_pass"] else None, 
            high_pass=self.meg_param["high_pass"] if self.meg_param["high_pass"] else None,
            to_print_interim_csv=self.to_print_interim_csv if self.to_print_interim_csv else None,
            preload=False, 
            tmin=self.meg_param["tmin"] if self.meg_param["tmin"] else None,
            tmax=self.meg_param["tmax"] if self.meg_param["tmax"] else None,
            decim=self.meg_param["decim"] if self.meg_param["decim"] else None
        ) 
        
        self.nchans, self.ntimes = signal_handler.get_nchans_ntimes()

        # set mne epoch for each session, each task
        # Specify a path to a epoch
        bids_path = mne_bids.BIDSPath(
            subject = f"{subject:02}",     # subject need to be 2-digit str (e.g. "01" to align folder name sub-01)  
            session = str(session),
            task = str(task),
            datatype = "meg",
            root = raw_data_path
        )

        if setting == TargetLabel.VOICED_PHONEME:
            ph_info:pd.DataFrame = pd.read_csv("./phoneme_data/phoneme_info.csv")   # file path is relative to root dir
            return signal_handler.prepare_one_epochs(bids_path, supplementary_meta = ph_info)

        elif setting in [TargetLabel.PLOT_WORD_ONSET,  TargetLabel.WORD_FREQ] :
            return signal_handler.prepare_one_epochs(bids_path, None)
        
        return
    
    def get_concated_epochs(self)-> Epochs:
        if self.concated_epochs is not None:
            return self.concated_epochs
        else:
            raise ValueError
    
    def create_batch_id(self, epoch, batch_size):
        """
        Adds a 'batch_id' column to the epoch

        Each batch of size `batch_size` is assigned a unique ID. Rows are grouped into batches,
        Parameters:
        -----------
        batch_size : int
        The size of each batch. Each batch will contain `batch_size` rows, and each batch
        will be assigned a unique ID.

        Returns:
        --------
        None
        """
        logger.info("create batch id")
         
        # Calculate the number of complete batches
        num_complete_batches =  len(epoch) // batch_size    # //= divde then take floor
        self.num_batch = int(num_complete_batches)
        logger.debug(f"num_batch: {self.num_batch}")
        epoch = epoch[:num_complete_batches * batch_size]    # drop last part to ensure #event is multiple of batch size
                                                            # because tensor expect each batch to have same dimension 
        # Create batch IDs
        batch_ids = np.repeat(np.arange(num_complete_batches), batch_size) 
        # this create arr like [0,0,0,..1,1,1...] each repeat batch_size times
        
        # Assign batch IDs to the DataFrame
        epoch.metadata['batch_id'] = batch_ids
        return epoch


    def extract_epochs_by_id(self, epochs, id):
        # Ensure the metadata exists
        if epochs.metadata is None:
            raise ValueError("No metadata found. Make sure metadata was properly assigned.")
        
        # Filter the epochs based on task, session, and subject
        filtered_epochs = epochs[
            (epochs.metadata["batch_id"] == id) 
        ]
        if not self.getitem_debug_printed:
            logger.debug(filtered_epochs)
        
        # Return the filtered epochs
        return filtered_epochs