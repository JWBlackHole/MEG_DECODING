import os
import sys
from loguru import logger
from datetime import date
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

def get_unique_file_name(file_name: str, dir: str = "./", verbose: bool = True):
    """Get a unique file name in a directory for saving file to avoid overwriting.
    example dir: "./results" (no need ending slash)
    
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    name, ext = os.path.splitext(file_name)
    i = 0
    while True:
        file_path = dir + "/" + f"{name}_{i}{ext}"
        if not os.path.exists(file_path):
            if(verbose):
                logger.info(f"Saving file to {file_path}...")
            return file_path
        i += 1

def add_comparison_column(pred_df: pd.DataFrame)->pd.DataFrame:
        """
        add column "TP/FP/TN/FN" from column "prediction" and "ground_truth"
        expected to have column: "prediction", "ground_truth"

        Parameters
        ---------
        pred_df: 
            prediction dataframe (must have columns named "prediction", "ground_truth")
        """
        if not all([col in pred_df.columns for col in ["prediction", "ground_truth"]]):
            logger.error("prediction dataframe must have columns named 'prediction' and 'ground_truth' in order to add comparison column. return original dataframe")
            return pred_df
        
        pred_df['TP/FP/TN/FN'] = pred_df.apply(
            lambda row: 'TP' if row['ground_truth'] and row['prediction'] else
                'FP' if not row['ground_truth'] and row['prediction'] else
                'TN' if not row['ground_truth'] and not row['prediction'] else
                'FN',
            axis=1
        )
        return pred_df
    
    
def get_eval_metrics(pred_df: pd.DataFrame, file_name: str="metrics", save_path: str = "./", description_str: str="") -> dict:
    """
    calculate evaluation matrics from column "TP/FP/TN/FN", then save matrics as .json

    Parameters
    ---------
    pred_df: 
        prediction dataframe (must have columns named "TP/FP/TN/FN")
    file_name: 
        file name of metrics (does not need file extension)
    save_path:
        dir to save metrics (no need ending slash)
    description_str:
        description str to put in metrics (allow to identify config of the training)
    
    """
    tp = len(pred_df[pred_df['TP/FP/TN/FN'] == 'TP'])
    fp = len(pred_df[pred_df['TP/FP/TN/FN'] == 'FP'])
    tn = len(pred_df[pred_df['TP/FP/TN/FN'] == 'TN'])
    fn = len(pred_df[pred_df['TP/FP/TN/FN'] == 'FN'])

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0       # 實際為True 的之中有多少被predcit出來
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0      # predcit 成True 的之中有多少確實是True
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0    # 猜中的比例
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'description': description_str,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

    
    # Save metrics to JSON 
    if ".json" not in file_name:
        file_name += ".json"
    file_path = get_unique_file_name(file_name, save_path)
    if file_path is not None:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        print("Error in saving metrics, skip saving")

    return metrics

def plot_loss_accu_across_epoch(train_losses: list, train_acc: list, test_losses: list, test_acc: list, total_epoch: int, save_path: str):
    """
    Plot training and test loss and accuracy across epochs.
    """
    verbose = False
    if verbose:
        logger.debug(f"len of train_losses: {len(train_losses)}")
        logger.debug(f"len of train_acc: {len(train_acc)}")
        logger.debug(f"len of test_losses: {len(test_losses)}")
        logger.debug(f"len of test_acc: {len(test_acc)}")
        epochs = range(total_epoch)

        logger.debug("train loss:")
        print([i for i in train_losses])
        logger.debug("train accuracies:")
        print([i for i in train_acc])
    
    plt.figure(figsize=(12, 10))

    # Subplot for losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, test_losses, label='Test Loss', color='dodgerblue',linewidth=2)
    plt.plot(epochs, train_losses, label='Training Loss', color='choclate', linestyle='--', linewidth=1.5, alpha=0.9)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss vs. Epoch')
    plt.legend()

    # Subplot for accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs, test_acc, label='Test Accuracy', color='dodgerblue',linewidth=2)
    plt.plot(epochs, train_acc, label='Training Accuracy', color='chocolate', linestyle='--', linewidth=1.5, alpha=0.9)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


class MyLogger:
    """
    set up for loguru logger (for prettier logging)
    """
    def __init__(self, logger, log_level, output = "console"):
        """
        Parameters
        ---------
        logger:
            expects loguru logger
        output: where to log, 
            "both" - both in console and log in file, 
            "file" - only log in file, "console" - only log in console,
            "no" - don't use logger for package messages
        """
        self.output = output
        self.log_level = log_level
        
        if self.output == "no":
            logger.remove()
            return
        if not (self.output in ["both", "file", "console"]):
            output = "console"

        
        logger.remove()
        if (self.output in ["both", "file"]):
            logger.add(
                f"./log/{date.today()}_log.log",
                level = self.log_level
            )
        if output == "console":
            logger.add(
                sys.stderr,
                level=self.log_level
            )


    def writePackageMsg(self, message):
        """
        (for redirect package msg to loguru logger, but not in used)
        """
        if message.strip():
            logger.log(self.log_level, message.strip())

    def flush(self):
        """
        (for redirect package msg to loguru logger, but not in used)
        """
        pass
