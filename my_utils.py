import os
import sys
from loguru import logger
from datetime import date

def get_unique_file_name(file_name: str, dir: str = "./", verbose: bool = True):
    """Get a unique file name in a directory for saving file to avoid overwriting.
    example dir: "./result" (no need ending slash)
    
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

class StreamToLogger:
    """
    redirect package message form stdout to loguru logger (for prettier logging)
    """
    def __init__(self, logger, log_level, output = "console"):
        """
        output: where to log, 
        "both" - vith in console and log in file, 
        "file" - only log in file, "console" - only log in console,
        "no" - don't use logger for package messages
        """
        self.output = output
        self.log_level = log_level
        self.logger = logger
        
        if self.output == "no":
            self.logger.remove()
            return
        if not (self.output in ["both", "file", "console"]):
            output = "console"

        
        self.logger.remove()
        if (self.output in ["both", "file"]):
            self.logger.add(
                f"./log/{date.today()}_log.log",
                level = self.log_level
            )
        if output == "console":
            self.logger.add(
                sys.stderr,
                level=self.log_level
            )


    def write(self, message):
        if message.strip():
            logger.log(self.log_level, message.strip())

    def flush(self):
        pass
