import os
from loguru import logger

def get_unique_file_name(file_name: str, dir: str = "./", verbose: bool = True):
    """Get a unique file name in a directory for saving file to avoid overwriting.
    example dir: "./result" (no need ending slash)
    
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    name, ext = os.path.splitext(file_name)
    i = 0
    while True:
        file_path = dir / f"{name}_{i}{ext}"
        if not file_path.exists():
            if(verbose):
                logger.info(f"Saving file to {file_path}...")
            return file_path
        i += 1