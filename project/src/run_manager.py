"""Run folder manager - creates timestamped run directories"""
import os
from datetime import datetime
from config import RESULT_BASE_DIR

def create_run_folder(base_dir=None):
    """Create a new timestamped run folder and return its path"""
    if base_dir is None:
        base_dir = RESULT_BASE_DIR
        
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base_dir, timestamp)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder