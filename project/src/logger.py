"""
Logging utility for AI Candidate Ranking System.
Handles creation of timestamped log files and consistent formatting.
"""
import logging
import os
from datetime import datetime
from config import PROJECT_ROOT

def setup_logger():
    """
    Configures and returns a logger that writes to both console and a timestamped file.
    Logs are stored in the 'logs/' directory.
    """
    # 1. Define logs directory
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 2. Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"run_{timestamp}.log"
    
    # 3. Create logger
    logger = logging.getLogger("AI_Ranker")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 4. Define log format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File Handler (Writes to logs/run_timestamp.log)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler (Writes to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Writing to: {log_file}")
    return logger

# Singleton instance for easy import
logger = setup_logger()
