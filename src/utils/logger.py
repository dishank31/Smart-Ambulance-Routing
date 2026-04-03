"""
Logging setup for the Smart Ambulance ML system.
Provides file + console logging with timestamps.
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_dir="logs", level=logging.DEBUG):
    """
    Create a logger with both file and console handlers.
    
    Args:
        name: Logger name (e.g., 'model_training', 'api')
        log_dir: Directory to store log files
        level: Logging level
    
    Returns:
        logging.Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # File handler — detailed logs
    log_file = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    
    # Console handler — info and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
