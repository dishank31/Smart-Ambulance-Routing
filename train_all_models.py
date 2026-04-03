"""
Master script to train all models end-to-end.
Run: python train_all_models.py
"""

import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.model_trainer import train_all
from src.utils.config import Config

if __name__ == "__main__":
    Config.ensure_dirs()
    print("Starting master training script...")
    train_all()
    print("Done!")
