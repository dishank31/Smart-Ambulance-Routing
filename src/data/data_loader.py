import pandas as pd
import os
from ..utils.config import Config

def load_severity_data():
    """Load severity dataset (from the real processed real data)."""
    path = Config.SEVERITY_CLEANED_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real data not found at {path}.")
    return pd.read_csv(path)

def load_eta_data():
    """Load ETA dataset."""
    path = Config.ETA_PROCESSED_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETA data not found at {path}.")
    return pd.read_csv(path)

def load_bed_data():
    """Load hospital beds dataset (from your originally processed bed data)."""
    path = Config.BED_PROCESSED_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bed data not found at {path}.")
    return pd.read_csv(path)
