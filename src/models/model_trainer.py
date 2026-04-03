import logging
from ..utils.logger import setup_logger
from .severity_classifier import train_severity_models
from .eta_predictor import train_eta_models
from .bed_predictor import train_bed_models

logger = setup_logger('master_training')

def train_all():
    logger.info("Starting Full Model Training Pipeline...")
    
    # By separating into function calls, memory is freed up between complex trainings
    # severity_stacking = train_severity_models() # Already trained successfully
    eta_stacking = train_eta_models()
    bed_stacking = train_bed_models()
    
    logger.info("Full pipeline complete. All models trained and registered.")
