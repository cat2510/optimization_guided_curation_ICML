"""
Module for saving predictions and results.
"""
import pandas as pd
import os
from typing import Dict, Optional
import logging

from modules.utils import ensure_dir

logger = logging.getLogger(__name__)


def save_predictions(
    predictions_df: pd.DataFrame,
    results_dir: str,
    filename: str = "predictions.csv"
) -> str:
    """
    Save predictions to CSV.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions
    results_dir : str
        Results directory
    filename : str
        Output filename
        
    Returns
    -------
    filepath : str
        Path to saved file
    """
    ensure_dir(results_dir)
    predictions_dir = os.path.join(results_dir, "predictions")
    ensure_dir(predictions_dir)
    
    filepath = os.path.join(predictions_dir, filename)
    predictions_df.to_csv(filepath, index=False)
    logger.info(f"✓ Saved predictions to: {filepath}")
    return filepath

