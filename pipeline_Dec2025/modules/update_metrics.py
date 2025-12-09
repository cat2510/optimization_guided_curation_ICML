"""
Module for updating metrics master file.
"""
import pandas as pd
import os
from typing import Dict, Optional
import logging

from modules.utils import ensure_dir

logger = logging.getLogger(__name__)


def update_metrics_master(
    metrics: Dict,
    results_dir: str,
    method: Optional[str] = None,
    metrics_file: Optional[str] = None,
    additional_fields: Optional[Dict] = None
) -> str:
    """
    Append metrics to master metrics file.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary
    results_dir : str
        Results directory
    method : str, optional
        Undersampling method (pushpull/doublefacility). If provided, 
        creates method-specific filename.
    metrics_file : str, optional
        Metrics filename. If not provided, generates method-specific name.
    additional_fields : dict, optional
        Additional fields to include (e.g., method, ratio, w)
        
    Returns
    -------
    filepath : str
        Path to metrics file
    """
    ensure_dir(results_dir)
    
    # Generate method-specific filename if method is provided
    if metrics_file is None:
        if method:
            metrics_file = f"metrics_master_{method}.csv"
        else:
            metrics_file = "metrics_master.csv"
    
    filepath = os.path.join(results_dir, metrics_file)
    
    # Combine metrics with additional fields
    row = metrics.copy()
    if additional_fields:
        row.update(additional_fields)
    
    # Convert to DataFrame
    row_df = pd.DataFrame([row])
    
    # Append to file (create if doesn't exist)
    if os.path.exists(filepath):
        row_df.to_csv(filepath, mode='a', header=False, index=False)
        logger.info(f"✓ Appended metrics to: {filepath}")
    else:
        row_df.to_csv(filepath, mode='w', header=True, index=False)
        logger.info(f"✓ Created metrics file: {filepath}")
    
    return filepath

