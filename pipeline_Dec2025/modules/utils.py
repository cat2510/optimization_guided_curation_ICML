"""
Utility functions for the OCT pipeline.
"""
import os
import yaml
import json
import pickle
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime


class TeeOutput:
    """
    Class to redirect stdout to both console and file.
    """
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    capture_print: bool = True
) -> Tuple[logging.Logger, Optional[TeeOutput]]:
    """
    Setup logging configuration with optional file logging.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file : str, optional
        Path to log file. If provided, logs will be written to file.
    capture_print : bool
        If True and log_file is provided, redirects print() to log file too.
        
    Returns
    -------
    logger : logging.Logger
        Logger instance
    tee_output : TeeOutput or None
        TeeOutput instance if capture_print is True, else None
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    tee_output = None
    if log_file:
        ensure_dir(os.path.dirname(log_file) if os.path.dirname(log_file) else '.')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Redirect print statements to file if requested
        if capture_print:
            tee_output = TeeOutput(log_file)
            sys.stdout = tee_output
            print(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
    
    return logging.getLogger(__name__), tee_output


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_extreme_points(ext: Dict[str, float], filepath: str) -> None:
    """Save extreme points to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    # Convert numpy types to native Python types for JSON serialization
    ext_serializable = {k: float(v) for k, v in ext.items()}
    with open(filepath, 'w') as f:
        json.dump(ext_serializable, f, indent=2)
    print(f"✓ Saved extreme points to: {filepath}")


def load_extreme_points(filepath: str) -> Optional[Dict[str, float]]:
    """Load extreme points from JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            ext = json.load(f)
        print(f"✓ Loaded extreme points from: {filepath}")
        return ext
    except Exception as e:
        print(f"⚠ Failed to load extreme points: {e}")
        return None


def get_extreme_points_cache_path(
    results_dir: str,
    method: str,
    final_ratio: float,
    top_k_case_ctrl: int,
    top_k_ctrl_ctrl: Optional[int] = None,
    L_pairs: Optional[int] = None
) -> str:
    """Generate cache path for extreme points based on parameters."""
    cache_dir = os.path.join(results_dir, "extreme_points")
    ensure_dir(cache_dir)
    
    # Create unique filename based on parameters
    if method == "pushpull":
        filename = f"extreme_points_{method}_ratio_{final_ratio:.2f}_topk_{top_k_case_ctrl}_Lpairs_{L_pairs}.json"
    elif method == "doublefacility":
        filename = f"extreme_points_{method}_ratio_{final_ratio:.2f}_topk_{top_k_case_ctrl}_topkctrl_{top_k_ctrl_ctrl}.json"
    else:
        filename = f"extreme_points_{method}_ratio_{final_ratio:.2f}.json"
    
    return os.path.join(cache_dir, filename)


def save_undersampled_data(df: pd.DataFrame, filepath: str) -> None:
    """Save undersampled training data."""
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"✓ Saved undersampled data to: {filepath}")


def load_undersampled_data(filepath: str) -> pd.DataFrame:
    """Load undersampled training data."""
    return pd.read_csv(filepath)


def get_undersampled_data_path(
    results_dir: str,
    method: str,
    final_ratio: float,
    w: float
) -> str:
    """Generate path for undersampled data."""
    ensure_dir(results_dir)
    filename = f"undersampled_{method}_ratio_{final_ratio:.2f}_w_{w:.2f}.csv"
    return os.path.join(results_dir, filename)


def get_cost_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of cost-related columns from dataframe.
    
    Matches columns containing "cost", "quarterly", or "increasing" in their name
    (case-insensitive), as defined in the notebooks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to search for cost columns
        
    Returns
    -------
    List[str]
        List of column names matching cost patterns
    """
    cost_cols = [
        col for col in df.columns 
        if ("cost" in col.lower() or 
            "quarterly" in col.lower() or 
            "increasing" in col.lower())
    ]
    return cost_cols


def get_exclude_cols_matching(
    df: pd.DataFrame,
    exclude_cols_matching: Optional[List[str]] = None,
    include_cost_columns: bool = True,
    include_cost_stratum: bool = True
) -> List[str]:
    """
    Build exclude_cols_matching list with COST_COLUMNS and cost_stratum_2018.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract cost columns from
    exclude_cols_matching : list, optional
        Base list of columns to exclude (will be extended)
    include_cost_columns : bool
        If True, automatically includes COST_COLUMNS
    include_cost_stratum : bool
        If True, automatically includes "cost_stratum_2018" if it exists
        
    Returns
    -------
    List[str]
        Combined list of columns to exclude
    """
    exclude_cols = list(exclude_cols_matching) if exclude_cols_matching else []
    
    if include_cost_columns:
        cost_cols = get_cost_columns(df)
        exclude_cols.extend(cost_cols)
    
    if include_cost_stratum and "cost_stratum_2018" in df.columns:
        if "cost_stratum_2018" not in exclude_cols:
            exclude_cols.append("cost_stratum_2018")
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for col in exclude_cols:
        if col not in seen:
            seen.add(col)
            result.append(col)
    
    return result

