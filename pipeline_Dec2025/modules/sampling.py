"""
Undersampling module with extreme point caching.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from balancing_functions.pushpull_sampler import PushPullSampler
from balancing_functions.doublefacility_sampler import DoubleFacilitySampler
from modules.utils import (
    save_extreme_points,
    load_extreme_points,
    get_extreme_points_cache_path,
    save_undersampled_data,
    get_undersampled_data_path
)

logger = logging.getLogger(__name__)


def apply_undersampling(
    df_cases: pd.DataFrame,
    df_controls: pd.DataFrame,
    exclude_cols_matching: list,
    method: str = "pushpull",
    final_ratio: float = 1.0,
    w: float = 0.5,
    K_factor: float = 3.0,
    top_k_case_ctrl: int = 20,
    top_k_ctrl_ctrl: int = 20,
    L_pairs: int = 20,
    results_dir: str = "results",
    save_extreme_points_flag: bool = True,
    load_extreme_points_flag: bool = True,
    verbose: bool = True,
    target_col: str = "highcost_gt_200000",  # Binary target column (same as binary_group)
    uid_col: str = "uid"  # Unique identifier column
) -> Tuple[pd.DataFrame, Dict[str, float], Dict]:
    """
    Apply undersampling (pushpull or doublefacility) with extreme point caching.
    
    Parameters
    ----------
    df_cases : pd.DataFrame
        Cases (minority class)
    df_controls : pd.DataFrame
        Controls (majority class)
    exclude_cols_matching : list
        Columns to exclude from feature matching
    method : str
        Undersampling method: "pushpull" or "doublefacility"
    final_ratio : float
        Ratio of controls to cases (1.0 = 1:1)
    w : float
        Weight for multi-objective optimization (0-1)
    K_factor : float
        Node pruning factor
    top_k_case_ctrl : int
        Top-k controls per case for edge pruning
    top_k_ctrl_ctrl : int
        Top-k controls per control (for doublefacility)
    L_pairs : int
        Number of far pairs for pushpull
    results_dir : str
        Directory for saving results
    save_extreme_points_flag : bool
        Whether to save computed extreme points
    load_extreme_points_flag : bool
        Whether to try loading cached extreme points
    verbose : bool
        Verbose output
    target_col : str
        Binary target column name (used as binary_group in sampler to exclude from features)
    uid_col : str
        Unique identifier column name
        
    Returns
    -------
    undersampled_df : pd.DataFrame
        Undersampled training data (cases + selected controls)
    ext : dict
        Extreme points dictionary
    result_dict : dict
        Additional results from sampling (f1, f2, status, etc.)
    """
    logger.info(f"Applying {method} undersampling with ratio={final_ratio:.2f}, w={w:.2f}")
    
    # Initialize sampler with target_col as binary_group (they should be the same)
    if method == "pushpull":
        sampler = PushPullSampler(
            binary_group=target_col,  # Use target_col instead of default
            uid_col=uid_col
        )
        ext_cache_path = get_extreme_points_cache_path(
            results_dir, method, final_ratio, top_k_case_ctrl, L_pairs=L_pairs
        )
    elif method == "doublefacility":
        sampler = DoubleFacilitySampler(
            binary_group=target_col,  # Use target_col instead of default
            uid_col=uid_col
        )
        ext_cache_path = get_extreme_points_cache_path(
            results_dir, method, final_ratio, top_k_case_ctrl, 
            top_k_ctrl_ctrl=top_k_ctrl_ctrl
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pushpull' or 'doublefacility'")
    
    # Try to load cached extreme points
    ext = None
    if load_extreme_points_flag:
        ext = load_extreme_points(ext_cache_path)
        if ext is not None:
            logger.info("Using cached extreme points")
    
    # Apply sampling - pass cached extreme points if available
    if method == "pushpull":
        undersampled_df, result_dict, ext = sampler.solve_pushpull_normalized_MILP(
            df_cases=df_cases,
            df_controls=df_controls,
            exclude_cols_matching=exclude_cols_matching,
            final_ratio=final_ratio,
            w=w,
            K_factor=K_factor,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            ext=ext,  # Pass cached extreme points (None if not cached)
            verbose=verbose
        )
    else:  # doublefacility
        undersampled_df, result_dict, ext = sampler.solve_double_facility_normalized_MILP(
            df_cases=df_cases,
            df_controls=df_controls,
            exclude_cols_matching=exclude_cols_matching,
            final_ratio=final_ratio,
            w=w,
            K_factor=K_factor,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            ext=ext,  # Pass cached extreme points (None if not cached)
            verbose=verbose
        )
    
    # Save extreme points if requested
    if save_extreme_points_flag and ext is not None:
        save_extreme_points(ext, ext_cache_path)
    
    # Save undersampled data
    undersampled_path = get_undersampled_data_path(results_dir, method, final_ratio, w)
    save_undersampled_data(undersampled_df, undersampled_path)
    
    logger.info(f"Undersampling complete. Final dataset: {len(undersampled_df):,} samples")
    
    return undersampled_df, ext, result_dict

