"""
Preprocessing module for feature detection and preparation.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_bin_flag_columns(df: pd.DataFrame) -> List[str]:
    """
    Get binary flag columns (columns starting with has_, ending with _adherent, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
        
    Returns
    -------
    list
        List of binary flag column names
    """
    return [
        col for col in df.columns 
        if col.startswith("has_") or "THRCLS" in col.upper()
        or col.endswith("_adherent") or col.startswith("early_")
        or col.startswith("is_")
    ]


def get_true_num_columns(df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
    """
    Get true numeric columns (cost, quarterly, claims related).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    categorical_cols : list
        List of categorical column names to exclude
        
    Returns
    -------
    list
        List of true numeric column names
    """
    return [
        col for col in df.columns
        if (
            ("cost" in col.lower() or 
             "quarterly" in col.lower() or 
             "claims" in col.lower())
            and col not in categorical_cols
        )
    ]


def filter_high_correlation_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    correlation_threshold: float = 0.5
) -> List[str]:
    """
    Filter out features with high correlation to target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    correlation_threshold : float
        Maximum absolute correlation allowed (default: 0.5)
        
    Returns
    -------
    filtered_cols : list
        Feature columns with high correlation features removed
    """
    # Get numeric columns only (correlation only works on numeric)
    numeric_cols = df[feature_cols + [target_col]].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    
    if target_col not in numeric_cols or len(numeric_cols) < 2:
        logger.info("Skipping correlation filtering (target not numeric or insufficient numeric columns)")
        return feature_cols
    
    # Compute correlations
    feature_numeric = [col for col in numeric_cols if col != target_col]
    if len(feature_numeric) == 0:
        return feature_cols
    
    corrs = df[feature_numeric + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    
    # Find high correlation columns
    high_corr_cols = corrs[corrs > correlation_threshold].index.tolist()
    high_corr_cols = [col for col in high_corr_cols if col != target_col]
    
    if high_corr_cols:
        logger.info(f"Removing {len(high_corr_cols)} features with correlation > {correlation_threshold} to target")
        logger.info(f"High correlation features: {high_corr_cols[:10]}...")  # Show first 10
        
        # Remove from feature_cols
        filtered_cols = [col for col in feature_cols if col not in high_corr_cols]
        return filtered_cols
    
    return feature_cols


def detect_feature_types(
    df: pd.DataFrame,
    target_col: str,
    uid_col: str = "uid",
    exclude_cols: Optional[List[str]] = None,
    use_model_pipeline_categories: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Automatically detect categorical and numeric columns.
    Optionally uses model_pipeline categorization (BIN_FLAG_COLUMNS, TRUE_NUM_COLUMNS).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target column name (to exclude)
    uid_col : str
        UID column name (to exclude)
    exclude_cols : list, optional
        Additional columns to exclude
    use_model_pipeline_categories : bool
        If True, uses model_pipeline.get_bin_flag_columns and get_true_num_columns
        
    Returns
    -------
    categorical_cols : list
        List of categorical column names
    numeric_cols : list
        List of numeric column names
    """
    exclude_cols = exclude_cols or []
    exclude_cols = [target_col, uid_col] + exclude_cols
    
    all_cols = [col for col in df.columns if col not in exclude_cols]
    
    if use_model_pipeline_categories:
        try:
            from model_pipeline import get_bin_flag_columns, get_true_num_columns
            
            # Detect categorical (object, category)
            categorical_cols = df[all_cols].select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            
            # Get binary flag columns (these are typically numeric but treated specially)
            bin_flag_cols = get_bin_flag_columns(df)
            bin_flag_cols = [col for col in bin_flag_cols if col in all_cols]
            
            # Get true numeric columns (cost, quarterly, claims)
            true_num_cols = get_true_num_columns(df, categorical_cols)
            true_num_cols = [col for col in true_num_cols if col in all_cols]
            
            # Remaining numeric columns (not in bin_flag or true_num)
            remaining_numeric = df[all_cols].select_dtypes(
                include=[np.number]
            ).columns.tolist()
            remaining_numeric = [
                col for col in remaining_numeric 
                if col not in bin_flag_cols and col not in true_num_cols
            ]
            
            # Combine: numeric_cols = true_num_cols + remaining_numeric
            # Binary flags are typically passed through as-is (not scaled)
            numeric_cols = true_num_cols + remaining_numeric
            
            logger.info(f"Detected {len(categorical_cols)} categorical columns")
            logger.info(f"Detected {len(bin_flag_cols)} binary flag columns")
            logger.info(f"Detected {len(true_num_cols)} true numeric columns (cost/claims)")
            logger.info(f"Detected {len(remaining_numeric)} other numeric columns")
            logger.info(f"Total numeric columns: {len(numeric_cols)}")
            
        except ImportError:
            logger.warning("model_pipeline not available, using simple detection")
            use_model_pipeline_categories = False
    
    if not use_model_pipeline_categories:
        # Simple detection
        categorical_cols = df[all_cols].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        numeric_cols = df[all_cols].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        logger.info(f"Detected {len(categorical_cols)} categorical columns")
        logger.info(f"Detected {len(numeric_cols)} numeric columns")
    
    return categorical_cols, numeric_cols

