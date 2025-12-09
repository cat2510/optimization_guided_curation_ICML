"""
Data loading and splitting module.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import model_pipeline for patient-level splitting
try:
    from model_pipeline import train_test_split_enrol
    HAS_MODEL_PIPELINE = True
except ImportError:
    HAS_MODEL_PIPELINE = False
    logger.warning("model_pipeline not available, using standard train_test_split")


def load_data(
    data_path: str,
    uid_col: str = "uid",
    file_format: str = "auto"  # "auto", "csv", "parquet"
) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Parameters
    ----------
    data_path : str
        Path to the data file
    uid_col : str
        Name of the unique identifier column
    file_format : str
        File format: "auto" (detect from extension), "csv", or "parquet"
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    # Resolve relative paths relative to the project root (parent of pipeline_Dec2025)
    if not os.path.isabs(data_path):
        # If relative, resolve relative to parent directory (where data files are)
        project_root = Path(__file__).parent.parent.parent
        data_path = os.path.join(project_root, data_path)
        # Normalize the path (handles .. and .)
        data_path = os.path.normpath(data_path)
    
    logger.info(f"Loading data from: {data_path}")
    
    if file_format == "auto":
        if data_path.endswith('.parquet'):
            file_format = "parquet"
        elif data_path.endswith('.csv'):
            file_format = "csv"
        else:
            file_format = "csv"  # default
    
    if file_format == "parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def split_features_labels(
    df: pd.DataFrame,
    target_col: str,
    uid_col: str = "uid",
    exclude_cols: Optional[List[str]] = None,
    exclude_cost_cols: bool = True,
    exclude_high_corr: bool = True,
    correlation_threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Split dataset into features and labels with optional feature filtering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    target_col : str
        Name of the target/label column
    uid_col : str
        Name of the unique identifier column
    exclude_cols : list, optional
        Additional columns to exclude from features
    exclude_cost_cols : bool
        If True, excludes columns with "cost" in name (except target)
    exclude_high_corr : bool
        If True, excludes features with high correlation to target
    correlation_threshold : float
        Maximum correlation allowed (default: 0.5)
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    feature_cols : list
        List of feature column names
    """
    exclude_cols = exclude_cols or []
    exclude_cols = [uid_col, target_col] + exclude_cols
    
    # Exclude cost columns if requested
    if exclude_cost_cols:
        cost_cols = [
            col for col in df.columns 
            if ("cost" in col.lower() or "quarterly" in col.lower() 
                or "increasing" in col.lower())
            and col != target_col
        ]
        exclude_cols.extend(cost_cols)
        logger.info(f"Excluding {len(cost_cols)} cost/quarterly columns")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter high correlation features if requested
    if exclude_high_corr:
        # Import here to avoid circular import at module level
        from .preprocessing import filter_high_correlation_features
        feature_cols = filter_high_correlation_features(
            df, feature_cols, target_col, correlation_threshold
        )
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Final features: {len(feature_cols)} columns")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_original: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    patient_level: bool = False,
    patient_id_col: str = "ENROLID"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create outcome-stratified train-test split.
    Optionally uses patient-level splitting (by patient ID) if model_pipeline is available.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    df_original : pd.DataFrame, optional
        Original dataframe (needed for patient-level splitting)
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify by outcome
    patient_level : bool
        If True and model_pipeline available, split by patient ID (ENROLID)
    patient_id_col : str
        Patient ID column name (default: "ENROLID")
        
    Returns
    -------
    X_train, X_test, y_train, y_test
        Train and test splits
    """
    if patient_level and HAS_MODEL_PIPELINE and df_original is not None:
        # Patient-level splitting using model_pipeline.train_test_split_enrol
        logger.info(f"Using patient-level splitting by {patient_id_col}")
        
        # Combine X and y back into a dataframe for splitting
        df_for_split = X.copy()
        df_for_split[y.name] = y
        
        # Ensure patient_id_col exists
        if patient_id_col not in df_original.columns:
            logger.warning(f"{patient_id_col} not found in original dataframe, falling back to row-level split")
            patient_level = False
        else:
            # Add patient ID to df_for_split (assuming same index)
            df_for_split[patient_id_col] = df_original[patient_id_col].values
            
            # Use train_test_split_enrol
            train_ids, test_ids, train_df, test_df = train_test_split_enrol(
                df=df_for_split,
                target_col=y.name,
                test_size=test_size,
                random_state=random_state,
                verbose=False
            )
            
            # Split back into X and y
            X_train = train_df[X.columns]
            X_test = test_df[X.columns]
            y_train = train_df[y.name]
            y_test = test_df[y.name]
            
            logger.info(f"Train set: {len(X_train):,} samples ({len(train_ids):,} unique patients)")
            logger.info(f"Test set: {len(X_test):,} samples ({len(test_ids):,} unique patients)")
            logger.info(f"Train target distribution:\n{y_train.value_counts()}")
            logger.info(f"Test target distribution:\n{y_test.value_counts()}")
            
            return X_train, X_test, y_train, y_test
    
    # Standard row-level splitting
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    logger.info(f"Train target distribution:\n{y_train.value_counts()}")
    logger.info(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def separate_cases_controls(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate dataset into cases (minority) and controls (majority).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with target column
    target_col : str
        Name of the binary target column (1 = case, 0 = control)
        
    Returns
    -------
    df_cases : pd.DataFrame
        Cases (minority class, target == 1)
    df_controls : pd.DataFrame
        Controls (majority class, target == 0)
    """
    df_cases = df[df[target_col] == 1].copy()
    df_controls = df[df[target_col] == 0].copy()
    
    logger.info(f"Cases (minority): {len(df_cases):,} samples")
    logger.info(f"Controls (majority): {len(df_controls):,} samples")
    logger.info(f"Ratio: {len(df_controls) / len(df_cases):.2f}:1")
    
    return df_cases, df_controls

