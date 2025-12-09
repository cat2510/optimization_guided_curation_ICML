"""
OCT training module with hyperparameter tuning.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging

from model_IAI import finetune_oct

logger = logging.getLogger(__name__)


def train_oct_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    depths: List[int] = [5, 7, 9],
    minbuckets: List[int] = [50, 100, 150],
    cps: List[float] = [0.001, 0.01, 0.05]
) -> Tuple[object, Tuple, pd.DataFrame, object, List[str]]:
    """
    Train OCT with hyperparameter tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    categorical_cols : list
        Categorical column names
    numeric_cols : list
        Numeric column names
    depths : list
        Max depth values to try
    minbuckets : list
        Min bucket sizes to try
    cps : list
        Complexity parameters to try
        
    Returns
    -------
    best_model : object
        Best trained OCT model
    best_params : tuple
        Best hyperparameters (depth, minbucket, cp)
    results_df : pd.DataFrame
        Grid search results
    preprocessor : object
        Fitted preprocessor
    feature_names : list
        Feature names after preprocessing
    """
    logger.info("Starting OCT hyperparameter tuning...")
    logger.info(f"Grid search: {len(depths)} depths × {len(minbuckets)} minbuckets × {len(cps)} cps = {len(depths) * len(minbuckets) * len(cps)} combinations")
    
    best_model, best_params, results_df, preprocessor, feature_names = finetune_oct(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        depths=depths,
        minbuckets=minbuckets,
        cps=cps
    )
    
    logger.info(f"Best parameters: depth={best_params[0]}, minbucket={best_params[1]}, cp={best_params[2]:.4f}")
    logger.info(f"Best F1 score: {results_df['f1'].max():.4f}")
    
    return best_model, best_params, results_df, preprocessor, feature_names

