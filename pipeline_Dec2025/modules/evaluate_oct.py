"""
OCT evaluation module.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import os
from model_IAI import evaluate_binary_oct

logger = logging.getLogger(__name__)


def evaluate_oct_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: object,
    feature_names: list,
    results_dir: Optional[str] = None,
    method: Optional[str] = None,
    ratio: Optional[float] = None,
    w: Optional[float] = None,
    compute_leaf_metrics: bool = False
) -> Dict[str, float]:
    """
    Evaluate OCT model on test set.
    
    Parameters
    ----------
    model : object
        Trained OCT model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    preprocessor : object
        Fitted preprocessor
    feature_names : list
        Feature names after preprocessing
    results_dir : str, optional
        Directory to save predictions
    method : str, optional
        Undersampling method (pushpull/doublefacility) for filename
    ratio : float, optional
        Ratio value for prediction filename
    w : float, optional
        Weight value for prediction filename
    compute_leaf_metrics : bool
        Whether to compute per-leaf metrics
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating OCT model on test set...")
    
    # Build method-specific predictions directory
    if results_dir:
        if method:
            predictions_dir = os.path.join(results_dir, "predictions", method)
            os.makedirs(predictions_dir, exist_ok=True)
        else:
            predictions_dir = os.path.join(results_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
    else:
        predictions_dir = None
    
    # Call evaluate_binary_oct - it will save to results_dir/predictions/
    # We'll handle the filename after
    metrics = evaluate_binary_oct(
        iai_model=model,
        X_test_df=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        compute_leaf_metrics=compute_leaf_metrics,
        results_dir=predictions_dir,
        ratio=ratio
    )
    
    # Rename the prediction file to include method and w if provided
    if predictions_dir and ratio is not None:
        import glob
        # Find the prediction file that was just created
        pattern = os.path.join(predictions_dir, f"oct_predictions_ratio_{ratio:.2f}.csv")
        if os.path.exists(pattern):
            # Build new filename with method and w
            if method and w is not None:
                new_name = f"oct_predictions_{method}_ratio_{ratio:.2f}_w_{w:.2f}.csv"
            elif method:
                new_name = f"oct_predictions_{method}_ratio_{ratio:.2f}.csv"
            else:
                new_name = None  # Keep original name
            
            if new_name:
                new_path = os.path.join(predictions_dir, new_name)
                os.rename(pattern, new_path)
                logger.info(f"✓ Saved predictions to: {new_path}")
    
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    
    return metrics

