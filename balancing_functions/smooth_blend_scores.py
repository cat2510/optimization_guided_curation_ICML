# blending.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    accuracy_score,
)

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


def smooth_blend_scores(
    risk_stratifier_scores: np.ndarray,
    full_model_scores: np.ndarray,
    matched_model_scores: np.ndarray,
    r0: float = 0.7,
    r1: float = 0.9,
) -> np.ndarray:
    """
    Compute a smooth linear blend of two sets of model predictions
    based on a risk‐stratifier score `r`.

    Parameters
    ----------
    risk_stratifier_scores : array of shape (n_samples,)
        The baseline risk score (e.g. logistic regression probabilities).
    full_model_scores : array of shape (n_samples,)
        The probabilities from the model trained on the full dataset.
    matched_model_scores : array of shape (n_samples,)
        The probabilities from the model trained on the matched dataset.
    r0 : float, default=0.7
        Lower cutoff for blending. When r <= r0, alpha = 0 (use full_model only).
    r1 : float, default=0.9
        Upper cutoff for blending. When r >= r1, alpha = 1 (use matched_model only).

    Returns
    -------
    p_blend : array of shape (n_samples,)
        The blended probabilities:
          p_blend = (1 - alpha) * full_model_scores + alpha * matched_model_scores
    """
    # compute interpolation weight alpha in [0,1]
    alpha = np.clip((risk_stratifier_scores - r0) / (r1 - r0), 0.0, 1.0)
    return (1 - alpha) * full_model_scores + alpha * matched_model_scores


def evaluate_predictions(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Compute common binary‐classification metrics from true labels and predicted probabilities.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred_proba : array of shape (n_samples,)
        Predicted probability of the positive class.
    threshold : float, default=0.5
        Probability cutoff to form binary predictions.

    Returns
    -------
    metrics : dict
        {
          "AUC": ...,
          "F1": ...,
          "Balanced Accuracy": ...,
          "MCC": ...,
          "Accuracy": ...
        }
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_pred_proba),
        "F1": f1_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def compare_models(
    y_true: np.ndarray, scores_dict: dict, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compare multiple prediction arrays on the same test set.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True binary labels.
    scores_dict : dict
        A mapping from model name to predicted‐probability array, e.g.
          {
            "Full XGB":      full_model_scores,
            "Matched XGB":   matched_model_scores,
            "Blended":       blended_scores
          }
    threshold : float, default=0.5
        Probability cutoff for F1, accuracy, etc.

    Returns
    -------
    df_metrics : pandas.DataFrame
        Rows are model names, columns are metrics.
    """
    records = []
    for name, proba in scores_dict.items():
        m = evaluate_predictions(y_true, proba, threshold=threshold)
        m["model"] = name
        records.append(m)
    df = pd.DataFrame.from_records(records).set_index("model")
    return df
