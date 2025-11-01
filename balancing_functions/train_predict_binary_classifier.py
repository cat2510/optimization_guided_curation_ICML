import numpy as np
import pandas as pd
from typing import Optional, Union, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """
    Find the threshold in [0,1] that maximizes `metric` on (y_true, y_score).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
      True binary labels.
    y_score : array-like of shape (n_samples,)
      Predicted probabilities.
    metric : {"f1", "balanced_accuracy"}
      Which metric to optimize.
    thresholds : array-like, optional
      List of thresholds to try. Default is np.linspace(0,1,101).

    Returns
    -------
    best_thr : float
      The threshold giving the highest metric.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_score = -np.inf
    best_thr = 0.5
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division="0")
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'f1' or 'balanced_accuracy'")
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr


def train_predict_binary_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    optimize_threshold: bool = False,
    threshold_metric: str = "f1",
    model_type: str = "logistic",
    impute_strategy: str = "mean",
    scale: bool = True,
    random_state: int = 42,
    evaluate_subset_enable: bool = False,
    weight_beta: float = 3.0,  # controls tail emphasis
    num_bins: int = 10,  # used only if bin_edges is None
    bin_edges: Optional[np.ndarray] = None,  # if provided, overrides num_bins
    save_paths: Optional[dict] = None,
) -> dict:
    """
    Trains & evaluates a binary classifier, returning:
      - Global metrics (AUC, F1, etc.)
      - bin-wise AUC over slices defined by `bin_edges` or `num_bins`
      - Weighted AUC emphasizing high-risk via exp(weight_beta * score)
      - If bin_edges was None, returns the computed edges in metrics["bin_edges"]
    """

    # 1) Prepare data
    X_train = X_train.drop(columns=["uid"], errors="ignore")
    X_test = X_test.drop(columns=["uid"], errors="ignore")

    # 2) Build & fit pipeline
    steps = []
    if model_type == "logistic":
        steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
        if scale:
            steps.append(("scaler", StandardScaler()))
        clf = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "xgboost":
        clf = XGBClassifier(
            tree_method="hist",
            max_depth=5,
            learning_rate=0.06,
            random_state=random_state,
            eval_metric="logloss",
            enable_categorical=True,
        )
    else:
        raise ValueError("Unsupported model_type. Choose 'logistic' or 'xgboost'.")
    steps.append(("clf", clf))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    # --- tune threshold on (X_val, y_val) if requested ---
    best_thr = 0.5
    if optimize_threshold:
        if X_val is None or y_val is None:
            raise ValueError(
                "Must provide X_val and y_val when optimize_threshold=True"
            )
        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        best_thr = find_best_threshold(y_val, y_proba_val, metric=threshold_metric)
        val_probs = y_proba_val  # <-- store for later

    # binary predictions on the test split
    y_pred = (y_proba_test >= best_thr).astype(int)

    # 4) Global metrics
    metrics = {
        "AUC": roc_auc_score(y_test, y_proba_test),
        "F1": f1_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
        "Test Proba": y_proba_test,
        "Train Proba": y_proba_train,
    }
    if optimize_threshold:
        metrics["Val Proba"] = val_probs
        metrics["best_threshold"] = best_thr
    # 5) Determine bin edges once, ensuring no clash
    if bin_edges is not None:
        edges = np.array(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("`bin_edges` must be a 1D array of length >= 2.")
        num_bins = edges.size - 1
    else:
        edges = np.percentile(y_proba_test, np.linspace(0, 100, num_bins + 1))
        metrics["bin_edges"] = edges.copy()

    # 6) Bin-wise AUC
    bin_aucs = {}
    for i in range(num_bins):
        low, high = edges[i], edges[i + 1]
        if i < num_bins - 1:
            mask = (y_proba_test >= low) & (y_proba_test < high)
        else:
            mask = (y_proba_test >= low) & (y_proba_test <= high)

        y_true_bin = y_test[mask]
        y_score_bin = y_proba_test[mask]

        if len(y_true_bin) >= 2 and len(np.unique(y_true_bin)) > 1:
            bin_aucs[f"Bin_{i+1}_AUC"] = roc_auc_score(y_true_bin, y_score_bin)
        else:
            bin_aucs[f"Bin_{i+1}_AUC"] = None

    metrics.update(bin_aucs)

    # 7) Weighted AUC (tail emphasis)
    w = np.exp(weight_beta * y_proba_test)
    w = w / np.mean(w)
    metrics["Weighted AUC"] = roc_auc_score(y_test, y_proba_test, sample_weight=w)

    # 8) Optional subgroup evaluation
    if evaluate_subset_enable:
        from sklearn.metrics import (
            roc_auc_score as _auc,
            f1_score as _f1,
            balanced_accuracy_score as _bal,
            matthews_corrcoef as _mcc,
        )

        for label in [0, 1]:
            mask = y_test == label
            y_t = y_test[mask]
            y_p = y_pred[mask]
            y_s = y_proba_test[mask]
            key = f"AUC (Label={label})"
            metrics[key] = _auc(y_t, y_s) if len(np.unique(y_t)) > 1 else None
            metrics[f"F1 (Label={label})"] = _f1(y_t, y_p)
            metrics[f"Balanced Acc (Label={label})"] = _bal(y_t, y_p)
            metrics[f"Accuracy (Label={label})"] = accuracy_score(y_t, y_p)
            metrics[f"MCC (Label={label})"] = _mcc(y_t, y_p)

    # 9) Optional saving of probabilities
    if save_paths:
        if "train" in save_paths:
            np.save(save_paths["train"], y_proba_train)
        if "test" in save_paths:
            np.save(save_paths["test"], y_proba_test)

    return metrics
