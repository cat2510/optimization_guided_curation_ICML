"""
OCT→Student Distillation Module

This module implements knowledge distillation from an Optimal Classification Tree (OCT)
to an XGBoost student model for imbalanced binary classification.

Key components:
1. OCT teacher probability interface
2. Distilled XGBoost student training
3. Minority-focused evaluation metrics
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Optional, Dict, Tuple, List, Union
from pathlib import Path

try:
    from interpretableai import iai
    IAI_AVAILABLE = True
except ImportError:
    IAI_AVAILABLE = False
    print("Warning: interpretableai not available. OCT model loading will be limited.")

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, recall_score, precision_score, f1_score,
    matthews_corrcoef, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Import utilities from existing modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model_nonIAI_utils import (
    get_preprocessor_with_impute, train_test_split_enrol,
    get_bin_flag_columns, get_true_num_columns
)


# ============================================================================
# OCT TEACHER PROBABILITY INTERFACE
# ============================================================================

class OCTTeacher:
    """
    Wrapper for OCT model that provides probability predictions.
    
    Supports:
    - Loading from saved IAI model file
    - Reconstructing from CSV splits (manual tree traversal)
    - Leaf empirical probabilities with Laplace smoothing
    """
    
    def __init__(self, model_path: Optional[str] = None, splits_csv: Optional[str] = None,
                 X_train: Optional[pd.DataFrame] = None, y_train: Optional[np.ndarray] = None,
                 preprocessor=None, feature_names: Optional[List[str]] = None):
        """
        Initialize OCT teacher.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved IAI OptimalTreeClassifier (pickle or IAI format)
        splits_csv : str, optional
            Path to CSV file with tree splits (node_id, feature, threshold)
        X_train : pd.DataFrame, optional
            Training data for computing leaf empirical probabilities
        y_train : np.ndarray, optional
            Training labels for computing leaf empirical probabilities
        preprocessor : sklearn transformer, optional
            Preprocessor used with OCT (must match student preprocessing)
        feature_names : list, optional
            Feature names after preprocessing
        """
        self.model = None
        self.splits_df = None
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.leaf_probs = {}  # Cache for leaf empirical probabilities
        self.X_train_processed = None
        self.y_train = None
        
        # Try to load model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        # Load splits if provided
        if splits_csv and os.path.exists(splits_csv):
            self.splits_df = pd.read_csv(splits_csv)
            print(f"✓ Loaded OCT splits from {splits_csv}: {len(self.splits_df)} nodes")
        
        # Store training data for leaf probability estimation
        if X_train is not None and y_train is not None and preprocessor is not None:
            self._prepare_training_data(X_train, y_train)
    
    def _load_model(self, model_path: str):
        """Load IAI model from file."""
        if not IAI_AVAILABLE:
            raise ImportError("interpretableai package required to load OCT models")
        
        try:
            # Try loading as pickle
            if model_path.endswith('.pkl') or model_path.endswith('.pickle'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Try IAI's load_learner
                self.model = iai.load_learner(model_path)
            print(f"✓ Loaded OCT model from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load model from {model_path}: {e}")
            print("   Will attempt to reconstruct from splits CSV")
    
    def _prepare_training_data(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Preprocess training data and compute leaf assignments."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor required to prepare training data")
        
        X_train_processed = self.preprocessor.transform(X_train)
        if self.feature_names:
            X_train_processed = pd.DataFrame(X_train_processed, columns=self.feature_names)
        else:
            # Infer feature names from preprocessor
            X_train_processed = pd.DataFrame(X_train_processed)
        
        self.X_train_processed = X_train_processed
        self.y_train = np.asarray(y_train)
        
        # Compute leaf assignments and empirical probabilities
        if self.model is not None:
            leaf_assignments = self.model.apply(X_train_processed)
        elif self.splits_df is not None:
            leaf_assignments = self._apply_splits_manual(X_train_processed)
        else:
            raise ValueError("Need either model or splits_df to compute leaf assignments")
        
        # Compute empirical probabilities per leaf with Laplace smoothing
        self._compute_leaf_probs(leaf_assignments)
    
    def _apply_splits_manual(self, X: pd.DataFrame) -> np.ndarray:
        """
        Manually apply tree splits to get leaf assignments.
        
        This reconstructs the tree from splits_df and assigns each sample to a leaf.
        """
        if self.splits_df is None:
            raise ValueError("splits_df required for manual tree application")
        
        n_samples = len(X)
        leaf_assignments = np.zeros(n_samples, dtype=int)
        
        # Build tree structure from splits
        # Node 1 is root, we traverse based on splits
        tree = {}
        for _, row in self.splits_df.iterrows():
            node_id = int(row['node_id'])
            feature = row['feature']
            threshold = float(row['threshold'])
            tree[node_id] = {'feature': feature, 'threshold': threshold}
        
        # Traverse tree for each sample
        for i in range(n_samples):
            node_id = 1  # Start at root
            path = []
            
            while node_id in tree:
                split_info = tree[node_id]
                feature = split_info['feature']
                threshold = split_info['threshold']
                
                if feature not in X.columns:
                    # Feature might be one-hot encoded, try to find it
                    matching_cols = [c for c in X.columns if feature in str(c)]
                    if not matching_cols:
                        # Default to left child if feature missing
                        node_id = node_id * 2  # Left child
                        path.append(0)
                        continue
                    feature = matching_cols[0]
                
                value = X.iloc[i][feature] if feature in X.columns else 0
                
                if pd.isna(value):
                    # Handle missing: default to left (always_left mode)
                    node_id = node_id * 2
                    path.append(0)
                elif value <= threshold:
                    node_id = node_id * 2  # Left child
                    path.append(0)
                else:
                    node_id = node_id * 2 + 1  # Right child
                    path.append(1)
            
            # Leaf ID is the path (binary encoding)
            leaf_id = int(''.join(map(str, path)), 2) if path else 0
            leaf_assignments[i] = leaf_id
        
        return leaf_assignments
    
    def _compute_leaf_probs(self, leaf_assignments: np.ndarray, laplace_smoothing: float = 1.0):
        """
        Compute empirical class probabilities per leaf with Laplace smoothing.
        
        P(class=1 | leaf) = (count_1 + laplace) / (count_total + 2*laplace)
        """
        unique_leaves = np.unique(leaf_assignments)
        
        for leaf_id in unique_leaves:
            mask = (leaf_assignments == leaf_id)
            leaf_samples = self.y_train[mask]
            
            if len(leaf_samples) == 0:
                # No training samples in this leaf, use prior
                p_positive = np.mean(self.y_train)
            else:
                count_positive = np.sum(leaf_samples == 1)
                count_total = len(leaf_samples)
                
                # Laplace smoothing
                p_positive = (count_positive + laplace_smoothing) / (count_total + 2 * laplace_smoothing)
            
            self.leaf_probs[leaf_id] = p_positive
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Returns:
        --------
        proba : np.ndarray, shape (n_samples, 2)
            Probabilities for [class_0, class_1]
        """
        # Preprocess if needed
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
            if self.feature_names:
                X_processed = pd.DataFrame(X_processed, columns=self.feature_names)
            else:
                X_processed = pd.DataFrame(X_processed)
        else:
            X_processed = X.copy()
        
        # Get predictions
        if self.model is not None:
            # Use IAI model's predict_proba if available
            try:
                proba_df = self.model.predict_proba(X_processed)
                if isinstance(proba_df, pd.DataFrame):
                    proba = proba_df.values
                else:
                    proba = proba_df
                
                # Ensure shape (n, 2)
                if proba.shape[1] == 1:
                    # Binary: expand to 2 columns
                    proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
                return proba
            except:
                # Fall back to leaf probabilities
                pass
        
        # Use leaf empirical probabilities
        if not self.leaf_probs:
            raise ValueError("Leaf probabilities not computed. Provide X_train and y_train.")
        
        leaf_assignments = self._apply_splits_manual(X_processed)
        n_samples = len(X_processed)
        proba = np.zeros((n_samples, 2))
        
        for i, leaf_id in enumerate(leaf_assignments):
            p_positive = self.leaf_probs.get(leaf_id, np.mean(self.y_train) if self.y_train is not None else 0.5)
            proba[i, 0] = 1 - p_positive
            proba[i, 1] = p_positive
        
        return proba


def oct_predict_proba(
    X: pd.DataFrame,
    model_path: Optional[str] = None,
    splits_csv: Optional[str] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[np.ndarray] = None,
    preprocessor=None,
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convenience function to get OCT teacher probabilities.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features to predict (raw, will be preprocessed)
    model_path : str, optional
        Path to saved OCT model
    splits_csv : str, optional
        Path to CSV with tree splits
    X_train : pd.DataFrame, optional
        Training data for leaf probability estimation
    y_train : np.ndarray, optional
        Training labels
    preprocessor : sklearn transformer, optional
        Preprocessor (must be fitted)
    feature_names : list, optional
        Feature names after preprocessing
    
    Returns:
    --------
    proba : np.ndarray, shape (n, 2)
        Class probabilities [P(class=0), P(class=1)]
    """
    teacher = OCTTeacher(
        model_path=model_path,
        splits_csv=splits_csv,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        feature_names=feature_names
    )
    return teacher.predict_proba(X)


# ============================================================================
# DISTILLED STUDENT TRAINING
# ============================================================================

def train_student_distilled(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    teacher_proba: np.ndarray,
    preprocessor=None,
    alpha: float = 0.3,
    temperature: float = 1.0,
    scale_pos_weight: Optional[float] = None,
    early_stop_metric: str = 'pr_auc',
    early_stop_rounds: int = 50,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost student with distillation from OCT teacher.
    
    Uses soft label blending: p_soft = (1-α)*y + α*p_teacher
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels (0/1)
    X_val : pd.DataFrame
        Validation features
    y_val : np.ndarray
        Validation labels
    teacher_proba : np.ndarray, shape (n_train, 2)
        Teacher probabilities for training samples
    preprocessor : sklearn transformer, optional
        Preprocessor (will be fitted if None)
    alpha : float, default 0.3
        Distillation strength (0 = no distillation, 1 = only teacher)
    temperature : float, default 1.0
        Temperature scaling for teacher probabilities (not used in current implementation)
    scale_pos_weight : float, optional
        XGBoost scale_pos_weight (auto-computed if None)
    early_stop_metric : str, default 'pr_auc'
        Metric for early stopping ('pr_auc', 'recall', 'auc', 'logloss')
    early_stop_rounds : int, default 50
        Early stopping patience
    n_estimators : int, default 500
        Maximum number of trees
    max_depth : int, default 6
        Tree depth
    learning_rate : float, default 0.1
        Learning rate
    random_state : int, default 42
        Random seed
    verbose : bool, default True
        Print progress
    
    Returns:
    --------
    model : xgb.XGBClassifier
        Trained student model
    metrics : dict
        Validation metrics
    """
    # Preprocess data
    if preprocessor is None:
        raise ValueError("Preprocessor required")
    
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Get feature names
    feature_names = []
    if hasattr(preprocessor, 'transformers_'):
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'cat' and hasattr(transformer, 'named_steps'):
                # Pipeline with OHE
                ohe = transformer.named_steps.get('ohe') or transformer.named_steps.get('onehotencoder')
                if ohe:
                    feature_names.extend(ohe.get_feature_names_out(columns))
            elif name == 'num' or name == 'binary':
                feature_names.extend(columns)
    else:
        feature_names = [f'f{i}' for i in range(X_train_processed.shape[1])]
    
    # Create soft labels
    y_train_soft = (1 - alpha) * y_train.astype(float) + alpha * teacher_proba[:, 1]
    y_train_soft = np.clip(y_train_soft, 0.0, 1.0)  # Ensure [0, 1]
    
    if verbose:
        print(f"Distillation: α={alpha:.2f}, soft label range: [{y_train_soft.min():.3f}, {y_train_soft.max():.3f}]")
        print(f"  Original labels: {np.bincount(y_train.astype(int))}")
    
    # Compute scale_pos_weight if not provided
    if scale_pos_weight is None:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        if verbose:
            print(f"Auto scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_processed, label=y_train_soft, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_processed, label=y_val, feature_names=feature_names)
    
    # Set up parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss',
        'random_state': random_state,
        'tree_method': 'hist'
    }
    
    # Train with early stopping
    # Note: XGBoost's early stopping uses eval_metric by default
    # For custom metrics, we'll evaluate after training and use a callback
    evals_result = {}
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Use standard training, then evaluate custom metrics
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stop_rounds,
        verbose_eval=verbose and 10  # Print every 10 rounds
    )
    
    # Convert to sklearn API for easier use
    model_sklearn = xgb.XGBClassifier()
    model_sklearn._Booster = model
    model_sklearn._le = None  # Binary classification, no label encoder needed
    model_sklearn.classes_ = np.array([0, 1])
    model_sklearn.n_classes_ = 2
    model_sklearn._n_features = X_train_processed.shape[1]
    
    # Evaluate on validation set
    y_val_pred_proba = model_sklearn.predict_proba(X_val_processed)[:, 1]
    metrics = compute_minority_metrics(y_val, y_val_pred_proba, verbose=verbose)
    
    return model_sklearn, metrics


# ============================================================================
# MINORITY-FOCUSED EVALUATION
# ============================================================================

def compute_minority_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Compute comprehensive metrics with focus on minority class.
    
    Returns:
    --------
    metrics : dict
        Dictionary with all metrics
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Threshold-free metrics
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    
    # Optimal threshold metrics (F1, MCC, G-mean)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    
    # MCC threshold
    mcc_scores = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_t)
        mcc_scores.append(mcc)
    best_mcc_idx = np.argmax(mcc_scores)
    best_mcc_threshold = thresholds[best_mcc_idx] if best_mcc_idx < len(thresholds) else 0.5
    
    # G-mean threshold
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    specificity = 1 - fpr
    gmean = np.sqrt(tpr * specificity)
    best_gmean_idx = np.argmax(gmean)
    best_gmean_threshold = thresholds[best_gmean_idx] if best_gmean_idx < len(thresholds) else 0.5
    
    # Compute metrics at each threshold
    y_pred_f1 = (y_proba >= best_f1_threshold).astype(int)
    y_pred_mcc = (y_proba >= best_mcc_threshold).astype(int)
    y_pred_gmean = (y_proba >= best_gmean_threshold).astype(int)
    
    tn_f1, fp_f1, fn_f1, tp_f1 = confusion_matrix(y_true, y_pred_f1).ravel()
    tn_mcc, fp_mcc, fn_mcc, tp_mcc = confusion_matrix(y_true, y_pred_mcc).ravel()
    tn_gmean, fp_gmean, fn_gmean, tp_gmean = confusion_matrix(y_true, y_pred_gmean).ravel()
    
    metrics = {
        'auc': float(auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier),
        'f1_optimal': {
            'threshold': float(best_f1_threshold),
            'f1': float(f1_scores[best_f1_idx]),
            'precision': float(tp_f1 / (tp_f1 + fp_f1)) if (tp_f1 + fp_f1) > 0 else 0.0,
            'recall': float(tp_f1 / (tp_f1 + fn_f1)) if (tp_f1 + fn_f1) > 0 else 0.0,
            'specificity': float(tn_f1 / (tn_f1 + fp_f1)) if (tn_f1 + fp_f1) > 0 else 0.0
        },
        'mcc_optimal': {
            'threshold': float(best_mcc_threshold),
            'mcc': float(mcc_scores[best_mcc_idx]),
            'precision': float(tp_mcc / (tp_mcc + fp_mcc)) if (tp_mcc + fp_mcc) > 0 else 0.0,
            'recall': float(tp_mcc / (tp_mcc + fn_mcc)) if (tp_mcc + fn_mcc) > 0 else 0.0,
            'specificity': float(tn_mcc / (tn_mcc + fp_mcc)) if (tn_mcc + fp_mcc) > 0 else 0.0
        },
        'gmean_optimal': {
            'threshold': float(best_gmean_threshold),
            'gmean': float(gmean[best_gmean_idx]),
            'precision': float(tp_gmean / (tp_gmean + fp_gmean)) if (tp_gmean + fp_gmean) > 0 else 0.0,
            'recall': float(tp_gmean / (tp_gmean + fn_gmean)) if (tp_gmean + fn_gmean) > 0 else 0.0,
            'specificity': float(tn_gmean / (tn_gmean + fp_gmean)) if (tn_gmean + fp_gmean) > 0 else 0.0
        }
    }
    
    if verbose:
        print(f"\n=== Evaluation Metrics ===")
        print(f"AUC: {auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"\nF1-optimal (threshold={best_f1_threshold:.4f}):")
        print(f"  Precision: {metrics['f1_optimal']['precision']:.4f}")
        print(f"  Recall: {metrics['f1_optimal']['recall']:.4f}")
        print(f"  F1: {metrics['f1_optimal']['f1']:.4f}")
        print(f"\nMCC-optimal (threshold={best_mcc_threshold:.4f}):")
        print(f"  Precision: {metrics['mcc_optimal']['precision']:.4f}")
        print(f"  Recall (minority): {metrics['mcc_optimal']['recall']:.4f}")
        print(f"  MCC: {metrics['mcc_optimal']['mcc']:.4f}")
        print(f"\nG-mean-optimal (threshold={best_gmean_threshold:.4f}):")
        print(f"  Recall: {metrics['gmean_optimal']['recall']:.4f}")
        print(f"  Specificity: {metrics['gmean_optimal']['specificity']:.4f}")
        print(f"  G-mean: {metrics['gmean_optimal']['gmean']:.4f}")
    
    return metrics


def check_calibration(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Check model calibration using reliability curve.
    
    Returns:
    --------
    calibration_info : dict
        Calibration metrics and curve data
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # ECE (Expected Calibration Error)
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    return {
        'ece': float(ece),
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist()
    }
