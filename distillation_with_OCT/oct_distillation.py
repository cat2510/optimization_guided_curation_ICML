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
    matthews_corrcoef, brier_score_loss, roc_curve
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
# RULE-BASED FEATURE EXTRACTION
# ============================================================================

def extract_oct_rule_features(
    teacher: OCTTeacher,
    X: pd.DataFrame,
    include_leaf_assignment: bool = True,
    include_rule_indicators: bool = True,
    include_rule_confidence: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract OCT's interpretable decision rules as features.
    
    This captures the structure of OCT's rules (which splits were triggered)
    rather than just the probabilities, allowing XGBoost to learn from
    the interpretable patterns while still discovering additional patterns.
    
    Parameters:
    -----------
    teacher : OCTTeacher
        Trained OCT teacher
    X : pd.DataFrame
        Features (raw, will be preprocessed)
    include_leaf_assignment : bool
        Include leaf assignment as a categorical feature
    include_rule_indicators : bool
        Include binary indicators for each rule (split) that was triggered
    include_rule_confidence : bool
        Include confidence scores based on leaf purity
    
    Returns:
    --------
    rule_features : pd.DataFrame
        Rule-based features to concatenate with original features
    rule_metadata : dict
        Metadata about the rules (for interpretation)
    """
    # Preprocess
    if teacher.preprocessor is not None:
        X_processed = teacher.preprocessor.transform(X)
        if teacher.feature_names:
            X_processed = pd.DataFrame(X_processed, columns=teacher.feature_names)
        else:
            X_processed = pd.DataFrame(X_processed)
    else:
        X_processed = X.copy()
    
    n_samples = len(X_processed)
    rule_features_dict = {}
    rule_metadata = {
        'n_rules': 0,
        'rule_names': [],
        'leaf_stats': {}
    }
    
    # Get leaf assignments
    if teacher.model is not None:
        leaf_assignments = teacher.model.apply(X_processed)
    elif teacher.splits_df is not None:
        leaf_assignments = teacher._apply_splits_manual(X_processed)
    else:
        raise ValueError("Need either model or splits_df to extract rules")
    
    # 1. Leaf assignment (categorical feature)
    if include_leaf_assignment:
        rule_features_dict['oct_leaf_id'] = leaf_assignments
        unique_leaves = np.unique(leaf_assignments)
        rule_metadata['leaf_stats'] = {
            'n_unique_leaves': len(unique_leaves),
            'leaf_ids': unique_leaves.tolist()
        }
    
    # 2. Rule indicators (which splits were triggered)
    if include_rule_indicators and teacher.splits_df is not None:
        # For each rule (split), create a binary indicator
        for _, row in teacher.splits_df.iterrows():
            node_id = int(row['node_id'])
            feature = row['feature']
            threshold = float(row['threshold'])
            
            # Find matching feature in processed data
            matching_cols = [c for c in X_processed.columns if feature in str(c)]
            if not matching_cols:
                # Try exact match
                if feature in X_processed.columns:
                    matching_cols = [feature]
            
            if matching_cols:
                feature_col = matching_cols[0]
                # Rule: feature <= threshold (left child) or > threshold (right child)
                # We'll create indicators for both paths
                rule_name_left = f'oct_rule_{node_id}_left'  # feature <= threshold
                rule_name_right = f'oct_rule_{node_id}_right'  # feature > threshold
                
                values = X_processed[feature_col].values
                rule_features_dict[rule_name_left] = (values <= threshold).astype(int)
                rule_features_dict[rule_name_right] = (values > threshold).astype(int)
                
                rule_metadata['rule_names'].extend([rule_name_left, rule_name_right])
                rule_metadata['n_rules'] += 2
    
    # 3. Rule confidence (based on leaf purity and size)
    if include_rule_confidence and teacher.leaf_probs:
        # Confidence = how "pure" the leaf is (how far from 0.5)
        # Also consider leaf size (larger leaves = more confident)
        leaf_confidences = np.zeros(n_samples)
        leaf_sizes = {}
        
        # Compute leaf sizes from training data if available
        if teacher.X_train_processed is not None:
            train_leaf_assignments = teacher.model.apply(teacher.X_train_processed) if teacher.model else teacher._apply_splits_manual(teacher.X_train_processed)
            for leaf_id in np.unique(train_leaf_assignments):
                leaf_sizes[leaf_id] = np.sum(train_leaf_assignments == leaf_id)
        
        for i, leaf_id in enumerate(leaf_assignments):
            p_positive = teacher.leaf_probs.get(leaf_id, 0.5)
            # Purity: distance from 0.5 (max at 0 or 1)
            purity = abs(p_positive - 0.5) * 2  # Scale to [0, 1]
            
            # Size confidence: larger leaves are more stable
            leaf_size = leaf_sizes.get(leaf_id, 1)
            size_confidence = min(1.0, np.log(leaf_size + 1) / np.log(100))  # Log scale, cap at 100 samples
            
            # Combined confidence
            leaf_confidences[i] = 0.7 * purity + 0.3 * size_confidence
        
        rule_features_dict['oct_rule_confidence'] = leaf_confidences
    
    # Convert to DataFrame
    if rule_features_dict:
        rule_features = pd.DataFrame(rule_features_dict, index=X.index if hasattr(X, 'index') else range(n_samples))
    else:
        rule_features = pd.DataFrame(index=X.index if hasattr(X, 'index') else range(n_samples))
    
    return rule_features, rule_metadata


def compute_rule_based_sample_weights(
    teacher: OCTTeacher,
    X: pd.DataFrame,
    y: np.ndarray,
    weight_strategy: str = 'confidence',
    min_weight: float = 1.0,
    max_weight: float = 2.0,
    confidence_exponent: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Compute sample weights based on OCT's rule confidence.
    
    Samples where OCT's rules are clear/confident get higher weights,
    encouraging XGBoost to pay more attention to these interpretable patterns.
    
    Default min_weight=1.0 (boost-only): low-confidence samples keep weight 1.0;
    only confident samples get weight > 1. This avoids downweighting most data
    when leaf purity is moderate (e.g. p_positive ~ 0.6 → low purity → small
    confidence), which would otherwise shrink the training signal.
    
    Parameters:
    -----------
    teacher : OCTTeacher
        Trained OCT teacher
    X : pd.DataFrame
        Features
    y : np.ndarray
        Labels
    weight_strategy : str
        'confidence': Weight by rule confidence (leaf purity + size)
        'agreement': Higher weight when OCT and label agree
        'minority_boost': Boost minority class samples in confident leaves
    min_weight : float
        Minimum sample weight (default 1.0 = no downweighting)
    max_weight : float
        Maximum sample weight (samples in high-purity leaves can get this)
    confidence_exponent : float, default 1.0
        Map confidence to weight via confidence ** exponent. Use > 1 (e.g. 2.0)
        to boost only high-confidence samples more strongly.
    
    Returns:
    --------
    weights : np.ndarray
        Sample weights for training
    """
    # Get rule features to compute confidence
    rule_features, _ = extract_oct_rule_features(
        teacher, X, include_rule_confidence=True, include_rule_indicators=False
    )
    
    if 'oct_rule_confidence' in rule_features.columns:
        confidences = rule_features['oct_rule_confidence'].values
    else:
        # Fallback: uniform weights
        return np.ones(len(X))
    
    # Steepen the confidence→weight curve when exponent > 1
    if confidence_exponent != 1.0:
        confidences = np.power(np.clip(confidences, 0.0, 1.0), confidence_exponent)

    if weight_strategy == 'confidence':
        # Weight by confidence directly
        weights = min_weight + (max_weight - min_weight) * confidences
    
    elif weight_strategy == 'agreement':
        # Higher weight when OCT prediction agrees with label
        leaf_assignments = rule_features['oct_leaf_id'].values if 'oct_leaf_id' in rule_features.columns else None
        if leaf_assignments is not None and teacher.leaf_probs:
            oct_predictions = np.array([teacher.leaf_probs.get(leaf_id, 0.5) for leaf_id in leaf_assignments])
            oct_pred_binary = (oct_predictions >= 0.5).astype(int)
            agreement = (oct_pred_binary == y).astype(float)
            # Combine confidence and agreement
            weights = min_weight + (max_weight - min_weight) * (0.5 * confidences + 0.5 * agreement)
        else:
            weights = min_weight + (max_weight - min_weight) * confidences
    
    elif weight_strategy == 'minority_boost':
        # Boost minority class samples in confident leaves; others at min_weight (default 1.0)
        weights = np.ones(len(X)) * min_weight
        minority_mask = (y == 1)
        weights[minority_mask] = min_weight + (max_weight - min_weight) * confidences[minority_mask]
        weights[~minority_mask] = min_weight + 0.5 * (max_weight - min_weight) * confidences[~minority_mask]
    
    else:
        weights = np.ones(len(X))
    
    return np.clip(weights, min_weight, max_weight)


# ============================================================================
# BOOSTER WRAPPER (sklearn-like API without mutating XGBClassifier)
# ============================================================================

class _BoosterWrapper:
    """Thin wrapper around xgb.Booster with predict_proba and feature_importances_."""

    def __init__(self, booster, feature_names: List[str], n_features: int):
        self._booster = booster
        self.feature_names = feature_names
        self._n_features = n_features

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X)
        dmat = xgb.DMatrix(X, feature_names=self.feature_names if len(self.feature_names) == X.shape[1] else None)
        p_pos = self._booster.predict(dmat)  # binary:logistic returns P(class=1)
        p_pos = np.clip(np.asarray(p_pos).ravel(), 0.0, 1.0)
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, X) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        score = self._booster.get_score(importance_type="weight")
        out = np.zeros(self._n_features)
        for i in range(self._n_features):
            key = f"f{i}"
            out[i] = score.get(key, 0.0)
        return out


# ============================================================================
# DISTILLED STUDENT TRAINING (RULE-BASED)
# ============================================================================

def train_student_distilled(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    teacher: OCTTeacher,
    preprocessor=None,
    use_rule_features: bool = True,
    use_sample_weights: bool = True,
    weight_strategy: str = 'confidence',
    weight_min: float = 1.0,
    weight_max: float = 2.0,
    confidence_exponent: float = 1.0,
    rule_feature_scale: float = 1.0,
    include_rule_confidence_feature: bool = True,
    scale_pos_weight: Optional[float] = None,
    early_stop_metric: str = 'pr_auc',
    early_stop_rounds: int = 50,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Union[xgb.XGBClassifier, _BoosterWrapper], Dict]:
    """
    Train XGBoost student with rule-based distillation from OCT teacher.
    
    Instead of blending probabilities, this uses OCT's interpretable decision rules:
    1. Extracts rule-based features (leaf assignments, rule indicators, confidence)
    2. Uses sample weighting based on rule confidence
    3. Allows XGBoost to learn from interpretable patterns while discovering additional patterns
    
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
    teacher : OCTTeacher
        Trained OCT teacher
    preprocessor : sklearn transformer, optional
        Preprocessor (will be fitted if None)
    use_rule_features : bool, default True
        Add OCT rule features (leaf assignment, rule indicators) to XGBoost
    use_sample_weights : bool, default True
        Use rule-based sample weighting
    weight_strategy : str, default 'confidence'
        Sample weighting strategy ('confidence', 'agreement', 'minority_boost')
    weight_min : float, default 1.0
        Minimum sample weight (1.0 = no downweighting; only boost confident samples)
    weight_max : float, default 2.0
        Maximum sample weight (samples in high-purity leaves can get this)
    confidence_exponent : float, default 1.0
        Use confidence ** exponent for weighting; > 1 strengthens effect for high-confidence samples
    rule_feature_scale : float, default 1.0
        Multiply rule features by this before concatenating; > 1 makes OCT rules count more in splits
    include_rule_confidence_feature : bool, default True
        If False, do not add oct_rule_confidence as a feature (only leaf id and rule indicators); sample weights can still use confidence
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
        Validation metrics and rule metadata
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
    
    # Extract rule-based features
    rule_features_train = None
    rule_features_val = None
    rule_metadata = {}
    
    if use_rule_features:
        if verbose:
            print("Extracting OCT rule-based features...")
        rule_features_train, rule_metadata = extract_oct_rule_features(
            teacher, X_train, include_leaf_assignment=True,
            include_rule_indicators=True, include_rule_confidence=include_rule_confidence_feature
        )
        rule_features_val, _ = extract_oct_rule_features(
            teacher, X_val, include_leaf_assignment=True,
            include_rule_indicators=True, include_rule_confidence=include_rule_confidence_feature
        )
        
        # Scale rule features so OCT rules can affect splits more (rule_feature_scale > 1)
        rtrain = rule_features_train.values.astype(np.float64) * rule_feature_scale
        rval = rule_features_val.values.astype(np.float64) * rule_feature_scale
        X_train_processed = np.hstack([X_train_processed, rtrain])
        X_val_processed = np.hstack([X_val_processed, rval])
        
        # Update feature names
        rule_feature_names = list(rule_features_train.columns)
        feature_names = feature_names + rule_feature_names
        
        if verbose:
            print(f"  Added {len(rule_feature_names)} rule features")
            if rule_feature_scale != 1.0:
                print(f"  Rule feature scale: {rule_feature_scale} (OCT rules emphasized in splits)")
            print(f"  Total features: {len(feature_names)}")
            print(f"  Rule metadata: {len(rule_metadata.get('rule_names', []))} rule indicators")
    
    # Compute sample weights based on rule confidence
    sample_weights = None
    if use_sample_weights:
        if verbose:
            print(f"Computing rule-based sample weights (strategy: {weight_strategy})...")
            if confidence_exponent != 1.0:
                print(f"  Confidence exponent: {confidence_exponent} (steeper boost for high-confidence samples)")
        sample_weights = compute_rule_based_sample_weights(
            teacher, X_train, y_train, weight_strategy=weight_strategy,
            min_weight=weight_min, max_weight=weight_max,
            confidence_exponent=confidence_exponent
        )
        if verbose:
            print(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
            print(f"  Mean weight: {sample_weights.mean():.3f}")
    
    # Compute scale_pos_weight if not provided
    if scale_pos_weight is None:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        if verbose:
            print(f"Auto scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Prepare DMatrix for XGBoost (use true labels, not soft labels)
    dtrain = xgb.DMatrix(X_train_processed, label=y_train, feature_names=feature_names, weight=sample_weights)
    dval = xgb.DMatrix(X_val_processed, label=y_val, feature_names=feature_names)
    
    # Set up parameters (aligned with baseline XGBClassifier in train_student.py for fair comparison)
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
    
    # Wrap Booster in an object with sklearn-like predict_proba and feature_importances_
    # (avoid mutating XGBClassifier.classes_ which is read-only in newer xgboost)
    model_sklearn = _BoosterWrapper(model, feature_names, n_features=X_train_processed.shape[1])
    
    # Evaluate on validation set
    y_val_pred_proba = model_sklearn.predict_proba(X_val_processed)[:, 1]
    metrics = compute_minority_metrics(y_val, y_val_pred_proba, verbose=verbose)
    
    # Add rule metadata to metrics
    metrics['rule_metadata'] = rule_metadata
    metrics['n_rule_features'] = len(rule_feature_names) if use_rule_features else 0
    metrics['used_sample_weights'] = use_sample_weights
    
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
