# -----------------------------------------------------------------------------
# IAI OPTIMAL CLASSIFICATION TREES 
import numpy as np
import pandas as pd
from interpretableai import iai
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix, brier_score_loss, roc_curve, roc_auc_score
from model_pipeline import get_preprocessor, train_test_split_enrol
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os
def train_opt_with_feature_names(X_train, treatments, outcomes,
                                 categorical_cols, numeric_cols,
                                 max_depth=5, minbucket=50, cp=0.001):
   
    # Step 1: Create and fit preprocessor
    from model_pipeline import get_preprocessor
    preprocessor = get_preprocessor(X_train, categorical_cols, numeric_cols)
    
    # Step 2: Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Step 3: Get feature names after transformation
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'ohe':
            # OneHotEncoder - get encoded feature names
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == 'num':
            # StandardScaler - keeps same names
            feature_names.extend(columns)
        elif name == 'remainder':
            # Passthrough features
            if preprocessor.remainder == 'passthrough':
                # Get columns not in other transformers
                all_cols = X_train.columns.tolist()
                used_cols = []
                for _, _, cols in preprocessor.transformers_[:-1]:
                    used_cols.extend(cols)
                remainder_cols = [c for c in all_cols if c not in used_cols]
                feature_names.extend(remainder_cols)
    
    # Step 4: Create DataFrames with proper feature names
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    (train_X, train_treatments, train_outcomes), (test_X, test_treatments, test_outcomes) = (
        iai.split_data('policy_maximize', X_train_df, treatments, outcomes, seed=123, train_proportion=0.5))
    reward_lnr = iai.CategoricalClassificationRewardEstimator(
        propensity_estimator=iai.RandomForestClassifier(),
        outcome_estimator=iai.RandomForestClassifier(),
        reward_estimator='direct_method',
        random_seed=123,
    )
    train_predictions, train_reward_score = reward_lnr.fit_predict(
        train_X, train_treatments, train_outcomes,
        propensity_score_criterion='auc', outcome_score_criterion='auc')
    train_rewards = train_predictions['reward']
    grid = iai.GridSearch(
    iai.OptimalTreePolicyMaximizer(
        random_seed=121,
        max_categoric_levels_before_warning=20,
    ),
    max_depth=range(6),
    )
    grid.fit(train_X, train_rewards)
    opt_learner = grid.get_learner()
    
    return opt_learner, preprocessor, feature_names


def train_oct_with_feature_names(X_train, y_train, 
                                 categorical_cols, numeric_cols,
                                 max_depth=5, minbucket=50, cp=0.001):
    """
    Train IAI with proper feature names by transforming data first
    
    This is the RECOMMENDED approach - transform first, then train IAI directly
    """
    
    # Step 1: Create and fit preprocessor
    preprocessor = get_preprocessor(X_train, categorical_cols, numeric_cols)
    
    # Step 2: Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Step 3: Get feature names after transformation
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'ohe':
            # OneHotEncoder - get encoded feature names
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == 'num':
            # StandardScaler - keeps same names
            feature_names.extend(columns)
        elif name == 'remainder':
            # Passthrough features
            if preprocessor.remainder == 'passthrough':
                # Get columns not in other transformers
                all_cols = X_train.columns.tolist()
                used_cols = []
                for _, _, cols in preprocessor.transformers_[:-1]:
                    used_cols.extend(cols)
                remainder_cols = [c for c in all_cols if c not in used_cols]
                feature_names.extend(remainder_cols)
    
    # Step 4: Create DataFrames with proper feature names
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    
    # Step 5: Train IAI directly (no pipeline needed)
    
    iai_model = iai.OptimalTreeClassifier(
        max_depth=max_depth,
        minbucket=minbucket,
        cp=cp,
        random_seed=42
    )
    
    iai_model.fit(X_train_df, y_train)
    
     
    return iai_model, preprocessor, feature_names


def finetune_oct(X_train, y_train, X_val, y_val, categorical_cols, numeric_cols,
                 depths=[5, 7,9],
                 minbuckets=[50, 100,150],
                 cps=[0.001, 0.01, 0.05]):
    """
    Hyperparameter tuning for IAI OptimalTreeClassifier
    Selects best hyperparameters based on F1 score
    """

    best_score = -1
    best_params = None
    best_model = None
    results = []
    
    preprocessor = get_preprocessor(X_train, categorical_cols, numeric_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    # Get feature names
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'ohe':
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == 'num':
            feature_names.extend(columns)
        elif name == 'remainder' and preprocessor.remainder == 'passthrough':
            all_cols = X_train.columns.tolist()
            used_cols = []
            for _, _, cols in preprocessor.transformers_[:-1]:
                used_cols.extend(cols)
            remainder_cols = [c for c in all_cols if c not in used_cols]
            feature_names.extend(remainder_cols)

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_transformed, columns=feature_names)

    # Grid search
    for depth, minbucket, cp in itertools.product(depths, minbuckets, cps):

        model = iai.OptimalTreeClassifier(
            max_depth=depth,
            minbucket=minbucket,
            cp=cp,
            random_seed=123
        )
        model.fit(X_train_df, y_train)

        y_pred = model.predict(X_val_df)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "depth": depth,
            "minbucket": minbucket,
            "cp": cp,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        if f1 > best_score:
            best_score = f1
            best_params = (depth, minbucket, cp)
            best_model = model

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)

    return best_model, best_params, results_df,preprocessor, feature_names


def _save_tree_splits(learner, out_path):
    """
    Extract tree splits (features and split values) from IAI OptimalTreeClassifier.
    
    Uses IAI's documented API methods. If direct extraction isn't available,
    this function will attempt alternative methods or provide guidance.
    Reference: https://docs.interpretable.ai/stable/IAI-Python/reference/#OptimalTreeClassifier
    """
    rows = []
    
    try:
        # Method 1: Try to get tree structure directly if available
        # Check for common tree export/access methods
        if hasattr(learner, 'to_dict'):
            tree_dict = learner.to_dict()
            # Parse dictionary structure if it contains node/split info
            if isinstance(tree_dict, dict):
                # Implementation depends on IAI's dict structure
                pass
        
        # Method 2: Traverse nodes using get_num_nodes() and node access methods
        num_nodes = learner.get_num_nodes()
        
        if num_nodes == 0:
            print("⚠ Tree has no nodes")
            return
        
        # Try to extract splits by checking each node
        # IAI nodes are typically 1-indexed
        for node_id in range(1, num_nodes + 1):
            try:
                # Check if node has children (internal nodes have splits)
                lower_child = learner.get_lower_child(node_id)
                
                # If node has children, try to get split information
                # Note: The exact method names depend on IAI's API
                # Common patterns: get_split_feature, get_feature, get_split_threshold, etc.
                
                # Try various possible method names for getting split feature
                feature = None
                threshold = None
                
                for method_name in ['get_split_feature', 'get_feature', 'get_node_feature']:
                    if hasattr(learner, method_name):
                        try:
                            feature = getattr(learner, method_name)(node_id)
                            break
                        except:
                            continue
                
                # Try various possible method names for getting split threshold
                for method_name in ['get_split_threshold', 'get_threshold', 'get_split_value', 'get_node_threshold']:
                    if hasattr(learner, method_name):
                        try:
                            threshold = getattr(learner, method_name)(node_id)
                            break
                        except:
                            continue
                
                if feature is not None:
                    rows.append({
                        "node_id": node_id,
                        "feature": feature,
                        "threshold": threshold,
                    })
                    
            except (AttributeError, ValueError, TypeError):
                # Node is a leaf or doesn't have accessible split info, skip
                continue
            except Exception:
                # Other errors - continue to next node
                continue
        
        # Method 3: If no splits found, try alternative approaches
        if not rows:
            # Try to get features used in the tree
            try:
                features_used = learner.get_features_used()
                print(f"⚠ Could not extract individual splits")
                print(f"   Tree uses {len(features_used)} feature(s): {features_used}")
                print("   Consider using learner.show_tree() for visualization")
            except:
                pass
        
        # Save splits if found
        if rows:
            splits_df = pd.DataFrame(rows)
            splits_path = out_path.replace(".json", "_splits.csv")
            splits_df.to_csv(splits_path, index=False)
            print(f"✓ Saved split table ({len(rows)} splits) to: {splits_path}")
        else:
            print(f"⚠ No splits extracted (tree has {num_nodes} node(s))")
            print("   This may indicate:")
            print("   1. Tree is a single leaf (no splits)")
            print("   2. IAI API methods differ from expected")
            print("   3. Check IAI docs for correct split extraction method")
            
    except Exception as e:
        print(f"⚠ Error extracting tree splits: {e}")
        print("   Available methods on learner:")
        methods = [m for m in dir(learner) if not m.startswith('_') and 'split' in m.lower()]
        if methods:
            print(f"   Split-related: {methods[:5]}")

def evaluate_binary_oct(
    iai_model,
    X_test_df,
    y_test,
    preprocessor,
    feature_names,
    results_dir=None,              # ← NEW: where to save predictions
    ratio=None                     # ← NEW: to encode file name
):
    print(f"Test dataset for OCT application: {len(X_test_df):,} samples")

    # ------------------------------------------------------------
    # Preprocessing + OCT Predictions
    # ------------------------------------------------------------
    try:
        X_test_processed = preprocessor.transform(X_test_df)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

        # Base predictions from OCT
        y_pred_default = iai_model.predict(X_test_processed)
        y_proba = iai_model.predict_proba(X_test_processed).iloc[:, 1]

        X_test_processed["predicted_proba"] = y_proba
        X_test_processed["predicted_class_default"] = y_pred_default
        leaf_assignments = iai_model.apply(X_test_processed)

        print("✓ Predictions completed")

    except Exception as e:
        print(f"✗ Error applying OCT: {e}")
        raise e

    X_test_processed["leaf_assignment"] = leaf_assignments
    X_test_processed["predicted_cost_stratum_default"] = y_pred_default

    y_test_series = pd.Series(y_test).reset_index(drop=True)

    # ------------------------------------------------------------
    # AUC metrics (threshold-free)
    # ------------------------------------------------------------
    auc = iai_model.score(X_test_processed, y_test_series, criterion="auc")
    pr_auc = average_precision_score(y_test_series, y_proba)

    # ------------------------------------------------------------
    # F1-optimal thresholding
    # ------------------------------------------------------------
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test_series, y_proba)
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_opt = (y_proba >= best_threshold).astype(int)
    X_test_processed["predicted_class_optf1"] = y_pred_opt

    # ------------------------------------------------------------
    # Balanced recall/specificity thresholds (no F1/Youden)
    # ------------------------------------------------------------
    balanced = best_balanced_threshold(y_test_series.values, y_proba)
    # ------------------------------------------------------------
    # **WRITE OUT PREDICTIONS TO DISK**
    # ------------------------------------------------------------
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/predictions", exist_ok=True)
        if ratio is not None:
            pred_path = f"{results_dir}/predictions/oct_predictions_ratio_{ratio:.2f}.csv"
            tree_path = f"{results_dir}/oct_tree_ratio_{ratio:.2f}.json"
        else:
            pred_path = f"{results_dir}/predictions/oct_predictions.csv"
            tree_path = f"{results_dir}/oct_tree.json"
        X_test_processed.to_csv(pred_path, index=False)       # ← NEW OUTPUT FILE
        print(f"✓ Saved OCT predictions to: {pred_path}")
        _save_tree_splits(iai_model, tree_path)
    # Confusion matrix for optimized threshold
    tn, fp, fn, tp = confusion_matrix(y_test_series, y_pred_opt).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    tn, fp, fn, tp = confusion_matrix(y_test_series, y_pred_default).ravel()
    sensitivity_default = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_default = tn / (tn + fp) if (tn + fp) else 0.0
    
    print(f"AUC score: {auc:.3f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Balanced (G-mean) recall: {balanced['gmean_opt']['recall']:.3f}")
    print(f"Balanced (G-mean) specificity: {balanced['gmean_opt']['specificity']:.3f}")
    print(f"Sensitivity (default): {sensitivity_default:.3f}")
    print(f"Specificity (default): {specificity_default:.3f}")
    print("Number of leaves:", len(pd.unique(leaf_assignments)))


    # ------------------------------------------------------------
    # Return dictionary for logging
    # ------------------------------------------------------------
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1_threshold": best_threshold,
        "optimal_f1": f1_scores[best_idx],
        "sensitivity_f1": sensitivity,
        "specificity_f1": specificity,
        "sensitivity_default": sensitivity_default,
        "specificity_default": specificity_default,
        "balanced_threshold_gmean": balanced["gmean_opt"]["threshold"],
        "balanced_recall_gmean": balanced["gmean_opt"]["recall"],
        "balanced_specificity_gmean": balanced["gmean_opt"]["specificity"],
        "balanced_threshold_minside": balanced["minside_opt"]["threshold"],
        "balanced_recall_minside": balanced["minside_opt"]["recall"],
        "balanced_specificity_minside": balanced["minside_opt"]["specificity"],
        
    }


# -------------------------------
# Expected Calibration Error (ECE)
# -------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += np.abs(avg_conf - avg_acc) * (mask.sum() / len(y_prob))
    return ece


# ------------------------------------------------------------
# Find threshold satisfying constraints (precision/recall)
# ------------------------------------------------------------
def find_feasible_threshold(y_true, y_prob, min_precision=0.8, min_recall=0.8):
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)

    valid = []
    # thresholds has len-1 relative to precision/recall curves
    thr_padded = list(thresholds) + [thresholds[-1]]

    for p, r, t in zip(precision_curve, recall_curve, thr_padded):
        if p >= min_precision and r >= min_recall:
            valid.append((t, p, r))

    if len(valid) == 0:
        return None

    # choose maximum F1 in feasible region
    best = max(valid, key=lambda x: 2 * x[1] * x[2] / (x[1] + x[2] + 1e-12))
    return {"threshold": best[0], "precision": best[1], "recall": best[2]}

def best_balanced_threshold(y_true, y_prob):
    """
    Find thresholds that balance recall and specificity without F1/Youden.
    Returns two candidates:
      - gmean_opt: maximizes sqrt(recall * specificity)
      - minside_opt: maximizes min(recall, specificity)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    recall = tpr

    gmean = np.sqrt(recall * specificity)
    idx_g = int(np.argmax(gmean))

    min_side = np.minimum(recall, specificity)
    idx_min = int(np.argmax(min_side))

    def pack(idx):
        return {
            "threshold": thresholds[idx],
            "recall": recall[idx],
            "specificity": specificity[idx],
            "gmean": np.sqrt(recall[idx] * specificity[idx]),
            "min_recall_spec": min(recall[idx], specificity[idx]),
        }

    return {"gmean_opt": pack(idx_g), "minside_opt": pack(idx_min)}


def calculate_auc_by_vanilla_subgroups(
    anchor_model_pred_path,
    y_test,
    test_model_pred_path=None,
    enrolid_col=None
):
    """
    Calculate AUC in subgroups defined by vanilla OCT leaf assignments.
    Uses leaf_assignment directly from predictions CSV files for reliability.
    
    Note: Vanilla OCT assigns constant probabilities within each leaf (tree behavior),
    so vanilla ROC AUC within subgroups will be 0.5 (no discrimination within leaf).
    The balanced OCT can have varying probabilities within the same leaves, allowing
    for discrimination. For fair comparison, also check calibration_error and mean_proba.
    
    Parameters
    ----------
    anchor_model_pred_path : str
        Path to anchor model OCT predictions CSV (must have 'leaf_assignment' and 'predicted_proba' columns)
    y_test : pd.Series or array-like
        True labels (must align with predictions by position/index)
    test_model_pred_path : str, optional
        Path to test model OCT predictions CSV (must have 'predicted_proba' column).
        If provided, will calculate metrics using balanced probabilities in vanilla subgroups.
    enrolid_col : str, optional
        (Deprecated - not used) Kept for backward compatibility.
    
    Returns
    -------
    dict
        Dictionary with:
        - 'subgroups': DataFrame with subgroup assignments and metrics including:
          * positive_rate: actual positive rate in subgroup
          * vanilla_mean_proba: mean vanilla probability (constant within leaf)
          * vanilla_calibration_error: |mean_proba - positive_rate|
          * balanced_mean_proba: mean balanced probability
          * balanced_calibration_error: |mean_proba - positive_rate|
          * vanilla_roc_auc: Will be 0.5 if constant (expected for trees)
          * balanced_roc_auc: Can be >0.5 if probabilities vary
        - 'vanilla_roc_auc_by_subgroup': dict mapping subgroup_id to vanilla ROC AUC
        - 'vanilla_pr_auc_by_subgroup': dict mapping subgroup_id to vanilla PR-AUC
        - 'balanced_roc_auc_by_subgroup': dict mapping subgroup_id to balanced ROC AUC (if provided)
        - 'balanced_pr_auc_by_subgroup': dict mapping subgroup_id to balanced PR-AUC (if provided)
    """
    # Load vanilla OCT predictions
    vanilla_preds = pd.read_csv(anchor_model_pred_path)
    
    if 'leaf_assignment' not in vanilla_preds.columns:
        raise ValueError(f"Vanilla predictions CSV must have 'leaf_assignment' column: {anchor_model_pred_path}")
    if 'predicted_proba' not in vanilla_preds.columns:
        raise ValueError(f"Vanilla predictions CSV must have 'predicted_proba' column: {anchor_model_pred_path}")
    
    # Load balanced OCT predictions if provided
    balanced_preds = pd.read_csv(test_model_pred_path)
    if 'predicted_proba' not in balanced_preds.columns:
        raise ValueError(f"Balanced predictions CSV must have 'predicted_proba' column: {test_model_pred_path}")
    
    # Prepare y_test as Series
    y_test = pd.Series(y_test)
    
    # Match predictions with y_test
    if enrolid_col is not None:
        # Match by ENROLID
        if enrolid_col not in vanilla_preds.columns:
            raise ValueError(f"ENROLID column '{enrolid_col}' not found in anchor model predictions")
        # Set ENROLID as index for matching
        vanilla_preds = vanilla_preds.set_index(enrolid_col)
        if balanced_preds is not None:
            if enrolid_col not in balanced_preds.columns:
                raise ValueError(f"ENROLID column '{enrolid_col}' not found in test model predictions")
            balanced_preds = balanced_preds.set_index(enrolid_col)
        # y_test should also have ENROLID as index
        if not isinstance(y_test.index, pd.Index) or enrolid_col not in str(y_test.index.name):
            # Try to set index if y_test has ENROLID column
            if hasattr(y_test, 'name') and y_test.name == enrolid_col:
                pass  # Already indexed
            else:
                raise ValueError(f"y_test should have ENROLID as index when enrolid_col is provided")
    
    # Align by index (assuming same order or mat ching indices)
    aligned_indices = vanilla_preds.index.intersection(y_test.index)
    if len(aligned_indices) == 0:
        # Try matching by position if indices don't match
        min_len = min(len(vanilla_preds), len(y_test))
        aligned_indices = vanilla_preds.index[:min_len]
        y_test_aligned = y_test.iloc[:min_len]
        print(f"⚠ Warning: Indices don't match. Using first {min_len} samples by position")
    else:
        y_test_aligned = y_test.loc[aligned_indices]
    
    vanilla_leaf_assignments = vanilla_preds.loc[aligned_indices, 'leaf_assignment']
    vanilla_probas = vanilla_preds.loc[aligned_indices, 'predicted_proba']
    
    if balanced_preds is not None:
        balanced_indices = balanced_preds.index.intersection(aligned_indices)
        if len(balanced_indices) != len(aligned_indices):
            print(f"⚠ Warning: Only {len(balanced_indices)}/{len(aligned_indices)} samples matched for balanced predictions")
        balanced_probas = balanced_preds.loc[balanced_indices, 'predicted_proba']
        # Align balanced with vanilla indices
        balanced_probas_aligned = pd.Series(index=aligned_indices, dtype=float)
        balanced_probas_aligned.loc[balanced_indices] = balanced_probas
    else:
        balanced_probas_aligned = None
    
    # Group by leaf_assignment and calculate metrics
    results = []
    
    for leaf_id in sorted(vanilla_leaf_assignments.unique()):
        # Get samples in this leaf (by position/index)
        leaf_mask = vanilla_leaf_assignments == leaf_id
        leaf_indices = np.where(leaf_mask)[0]  # Get integer positions
        
        if len(leaf_indices) == 0:
            continue
        
        # Get labels and probabilities for this subgroup
        y_subgroup = y_test_aligned.iloc[leaf_indices]
        proba_vanilla_subgroup = vanilla_probas.iloc[leaf_indices]
        
        # Calculate vanilla metrics for this subgroup (ROC AUC and PR-AUC)
        vanilla_roc_auc = np.nan
        vanilla_pr_auc = np.nan
        if len(y_subgroup.unique()) < 2:
            n_pos = len(y_subgroup[y_subgroup == 1])
            n_neg = len(y_subgroup[y_subgroup == 0])
        else:
            try:
                vanilla_roc_auc = roc_auc_score(y_subgroup, proba_vanilla_subgroup)
                vanilla_pr_auc = average_precision_score(y_subgroup, proba_vanilla_subgroup)
                n_pos = int((y_subgroup == 1).sum())
                n_neg = int((y_subgroup == 0).sum())
            except ValueError:
                n_pos = int((y_subgroup == 1).sum())
                n_neg = int((y_subgroup == 0).sum())
        
        # Calculate balanced metrics if provided (ROC AUC and PR-AUC)
        balanced_roc_auc = np.nan
        balanced_pr_auc = np.nan
        test_model_proba_mean = np.nan
        test_model_proba_std = np.nan
        test_model_proba_min = np.nan
        test_model_proba_max = np.nan
        if balanced_probas_aligned is not None:
            proba_balanced_subgroup = balanced_probas_aligned.iloc[leaf_indices]
            # Remove NaN values (samples not in balanced predictions)
            valid_mask = ~proba_balanced_subgroup.isna()
            if valid_mask.sum() > 0:
                valid_positions = np.where(valid_mask)[0]
                valid_leaf_indices = leaf_indices[valid_positions]
                y_subgroup_balanced = y_test_aligned.iloc[valid_leaf_indices]
                proba_balanced_subgroup_valid = proba_balanced_subgroup.iloc[valid_positions]
                
                # Calculate probability distribution statistics
                test_model_proba_mean = float(proba_balanced_subgroup_valid.mean())
                test_model_proba_std = float(proba_balanced_subgroup_valid.std())
                test_model_proba_min = float(proba_balanced_subgroup_valid.min())
                test_model_proba_max = float(proba_balanced_subgroup_valid.max())
                
                if len(y_subgroup_balanced.unique()) >= 2:
                    try:
                        balanced_roc_auc = roc_auc_score(y_subgroup_balanced, proba_balanced_subgroup_valid)
                        balanced_pr_auc = average_precision_score(y_subgroup_balanced, proba_balanced_subgroup_valid)
                    except ValueError:
                        pass
        
        results.append({
            'subgroup_id': leaf_id,
            'n_samples': len(leaf_indices),
            'n_positive': n_pos,
            'n_negative': n_neg,
            'positive_rate': n_pos / len(leaf_indices) if len(leaf_indices) > 0 else 0,
            'anchor_model_roc_auc': vanilla_roc_auc,  # Will be 0.5 if constant (expected for trees)
            'anchor_model_pr_auc': vanilla_pr_auc,
            'test_model_roc_auc': balanced_roc_auc,  # Can be >0.5 if probabilities vary within leaf
            'test_model_pr_auc': balanced_pr_auc,
            'test_model_proba_mean': test_model_proba_mean,  # Mean probability in subgroup
            'test_model_proba_std': test_model_proba_std,  # Std of probabilities (low = homogeneous)
            'test_model_proba_range': test_model_proba_max - test_model_proba_min  # Range (high = heterogeneous)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['n_samples'] > 0].sort_values('subgroup_id')
    
    # Calculate overall AUC for comparison (on all test samples, not just within subgroups)
    overall_vanilla_auc = roc_auc_score(y_test_aligned, vanilla_probas)
    overall_vanilla_pr_auc = average_precision_score(y_test_aligned, vanilla_probas)
    
    overall_balanced_auc = np.nan
    overall_balanced_pr_auc = np.nan
    if balanced_probas_aligned is not None:
        valid_balanced = ~balanced_probas_aligned.isna()
        if valid_balanced.sum() > 0:
            overall_balanced_auc = roc_auc_score(y_test_aligned[valid_balanced], balanced_probas_aligned[valid_balanced])
            overall_balanced_pr_auc = average_precision_score(y_test_aligned[valid_balanced], balanced_probas_aligned[valid_balanced])
    
    # Create summary dictionaries
    vanilla_roc_auc_dict = dict(zip(results_df['subgroup_id'], results_df['anchor_model_roc_auc']))
    vanilla_pr_auc_dict = dict(zip(results_df['subgroup_id'], results_df['anchor_model_pr_auc']))
    balanced_roc_auc_dict = dict(zip(results_df['subgroup_id'], results_df['test_model_roc_auc'])) if balanced_preds is not None else {}
    balanced_pr_auc_dict = dict(zip(results_df['subgroup_id'], results_df['test_model_pr_auc'])) if balanced_preds is not None else {}
    
    return {
        'subgroups': results_df,
        'overall_anchor_model_roc_auc': overall_vanilla_auc,
        'overall_anchor_model_pr_auc': overall_vanilla_pr_auc,
        'overall_test_model_roc_auc': overall_balanced_auc,
        'overall_test_model_pr_auc': overall_balanced_pr_auc,
        'anchor_model_roc_auc_by_subgroup': vanilla_roc_auc_dict,
        'anchor_model_pr_auc_by_subgroup': vanilla_pr_auc_dict,
        'test_model_roc_auc_by_subgroup': balanced_roc_auc_dict,
        'test_model_pr_auc_by_subgroup': balanced_pr_auc_dict
    }


