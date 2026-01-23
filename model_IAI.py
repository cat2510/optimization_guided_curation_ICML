# -----------------------------------------------------------------------------
# IAI OPTIMAL CLASSIFICATION TREES 
import numpy as np
import pandas as pd
from interpretableai import iai
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix, brier_score_loss, roc_curve, roc_auc_score
from sklearn.exceptions import NotFittedError
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
                 cps=[1e-6,1e-5,1e-4]):
    """
    Hyperparameter tuning for IAI OptimalTreeClassifier
    Selects best hyperparameters based on F1 score
    """
    print(f"Finetuning OCT with depths: {depths}, minbuckets: {minbuckets}, cps: {cps}, for best PR-AUC!!!")
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
            # Only get feature names if OneHotEncoder exists, has columns, and is fitted
            # Skip if columns is empty (old get_preprocessor bug) or transformer not fitted
            if columns:  # Only process if there are actual columns to encode
                try:
                    # Check if fitted by trying to access categories_ or calling get_feature_names_out
                    ohe_features = transformer.get_feature_names_out(columns)
                    feature_names.extend(ohe_features)
                except (NotFittedError, AttributeError, ValueError):
                    # Skip if not fitted or other error (shouldn't happen with fixed get_preprocessor)
                    pass
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

        #precision = precision_score(y_val, y_pred, zero_division=0)
        #recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_val, y_pred)

        results.append({
            "depth": depth,
            "minbucket": minbucket,
            "cp": cp,
            "f1": f1,
            "pr_auc": pr_auc
        })

        if pr_auc > best_score:
            best_score = pr_auc
            best_params = (depth, minbucket, cp)
            best_model = model

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print(f"Best params: {best_params} @ PR-AUC: {best_score:.3f}")
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

from sklearn.metrics import matthews_corrcoef

def best_mcc_threshold(y_true, y_proba):
    """
    Find threshold t that maximizes MCC for predictions 1{p >= t}.

    Returns
    -------
    dict with keys:
      - threshold
      - mcc
      - y_pred
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # If y_true has only one class, MCC is undefined -> return NaN
    if np.unique(y_true).size < 2:
        return {"threshold": np.nan, "mcc": np.nan, "y_pred": np.zeros_like(y_true)}

    # If all probabilities identical, any threshold yields constant predictions -> MCC will be 0 (or NaN)
    if np.all(y_proba == y_proba[0]):
        y_pred = (y_proba >= y_proba[0]).astype(int)  # all 1s
        mcc = matthews_corrcoef(y_true, y_pred) if np.unique(y_pred).size > 1 else 0.0
        return {"threshold": float(y_proba[0]), "mcc": float(mcc), "y_pred": y_pred}

    # Candidate thresholds: midpoints between sorted unique probabilities
    uniq = np.unique(y_proba)
    uniq.sort()

    # thresholds that induce distinct labelings for rule (p >= t):
    # include extremes so we can produce all-1 and all-0 predictions
    candidates = np.concatenate((
        [uniq[0] - 1e-12],                    # all predicted 1
        (uniq[:-1] + uniq[1:]) / 2.0,         # changes happen between uniq values
        [uniq[-1] + 1e-12],                   # all predicted 0
    ))

    best = {"threshold": np.nan, "mcc": -np.inf, "y_pred": None}

    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        # MCC is defined even if y_pred is constant, but it becomes 0.0 in sklearn when denominator is 0
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best["mcc"]:
            best = {"threshold": float(t), "mcc": float(mcc), "y_pred": y_pred}

    return best

def evaluate_binary_oct(
    iai_model,
    X_test_df,
    y_test,
    preprocessor,
    feature_names,
    results_dir=None,
    ratio=None
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

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    # ------------------------------------------------------------
    # Balanced recall/specificity thresholds (Gmean or min-side)
    # ------------------------------------------------------------
    balanced = best_balanced_threshold(y_test_series.values, y_proba)

    # ------------------------------------------------------------
    # MCC (balanced precision/recall trade-off) at multiple thresholds
    # ------------------------------------------------------------
    mcc_best = best_mcc_threshold(y_test_series.values, y_proba.values if hasattr(y_proba, "values") else y_proba)
    best_mcc_threshold_value = mcc_best["threshold"]
    best_mcc_value = mcc_best["mcc"]
    y_pred_opt_mcc = mcc_best["y_pred"]
    
    # Compute recall, precision, and specificity at best MCC threshold from confusion matrix
    tn_mcc, fp_mcc, fn_mcc, tp_mcc = confusion_matrix(y_test_series, y_pred_opt_mcc).ravel()
    recall_mcc = tp_mcc / (tp_mcc + fn_mcc) if (tp_mcc + fn_mcc) else 0.0
    precision_mcc = tp_mcc / (tp_mcc + fp_mcc) if (tp_mcc + fp_mcc) else 0.0
    specificity_mcc = tn_mcc / (tn_mcc + fp_mcc) if (tn_mcc + fp_mcc) else 0.0
    
    # ------------------------------------------------------------
    # WRITE OUT PREDICTIONS TO DISK
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
        X_test_processed.to_csv(pred_path, index=False)
        print(f"✓ Saved OCT predictions to: {pred_path}")
        _save_tree_splits(iai_model, tree_path)

    print(f"AUC score: {auc:.3f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}")
    print(f"Best MCC: {best_mcc_value:.3f} @ threshold={best_mcc_threshold_value:.6f}")
    print(f"Sensitivity (Recall) @MCC*: {recall_mcc:.3f}")
    print(f"Specificity @MCC*: {specificity_mcc:.3f}")
    print(f"Balanced (G-mean) recall: {balanced['gmean_opt']['recall']:.3f}")
    print(f"Balanced (G-mean) specificity: {balanced['gmean_opt']['specificity']:.3f}")
    print("Number of leaves:", len(pd.unique(leaf_assignments)))

    # ------------------------------------------------------------
    # Return dictionary for logging
    # ------------------------------------------------------------
    # Compute precision at G-mean threshold
    precision_gmean = None
    if 'gmean_opt' in balanced and 'threshold' in balanced['gmean_opt']:
        y_pred_gmean = (y_proba >= balanced['gmean_opt']['threshold']).astype(int)
        tn_gmean, fp_gmean, fn_gmean, tp_gmean = confusion_matrix(y_test_series, y_pred_gmean).ravel()
        precision_gmean = tp_gmean / (tp_gmean + fp_gmean) if (tp_gmean + fp_gmean) else 0.0
    
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": best_mcc_value,
        "best_mcc_threshold": best_mcc_threshold_value,
        "recall_mcc": float(recall_mcc),
        "precision_mcc": float(precision_mcc),
        "optimal_f1": float(f1_scores[best_idx]),
        "balanced_recall_gmean": float(balanced["gmean_opt"]["recall"]),
        "balanced_specificity_gmean": float(balanced["gmean_opt"]["specificity"]),
        "precision_gmean": float(precision_gmean) if precision_gmean is not None else None,
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



