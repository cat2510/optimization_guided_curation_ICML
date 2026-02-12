# -----------------------------------------------------------------------------
# IAI OPTIMAL CLASSIFICATION TREES 
import numpy as np
import pandas as pd
from interpretableai import iai
import itertools
import matplotlib.pyplot as plt
import os,time
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Resource tracking will be limited.")

from sklearn.metrics import roc_curve, roc_auc_score,f1_score, average_precision_score, precision_recall_curve, confusion_matrix, matthews_corrcoef
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
try:
    from scipy import sparse
except Exception:
    sparse = None
def train_test_split_enrol(df, target_col, test_size=0.3, random_state=42,verbose=True):
    """
    Splits df by ENROLID into train/test, stratifying on target_col.
    Returns: train_df, test_df
    """
    # Ensure ENROLID is unique per row
    assert df["ENROLID"].is_unique, "DataFrame must have one row per ENROLID"

    # Stratified split on ENROLID
    train_ids, test_ids = train_test_split(
        df["ENROLID"],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    train = df[df["ENROLID"].isin(train_ids)].reset_index(drop=True)
    test  = df[df["ENROLID"].isin(test_ids)].reset_index(drop=True)
    if verbose:
        # Debug prints
        print("Train/Test shapes:", train.shape, test.shape)
        print("Train distribution of {}:".format(target_col))
        print(train[target_col].value_counts(normalize=True))
        print("Test distribution of {}:".format(target_col))
        print(test[target_col].value_counts(normalize=True))

    return train_ids, test_ids, train, test

def get_cat_columns(df):
    cols= df.select_dtypes(include=["object","category","string"]).columns.tolist()
    return [col for col in cols if col != "ENROLID"]
def get_bin_flag_columns(df):
    return [col for col in df.columns if col.startswith("has_") or "THRCLS" in col.upper()
    or col.endswith("adherent") or col.startswith("early_")
    or col.startswith("is_") or "is_increasing" in col.lower() or "is_decreasing" in col.lower()]
def get_true_num_columns(df, CAT_COLUMNS,BIN_FLAG_COLUMNS):
    return [
        col for col in df.columns
        if (col not in ['ENROLID']
            and col not in CAT_COLUMNS+BIN_FLAG_COLUMNS
        )
    ]

def train_oct_with_feature_names(X_train, y_train, 
                                 categorical_cols, numeric_cols,
                                 max_depth=5, minbucket=50, cp=0.001):
    """
    Train IAI with proper feature names by transforming data first
    
    This is the RECOMMENDED approach - transform first, then train IAI directly
    """
    
    # Step 1: Create and fit preprocessor
    preprocessor = get_preprocessor_with_impute(X_train, categorical_cols, numeric_cols)
    
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
        random_seed=42,
        missingdatamode='always_left'  # Handle missing data: always_left, always_right, or separate_class
    )
    
    iai_model.fit(X_train_df, y_train)
    
     
    return iai_model, preprocessor, feature_names


def get_preprocessor_with_impute(X_train, categorical_cols, numeric_cols, binary_cols=None, verbose=True):
    """
    Build preprocessor with conditional imputation.
    Only includes SimpleImputer if there are missing values in the data.
    
    Binary flag columns are passed through without scaling (they're already 0/1).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training data to check for missing values
    categorical_cols : list
        List of categorical column names
    numeric_cols : list
        List of numeric column names (will be scaled)
    binary_cols : list, optional
        List of binary flag column names (0/1). These will be passed through
        without scaling. If None, binary columns are dropped.
    verbose : bool
        Whether to print preprocessing details
    """
    # Filter columns to only those present in X_train
    # This handles cases where column lists include columns not in the actual data
    cat_cols_present = [col for col in categorical_cols if col in X_train.columns] if categorical_cols else []
    num_cols_present = [col for col in numeric_cols if col in X_train.columns] if numeric_cols else []
    binary_cols_present = [col for col in binary_cols if col in X_train.columns] if binary_cols else []
    
    # Check for missing values (only on columns that actually exist)
    cat_has_missing = False
    num_has_missing = False
    
    if cat_cols_present:
        cat_has_missing = X_train[cat_cols_present].isnull().any().any()
    
    if num_cols_present:
        num_has_missing = X_train[num_cols_present].isnull().any().any()
    
    if verbose:
        print("→ Building preprocessor w/ conditional imputation:")
        if cat_cols_present:
            impute_status = "impute(most_frequent) + " if cat_has_missing else ""
            print(f"   • Cat: {impute_status}OHE on: {cat_cols_present}")
        if num_cols_present:
            impute_status = "impute(median) + " if num_has_missing else ""
            print(f"   • Num: {impute_status}scale on: {num_cols_present}")
        if binary_cols_present:
            print(f"   • Binary: passthrough (no scaling) on: {binary_cols_present}")

    transformers = []
    if cat_cols_present:
        cat_steps = []
        if cat_has_missing:
            cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        cat_steps.append(("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")))
        cat_pipe = Pipeline(steps=cat_steps)
        transformers.append(("cat", cat_pipe, cat_cols_present))  # Use filtered list
    
    if num_cols_present:
        num_steps = []
        if num_has_missing:
            num_steps.append(("impute", SimpleImputer(strategy="median")))
        num_steps.append(("scale", StandardScaler()))
        num_pipe = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipe, num_cols_present))  # Use filtered list
    
    # Binary columns: passthrough (no imputation, no scaling)
    if binary_cols_present:
        # Use FunctionTransformer with identity function for passthrough
        transformers.append(("binary", FunctionTransformer(), binary_cols_present))

    return ColumnTransformer(transformers=transformers, remainder="drop")




def finetune_oct(
    X_train, y_train, X_val, y_val,
    categorical_cols, numeric_cols,
    binary_cols=None,
    depths=(5, 7, 9),
    minbuckets=(50, 100, 150),
    cps=(1e-6, 1e-5, 1e-4, 1e-3),
    # NEW:
    tree_kind="oct",  # {"oct", "oct_h", "both"}
    hyperplane_configs=None,  # list[dict] for OCT-H, e.g. [{"sparsity":"all"}]
    ls_num_hyper_restarts=5,  # hyperplane search restarts (optional speed/quality knob)
    missingdatamode_oct="always_left",
    verbose=True,
    random_seed=123,
):
    """
    Hyperparameter tuning for IAI OptimalTreeClassifier with conditional imputation.

    NEW:
      - tree_kind: "oct" (axis-aligned), "oct_h" (hyperplanes), or "both" (compete in same search)
      - hyperplane_configs: list of dict configs for IAI hyperplane_config
          default when OCT-H is used: [{"sparsity": "all"}]
      - ls_num_hyper_restarts: number of random restarts for hyperplane optimization
      - missingdatamode_oct: missing-data mode for vanilla OCT (OCT-H should not rely on missingdatamode)
    """
    tree_kind = tree_kind.lower().strip()
    if tree_kind not in {"oct", "oct_h", "both"}:
        raise ValueError("tree_kind must be one of {'oct','oct_h','both'}")

    if tree_kind in {"oct_h", "both"}:
        if hyperplane_configs is None:
            hyperplane_configs = [{"sparsity": "all"}]  # IAI default “turn on hyperplanes”
    else:
        hyperplane_configs = []

    # Build model-variant grid:
    # None => vanilla OCT
    # dict => OCT-H enabled with that hyperplane_config
    variant_grid = []
    if tree_kind in {"oct", "both"}:
        variant_grid.append(None)
    if tree_kind in {"oct_h", "both"}:
        variant_grid.extend(list(hyperplane_configs))

    if verbose:
        print(
            f"Finetuning IAI OCT with variants={['oct' if v is None else 'oct_h' for v in variant_grid]}, "
            f"depths={list(depths)}, minbuckets={list(minbuckets)}, cps={list(cps)} (best PR-AUC)"
        )

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []

    # ── Preprocess (fit on TRAIN only) ──
    preprocessor = get_preprocessor_with_impute(
        X_train, categorical_cols, numeric_cols, binary_cols=binary_cols, verbose=verbose
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    # If the transformer yields sparse output, densify (IAI typically expects dense tabular input)
    if sparse is not None and sparse.issparse(X_train_transformed):
        X_train_transformed = X_train_transformed.toarray()
        X_val_transformed = X_val_transformed.toarray()

    # Robust missing check without forcing huge DataFrames
    train_has_nan = np.isnan(X_train_transformed).any()
    val_has_nan = np.isnan(X_val_transformed).any()
    if train_has_nan or val_has_nan:
        msg = "⚠️ Warning: Missing values still present after preprocessing."
        if verbose:
            print(msg)

    # ── Feature name extraction (your original logic) ──
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            try:
                if hasattr(transformer, 'named_steps') and 'ohe' in transformer.named_steps:
                    ohe = transformer.named_steps['ohe']
                    ohe_features = ohe.get_feature_names_out(columns)
                    feature_names.extend(ohe_features)
                elif hasattr(transformer, 'get_feature_names_out'):
                    ohe_features = transformer.get_feature_names_out(columns)
                    feature_names.extend(ohe_features)
            except (NotFittedError, AttributeError, ValueError) as e:
                if verbose:
                    print(f"  ⚠️ Could not extract categorical feature names: {e}")
        elif name == 'ohe':
            if columns:
                try:
                    ohe_features = transformer.get_feature_names_out(columns)
                    feature_names.extend(ohe_features)
                except (NotFittedError, AttributeError, ValueError):
                    pass
        elif name == 'num':
            feature_names.extend(columns)
        elif name == 'binary':
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

    # OCT-H should not see missing values at all (IAI notes missing not directly supported w/ hyperplanes)
    # If any NaNs remain, we allow vanilla OCT to proceed with missingdatamode, but fail for OCT-H variants.
    # :contentReference[oaicite:2]{index=2}
    # (If you want a softer behavior, you can skip OCT-H variants instead of raising.)
    for depth, minbucket, cp, variant in itertools.product(depths, minbuckets, cps, variant_grid):
        is_hyperplane = variant is not None

        if is_hyperplane and (train_has_nan or val_has_nan):
            raise ValueError(
                "OCT-H (hyperplane splits) selected but NaNs remain after preprocessing. "
                "Ensure get_preprocessor_with_impute fully imputes all missing values."
            )

        model_kwargs = dict(
            max_depth=depth,
            minbucket=minbucket,
            cp=cp,
            random_seed=random_seed,
        )

        if is_hyperplane:
            # Enable OCT-H via hyperplane_config (Python example in IAI docs)
            # :contentReference[oaicite:3]{index=3}
            model_kwargs["hyperplane_config"] = variant
            model_kwargs["ls_num_hyper_restarts"] = ls_num_hyper_restarts
            # missingdatamode is irrelevant if there are no NaNs; keep it conservative
            model_kwargs["missingdatamode"] = "none"
        else:
            # Vanilla OCT can optionally route NaNs if they exist
            model_kwargs["missingdatamode"] = missingdatamode_oct

        model = iai.OptimalTreeClassifier(**model_kwargs)
        model.fit(X_train_df, y_train)

        y_pred = model.predict(X_val_df)
        y_val_proba = model.predict_proba(X_val_df).iloc[:, 1]

        f1 = f1_score(y_val, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_val, y_val_proba)

        results.append({
            "variant": "oct_h" if is_hyperplane else "oct",
            "hyperplane_config": repr(variant) if is_hyperplane else None,
            "depth": depth,
            "minbucket": minbucket,
            "cp": cp,
            "f1": f1,
            "pr_auc": pr_auc,
        })

        if pr_auc > best_score:
            best_score = pr_auc
            best_params = {
                "variant": "oct_h" if is_hyperplane else "oct",
                "hyperplane_config": variant,
                "depth": depth,
                "minbucket": minbucket,
                "cp": cp,
            }
            best_model = model

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"Best params: {best_params} @ PR-AUC: {best_score:.4f}")

    return best_model, best_params, results_df, preprocessor, feature_names
# Alias for backward compatibility
def finetune_oct_impute(X_train, y_train, X_val, y_val, categorical_cols, numeric_cols,
                        binary_cols=None,
                        depths=[5, 7, 9], minbuckets=[50, 100, 150], cps=[1e-5, 1e-4, 1e-3],
                        verbose=True, random_seed=123):
    """
    Alias for finetune_oct() for backward compatibility.
    Same functionality, different default hyperparameters.
    """
    return finetune_oct(X_train, y_train, X_val, y_val, categorical_cols, numeric_cols,
                       binary_cols=binary_cols,
                       depths=depths, minbuckets=minbuckets, cps=cps,
                       verbose=verbose, random_seed=random_seed)


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
    save_suffix=None,
    X_val_df=None,
    y_val=None
):
    """
    Evaluate OCT model on test set.
    
    Parameters
    ----------
    iai_model : IAI OptimalTreeClassifier
        Trained model
    X_test_df : DataFrame
        Test set features for evaluation
    y_test : array-like
        Test set labels
    preprocessor : sklearn Pipeline
        Fitted preprocessor
    feature_names : list
        Feature names after preprocessing
    results_dir : str, optional
        Directory to save results
    ratio : float, optional
        Matching ratio (for file naming)
    X_val_df : DataFrame, optional
        Validation set features for threshold selection. If provided, thresholds
        (MCC, G-mean, F1) will be computed on validation set and applied to test set.
        This is recommended for proper evaluation: use validation set during
        hyperparameter tuning, and test set only for final evaluation.
    y_val : array-like, optional
        Validation set labels (required if X_val_df is provided)
    
    Returns
    -------
    dict : Evaluation metrics
    """
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
        out = X_test_processed.copy()

        out["predicted_proba"] = y_proba
        out["predicted_class_default"] = y_pred_default
        leaf_assignments = iai_model.apply(X_test_processed)

        print("✓ Predictions completed")

    except Exception as e:
        print(f"✗ Error applying OCT: {e}")
        raise e

    out["leaf_assignment"] = leaf_assignments
    out["predicted_cost_stratum_default"] = y_pred_default

    y_test_series = pd.Series(y_test).reset_index(drop=True)

    # ------------------------------------------------------------
    # AUC metrics (threshold-free)
    # ------------------------------------------------------------
    auc = iai_model.score(X_test_processed, y_test_series, criterion="auc")
    pr_auc = average_precision_score(y_test_series, y_proba)

    # ------------------------------------------------------------
    # Threshold selection: use validation set if provided, otherwise test set
    # ------------------------------------------------------------
    if X_val_df is not None and y_val is not None:
        # Compute thresholds on validation set (proper evaluation)
        print(f"Computing optimal thresholds on validation set ({len(X_val_df):,} samples)")
        X_val_processed = preprocessor.transform(X_val_df)
        X_val_processed = pd.DataFrame(X_val_processed, columns=feature_names)
        y_proba_val = iai_model.predict_proba(X_val_processed).iloc[:, 1]
        y_val_series = pd.Series(y_val).reset_index(drop=True)
        
        # F1-optimal thresholding on validation set
        precision_curve_val, recall_curve_val, thresholds_val = precision_recall_curve(y_val_series, y_proba_val)
        f1_scores_val = 2 * precision_curve_val * recall_curve_val / (precision_curve_val + recall_curve_val + 1e-10)
        best_idx_val = int(np.argmax(f1_scores_val))
        best_threshold_f1 = float(thresholds_val[best_idx_val]) if best_idx_val < len(thresholds_val) else 0.5
        
        # Balanced recall/specificity thresholds on validation set
        balanced = best_balanced_threshold(y_val_series.values, y_proba_val.values if hasattr(y_proba_val, "values") else y_proba_val)
        
        # MCC threshold on validation set
        mcc_best = best_mcc_threshold(y_val_series.values, y_proba_val.values if hasattr(y_proba_val, "values") else y_proba_val)
        best_mcc_threshold_value = mcc_best["threshold"]
        
        # Apply validation-set thresholds to test set for evaluation
        y_pred_opt_mcc = (y_proba >= best_mcc_threshold_value).astype(int)
        print(f"  Applied validation-set thresholds to test set for evaluation")
    else:
        # Compute thresholds on test set (less rigorous, but sometimes used for final reporting)
        print(f"Computing optimal thresholds on test set (not recommended for hyperparameter tuning)")
        # F1-optimal thresholding
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_test_series, y_proba)
        f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-10)
        best_idx = int(np.argmax(f1_scores))
        best_threshold_f1 = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        
        # Balanced recall/specificity thresholds
        balanced = best_balanced_threshold(y_test_series.values, y_proba.values if hasattr(y_proba, "values") else y_proba)
        
        # MCC threshold
        mcc_best = best_mcc_threshold(y_test_series.values, y_proba.values if hasattr(y_proba, "values") else y_proba)
        best_mcc_threshold_value = mcc_best["threshold"]
        y_pred_opt_mcc = mcc_best["y_pred"]
    
    # Evaluate metrics on test set using selected thresholds
    if X_val_df is not None:
        # Thresholds computed on validation set, evaluate on test set
        best_mcc_value = matthews_corrcoef(y_test_series, y_pred_opt_mcc)
        # Compute F1 on test set using validation-set F1 threshold
        y_pred_f1 = (y_proba >= best_threshold_f1).astype(int)
        optimal_f1 = f1_score(y_test_series, y_pred_f1, zero_division=0)
    else:
        # Thresholds computed on test set
        best_mcc_value = mcc_best["mcc"]
        optimal_f1 = float(f1_scores[best_idx])
    
    # Compute recall, precision, and specificity at best MCC threshold from confusion matrix
    tn_mcc, fp_mcc, fn_mcc, tp_mcc = confusion_matrix(y_test_series, y_pred_opt_mcc).ravel()
    recall_mcc = tp_mcc / (tp_mcc + fn_mcc) if (tp_mcc + fn_mcc) else 0.0
    precision_mcc = tp_mcc / (tp_mcc + fp_mcc) if (tp_mcc + fp_mcc) else 0.0
    specificity_mcc = tn_mcc / (tn_mcc + fp_mcc) if (tn_mcc + fp_mcc) else 0.0
    
    # Compute precision, recall, and specificity at G-mean threshold on test set
    precision_gmean = None
    balanced_recall_gmean_test = None
    balanced_specificity_gmean_test = None
    if 'gmean_opt' in balanced and 'threshold' in balanced['gmean_opt']:
        gmean_threshold = balanced['gmean_opt']['threshold']
        y_pred_gmean = (y_proba >= gmean_threshold).astype(int)
        tn_gmean, fp_gmean, fn_gmean, tp_gmean = confusion_matrix(y_test_series, y_pred_gmean).ravel()
        precision_gmean = tp_gmean / (tp_gmean + fp_gmean) if (tp_gmean + fp_gmean) else 0.0
        balanced_recall_gmean_test = tp_gmean / (tp_gmean + fn_gmean) if (tp_gmean + fn_gmean) else 0.0
        balanced_specificity_gmean_test = tn_gmean / (tn_gmean + fp_gmean) if (tn_gmean + fp_gmean) else 0.0
    
    # ------------------------------------------------------------
    # WRITE OUT PREDICTIONS TO DISK
    # ------------------------------------------------------------
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/predictions", exist_ok=True)
        if save_suffix is not None:
            pred_path = f"{results_dir}/predictions/oct_predictions_{save_suffix}.csv"
            tree_path = f"{results_dir}/oct_tree_{save_suffix}.json"
        else:
            pred_path = f"{results_dir}/predictions/oct_predictions.csv"
            tree_path = f"{results_dir}/oct_tree.json"
        # after creating out = X_test_processed.copy()
        if "ENROLID" in X_test_df.columns:
            out.insert(0, "ENROLID", X_test_df.reset_index(drop=True)["ENROLID"].values)
        out.to_csv(pred_path, index=False)
        print(f"✓ Saved OCT predictions to: {pred_path}")
        _save_tree_splits(iai_model, tree_path)

    print(f"AUC score: {auc:.3f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}")
    if X_val_df is not None:
        print(f"Best MCC (test set, threshold from val): {best_mcc_value:.3f} @ threshold={best_mcc_threshold_value:.6f}")
    else:
        print(f"Best MCC: {best_mcc_value:.3f} @ threshold={best_mcc_threshold_value:.6f}")
    print(f"Sensitivity (Recall) @MCC*: {recall_mcc:.3f}")
    print(f"Specificity @MCC*: {specificity_mcc:.3f}")
    if X_val_df is not None and balanced_recall_gmean_test is not None:
        print(f"Balanced (G-mean) recall (test set, threshold from val): {balanced_recall_gmean_test:.3f}")
        print(f"Balanced (G-mean) specificity (test set, threshold from val): {balanced_specificity_gmean_test:.3f}")
    else:
        print(f"Balanced (G-mean) recall: {balanced['gmean_opt']['recall']:.3f}")
        print(f"Balanced (G-mean) specificity: {balanced['gmean_opt']['specificity']:.3f}")
    print("Number of leaves:", len(pd.unique(leaf_assignments)))

    # ------------------------------------------------------------
    # Return dictionary for logging
    # ------------------------------------------------------------
    # Use test-set metrics when validation set was provided, otherwise use original values
    if X_val_df is not None and balanced_recall_gmean_test is not None:
        balanced_recall_gmean = balanced_recall_gmean_test
        balanced_specificity_gmean = balanced_specificity_gmean_test
    else:
        balanced_recall_gmean = balanced["gmean_opt"]["recall"]
        balanced_specificity_gmean = balanced["gmean_opt"]["specificity"]
    
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": best_mcc_value,
        "best_mcc_threshold": best_mcc_threshold_value,
        "recall_mcc": float(recall_mcc),
        "precision_mcc": float(precision_mcc),
        "optimal_f1": float(optimal_f1),
        "balanced_recall_gmean": float(balanced_recall_gmean),
        "balanced_specificity_gmean": float(balanced_specificity_gmean),
        "precision_gmean": float(precision_gmean) if precision_gmean is not None else None,
    }

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


def format_time(seconds):
    """Format time in a readable way."""
    if seconds < 0.1:
        return f"{seconds:.4f}"
    elif seconds < 1.0:
        return f"{seconds:.3f}"
    else:
        return f"{seconds:.2f}"

def get_resource_usage():
    """Get current CPU and memory usage."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': memory_info.rss / (1024 * 1024),  # RSS in MB
            'memory_percent': process.memory_percent(),
        }
    else:
        return {
            'cpu_percent': None,
            'memory_mb': None,
            'memory_percent': None,
        }



