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


def recall_at_specificity(y_true, y_score, target_specificity: float = 0.60):
    """
    Return (recall, specificity, threshold) where specificity >= target_specificity and
    recall is maximized among those thresholds.

    Uses ROC curve (thresholds on y_score). If no threshold reaches target specificity,
    returns the closest (max specificity) operating point.

    When all scores are constant (degenerate classifier), returns (0, 1, threshold)
    since predict-all-negative is the only way to achieve high specificity.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if np.ptp(y_score) == 0:
        c = float(np.max(y_score))
        return 0.0, 1.0, (c + 1.0 if np.isfinite(c) else np.inf)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr

    # Candidate indices satisfying specificity constraint
    idx = np.where(specificity >= target_specificity)[0]
    if idx.size > 0:
        # pick the one with max recall (tpr) among feasible points
        best_i = idx[np.argmax(tpr[idx])]
        return float(tpr[best_i]), float(specificity[best_i]), float(thresholds[best_i])

    print("Constraint not achievable, pick point with maximal specificity (i.e., minimal FPR)")
    best_i = int(np.argmax(specificity))
    return float(tpr[best_i]), float(specificity[best_i]), float(thresholds[best_i])



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


# Tolerance for Gower v2 binary columns (must match public.precompute_gower_distances validation)
GOWER_BINARY_ATOL = 1e-6
GOWER_BINARY_RTOL = 1e-5


def is_binary_01_series(
    s: pd.Series,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> bool:
    """
    True if non-missing values are in {0, 1} (or within atol/rtol of 0 or 1).

    Default atol=rtol=0: strict — used by get_bin_flag_columns for value-based detection
    (exact 0/1 after coercion; bool OK).

    Use atol=GOWER_BINARY_ATOL, rtol=GOWER_BINARY_RTOL to match the Gower distance kernel
    (e.g. validating has_* columns on the cases∪controls matrix in preprocessing).
    """
    non_null = s.dropna()
    if non_null.empty:
        return False

    if atol == 0.0 and rtol == 0.0:
        if pd.api.types.is_bool_dtype(non_null):
            vals = set(non_null.astype(int).unique())
            return vals.issubset({0, 1})
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False
        vals = set(coerced.unique())
        return vals.issubset({0, 1})

    arr = np.asarray(pd.to_numeric(non_null, errors="coerce"), dtype=np.float64)
    if np.any(np.isnan(arr)):
        return False
    close_0 = np.abs(arr) <= atol + rtol * np.maximum(np.abs(arr), 1e-12)
    close_1 = np.abs(arr - 1.0) <= atol + rtol * np.maximum(np.abs(arr), 1e-12)
    return bool(np.all(close_0 | close_1))


def get_bin_flag_columns_with_provenance(df: pd.DataFrame):
    """
    Same bin columns as get_bin_flag_columns, plus which ones were confirmed by strict
    is_binary_01_series on df (not only by has_* naming).

    Columns in the second return value need no second pass in Gower matrix build; others
    (has_* with non–0/1 values on df) are checked with Gower tolerance on cases∪controls.
    """
    bin_cols = []
    verified_strict: set = set()
    for col in df.columns:
        name_match = col.startswith("has_")
        value_match = is_binary_01_series(df[col])
        if name_match or value_match:
            bin_cols.append(col)
            if value_match:
                verified_strict.add(col)
    return bin_cols, frozenset(verified_strict)


def get_bin_flag_columns(df: pd.DataFrame):
    return get_bin_flag_columns_with_provenance(df)[0]
    
def get_true_num_columns(df, CAT_COLUMNS,BIN_FLAG_COLUMNS):
    return [
        col for col in df.columns
        if (col not in ['ENROLID']
            and col not in CAT_COLUMNS+BIN_FLAG_COLUMNS
        )
    ]


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
        # feature_names_out required for ColumnTransformer.get_feature_names_out() (sklearn >= 1.2)
        transformers.append(
            ("binary", FunctionTransformer(feature_names_out="one-to-one"), binary_cols_present)
        )

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
    fit_sample_weight=None,
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
      - fit_sample_weight: forwarded to IAI fit(..., sample_weight=...)
          Use "autobalance" for IAI built-in class balancing on imbalanced classification.
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

    tuning_start_time = time.perf_counter()
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

    for depth, minbucket, cp, variant in itertools.product(depths, minbuckets, cps, variant_grid):
        config_fit_start_time = time.perf_counter()
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
        fit_kwargs = {}
        if fit_sample_weight is not None:
            fit_kwargs["sample_weight"] = fit_sample_weight
        model.fit(X_train_df, y_train, **fit_kwargs)
        config_fit_time_seconds = float(time.perf_counter() - config_fit_start_time)

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
            "fit_time_seconds": config_fit_time_seconds,
        })

        if pr_auc > best_score:
            best_score = pr_auc
            best_params = {
                "variant": "oct_h" if is_hyperplane else "oct",
                "hyperplane_config": variant,
                "depth": depth,
                "minbucket": minbucket,
                "cp": cp,
                "best_fit_time_seconds": config_fit_time_seconds,
            }
            best_model = model

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    tuning_time_seconds = float(time.perf_counter() - tuning_start_time)
    if isinstance(best_params, dict):
        best_params["tuning_time_seconds"] = tuning_time_seconds
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

    # If all probabilities identical: prefer predict-all-negative (consistent with best_balanced_threshold)
    if np.ptp(y_proba) == 0:
        c = float(np.max(y_proba))
        thresh = c + 1.0 if np.isfinite(c) else np.inf
        y_pred = np.zeros_like(y_true, dtype=int)
        return {"threshold": thresh, "mcc": 0.0, "y_pred": y_pred}

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


def _transform_for_iai(preprocessor, X_df, feature_names):
    """
    Apply fitted preprocessor and return dense DataFrame with feature names.
    """
    X_processed = preprocessor.transform(X_df)
    if sparse is not None and sparse.issparse(X_processed):
        X_processed = X_processed.toarray()
    return pd.DataFrame(X_processed, columns=feature_names)


def _safe_tree_counts(iai_model):
    """
    Best-effort tree size extraction for reporting.
    Returns (n_leaves, n_splits) as ints or None when unavailable.
    """
    n_leaves = None
    n_splits = None
    try:
        if hasattr(iai_model, "get_num_leaves"):
            n_leaves = int(iai_model.get_num_leaves())
    except Exception:
        n_leaves = None
    try:
        if hasattr(iai_model, "get_num_nodes"):
            n_nodes = int(iai_model.get_num_nodes())
            if n_nodes >= 1:
                if n_leaves is not None:
                    n_splits = max(0, n_nodes - n_leaves)
                else:
                    n_splits = max(0, (n_nodes - 1) // 2)
    except Exception:
        n_splits = None
    return n_leaves, n_splits


def refit_leaf_probabilities_on_dataset(
    iai_model,
    X_refit,
    y_refit,
    preprocessor,
    feature_names,
    *,
    fallback="global_rate",
):
    """
    Recompute leaf probabilities from a refit dataset without retraining tree.

    Parameters
    ----------
    fallback : str
        Currently supports "global_rate" only.

    Returns
    -------
    dict with keys:
      - leaf_prob_map: dict[int, float]
      - fallback_probability: float
      - stats: dict with counts and coverage diagnostics
    """
    if fallback != "global_rate":
        raise ValueError("Only fallback='global_rate' is currently supported.")

    X_refit_processed = _transform_for_iai(preprocessor, X_refit, feature_names)
    leaves = np.asarray(iai_model.apply(X_refit_processed), dtype=int)
    y_arr = np.asarray(y_refit).astype(int)
    if len(leaves) != len(y_arr):
        raise ValueError("Length mismatch between routed leaves and y_refit.")

    global_rate = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.0
    leaf_prob_map = {}
    leaf_counts = {}
    leaf_pos_counts = {}

    for leaf_id in np.unique(leaves):
        mask = leaves == leaf_id
        n_leaf = int(np.sum(mask))
        n_pos = int(np.sum(y_arr[mask]))
        p_leaf = float(n_pos / n_leaf) if n_leaf > 0 else global_rate
        leaf_prob_map[int(leaf_id)] = p_leaf
        leaf_counts[int(leaf_id)] = n_leaf
        leaf_pos_counts[int(leaf_id)] = n_pos

    stats = {
        "n_samples_refit": int(len(y_arr)),
        "n_unique_leaves_refit": int(len(leaf_prob_map)),
        "global_positive_rate_refit": global_rate,
        "leaf_counts": leaf_counts,
        "leaf_positive_counts": leaf_pos_counts,
    }
    return {
        "leaf_prob_map": leaf_prob_map,
        "fallback_probability": global_rate,
        "stats": stats,
    }


def scores_from_leaf_probability_map(leaf_assignments, leaf_prob_map, fallback_p):
    """
    Build probability scores from leaf IDs and a leaf->probability map.
    """
    leaves = np.asarray(leaf_assignments, dtype=int)
    return np.asarray([leaf_prob_map.get(int(l), float(fallback_p)) for l in leaves], dtype=float)

def evaluate_binary_oct(
    iai_model,
    X_test_df,
    y_test,
    preprocessor,
    feature_names,
    results_dir=None,
    save_suffix=None,
    X_val_df=None,
    y_val=None,
    leaf_probability_map=None,
    leaf_prob_fallback="global_rate",
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
    NEW METRIC:
      - recall_at_specificity_0.6: maximum recall achievable at specificity >= 0.6
        (and reports the selected threshold + achieved specificity).
    """
    eval_start_time = time.perf_counter()
    print(f"Test dataset for OCT application: {len(X_test_df):,} samples")
    if leaf_probability_map is not None and leaf_prob_fallback != "global_rate":
        raise ValueError("Only leaf_prob_fallback='global_rate' is currently supported.")

    # ------------------------------------------------------------
    # Preprocessing + OCT Predictions
    # ------------------------------------------------------------
    try:
        X_test_processed = _transform_for_iai(preprocessor, X_test_df, feature_names)

        # Base predictions from OCT
        y_pred_default = iai_model.predict(X_test_processed)
        leaf_assignments = iai_model.apply(X_test_processed)
        if leaf_probability_map is None:
            y_proba = iai_model.predict_proba(X_test_processed).iloc[:, 1]
            missing_leaf_count_test = 0
            fallback_probability = None
        else:
            fallback_probability = float(np.mean(np.asarray(y_test).astype(int)))
            y_proba = scores_from_leaf_probability_map(
                leaf_assignments=leaf_assignments,
                leaf_prob_map=leaf_probability_map,
                fallback_p=fallback_probability,
            )
            missing_leaf_count_test = int(
                np.sum(~np.isin(np.asarray(leaf_assignments, dtype=int), list(leaf_probability_map.keys())))
            )
        out = X_test_processed.copy()

        out["predicted_proba"] = y_proba
        out["predicted_class_default"] = y_pred_default

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
    if leaf_probability_map is None:
        auc = iai_model.score(X_test_processed, y_test_series, criterion="auc")
    else:
        auc = roc_auc_score(y_test_series, y_proba)
    pr_auc = average_precision_score(y_test_series, y_proba)

    # ------------------------------------------------------------
    # Threshold selection: use validation set if provided, otherwise test set
    # ------------------------------------------------------------
    if X_val_df is not None and y_val is not None:
        # Compute thresholds on validation set (proper evaluation)
        print(f"Computing optimal thresholds on validation set ({len(X_val_df):,} samples)")
        X_val_processed = _transform_for_iai(preprocessor, X_val_df, feature_names)
        leaf_assignments_val = iai_model.apply(X_val_processed)
        if leaf_probability_map is None:
            y_proba_val = iai_model.predict_proba(X_val_processed).iloc[:, 1]
            missing_leaf_count_val = 0
        else:
            fallback_probability_val = float(np.mean(np.asarray(y_val).astype(int)))
            y_proba_val = scores_from_leaf_probability_map(
                leaf_assignments=leaf_assignments_val,
                leaf_prob_map=leaf_probability_map,
                fallback_p=fallback_probability_val,
            )
            missing_leaf_count_val = int(
                np.sum(~np.isin(np.asarray(leaf_assignments_val, dtype=int), list(leaf_probability_map.keys())))
            )
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
    # NEW: recall at specificity >= 0.6 (on test set)
    # ------------------------------------------------------------
    recall_at_spec_06, achieved_spec_06, threshold_spec_06 = recall_at_specificity(
        y_test_series.values,
        y_proba.values if hasattr(y_proba, "values") else y_proba,
        target_specificity=0.60
    )

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
        # Save interactive tree HTML in the same folder as splits
        if hasattr(iai_model, "write_html"):
            html_path = tree_path.replace(".json", ".html")
            iai_model.write_html(html_path)
            print(f"✓ Saved OCT tree visualization to: {html_path}")

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
    print(f"Recall @ specificity>=0.60: {recall_at_spec_06:.3f} (achieved spec={achieved_spec_06:.3f}, thr={threshold_spec_06:.6f})")
    print("Number of leaves:", len(pd.unique(leaf_assignments)))
    if leaf_probability_map is not None:
        print(f"Leaf override active: {len(leaf_probability_map)} mapped leaves")
        print(f"Missing mapped leaves in test routing (fallback used): {missing_leaf_count_test}")
        if X_val_df is not None and y_val is not None:
            print(f"Missing mapped leaves in val routing (fallback used): {missing_leaf_count_val}")

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
    
    evaluation_time_seconds = float(time.perf_counter() - eval_start_time)
    n_leaves_tree, n_splits_tree = _safe_tree_counts(iai_model)
    n_leaves_routed = int(len(pd.unique(leaf_assignments)))
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": best_mcc_value,
        "recall_mcc": float(recall_mcc),
        "specificity_mcc": float(specificity_mcc),
        "optimal_f1": float(optimal_f1),
        "balanced_recall_gmean": float(balanced_recall_gmean),
        "balanced_specificity_gmean": float(balanced_specificity_gmean),
        "recall_at_specificity_0.6": float(recall_at_spec_06),
        "achieved_specificity_0.6": float(achieved_spec_06),
        "threshold_specificity_0.6": float(threshold_spec_06),
        "number_of_leaves": int(n_leaves_tree) if n_leaves_tree is not None else n_leaves_routed,
        "number_of_splits": int(n_splits_tree) if n_splits_tree is not None else None,
        "n_routed_leaves_test": n_leaves_routed,
        "leaf_probability_override": bool(leaf_probability_map is not None),
        "missing_refit_map_leaves_test_count": int(missing_leaf_count_test) if leaf_probability_map is not None else 0,
        "missing_refit_map_leaves_val_count": int(missing_leaf_count_val) if (leaf_probability_map is not None and X_val_df is not None and y_val is not None) else 0,
        "evaluation_time_seconds": evaluation_time_seconds,
    }

def best_balanced_threshold(y_true, y_prob):
    """
    Find thresholds that balance recall and specificity without F1/Youden.
    Returns two candidates:
      - gmean_opt: maximizes sqrt(recall * specificity)
      - minside_opt: maximizes min(recall, specificity)

    When all scores are constant (e.g. tree with no splits), roc_curve may return
    only one operating point. We prefer "predict all negative" (recall=0, spec=1)
    over "predict all positive" (recall=1, spec=0) since a degenerate tree typically
    predicts the majority class.
    """
    y_prob = np.asarray(y_prob).astype(float)
    if np.ptp(y_prob) == 0:
        # All scores identical: degenerate classifier. Prefer predict-all-negative.
        c = float(np.max(y_prob))
        thresh_inf = c + 1.0 if np.isfinite(c) else np.inf
        return {
            "gmean_opt": {
                "threshold": thresh_inf,
                "recall": 0.0,
                "specificity": 1.0,
                "gmean": 0.0,
                "min_recall_spec": 0.0,
            },
            "minside_opt": {
                "threshold": thresh_inf,
                "recall": 0.0,
                "specificity": 1.0,
                "gmean": 0.0,
                "min_recall_spec": 0.0,
            },
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    recall = tpr

    gmean = np.sqrt(recall * specificity)
    max_g = np.max(gmean)
    ties_g = np.flatnonzero(gmean >= max_g - 1e-12)
    idx_g = int(ties_g[np.argmax(thresholds[ties_g])])  # tie-break: prefer highest threshold

    min_side = np.minimum(recall, specificity)
    max_m = np.max(min_side)
    ties_m = np.flatnonzero(min_side >= max_m - 1e-12)
    idx_min = int(ties_m[np.argmax(thresholds[ties_m])])

    def pack(idx):
        return {
            "threshold": float(thresholds[idx]),
            "recall": float(recall[idx]),
            "specificity": float(specificity[idx]),
            "gmean": float(np.sqrt(recall[idx] * specificity[idx])),
            "min_recall_spec": float(min(recall[idx], specificity[idx])),
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



