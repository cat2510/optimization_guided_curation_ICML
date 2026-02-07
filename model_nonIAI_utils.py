from sklearn import calibration
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

def get_bin_flag_columns(df):
    return [col for col in df.columns if col.startswith("has_") or "THRCLS" in col.upper()
    or col.endswith("adherent") or col.startswith("early_")
    or col.startswith("is_")]
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
        # Use FunctionTransformer with identity function for passthrough
        transformers.append(("binary", FunctionTransformer(), binary_cols_present))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def get_histgb_pipeline(df,categorical_cols, numeric_cols,preprocessor=None, n_estimators=200, max_iter=200,
                                random_state=42, n_jobs=-1, calibration_method=None,balance_classes=True):

    if preprocessor is None:
            preprocessor = get_preprocessor_with_impute(df, categorical_cols, numeric_cols, verbose=False)
    # 2) wrap a gradient booster in a calibrator to get *better-spread* probabilities
    gb_kwargs = dict(max_iter=max_iter, random_state=random_state)
    if balance_classes:
        print("Class weights balanced for XGBoost model...")
        gb_kwargs['class_weight'] = "balanced"
    gb = HistGradientBoostingClassifier(**gb_kwargs)
    if calibration_method:
        print("Scaling prediction probabilities with CalibratedClassifierCV " + calibration_method)
        calibrated_gb = CalibratedClassifierCV(gb, cv=5, method=calibration_method) # other methods: "isotonic", "sigmoid"
    else:
        calibrated_gb = gb
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_gb)
    ])

    return pipeline

def get_logistic_pipeline(df,categorical_cols, numeric_cols,preprocessor=None, C=1.0, max_iter=500, class_weight=None,scale_post_preprocess=True,calibrate=False):

    """
    Returns a sklearn Pipeline with a preprocessor and logistic regression.
    """
    if preprocessor is None:
        preprocessor = get_preprocessor_with_impute(df, categorical_cols, numeric_cols, verbose=False)
    logreg = LogisticRegression(
        solver="lbfgs",
        #multi_class="multinomial",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=42
    )

    steps = [("preprocessor", preprocessor)]
    
    if scale_post_preprocess:
        steps.append(("post_scaler", StandardScaler()))
    
    if calibrate:
        print("Calibrated Logistic Stratifier...")
        clf_cal = CalibratedClassifierCV(logreg, method="isotonic", cv=5)
        steps.append(("classifier", clf_cal))

    else:
        steps.append(("classifier", logreg))


    pipeline = Pipeline(steps)
    return pipeline


def get_random_forest_pipeline(df,categorical_cols, numeric_cols,preprocessor=None, n_estimators=200, max_depth=None,
                                min_samples_leaf=1, class_weight=None, n_jobs=-1):
    """
    Returns a sklearn Pipeline with a preprocessor and random forest classifier.
    """
    if preprocessor is None:
        preprocessor = get_preprocessor_with_impute(df, categorical_cols, numeric_cols, verbose=False)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
        n_jobs=n_jobs
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])
    return pipeline


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


def best_mcc_threshold(y_true, y_proba):
    """
    Find threshold t that maximizes MCC for predictions 1{p >= t}.
    Returns dict with threshold, mcc, and y_pred.
    """
    from sklearn.metrics import matthews_corrcoef
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best_mcc = -1
    best_t = 0.5
    best_y_pred = None
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t
            best_y_pred = y_pred
    
    return {"threshold": float(best_t), "mcc": float(best_mcc), "y_pred": best_y_pred}


def best_balanced_threshold(y_true, y_prob):
    """
    Find thresholds that balance recall and specificity without F1/Youden.
    Returns two candidates:
      - gmean_opt: maximizes sqrt(recall * specificity)
      - minside_opt: maximizes min(recall, specificity)
    """
    from sklearn.metrics import roc_curve
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


def evaluate_binary_model(
    model,
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
    Evaluate model on test set.
    
    Parameters
    ----------
    model : model
        Trained model (sklearn-compatible with predict_proba)
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
    save_suffix : str, optional
        Suffix for saved files
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
    print(f"Test dataset for model evaluation: {len(X_test_df):,} samples")

    # ------------------------------------------------------------
    # Preprocessing + Model Predictions
    # ------------------------------------------------------------
    try:
        X_test_processed = preprocessor.transform(X_test_df)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

        # Get predictions
        y_pred_default = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]
        out = X_test_processed.copy()

        out["predicted_proba"] = y_proba
        out["predicted_class_default"] = y_pred_default
        
        # Try to get leaf assignments if model supports it (OCT models)
        try:
            if hasattr(model, 'apply'):
                leaf_assignments = model.apply(X_test_processed)
                out["leaf_assignment"] = leaf_assignments
        except:
            pass

        print("✓ Predictions completed")

    except Exception as e:
        print(f"✗ Error applying model: {e}")
        raise e

    y_test_series = pd.Series(y_test).reset_index(drop=True)

    # ------------------------------------------------------------
    # AUC metrics (threshold-free)
    # ------------------------------------------------------------
    auc = roc_auc_score(y_test_series, y_proba)
    pr_auc = average_precision_score(y_test_series, y_proba)

    # ------------------------------------------------------------
    # Threshold selection: use validation set if provided, otherwise test set
    # ------------------------------------------------------------
    if X_val_df is not None and y_val is not None:
        # Compute thresholds on validation set (proper evaluation)
        print(f"Computing optimal thresholds on validation set ({len(X_val_df):,} samples)")
        X_val_processed = preprocessor.transform(X_val_df)
        X_val_processed = pd.DataFrame(X_val_processed, columns=feature_names)
        y_proba_val = model.predict_proba(X_val_processed)[:, 1]
        y_val_series = pd.Series(y_val).reset_index(drop=True)
        
        # F1-optimal thresholding on validation set
        precision_curve_val, recall_curve_val, thresholds_val = precision_recall_curve(y_val_series, y_proba_val)
        f1_scores_val = 2 * precision_curve_val * recall_curve_val / (precision_curve_val + recall_curve_val + 1e-10)
        best_idx_val = int(np.argmax(f1_scores_val))
        best_threshold_f1 = float(thresholds_val[best_idx_val]) if best_idx_val < len(thresholds_val) else 0.5
        
        # Balanced recall/specificity thresholds on validation set
        balanced = best_balanced_threshold(y_val_series.values, y_proba_val)
        
        # MCC threshold on validation set
        mcc_best = best_mcc_threshold(y_val_series.values, y_proba_val)
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
        balanced = best_balanced_threshold(y_test_series.values, y_proba)
        
        # MCC threshold
        mcc_best = best_mcc_threshold(y_test_series.values, y_proba)
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
            pred_path = f"{results_dir}/predictions/predictions_{save_suffix}.csv"
        else:
            pred_path = f"{results_dir}/predictions/predictions.csv"
        # after creating out = X_test_processed.copy()
        if "ENROLID" in X_test_df.columns:
            out.insert(0, "ENROLID", X_test_df.reset_index(drop=True)["ENROLID"].values)
        out.to_csv(pred_path, index=False)
        print(f"✓ Saved predictions to: {pred_path}")

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



def evaluate_model_auc(clf, X_test, y_test, optimal_threshold=True):
    """
    Evaluates a fitted classifier pipeline by printing ROC AUC
    and returns metrics for logging.
    """

    y_test = np.asarray(y_test)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    if optimal_threshold:
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # compute P/R/F1 with chosen threshold
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp + 1e-10)   # avoid division by zero

    print(f"Binary AUC: {auc:.3f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}")
    #print("Classification Report:")
   # print(classification_report(y_test, y_pred, digits=3))

    # return dictionary for logging
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "threshold": threshold,
    }
