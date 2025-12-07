# -----------------------------------------------------------------------------
# IAI OPTIMAL CLASSIFICATION TREES 
import numpy as np
import pandas as pd
from interpretableai import iai
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix, brier_score_loss
from model_pipeline import get_preprocessor, train_test_split_enrol
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

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



def evaluate_binary_oct(
    iai_model,
    X_test_df,
    y_test,
    preprocessor,
    feature_names,
    compute_leaf_metrics=False,
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
    # **WRITE OUT PREDICTIONS TO DISK**
    # ------------------------------------------------------------
    if results_dir is not None:
        pred_path = f"{results_dir}/oct_predictions_ratio_{ratio:.2f}.csv"
        X_test_processed.to_csv(pred_path, index=False)       # ← NEW OUTPUT FILE
        print(f"✓ Saved OCT predictions to: {pred_path}")
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
    print(f"Sensitivity (default): {sensitivity_default:.3f}")
    print(f"Specificity (default): {specificity_default:.3f}")
    print("Number of leaves:", len(pd.unique(leaf_assignments)))

    # ------------------------------------------------------------
    # Optional: Per-leaf performance (using optimized predictions)
    # ------------------------------------------------------------
    leaf_metrics_df = None
    if compute_leaf_metrics:
        rows = []
        y_pred_series_opt = pd.Series(y_pred_opt)

        for leaf_id, idxs in X_test_processed.groupby("leaf_assignment").groups.items():
            idxs = list(idxs)

            y_true_leaf = y_test_series.iloc[idxs]
            y_pred_leaf = y_pred_series_opt.iloc[idxs]

            tp_l = int(((y_true_leaf == 1) & (y_pred_leaf == 1)).sum())
            tn_l = int(((y_true_leaf == 0) & (y_pred_leaf == 0)).sum())
            fp_l = int(((y_true_leaf == 0) & (y_pred_leaf == 1)).sum())
            fn_l = int(((y_true_leaf == 1) & (y_pred_leaf == 0)).sum())
            n = len(idxs)

            precision_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) else 0.0
            recall_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) else 0.0
            f1_l = (2 * precision_l * recall_l / (precision_l + recall_l)) if (precision_l + recall_l) else 0.0
            accuracy_l = (tp_l + tn_l) / n if n else 0.0

            rows.append({
                "leaf_id": leaf_id,
                "n": n,
                "accuracy": accuracy_l,
                "precision": precision_l,
                "recall": recall_l,
                "f1": f1_l,
            })

        leaf_metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
        print("Per-leaf metrics (top 10):")
        print(leaf_metrics_df.head(10))

    # ------------------------------------------------------------
    # Return dictionary for logging
    # ------------------------------------------------------------
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "optimal_threshold": best_threshold,
        "optimal_f1": f1_scores[best_idx],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sensitivity_default": sensitivity_default,
        "specificity_default": specificity_default,
        "leaf_metrics": leaf_metrics_df if compute_leaf_metrics else None,
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


# ============================================================
#                 MAIN EVALUATION FUNCTION
# ============================================================
def evaluate_binary_oct_calibration(
    iai_model,
    X_test_df,
    y_test,
    preprocessor,
    feature_names,
    compute_leaf_metrics=False,
    constraint_precision=0.8,
    constraint_recall=0.8,
    plot_calibration=True
):
    print(f"Test dataset for OCT application: {len(X_test_df):,} samples")

    # ------------------------------------------------------------
    # Preprocessing + OCT Predictions
    # ------------------------------------------------------------
    X_test_processed = preprocessor.transform(X_test_df)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    y_pred_default = iai_model.predict(X_test_processed)
    y_proba = iai_model.predict_proba(X_test_processed).iloc[:, 1]

    X_test_processed["predicted_proba"] = y_proba
    leaf_assignments = iai_model.apply(X_test_processed)
    X_test_processed["leaf_assignment"] = leaf_assignments

    y_test_series = pd.Series(y_test).reset_index(drop=True)
    print("✓ Predictions completed")

    # ------------------------------------------------------------
    # Calibration evaluation
    # ------------------------------------------------------------
    brier = brier_score_loss(y_test_series, y_proba)
    ece = expected_calibration_error(y_test_series.values, y_proba)

    print(f"Brier score: {brier:.4f}")
    print(f"ECE: {ece:.4f}")

    if plot_calibration:
        prob_true, prob_pred = calibration_curve(y_test_series, y_proba, n_bins=10)
        plt.figure(figsize=(5, 5))
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Calibration Curve (Reliability Diagram)")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------
    # Threshold-free metrics (AUC)
    # ------------------------------------------------------------
    auc = iai_model.score(X_test_processed, y_test_series, criterion="auc")
    pr_auc = average_precision_score(y_test_series, y_proba)
    print(f"AUC score: {auc:.3f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}")

    # ------------------------------------------------------------
    # F1-optimal threshold
    # ------------------------------------------------------------
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test_series, y_proba)
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-12)

    best_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    y_pred_f1 = (y_proba >= best_f1_threshold).astype(int)

    print(f"Optimal-threshold F1={best_f1:.3f} at threshold={best_f1_threshold:.3f}")
    print(f"Confusion matrix (F1 threshold):\n{confusion_matrix(y_test_series, y_pred_f1)}")

    # ------------------------------------------------------------
    # Feasible threshold search (precision≥X, recall≥Y)
    # ------------------------------------------------------------
    feasible = find_feasible_threshold(
        y_test_series.values,
        y_proba,
        min_precision=constraint_precision,
        min_recall=constraint_recall
    )

    if feasible is None:
        print(f"⚠ No feasible threshold found with precision≥{constraint_precision} and recall≥{constraint_recall}")
        feasible_threshold = 0.5
    else:
        feasible_threshold = feasible["threshold"]
        print(f"Feasible threshold found: {feasible_threshold:.3f}")
    print(f"   Precision={feasible['precision']:.3f}")
    print(f"   Recall={feasible['recall']:.3f}")

    y_pred_feasible = (y_proba >= feasible_threshold).astype(int)
    print("Confusion matrix (feasible threshold):")
    print(confusion_matrix(y_test_series, y_pred_feasible))
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
        "optimal_threshold": best_f1_threshold,
        "optimal_f1": best_f1,
        "feasible_threshold": feasible_threshold,
    }
