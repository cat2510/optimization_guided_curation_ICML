import model_IAI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import time
import warnings

def create_cost_strata_(df_prebalanced, n_strata=8, stratifier_stage_cutoff=3):
    """
    Steps 3-4: Create cost strata and train OCT model following your pipeline.
    
    Parameters:
    -----------
    df_prebalanced : pd.DataFrame
        Pre-balanced training data (with >$200k 1:20 ratio)
    feature_cols : list
        All feature columns
    CAT_COLUMNS, TRUE_NUM_COLUMNS : lists
        Categorical and numeric column definitions
    n_strata : int
        Number of cost strata (default: 8)
    stratifier_stage_cutoff : int
        Stage cutoff for stratifier training (default: 3)
    
    Returns:
    --------
    dict : Contains all pipeline components and results
    """

    # STEP 3: Create 8 cost strata using high-stage (>3) enrollees
    print(f"\nStep 3: Creating {n_strata} cost strata based on high-stage (>{stratifier_stage_cutoff}) enrollees")
    print("-"*60)
    
    # Filter to high-stage enrollees for strata definition
    high_stage_data = df_prebalanced[df_prebalanced["stage_2017"] > stratifier_stage_cutoff].copy()
    print(f"High-stage data for strata definition: {len(high_stage_data):,} samples")
    
    # Create quantile bins based on high-stage data only
    try:
        # Get the quantile boundaries from high-stage data
        _, bin_edges = pd.qcut(
            high_stage_data["annual_cost17"], 
            q=n_strata, 
            labels=False,
            duplicates='drop',
            retbins=True
        )
        
        print(f"Cost strata boundaries (from high-stage data):")
        for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            print(f"  Stratum {i}: ${lower:,.0f} - ${upper:,.0f}")
        
        # Apply these boundaries to the FULL pre-balanced dataset
        print(f"\nApplying strata definitions to full pre-balanced dataset ({len(df_prebalanced):,} samples)...")
        
        # Use pd.cut with the boundaries from high-stage data
        df_prebalanced_with_strata = df_prebalanced.copy()
        df_prebalanced_with_strata['cost_stratum'] = pd.cut(
            df_prebalanced_with_strata["annual_cost17"],
            bins=bin_edges,
            labels=[i for i in range(n_strata)],
            include_lowest=True,
            duplicates='drop'
        )
        
        # Handle any NaN values (costs outside the range)
        nan_mask = df_prebalanced_with_strata['cost_stratum'].isna()
        if nan_mask.sum() > 0:
            print(f"Warning: {nan_mask.sum()} samples fall outside strata boundaries")
            # Assign extreme values to boundary strata
            extreme_low = df_prebalanced_with_strata[nan_mask]['annual_cost17'] < bin_edges[0]
            extreme_high = df_prebalanced_with_strata[nan_mask]['annual_cost17'] > bin_edges[-1]
            
            df_prebalanced_with_strata.loc[nan_mask & extreme_low, 'cost_stratum'] = 0
            df_prebalanced_with_strata.loc[nan_mask & extreme_high, 'cost_stratum'] = n_strata - 1
        
        # Convert to int
        df_prebalanced_with_strata['cost_stratum'] = df_prebalanced_with_strata['cost_stratum'].astype(int)
        
    except Exception as e:
        print(f"Error in quantile creation: {e}")
        raise e

    
    # Show strata distribution in full dataset
    print(f"\nCost strata distribution in full pre-balanced dataset:")
    strata_dist = df_prebalanced_with_strata['cost_stratum'].value_counts().sort_index()
    for stratum, count in strata_dist.items():
        pct = count / len(df_prebalanced_with_strata) * 100
        print(f"  Stratum {stratum}: {count:,} samples ({pct:.1f}%)")
    
    return df_prebalanced_with_strata

def train_stratifier_oct(df_prebalanced, stratification_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, bias_feature=None, target = "cost_stratum_2018",
best_params = None):
    if bias_feature:
        stratifier_train_pd = df_prebalanced[df_prebalanced[bias_feature]==1].copy()
        print(f"\n Training binary OCT M1 biased on {bias_feature} data with {len(stratifier_train_pd):,} samples")

    else:
        stratifier_train_pd = df_prebalanced
        print(f"\n Training binary OCT M1 on all train data with {len(stratifier_train_pd):,} samples")
    print("-"*60)
    
    # Prepare features (include stage columns for now - you can modify this)
    print(f"Using {len(stratification_cols)} stratification features for M1")
    # Prepare features and target for OCT training
    X_train_stratifier = stratifier_train_pd[stratification_cols]
    y_train_stratifier = stratifier_train_pd[target]
    
    print(f"M1 training set target {target} distribution:")
    oct_class_dist = y_train_stratifier.value_counts().sort_index()
    for stratum, count in oct_class_dist.items():
        pct = count / len(y_train_stratifier) * 100
        print(f"  Class {stratum}: {count:,} samples ({pct:.1f}%)")
    
    # Train the OCT model
    print(f"\nTraining M1 OCT...")
    try:
        if best_params:
            max_depth=best_params["max_depth"]
            minbucket=best_params["minbucket"]
            cp=best_params["cp"]
        else:
            max_depth=9
            minbucket=100
            cp=1e-5
        print("Using preset params: ", max_depth, minbucket, cp)

        model, preprocessor, feature_names = model_IAI.train_oct_with_feature_names(
            X_train_stratifier, y_train_stratifier, 
            categorical_cols=CAT_COLUMNS,
            numeric_cols=TRUE_NUM_COLUMNS,
            max_depth=max_depth,
            minbucket=minbucket,
            cp=cp
        )
        
        # Test preprocessing
        X_train_processed = preprocessor.fit_transform(X_train_stratifier)
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
        print(f"✓ OCT training completed successfully")
        print(f"✓ Processed training data shape: {X_train_processed.shape}")
        print(f"✓ Feature names: {len(feature_names)}")
        
    except Exception as e:
        print(f"✗ Error training OCT: {e}")
        raise e
    
    if bias_feature:
        # STEP 4b: Apply trained OCT to full train data
        print(f"\n Inference Step: Applying trained OCT to full train dataset")
        print("-"*60)
        
        # Prepare full dataset for prediction
        X_full = df_prebalanced[stratification_cols]
        print(f"Full dataset for OCT application: {len(X_full):,} samples")
        
        try:
            # Transform full dataset using the same preprocessor
            X_full_processed = preprocessor.transform(X_full)
            print(f"✓ Full data preprocessing completed: {X_full_processed.shape}")
            
            # Create DataFrame for leaf assignment
            X_full_processed_df = pd.DataFrame(X_full_processed, columns=feature_names)
            # Get multi-class predictions (cost strata)
            cost_stratum_predictions = model.predict(X_full_processed_df)
            print(f"✓ Cost stratum predictions completed")
        
            leaf_assignments = model.apply(X_full_processed_df)
        except Exception as e:
            print(f"✗ Error applying OCT to full dataset: {e}")
            raise e
    else:
        leaf_assignments = model.apply(X_train_processed_df)
        cost_stratum_predictions = model.predict(X_train_processed_df)


    print(f"✓ Leaf assignments completed")
    print(f"✓ Number of unique leaves: {len(np.unique(leaf_assignments))}")
    print(f"✓ Number of unique predicted strata: {len(np.unique(cost_stratum_predictions))}")

    # Add leaf assignments to the full dataset
    df_prebalanced['leaf_assignment'] = leaf_assignments
    df_prebalanced['predicted_cost_stratum'] = cost_stratum_predictions
    # Create a summary showing the relationship
    leaf_stratum_summary = df_prebalanced.groupby(['leaf_assignment', 'predicted_cost_stratum']).size().reset_index(name='count')
    print(f"Total unique leaf-class combinations: {len(leaf_stratum_summary)}")
        
        
        
    # Return all components for next steps
    results = {
        'df_with_leaves': df_prebalanced,
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'stratification_cols': stratification_cols,
        'cost_stratum_predictions': cost_stratum_predictions,  
        'leaf_assignments': leaf_assignments,
    }
    
    
    return results


# -----------------------------------------------------------
# Utility: Evaluate multi-class predictions
# -----------------------------------------------------------
def evaluate_multiclass(y_true, y_pred, y_proba, average_auc=True):
    """
    y_proba: array (n_samples, n_classes) or None
    Returns dict of metrics (robust to missing classes in a fold).
    """
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class recall (optional diagnostic)
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    for cls_label, cls_stats in report.items():
        if cls_label.isdigit():  # class keys are '0','1','2','3'
            metrics[f"recall_class_{cls_label}"] = cls_stats["recall"]
            metrics[f"precision_class_{cls_label}"] = cls_stats["precision"]
    
    if y_proba is not None and average_auc:
        # Macro & weighted multiclass AUC (OvR)
        try:
            macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            weighted_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            metrics["auc_macro_ovr"] = macro_auc
            metrics["auc_weighted_ovr"] = weighted_auc
        except ValueError:
            # One or more classes missing in this validation fold
            metrics["auc_macro_ovr"] = np.nan
            metrics["auc_weighted_ovr"] = np.nan
    
    return metrics


# -----------------------------------------------------------
# Hyperparameter tuning function
# -----------------------------------------------------------
def tune_oct_multiclass(
    df_prebalanced,
    stratification_cols,
    CAT_COLUMNS,
    TRUE_NUM_COLUMNS,
    target="cost_stratum_2018",
    bias_feature=None,
    param_grid=None,
    n_splits=5,
    random_state=42,
    refit_metric="f1_macro",
    verbose=True,
    max_models=None,
    random_sample=False,
    random_sample_size=20
):
    """
    Perform cross-validated grid (or random) search over OCT hyperparameters.
    
    Parameters
    ----------
    df_prebalanced : DataFrame
        Training dataframe (already pre-filtered / balanced upstream if desired).
    stratification_cols : list[str]
        Feature column names used for the OCT.
    CAT_COLUMNS : list[str]
        Categorical feature names (subset of stratification_cols).
    TRUE_NUM_COLUMNS : list[str]
        Numeric feature names (subset of stratification_cols).
    target : str
        Multi-class target column (0..K-1).
    bias_feature : str or None
        If provided, subset data where bias_feature == 1 is used for training/tuning.
    param_grid : dict or None
        Keys: 'max_depth', 'minbucket', 'cp'; values: list of candidate values.
    n_splits : int
        Number of StratifiedKFold folds.
    refit_metric : str
        Aggregated metric used to pick best model.
    max_models : int or None
        Hard cap on number of (hyperparam combinations) to evaluate (after random sampling).
    random_sample : bool
        If True, sample random_sample_size combinations from full grid.
    random_sample_size : int
        Number of random parameter sets to sample if random_sample=True.
    
    Returns
    -------
    dict with:
        'tuning_results' : DataFrame (aggregated metrics per combo)
        'cv_details'     : DataFrame (fold-level metrics per combo)
        'best_params'    : dict
        'best_model'     : fitted model on all (biased or full) training data
        'preprocessor'   : fitted preprocessor for best model
        'feature_names'  : list
        'train_leaf_assignments' : array
        'train_predictions'      : array
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [5, 7, 9],
            "minbucket": [100, 150,200,250],
            "cp": [5e-4, 1e-4, 1e-3,1e-5]
        }
    
    # Subset for bias if requested
    if bias_feature:
        data = df_prebalanced[df_prebalanced[bias_feature] == 1].copy()
        if verbose:
            print(f"Using biased subset ( {bias_feature} == 1 ): {len(data):,} rows")
    else:
        data = df_prebalanced.copy()
        if verbose:
            print(f"Using full dataset for tuning: {len(data):,} rows")
    
    # Basic target checks
    y = data[target].astype(int).values
    X = data[stratification_cols].copy()
    
    class_dist = pd.Series(y).value_counts().sort_index()
    if verbose:
        print("Target distribution (tuning base):")
        for cls, cnt in class_dist.items():
            print(f"  Class {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")
    
    # Build parameter combinations
    keys = list(param_grid.keys())
    all_combos = list(product(*[param_grid[k] for k in keys]))
    if verbose:
        print(f"Total parameter combinations before sampling: {len(all_combos)}")
    
    # Optional random sampling of combos
    if random_sample and len(all_combos) > random_sample_size:
        rng = np.random.default_rng(random_state)
        sampled_indices = rng.choice(len(all_combos), size=random_sample_size, replace=False)
        all_combos = [all_combos[i] for i in sampled_indices]
        if verbose:
            print(f"Randomly sampled {len(all_combos)} parameter combinations")
    
    # Truncate if max_models specified
    if max_models is not None:
        all_combos = all_combos[:max_models]
        if verbose:
            print(f"Evaluating only first {len(all_combos)} combinations due to max_models cap.")
    
    # Prepare CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    combo_counter = 0
    
    for combo in all_combos:
        params = dict(zip(keys, combo))
        combo_counter += 1
        start_combo_time = time.time()
        
        if verbose:
            print(f"\n[{combo_counter}/{len(all_combos)}] Evaluating params: {params}")
        
        fold_idx = 0
        for train_index, valid_index in skf.split(X, y):
            fold_idx += 1
            X_train_fold = X.iloc[train_index]
            y_train_fold = y[train_index]
            X_valid_fold = X.iloc[valid_index]
            y_valid_fold = y[valid_index]
            
            # Train a model for this fold
            try:
                model, preprocessor, feature_names = model_IAI.train_oct_with_feature_names(
                    X_train_fold, y_train_fold,
                    categorical_cols=CAT_COLUMNS,
                    numeric_cols=TRUE_NUM_COLUMNS,
                    max_depth=params["max_depth"],
                    minbucket=params["minbucket"],
                    cp=params["cp"]
                )
            except Exception as e:
                warnings.warn(f"Training failed for params {params} fold {fold_idx}: {e}")
                # Record NaN metrics so we can later filter
                fold_results.append({
                    **params,
                    "fold": fold_idx,
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "f1_weighted": np.nan,
                    "auc_macro_ovr": np.nan,
                    "auc_weighted_ovr": np.nan
                })
                continue
            
            # Transform validation set
            X_valid_processed = preprocessor.transform(X_valid_fold)
            X_valid_processed_df = pd.DataFrame(X_valid_processed, columns=feature_names)
            
            # Predictions & probs
            y_valid_pred = model.predict(X_valid_processed_df)
            # model.predict_proba may return (n, K) or DataFrame; adapt:
            try:
                proba = model.predict_proba(X_valid_processed_df)
                # If returns dict or DataFrame, unify to numpy array
                if isinstance(proba, pd.DataFrame):
                    y_valid_proba = proba.values
                elif isinstance(proba, dict):
                    # improbable for multi-class; skip
                    y_valid_proba = np.column_stack([proba[k] for k in sorted(proba.keys())])
                else:
                    y_valid_proba = np.array(proba)
            except Exception:
                y_valid_proba = None
            
            metrics = evaluate_multiclass(y_valid_fold, y_valid_pred, y_valid_proba)
            fold_results.append({
                **params,
                "fold": fold_idx,
                **metrics
            })
        
        elapsed = time.time() - start_combo_time
        if verbose:
            # Show interim aggregate for this combo
            tmp_df = pd.DataFrame([r for r in fold_results if all(r.get(k) == params[k] for k in keys)])
            agg_f1 = tmp_df["f1_macro"].mean()
            agg_acc = tmp_df["accuracy"].mean()
            print(f"  -> Interim avg acc={agg_acc:.4f}, f1_macro={agg_f1:.4f} (time {elapsed:.1f}s)")
    
    cv_details_df = pd.DataFrame(fold_results)
    
    # Aggregate across folds
    agg_funcs = {
        "accuracy": "mean",
        "f1_macro": "mean",
        "f1_weighted": "mean",
        "auc_macro_ovr": "mean",
        "auc_weighted_ovr": "mean"
    }
    # Include per-class metrics if present
    per_class_cols = [c for c in cv_details_df.columns if c.startswith("recall_class_") or c.startswith("precision_class_")]
    for c in per_class_cols:
        agg_funcs[c] = "mean"
    
    tuning_results = (cv_details_df
                      .groupby(keys, as_index=False)
                      .agg(agg_funcs)
                      .sort_values(refit_metric, ascending=False))
    
    if verbose:
        print("\n=== Hyperparameter Tuning Summary (top 10 by refit metric) ===")
        display_cols = keys + [refit_metric, "accuracy", "f1_weighted", "auc_macro_ovr", "auc_weighted_ovr"]
        print(tuning_results[display_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    if tuning_results.empty:
        raise RuntimeError("No successful model trainings; tune search failed.")
    
    best_row = tuning_results.iloc[0]
    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "minbucket": int(best_row["minbucket"]),
        "cp": float(best_row["cp"]),  # leave as float
    }
    
    if verbose:
        print(f"\nBest params by {refit_metric}: {best_params}")
    
    # -------------------------------------------------------
    # Refit final model on full (possibly biased) training set
    # -------------------------------------------------------
    if verbose:
        print("Refitting best model on full training data...")
    final_model, final_preprocessor, final_feature_names = model_IAI.train_oct_with_feature_names(
        X, y,
        categorical_cols=CAT_COLUMNS,
        numeric_cols=TRUE_NUM_COLUMNS,
        max_depth=best_params["max_depth"],
        minbucket=best_params["minbucket"],
        cp=best_params["cp"]
    )
    
    # Leaf assignments + predictions on train (for potential diagnostics)
    X_full_processed = final_preprocessor.transform(X)
    X_full_processed_df = pd.DataFrame(X_full_processed, columns=final_feature_names)
    train_preds = final_model.predict(X_full_processed_df)
    train_leaf_ids = final_model.apply(X_full_processed_df)
    
    results = {
        "tuning_results": tuning_results.reset_index(drop=True),
        "cv_details": cv_details_df,
        "best_params": best_params,
        "best_model": final_model,
        "preprocessor": final_preprocessor,
        "feature_names": final_feature_names,
        "train_leaf_assignments": train_leaf_ids,
        "train_predictions": train_preds
    }
    return results


# -----------------------------------------------------------
# Wrapper function that integrates tuning + final enrichment
# (Optional convenience if you want to match your previous API)
# -----------------------------------------------------------
def train_stratifier_oct_tuned(
    df_prebalanced,
    stratification_cols,
    CAT_COLUMNS,
    TRUE_NUM_COLUMNS,
    bias_feature=None,
    target="cost_stratum_2018",
    param_grid=None,
    oct_seed = 42,
    refit_metric="f1_macro",
    **tune_kwargs
):
    """
    High-level convenience: run hyperparameter tuning and then
    add leaf assignments & predictions back into df_prebalanced.

    Returns same structure as your original function + tuning info.
    """
    tuning_out = tune_oct_multiclass(
        df_prebalanced=df_prebalanced,
        stratification_cols=stratification_cols,
        CAT_COLUMNS=CAT_COLUMNS,
        TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
        target=target,
        bias_feature=bias_feature,
        param_grid=param_grid,
        refit_metric=refit_metric,
        random_state = oct_seed,
        **tune_kwargs
    )
    
    model = tuning_out["best_model"]
    preprocessor = tuning_out["preprocessor"]
    feature_names = tuning_out["feature_names"]
    
    # Apply to full dataset (not just biased subset) for downstream use
    X_all = df_prebalanced[stratification_cols]
    X_all_processed = preprocessor.transform(X_all)
    X_all_processed_df = pd.DataFrame(X_all_processed, columns=feature_names)
    
    predictions_all = model.predict(X_all_processed_df)
    leaf_ids_all = model.apply(X_all_processed_df)
    
    df_prebalanced = df_prebalanced.copy()
    df_prebalanced["predicted_cost_stratum"] = predictions_all
    df_prebalanced["leaf_assignment"] = leaf_ids_all
    
    leaf_stratum_summary = (
        df_prebalanced
        .groupby(["leaf_assignment", "predicted_cost_stratum"])
        .size().reset_index(name="count")
    )
    
    if tune_kwargs.get("verbose", True):
        print(f"\nPost-fit: {len(np.unique(leaf_ids_all))} unique leaves")
        print(f"Leaf-stratum rows: {len(leaf_stratum_summary)}")
    
    results = {
        "df_with_leaves": df_prebalanced,
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "stratification_cols": stratification_cols,
        "cost_stratum_predictions": predictions_all,
        "leaf_assignments": leaf_ids_all,
        "leaf_stratum_summary": leaf_stratum_summary,
        "tuning_results": tuning_out["tuning_results"],
        "cv_details": tuning_out["cv_details"],
        "best_params": tuning_out["best_params"]
    }
    return results