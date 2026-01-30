#!/usr/bin/env python3
"""
Statistical Significance Test for Baseline vs. Curated OCT
==========================================================
This script performs statistical significance testing by:
1. Using multiple train-test splits (different random seeds) instead of bootstrapping
2. Training both baseline (full imbalanced) and curated (k-center undersampled) OCT models
3. Computing confidence intervals from the distribution across splits
4. Computing p-values for the improvement of curated over baseline

This approach uses repeated holdout validation rather than bootstrapping.
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from scipy import stats
from scipy.stats import t as t_dist

# Import custom modules
import model_IAI
import kcenter_hyperparameter_search_global
from kcenter_hyperparameter_search_global import run_global_kcenter_matching, build_undersampled_dataset
from model_IAI import finetune_oct, evaluate_binary_oct
from precompute_distances import compute_distances_batched, save_distances_hdf5
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset configuration
DATASET_PATH = "creditcard.csv"
TARGET_COL = "Class"
RESULTS_DIR = "./uci_statistical_significance_results"

# Number of random seeds (train-test splits) to use
N_SPLITS = 10  # Use 20 different train-test splits
SEED_START = 42  # Starting seed value

# OCT hyperparameters
OCT_DEPTHS = [5,7]
OCT_MINBUCKETS = [25,50, 100, 150]
OCT_CPS = [0.00001, 0.0001, 0.001, 0.01]

# K-center matching hyperparameters (can use fixed best config or grid search)
# Option 1: Use fixed best configuration (faster)
USE_FIXED_BEST_CONFIG = True
FIXED_BEST_CONFIG = {
    'case_weighting': "boundary",
    'use_adaptive_pool': True,
    'seed_method': "density",
}

# Option 2: Run grid search for each split (slower but more thorough)
USE_GRID_SEARCH = False
GRID_SEARCH_CONFIGS = {
    'case_weighting': ["boundary", None],
    'use_adaptive_pool': [True, False],
    'seed_method': ["density", "smart", "centroid", "random"],
}

MATCHING_RATIO = 1  # 1:1 matching

# Confidence interval level
ALPHA = 0.05  # 95% confidence intervals

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_dataset_for_kcenter(X, y, dataset_name):
    """Prepare dataset for k-center matching."""
    df = X.copy()
    df['ENROLID'] = range(1, len(df) + 1)
    
    y_numeric = pd.to_numeric(y, errors='coerce')
    unique_vals = y_numeric.dropna().unique()
    
    if set(unique_vals).issubset({-1, 1}):
        df['target'] = ((y_numeric == 1).astype(int))
    elif set(unique_vals).issubset({0, 1}):
        df['target'] = y_numeric.astype(int)
    else:
        value_counts = y_numeric.value_counts()
        minority_value = value_counts.idxmin()
        df['target'] = (y_numeric == minority_value).astype(int)
    
    cols = ['ENROLID', 'target'] + [c for c in df.columns if c not in ['ENROLID', 'target']]
    df = df[cols]
    
    return df


def setup_feature_columns(df):
    """Setup feature columns, handling categorical and numeric types."""
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['ENROLID', 'target']]
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['ENROLID', 'target']]
    
    # Handle special columns (e.g., Amount, Time)
    feature_cols = numeric_cols.copy()
    
    # Remove Time if present (not useful for distance)
    if 'Time' in feature_cols:
        feature_cols.remove('Time')
    
    # Use log-transformed Amount if present
    if 'Amount' in feature_cols:
        if 'Amount_log' not in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])
        feature_cols.remove('Amount')
        if 'Amount_log' not in feature_cols:
            feature_cols.append('Amount_log')
    
    return df, feature_cols, cat_cols, numeric_cols, []


def get_preprocessor_with_impute(categorical_cols, numeric_cols):
    """Build preprocessor with imputation."""
    transformers = []
    
    if categorical_cols:
        cat_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))
    
    if numeric_cols:
        num_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    
    return ColumnTransformer(transformers=transformers, remainder="drop")


def precompute_case_control_distances(
    train_df, target_col, feature_cols,
    cat_columns, true_num_columns,
    dataset_name, seed
):
    """Precompute case-control distances for k-center matching."""
    cases = train_df[train_df[target_col] == 1].copy()
    controls = train_df[train_df[target_col] == 0].copy()
    
    X_cases = cases[feature_cols].copy()
    X_controls = controls[feature_cols].copy()
    
    # Fit preprocessing on controls only
    preprocessor = get_preprocessor_with_impute(
        categorical_cols=[c for c in cat_columns if c in X_controls.columns],
        numeric_cols=[c for c in true_num_columns if c in X_controls.columns],
    )
    X_controls_processed = preprocessor.fit_transform(X_controls)
    X_cases_processed = preprocessor.transform(X_cases)
    
    # Compute distances
    distances = compute_distances_batched(
        X_controls_processed, X_cases_processed,
        batch_size=1000, dtype=np.float32
    )
    
    # Save to H5
    h5_path = f"{RESULTS_DIR}/distances_{dataset_name}_seed_{seed}.h5"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_distances_hdf5(
        distances,
        majority_ids=controls['ENROLID'].values,
        minority_ids=cases['ENROLID'].values,
        filepath=h5_path
    )
    
    return h5_path, preprocessor


def create_train_test_split(df, target_col='target', test_size=0.3, val_size=0.5, random_state=123):
    """Create train/val/test splits."""
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=temp_df[target_col],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def compute_confidence_interval(values, alpha=0.05):
    """Compute confidence interval using t-distribution."""
    n = len(values)
    if n < 2:
        return None, None, None
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)  # Standard error
    
    # t-critical value for (1-alpha) confidence
    t_crit = t_dist.ppf(1 - alpha/2, df=n-1)
    
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    
    return mean, ci_lower, ci_upper


def compute_paired_t_test(baseline_values, curated_values):
    """Compute paired t-test for difference (curated - baseline)."""
    differences = np.array(curated_values) - np.array(baseline_values)
    n = len(differences)
    
    if n < 2:
        return None, None
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # t-statistic
    t_stat = mean_diff / se_diff if se_diff > 0 else 0
    
    # p-value (one-sided: H0: mean_diff <= 0, H1: mean_diff > 0)
    p_value = 1 - t_dist.cdf(t_stat, df=n-1)
    
    return t_stat, p_value


def run_single_split(
    df, seed, feature_cols, cat_columns, true_num_columns,
    dataset_name="fraud"
):
    """Run a single train-test split: train both models and evaluate."""
    print(f"\n{'='*80}")
    print(f"SPLIT {seed - SEED_START + 1}/{N_SPLITS} (seed={seed})")
    print(f"{'='*80}")
    
    # Create splits
    train_df, val_df, test_df = create_train_test_split(
        df, target_col='target', random_state=seed
    )
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"Train: {len(train_df):,} (minority: {(y_train == 1).sum():,})")
    print(f"Val: {len(val_df):,} (minority: {(y_val == 1).sum():,})")
    print(f"Test: {len(test_df):,} (minority: {(y_test == 1).sum():,})")
    
    # ========================================================================
    # TRAIN BASELINE MODEL (full imbalanced training data)
    # ========================================================================
    print(f"\n--- Training Baseline OCT (full imbalanced data) ---")
    baseline_model, baseline_params, _, baseline_preprocessor, baseline_feature_names = finetune_oct(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=cat_columns,
        numeric_cols=true_num_columns,
        depths=OCT_DEPTHS,
        minbuckets=OCT_MINBUCKETS,
        cps=OCT_CPS,
    )
    
    # Evaluate baseline on test set (using validation for threshold tuning)
    baseline_metrics = evaluate_binary_oct(
        baseline_model, X_test, y_test,
        baseline_preprocessor, baseline_feature_names,
        results_dir=None, save_suffix=None,
        X_val_df=X_val, y_val=y_val
    )
    
    baseline_roc_auc = baseline_metrics.get('auc', 0)
    baseline_pr_auc = baseline_metrics.get('pr_auc', 0)
    baseline_mcc = baseline_metrics.get('best_mcc', 0)
    
    print(f"Baseline - ROC-AUC: {baseline_roc_auc:.4f}, PR-AUC: {baseline_pr_auc:.4f}, MCC: {baseline_mcc:.4f}")
    
    # ========================================================================
    # TRAIN CURATED MODEL (k-center undersampled data)
    # ========================================================================
    print(f"\n--- Training Curated OCT (k-center undersampled data) ---")
    
    # Precompute distances
    h5_path, _ = precompute_case_control_distances(
        train_df, 'target', feature_cols,
        cat_columns, true_num_columns,
        dataset_name, seed
    )
    
    # Select best configuration
    if USE_FIXED_BEST_CONFIG:
        best_config = FIXED_BEST_CONFIG
        print(f"Using fixed best config: {best_config}")
    elif USE_GRID_SEARCH:
        # Run grid search to find best config (based on validation PR-AUC)
        print("Running grid search for best configuration...")
        best_config = None
        best_val_pr_auc = -1
        
        for case_weighting, use_adaptive_pool, seed_method in product(
            GRID_SEARCH_CONFIGS['case_weighting'],
            GRID_SEARCH_CONFIGS['use_adaptive_pool'],
            GRID_SEARCH_CONFIGS['seed_method']
        ):
            config = {
                'case_weighting': case_weighting,
                'use_adaptive_pool': use_adaptive_pool,
                'seed_method': seed_method,
            }
            
            # Run k-center matching
            matching_result = run_global_kcenter_matching(
                train_pd=train_df,
                target_col='target',
                feature_cols=feature_cols,
                pn_h5_path=h5_path,
                matching_ratio=MATCHING_RATIO,
                case_weighting=case_weighting,
                use_adaptive_pool=use_adaptive_pool,
                seed_method=seed_method,
                CAT_COLUMNS=cat_columns,
                TRUE_NUM_COLUMNS=true_num_columns,
                COST_COLUMNS=None,
            )
            
            # Build undersampled dataset
            undersampled_train = build_undersampled_dataset(
                train_pd=train_df,
                matching_result=matching_result,
                target_col='target',
                matching_ratio=MATCHING_RATIO,
            )
            
            # Train OCT
            curated_model_temp, _, _, preprocessor_temp, feature_names_temp = finetune_oct(
                X_train=undersampled_train[feature_cols],
                y_train=undersampled_train['target'],
                X_val=X_val,
                y_val=y_val,
                categorical_cols=cat_columns,
                numeric_cols=true_num_columns,
                depths=OCT_DEPTHS,
                minbuckets=OCT_MINBUCKETS,
                cps=OCT_CPS,
            )
            
            # Evaluate on validation set
            X_val_processed = preprocessor_temp.transform(X_val)
            X_val_processed = pd.DataFrame(X_val_processed, columns=feature_names_temp)
            y_val_pred_proba = curated_model_temp.predict_proba(X_val_processed).iloc[:, 1]
            val_pr_auc = average_precision_score(y_val, y_val_pred_proba)
            
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_config = config
        
        print(f"Best config (val PR-AUC={best_val_pr_auc:.4f}): {best_config}")
    else:
        raise ValueError("Either USE_FIXED_BEST_CONFIG or USE_GRID_SEARCH must be True")
    
    # Run k-center matching with best config
    matching_result = run_global_kcenter_matching(
        train_pd=train_df,
        target_col='target',
        feature_cols=feature_cols,
        pn_h5_path=h5_path,
        matching_ratio=MATCHING_RATIO,
        case_weighting=best_config['case_weighting'],
        use_adaptive_pool=best_config['use_adaptive_pool'],
        seed_method=best_config['seed_method'],
        CAT_COLUMNS=cat_columns,
        TRUE_NUM_COLUMNS=true_num_columns,
        COST_COLUMNS=None,
    )
    
    # Build undersampled dataset
    undersampled_train = build_undersampled_dataset(
        train_pd=train_df,
        matching_result=matching_result,
        target_col='target',
        matching_ratio=MATCHING_RATIO,
    )
    
    print(f"Undersampled training data: {len(undersampled_train):,} samples")
    print(f"  - Minority: {(undersampled_train['target'] == 1).sum():,}")
    print(f"  - Majority: {(undersampled_train['target'] == 0).sum():,}")
    
    # Train curated OCT
    curated_model, curated_params, _, curated_preprocessor, curated_feature_names = finetune_oct(
        X_train=undersampled_train[feature_cols],
        y_train=undersampled_train['target'],
        X_val=X_val,
        y_val=y_val,
        categorical_cols=cat_columns,
        numeric_cols=true_num_columns,
        depths=OCT_DEPTHS,
        minbuckets=OCT_MINBUCKETS,
        cps=OCT_CPS,
    )
    
    # Evaluate curated on test set (using validation for threshold tuning)
    curated_metrics = evaluate_binary_oct(
        curated_model, X_test, y_test,
        curated_preprocessor, curated_feature_names,
        results_dir=None, save_suffix=None,
        X_val_df=X_val, y_val=y_val
    )
    
    curated_roc_auc = curated_metrics.get('auc', 0)
    curated_pr_auc = curated_metrics.get('pr_auc', 0)
    curated_mcc = curated_metrics.get('best_mcc', 0)
    
    print(f"Curated - ROC-AUC: {curated_roc_auc:.4f}, PR-AUC: {curated_pr_auc:.4f}, MCC: {curated_mcc:.4f}")
    
    # Return results
    return {
        'seed': seed,
        'baseline_roc_auc': baseline_roc_auc,
        'baseline_pr_auc': baseline_pr_auc,
        'baseline_mcc': baseline_mcc,
        'curated_roc_auc': curated_roc_auc,
        'curated_pr_auc': curated_pr_auc,
        'curated_mcc': curated_mcc,
        'diff_roc_auc': curated_roc_auc - baseline_roc_auc,
        'diff_pr_auc': curated_pr_auc - baseline_pr_auc,
        'diff_mcc': curated_mcc - baseline_mcc,
        'best_config': str(best_config),
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TEST: BASELINE vs. CURATED OCT")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of splits: {N_SPLITS}")
    print(f"Using {'fixed best config' if USE_FIXED_BEST_CONFIG else 'grid search'}")
    print(f"{'='*80}\n")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    df_raw = pd.read_csv(DATASET_PATH)
    X = df_raw.drop(TARGET_COL, axis=1)
    y = df_raw[TARGET_COL]
    
    # Prepare dataset
    df = prepare_dataset_for_kcenter(X, y, "fraud")
    df, feature_cols, cat_columns, true_num_columns, bin_columns = setup_feature_columns(df)
    
    print(f"\nDataset prepared:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Minority class: {(df['target'] == 1).sum():,} ({(df['target'] == 1).mean()*100:.2f}%)")
    
    # Run all splits
    all_results = []
    seeds = range(SEED_START, SEED_START + N_SPLITS)
    
    for seed in seeds:
        try:
            result = run_single_split(
                df, seed, feature_cols, cat_columns, true_num_columns, "fraud"
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ ERROR in split {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) == 0:
        print("\n✗ No successful splits. Exiting.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = f"{RESULTS_DIR}/all_splits_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # ========================================================================
    # COMPUTE CONFIDENCE INTERVALS AND P-VALUES
    # ========================================================================
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")
    
    # Extract metrics
    baseline_roc_aucs = results_df['baseline_roc_auc'].values
    baseline_pr_aucs = results_df['baseline_pr_auc'].values
    baseline_mccs = results_df['baseline_mcc'].values
    
    curated_roc_aucs = results_df['curated_roc_auc'].values
    curated_pr_aucs = results_df['curated_pr_auc'].values
    curated_mccs = results_df['curated_mcc'].values
    
    diff_roc_aucs = results_df['diff_roc_auc'].values
    diff_pr_aucs = results_df['diff_pr_auc'].values
    diff_mccs = results_df['diff_mcc'].values
    
    # Compute confidence intervals
    print("CONFIDENCE INTERVALS (95%):")
    print("-" * 80)
    
    # Baseline CIs
    baseline_roc_mean, baseline_roc_ci_lower, baseline_roc_ci_upper = compute_confidence_interval(
        baseline_roc_aucs, ALPHA
    )
    baseline_pr_mean, baseline_pr_ci_lower, baseline_pr_ci_upper = compute_confidence_interval(
        baseline_pr_aucs, ALPHA
    )
    baseline_mcc_mean, baseline_mcc_ci_lower, baseline_mcc_ci_upper = compute_confidence_interval(
        baseline_mccs, ALPHA
    )
    
    print(f"\nBaseline OCT:")
    print(f"  ROC-AUC: {baseline_roc_mean:.4f} [{baseline_roc_ci_lower:.4f}, {baseline_roc_ci_upper:.4f}]")
    print(f"  PR-AUC:  {baseline_pr_mean:.4f} [{baseline_pr_ci_lower:.4f}, {baseline_pr_ci_upper:.4f}]")
    print(f"  MCC:     {baseline_mcc_mean:.4f} [{baseline_mcc_ci_lower:.4f}, {baseline_mcc_ci_upper:.4f}]")
    
    # Curated CIs
    curated_roc_mean, curated_roc_ci_lower, curated_roc_ci_upper = compute_confidence_interval(
        curated_roc_aucs, ALPHA
    )
    curated_pr_mean, curated_pr_ci_lower, curated_pr_ci_upper = compute_confidence_interval(
        curated_pr_aucs, ALPHA
    )
    curated_mcc_mean, curated_mcc_ci_lower, curated_mcc_ci_upper = compute_confidence_interval(
        curated_mccs, ALPHA
    )
    
    print(f"\nCurated OCT:")
    print(f"  ROC-AUC: {curated_roc_mean:.4f} [{curated_roc_ci_lower:.4f}, {curated_roc_ci_upper:.4f}]")
    print(f"  PR-AUC:  {curated_pr_mean:.4f} [{curated_pr_ci_lower:.4f}, {curated_pr_ci_upper:.4f}]")
    print(f"  MCC:     {curated_mcc_mean:.4f} [{curated_mcc_ci_lower:.4f}, {curated_mcc_ci_upper:.4f}]")
    
    # Difference CIs
    diff_roc_mean, diff_roc_ci_lower, diff_roc_ci_upper = compute_confidence_interval(
        diff_roc_aucs, ALPHA
    )
    diff_pr_mean, diff_pr_ci_lower, diff_pr_ci_upper = compute_confidence_interval(
        diff_pr_aucs, ALPHA
    )
    diff_mcc_mean, diff_mcc_ci_lower, diff_mcc_ci_upper = compute_confidence_interval(
        diff_mccs, ALPHA
    )
    
    print(f"\nDifference (Curated - Baseline):")
    print(f"  ROC-AUC: {diff_roc_mean:+.4f} [{diff_roc_ci_lower:+.4f}, {diff_roc_ci_upper:+.4f}]")
    print(f"  PR-AUC:  {diff_pr_mean:+.4f} [{diff_pr_ci_lower:+.4f}, {diff_pr_ci_upper:+.4f}]")
    print(f"  MCC:     {diff_mcc_mean:+.4f} [{diff_mcc_ci_lower:+.4f}, {diff_mcc_ci_upper:+.4f}]")
    
    # Compute p-values (one-sided: H0: curated <= baseline, H1: curated > baseline)
    print(f"\n{'='*80}")
    print("P-VALUES (One-sided: H0: Curated <= Baseline, H1: Curated > Baseline)")
    print(f"{'='*80}\n")
    
    t_roc, p_roc = compute_paired_t_test(baseline_roc_aucs, curated_roc_aucs)
    t_pr, p_pr = compute_paired_t_test(baseline_pr_aucs, curated_pr_aucs)
    t_mcc, p_mcc = compute_paired_t_test(baseline_mccs, curated_mccs)
    
    print(f"ROC-AUC improvement:")
    print(f"  t-statistic: {t_roc:.4f}")
    print(f"  p-value: {p_roc:.6f} {'***' if p_roc < 0.001 else '**' if p_roc < 0.01 else '*' if p_roc < 0.05 else '(ns)'}")
    
    print(f"\nPR-AUC improvement:")
    print(f"  t-statistic: {t_pr:.4f}")
    print(f"  p-value: {p_pr:.6f} {'***' if p_pr < 0.001 else '**' if p_pr < 0.01 else '*' if p_pr < 0.05 else '(ns)'}")
    
    print(f"\nMCC improvement:")
    print(f"  t-statistic: {t_mcc:.4f}")
    print(f"  p-value: {p_mcc:.6f} {'***' if p_mcc < 0.001 else '**' if p_mcc < 0.01 else '*' if p_mcc < 0.05 else '(ns)'}")
    
    # Summary table
    summary_data = {
        'Metric': ['ROC-AUC', 'PR-AUC', 'MCC'],
        'Baseline_Mean': [baseline_roc_mean, baseline_pr_mean, baseline_mcc_mean],
        'Baseline_CI_Lower': [baseline_roc_ci_lower, baseline_pr_ci_lower, baseline_mcc_ci_lower],
        'Baseline_CI_Upper': [baseline_roc_ci_upper, baseline_pr_ci_upper, baseline_mcc_ci_upper],
        'Curated_Mean': [curated_roc_mean, curated_pr_mean, curated_mcc_mean],
        'Curated_CI_Lower': [curated_roc_ci_lower, curated_pr_ci_lower, curated_mcc_ci_lower],
        'Curated_CI_Upper': [curated_roc_ci_upper, curated_pr_ci_upper, curated_mcc_ci_upper],
        'Difference_Mean': [diff_roc_mean, diff_pr_mean, diff_mcc_mean],
        'Difference_CI_Lower': [diff_roc_ci_lower, diff_pr_ci_lower, diff_mcc_ci_lower],
        'Difference_CI_Upper': [diff_roc_ci_upper, diff_pr_ci_upper, diff_mcc_ci_upper],
        'P_Value': [p_roc, p_pr, p_mcc],
    }
    summary_df = pd.DataFrame(summary_data)
    
    summary_path = f"{RESULTS_DIR}/statistical_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Successful splits: {len(all_results)}/{N_SPLITS}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
