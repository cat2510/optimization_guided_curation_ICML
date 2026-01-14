#!/usr/bin/env python3
"""
K-Center Matching Hyperparameter Search (Global Undersampling)
================================================================
This script performs a grid search over k-center matching hyperparameters:
- Case weighting: None, "boundary", "uncertainty", "density_inverse"
- Adaptive pool: True/False
- Seed method: "smart", "centroid", "density", "random"

For each configuration, it:
1. Runs GLOBAL k-center matching (single call per config) to create undersampled training data
2. Trains an OCT model on the undersampled data
3. Evaluates and saves results

NOTE: This version does NOT use per-leaf stratification - it undersamples globally.
"""

import os
import sys
import pickle
import importlib
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime

# Import custom modules
import model_pipeline
import model_IAI
import two_stage_kcenter_match
from two_stage_kcenter_match import two_stage_kcenter_then_match
from precompute_distances import precompute_leaf_dnn_memmap
from model_pipeline import get_preprocessor, get_bin_flag_columns
from sklearn.impute import SimpleImputer

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hyperparameter grid
MATCHING_RATIO = 1  # Fixed for now, can be added to grid if needed
HYPERPARAMETER_GRID = {
    'case_weighting': [None, "boundary", "uncertainty", "density_inverse"],
    'use_adaptive_pool': [True, False],
    'seed_method': ["smart", "centroid", "density", "random"],
}

# OCT hyperparameters for model training
OCT_DEPTHS = [7, 9]
OCT_MINBUCKETS = [50, 100, 120, 150]
OCT_CPS = [0.00001, 0.0001, 0.001, 0.01]

# Paths
PN_H5_PATH = "./precomputed_distances/distances_majority_minority.h5"
RESULTS_DIR = "./kcenter_hyperparameter_search_results_global"
DNN_OUT_DIR = "./precomputed_distances/global_dnn"

# Target column
TARGET_COL = "highcost_gt_200000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_probabilities_for_uncertainty(
    train_pd,
    feature_cols,
    target_col,
    CAT_COLUMNS,
    TRUE_NUM_COLUMNS,
    X_cases,
):
    """
    Train a simple model on full training data to get probabilities for uncertainty weighting.
    Uses a Random Forest for quick training.
    """
    from sklearn.ensemble import RandomForestClassifier
    from model_pipeline import get_preprocessor
    
    print(f"\n  Training Random Forest for uncertainty weighting...")
    
    # Preprocess training data
    preprocessor = get_preprocessor(
        df=train_pd[feature_cols],
        categorical_cols=CAT_COLUMNS,
        numeric_cols=TRUE_NUM_COLUMNS,
        verbose=False
    )
    X_train_processed = preprocessor.transform(train_pd[feature_cols])
    y_train = train_pd[target_col]
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_processed, y_train)
    
    # Get probabilities for cases
    X_cases_processed = preprocessor.transform(X_cases)
    predicted_probs = rf.predict_proba(X_cases_processed)
    
    print(f"    ✓ Got probabilities for {len(predicted_probs)} cases")
    
    return predicted_probs


def run_global_kcenter_matching(
    train_pd,
    target_col,
    feature_cols,
    pn_h5_path,
    matching_ratio,
    case_weighting,
    use_adaptive_pool,
    seed_method,
    CAT_COLUMNS=None,
    TRUE_NUM_COLUMNS=None,
    COST_COLUMNS=None,
):
    """
    Run global k-center matching (single call) with specified hyperparameters.
    
    Returns:
        dict: Matching results with selected controls and costs
    """
    print(f"\n{'='*80}")
    print(f"GLOBAL K-CENTER MATCHING CONFIGURATION:")
    print(f"  case_weighting: {case_weighting}")
    print(f"  use_adaptive_pool: {use_adaptive_pool}")
    print(f"  seed_method: {seed_method}")
    print(f"  matching_ratio: 1:{matching_ratio}")
    print(f"{'='*80}\n")
    
    # Extract cases and controls globally
    cases = train_pd[train_pd[target_col] == 1].copy()
    controls = train_pd[train_pd[target_col] == 0].copy()
    
    n_cases = len(cases)
    n_controls = len(controls)
    
    print(f"Global Statistics:")
    print(f"  Cases (minority): {n_cases:,}")
    print(f"  Controls (majority): {n_controls:,}")
    print(f"  Ratio: {n_controls/n_cases if n_cases > 0 else float('inf'):.2f}:1")
    
    if n_cases == 0:
        raise ValueError("No cases in training data!")
    
    if n_controls == 0:
        raise ValueError("No controls in training data!")
    
    # Handle edge case: More cases than controls
    if n_cases > n_controls:
        print(f"\n  ⚠️ WARNING: More cases ({n_cases:,}) than controls ({n_controls:,})")
        print(f"     Strategy: SKIP k-center matching - keep ALL minority and ALL majority as-is")
        
        return {
            'n_cases_total': n_cases,
            'n_cases_matched': 0,
            'n_controls': n_controls,
            'M': 0,
            'selected_control_enrolids': controls["ENROLID"].to_numpy(),
            'match_costs': np.array([]),
            'candidate_majority_enrolids': controls["ENROLID"].to_numpy(),
            'case_to_control_map': {},
            'skipped_reason': 'more_cases_than_controls'
        }
    
    # Preprocess controls to get feature matrix
    exclude_cols = ["cost_stratum_2018"] + (COST_COLUMNS or [])
    drop_cols = ['ENROLID', target_col] + exclude_cols
    feature_cols_controls = [c for c in controls.columns if c not in drop_cols]
    
    numeric_cols = controls[feature_cols_controls].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = controls[feature_cols_controls].select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Impute and preprocess
    controls_preprocessed = controls[feature_cols_controls].copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        controls_preprocessed[numeric_cols] = imputer.fit_transform(controls_preprocessed[numeric_cols])
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        controls_preprocessed[categorical_cols] = cat_imputer.fit_transform(controls_preprocessed[categorical_cols])
    
    bin_feats = get_bin_flag_columns(controls_preprocessed)
    num_feats = [c for c in numeric_cols if c not in bin_feats]
    
    preprocessor_controls = get_preprocessor(
        df=controls_preprocessed,
        categorical_cols=categorical_cols,
        numeric_cols=num_feats,
        verbose=False
    )
    X_majority = preprocessor_controls.fit_transform(controls_preprocessed)
    
    print(f"\n  Preprocessing:")
    print(f"    Features: {len(feature_cols_controls)}")
    print(f"    Preprocessed shape: {X_majority.shape}")
    
    # Precompute or load control-control distances globally
    print(f"\n  Preparing control-control distances (global)...")
    majority_enrolids = controls["ENROLID"].to_numpy()

    # Paths used by precompute_leaf_dnn_memmap for the global run
    dnn_matrix_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_enrolids.npy")

    if os.path.exists(dnn_matrix_npy) and os.path.exists(dnn_enrolids_npy):
        print(f"    ✓ Found existing global d_nn files, loading paths:")
        print(f"      d_nn matrix: {dnn_matrix_npy}")
        print(f"      d_nn enrolids: {dnn_enrolids_npy}")
    else:
        print(f"    ⚙️  Computing global control-control distances (this is done once)...")
        dnn_matrix_npy, dnn_enrolids_npy = precompute_leaf_dnn_memmap(
            X_majority_leaf=X_majority,
            majority_enrolids_leaf=majority_enrolids,
            out_dir=DNN_OUT_DIR,
            leaf_id="global",  # Use \"global\" as leaf_id
            batch_size=750,
        )
        print(f"    ✓ Control-control distances computed and saved")
    
    # Normal case: n_controls >= n_cases
    M = max(n_cases * matching_ratio, min(8000, n_controls))
    
    print(f"\n  K-Center Configuration:")
    print(f"    M (candidate pool size): {M:,} / {n_controls:,} ({M/n_controls*100:.1f}%)")
    print(f"    Cases to match: {n_cases:,}")
    print(f"    Seed method: {seed_method}")
    print(f"    Adaptive pool: {use_adaptive_pool}")
    print(f"    Case weighting: {case_weighting}")
    
    # Get probabilities if needed for uncertainty weighting
    predicted_probs = None
    if case_weighting == "uncertainty":
        print(f"\n  Computing probabilities for uncertainty weighting...")
        predicted_probs = get_model_probabilities_for_uncertainty(
            train_pd=train_pd,
            feature_cols=feature_cols,
            target_col=target_col,
            CAT_COLUMNS=CAT_COLUMNS,
            TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
            X_cases=cases[feature_cols],
        )
    
    # Run two-stage k-center matching
    print(f"\n  Running two-stage k-center matching (1:{matching_ratio})...")
    try:
        out = two_stage_kcenter_then_match(
            leaf_controls_enrolids=controls["ENROLID"].to_numpy(),
            leaf_cases_enrolids=cases["ENROLID"].to_numpy(),
            leaf_nn_matrix_npy=dnn_matrix_npy,
            leaf_nn_enrolids_npy=dnn_enrolids_npy,
            pn_h5_path=pn_h5_path,
            M=M,
            use_adaptive_pool=use_adaptive_pool,
            tau=None,  # Auto-compute from 95th percentile
            plateau_eps=0.01,
            force_nearest_per_case=True,
            force_topm=1,
            assignment_topk_start=None,  # Exact matching
            seed_method=seed_method,
            matching_ratio=matching_ratio,
            X_majority_leaf=X_majority,
            case_weighting=case_weighting,
            predicted_probs=predicted_probs,
        )
        
        # Extract results
        selected_control_enrolids = out["selected_control_enrolids"]
        all_match_costs = out["match_costs"]
        candidate_majority_enrolids = out["candidate_majority_enrolids"]
        case_to_control_map = out["case_to_control_map"]
        
        print(f"    ✓ Matching complete!")
        print(f"    Cases matched: {n_cases:,}")
        print(f"    Total selected controls: {len(selected_control_enrolids):,} (unique: {len(set(selected_control_enrolids)):,})")
        print(f"    Mean matching cost: {all_match_costs.mean():.4f}")
        
        return {
            'n_cases_total': n_cases,
            'n_cases_matched': n_cases,
            'n_controls': n_controls,
            'M': M,
            'matching_ratio': matching_ratio,
            'selected_control_enrolids': selected_control_enrolids,
            'match_costs': all_match_costs,
            'candidate_majority_enrolids': candidate_majority_enrolids,
            'case_to_control_map': case_to_control_map,
            'skipped_reason': None
        }
        
    except Exception as e:
        print(f"    ✗ ERROR in global matching: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_undersampled_dataset(train_pd, matching_result, target_col, matching_ratio):
    """Build undersampled training dataset from matching results."""
    print(f"\n{'='*80}")
    print("BUILDING UNDERSAMPLED TRAINING DATASET")
    print(f"{'='*80}\n")
    
    # Collect all minority samples (keep ALL)
    all_minority = train_pd[train_pd[target_col] == 1].copy()
    print(f"✓ Collected all minority samples: {len(all_minority):,}")
    
    # Collect majority samples from matching result
    if matching_result.get('skipped_reason') == 'more_cases_than_controls':
        # For imbalanced case, keep all controls
        selected_majority_enrolids = matching_result['selected_control_enrolids'].tolist()
    else:
        # For matched case, use the selected controls
        selected_majority_enrolids = matching_result['selected_control_enrolids'].tolist()
    
    # Handle control reuse in 1:k matching
    if matching_ratio > 1:
        print(f"\n⚠️ 1:{matching_ratio} matching: Controls may be reused")
        print(f"   Total selections: {len(selected_majority_enrolids):,}")
        print(f"   Unique controls: {len(set(selected_majority_enrolids)):,}")
        print(f"   Reused controls: {len(selected_majority_enrolids) - len(set(selected_majority_enrolids)):,}")
    
    # Get unique majority samples
    unique_majority_enrolids = list(set(selected_majority_enrolids))
    selected_majority = train_pd[
        (train_pd[target_col] == 0) & 
        (train_pd['ENROLID'].isin(unique_majority_enrolids))
    ].copy()
    
    print(f"✓ Collected selected majority samples: {len(selected_majority):,}")
    
    # Combine minority and majority
    undersampled_training_data = pd.concat([all_minority, selected_majority], axis=0, ignore_index=True)
    
    print(f"\n✓ Undersampled dataset created:")
    print(f"   Total samples: {len(undersampled_training_data):,}")
    print(f"   Minority: {len(all_minority):,}")
    print(f"   Majority: {len(selected_majority):,}")
    print(f"   Ratio (maj:min): {len(selected_majority)/len(all_minority):.2f}:1")
    print(f"\nClass distribution:")
    print(undersampled_training_data[target_col].value_counts().sort_index())
    
    return undersampled_training_data


def train_and_evaluate_oct(
    undersampled_training_data,
    X_val, y_val,
    X_test, y_test,
    feature_cols,
    target_col,
    CAT_COLUMNS,
    TRUE_NUM_COLUMNS,
    config_name,
    results_dir,
):
    """Train OCT model and evaluate on test set."""
    print(f"\n{'='*80}")
    print(f"TRAINING OCT MODEL: {config_name}")
    print(f"{'='*80}\n")
    
    from model_IAI import finetune_oct, evaluate_binary_oct
    
    # Train OCT
    balanced_model, balanced_params, _, preprocessor, feature_names = finetune_oct(
        X_train=undersampled_training_data[feature_cols],
        y_train=undersampled_training_data[target_col],
        X_val=X_val,
        y_val=y_val,
        categorical_cols=CAT_COLUMNS,
        numeric_cols=TRUE_NUM_COLUMNS,
        depths=OCT_DEPTHS,
        minbuckets=OCT_MINBUCKETS,
        cps=OCT_CPS,
    )
    
    # Evaluate
    metrics = evaluate_binary_oct(
        balanced_model, X_test, y_test, preprocessor, feature_names,
        results_dir=results_dir, ratio=1.0
    )
    
    print(f"\n✓ Model training complete:")
    print(f"   Best params: {balanced_params}")
    
    if isinstance(metrics, dict):
        print(f"   AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"   AUC: {metrics.get('auc', 'N/A')}")
        print(f"   PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"   PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        print(f"   Optimal F1: {metrics.get('optimal_f1', 'N/A'):.4f}" if isinstance(metrics.get('optimal_f1'), (int, float)) else f"   Optimal F1: {metrics.get('optimal_f1', 'N/A')}")
        print(f"   Sensitivity (G-mean threshold): {metrics.get('balanced_recall_gmean', 'N/A'):.4f}" if isinstance(metrics.get('balanced_recall_gmean'), (int, float)) else f"   Sensitivity: {metrics.get('sensitivity_f1', 'N/A')}")
        print(f"   Specificity (G-mean threshold): {metrics.get('balanced_specificity_gmean', 'N/A'):.4f}" if isinstance(metrics.get('balanced_specificity_gmean'), (int, float)) else f"   Specificity: {metrics.get('specificity_f1', 'N/A')}")
    else:
        print(f"   WARNING: metrics is not a dict, type={type(metrics)}")
    
    return balanced_model, balanced_params, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print("K-CENTER HYPERPARAMETER SEARCH (GLOBAL UNDERSAMPLING)")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"✓ Results will be saved to: {RESULTS_DIR}/\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print(f"{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}\n")
    
    # Load Spark data
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("KCenterSearchGlobal").getOrCreate()
    df_2018 = spark.read.format("parquet").load("0917_2017_18_with_2017_cost.parquet")
    df_og = df_2018.toPandas()
    print(f"✓ Loaded original data: {df_og.shape}")
    
    # Setup columns (same as notebook)
    BIN_FLAG_COLUMNS = model_pipeline.get_bin_flag_columns(df_og) + [
        'lab_monitoring_adherent', 'nephrology_consult_adherent', 'early_nephrology_referral'
    ]
    STAGE_COLUMNS = [col for col in df_og.columns if "stage" in col.lower()]
    CAT_COLUMNS = df_og.select_dtypes(include=["object", "category"]).columns.tolist()
    TRUE_NUM_COLUMNS = model_pipeline.get_true_num_columns(df_og, CAT_COLUMNS) + [
        'util_2017', 'total_increasing_quarters_2017', 'total_lab_tests', 
        'ckd_visit_count', 'quarters_with_labs', 'nephrology_visit_count', 
        'days_to_nephrology', 'MEDIAN_INCOME'
    ]
    COST_COLUMNS = [
        col for col in df_og.columns 
        if "cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower()
    ]
    
    # Create cost stratum
    def make_cost_stratum_3class(df):
        cost_stratum = pd.Series(0, index=df.index)
        cost_stratum[(df['highcost_gt_50000'] == 1) & (df['highcost_gt_100000'] == 0)] = 1
        cost_stratum[(df['highcost_gt_100000'] == 1) & (df['highcost_gt_200000'] == 0)] = 2
        cost_stratum[df['highcost_gt_200000'] == 1] = 3
        return cost_stratum
    
    df_og['cost_stratum_2018'] = make_cost_stratum_3class(df_og)
    
    # Setup feature columns
    cutoff_columns = [col for col in df_og.columns if col.startswith('highcost_gt_')]
    feature_cols = [
        c for c in df_og.columns
        if c not in (['annual_cost_2017', 'annual_cost_2018_deflated', "ENROLID", "cost_stratum_2018"] + cutoff_columns)
    ]
    
    # Remove high-correlation features
    numeric_cols = df_og[feature_cols + ["cost_stratum_2018"]].select_dtypes(include=["number"]).columns
    corrs = df_og[numeric_cols].corr()["cost_stratum_2018"].abs().sort_values(ascending=False)
    high_corr_cols = corrs[corrs > 0.5].index.tolist()
    high_corr_cols = [col for col in high_corr_cols if col != "cost_stratum_2018"]
    feature_cols = [col for col in feature_cols if col not in high_corr_cols]
    
    print(f"✓ Feature columns: {len(feature_cols)}")
    
    # Train/test split
    train_ids, test_ids, train_pd, test_pd = model_pipeline.train_test_split_enrol(
        df_og, target_col="cost_stratum_2018", test_size=0.3, verbose=False, random_state=123
    )
    val_ids, test_ids, val_pd, test_pd = model_pipeline.train_test_split_enrol(
        test_pd, target_col=TARGET_COL, test_size=0.5, verbose=False
    )
    
    X_test = test_pd[feature_cols]
    y_test = test_pd[TARGET_COL]
    X_val = val_pd[feature_cols]
    y_val = val_pd[TARGET_COL]
    
    print(f"✓ Train: {train_pd.shape}, Val: {val_pd.shape}, Test: {test_pd.shape}")
    
    # ========================================================================
    # HYPERPARAMETER GRID SEARCH
    # ========================================================================
    print(f"\n{'='*80}")
    print("HYPERPARAMETER GRID SEARCH")
    print(f"{'='*80}\n")
    
    # Generate all combinations
    param_names = list(HYPERPARAMETER_GRID.keys())
    param_values = list(HYPERPARAMETER_GRID.values())
    all_combinations = list(product(*param_values))
    
    print(f"Total configurations to test: {len(all_combinations)}")
    print(f"Hyperparameters:")
    for param, values in HYPERPARAMETER_GRID.items():
        print(f"  {param}: {values}")
    print()
    
    # Store all results
    all_results = []
    metrics_master_path = f"{RESULTS_DIR}/metrics_master.csv"
    
    # Loop through all combinations
    for idx, param_combo in enumerate(all_combinations, 1):
        config = dict(zip(param_names, param_combo))
        config_name = f"cw_{config['case_weighting']}_pool_{config['use_adaptive_pool']}_seed_{config['seed_method']}"
        
        print(f"\n{'#'*80}")
        print(f"CONFIGURATION {idx}/{len(all_combinations)}: {config_name}")
        print(f"{'#'*80}")
        
        try:
            # Run global k-center matching
            matching_result = run_global_kcenter_matching(
                train_pd=train_pd,
                target_col=TARGET_COL,
                feature_cols=feature_cols,
                pn_h5_path=PN_H5_PATH,
                matching_ratio=MATCHING_RATIO,
                case_weighting=config['case_weighting'],
                use_adaptive_pool=config['use_adaptive_pool'],
                seed_method=config['seed_method'],
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                COST_COLUMNS=COST_COLUMNS,
            )
            
            # Build undersampled dataset
            undersampled_training_data = build_undersampled_dataset(
                train_pd=train_pd,
                matching_result=matching_result,
                target_col=TARGET_COL,
                matching_ratio=MATCHING_RATIO,
            )
            
            # Save undersampled dataset
            undersample_path = f"{RESULTS_DIR}/undersampled_{config_name}.csv"
            undersampled_training_data.to_csv(undersample_path, index=False)
            print(f"\n✓ Saved undersampled dataset: {undersample_path}")
            
            # Train and evaluate OCT
            balanced_model, balanced_params, metrics = train_and_evaluate_oct(
                undersampled_training_data=undersampled_training_data,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                feature_cols=feature_cols,
                target_col=TARGET_COL,
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                config_name=config_name,
                results_dir=RESULTS_DIR,
            )
            
            # Collect results
            # Handle balanced_params (tuple of depth, minbucket, cp)
            if isinstance(balanced_params, tuple) and len(balanced_params) == 3:
                params_dict = {
                    'best_depth': balanced_params[0],
                    'best_minbucket': balanced_params[1],
                    'best_cp': balanced_params[2],
                }
            else:
                params_dict = {'best_params': str(balanced_params)}
            
            row = {
                'config_name': config_name,
                'case_weighting': config['case_weighting'],
                'use_adaptive_pool': config['use_adaptive_pool'],
                'seed_method': config['seed_method'],
                'matching_ratio': MATCHING_RATIO,
                'n_train_samples': len(undersampled_training_data),
                'n_train_minority': (undersampled_training_data[TARGET_COL] == 1).sum(),
                'n_train_majority': (undersampled_training_data[TARGET_COL] == 0).sum(),
                **params_dict,
            }
            
            # Add metrics (handle both dict and non-dict cases)
            if isinstance(metrics, dict):
                row.update(metrics)
            else:
                row['metrics_error'] = str(metrics)
            
            all_results.append(row)
            
            # Save incrementally
            pd.DataFrame([row]).to_csv(
                metrics_master_path,
                mode='a',
                header=not os.path.exists(metrics_master_path),
                index=False,
            )
            
            print(f"\n✓ Configuration {idx}/{len(all_combinations)} complete")
            if isinstance(metrics, dict):
                auc = metrics.get('auc', 0)
                opt_f1 = metrics.get('optimal_f1', 0)
                sens = metrics.get('balanced_recall_gmean', 0)
                spec = metrics.get('balanced_specificity_gmean', 0)
                print(f"  AUC: {auc:.4f}" if isinstance(auc, (int, float)) else f"  AUC: {auc}")
                print(f"  Optimal F1: {opt_f1:.4f}" if isinstance(opt_f1, (int, float)) else f"  Optimal F1: {opt_f1}")
                print(f"  Sensitivity: {sens:.4f}" if isinstance(sens, (int, float)) else f"  Sensitivity: {sens}")    
                print(f"  Specificity: {spec:.4f}" if isinstance(spec, (int, float)) else f"  Specificity: {spec}")
            
        except Exception as e:
            print(f"\n✗ ERROR in configuration {config_name}:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            
            # Log error
            row = {
                'config_name': config_name,
                'case_weighting': config['case_weighting'],
                'use_adaptive_pool': config['use_adaptive_pool'],
                'seed_method': config['seed_method'],
                'error': str(e),
            }
            all_results.append(row)
            continue
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Results saved to: {metrics_master_path}")
    
    # Show best configurations
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort by AUC (primary metric from evaluate_binary_oct)
        if 'auc' in results_df.columns:
            results_df_sorted = results_df.sort_values('auc', ascending=False)
            print(f"\nTop 5 configurations by AUC:")
            
            # Display relevant metrics
            display_cols = ['config_name']
            for col in ['auc', 'pr_auc', 'optimal_f1', 'balanced_recall_gmean', 'balanced_specificity_gmean']:
                if col in results_df.columns:
                    display_cols.append(col)
            
            print(results_df_sorted[display_cols].head())
            
            # Also show by optimal F1 if available
            if 'balanced_recall_gmean' in results_df.columns:
                print(f"\n{'='*80}")
                print("TOP 10 CONFIGURATIONS BY BALANCED RECALL G-MEAN")
                print(f"{'='*80}\n")
                results_df_sorted_f1 = results_df.sort_values('balanced_recall_gmean', ascending=False)
                print(results_df_sorted_f1[display_cols].head(10).to_string(index=False))
            
        else:
            print(f"\nAll configurations completed. Available columns: {list(results_df.columns)}")
            print(results_df.head())
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
