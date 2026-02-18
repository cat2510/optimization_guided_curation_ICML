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
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Resource tracking will be limited.")

# Import custom modules
import public.model_IAI
import public.two_stage_kcenter_match
from public.two_stage_kcenter_match import two_stage_kcenter_then_match
from public.precompute_distances import precompute_leaf_dnn_memmap
from public.model_IAI import get_preprocessor_with_impute,train_test_split_enrol, get_bin_flag_columns, get_cat_columns, get_true_num_columns
from sklearn.impute import SimpleImputer

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_TEST_SEED = 123
# Hyperparameter grid
MATCHING_RATIO = 1  # Fixed for now, can be added to grid if needed
HYPERPARAMETER_GRID = {
    'case_weighting': ["boundary",None], #, "boundary",None
    'use_adaptive_pool': [True,False],
    'seed_method': ["density","smart","centroid","random"],#"smart", "centroid", "density", 
}

# OCT hyperparameters for model training
OCT_DEPTHS = [7, 9]
OCT_MINBUCKETS = [50, 100, 120, 150]
OCT_CPS = [0.00001, 0.0001, 0.001, 0.01]

# Paths
PN_H5_PATH = f"./precomputed_distances/distances_majority_minority.h5" #_seed_{TRAIN_TEST_SEED}
RESULTS_DIR = f"./kcenter_hyperparameter_search_results_global_seed_{TRAIN_TEST_SEED}_matching_ratio_{MATCHING_RATIO}"
DNN_OUT_DIR = f"./precomputed_distances/global_dnn_seed_{TRAIN_TEST_SEED}"

# Target column
TARGET_COL = "highcost_gt_200000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_resource_usage():
    """
    Get current CPU and memory usage.
    
    Returns:
        dict: Dictionary with 'cpu_percent', 'memory_mb', 'memory_percent'
    """
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
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


def format_time(seconds):
    """
    Format time in a readable way.
    For times < 0.1s, report with more precision.
    For times >= 0.1s, report with 2 decimal places.
    """
    if seconds < 0.1:
        return f"{seconds:.4f}"
    elif seconds < 1.0:
        return f"{seconds:.3f}"
    else:
        return f"{seconds:.2f}"


def run_global_kcenter_matching(
    train_pd,
    target_col,
    feature_cols,
    pn_h5_path,
    matching_ratio,
    case_weighting,
    use_adaptive_pool,
    seed_method,
    candidate_pool_size=None,
    CAT_COLUMNS=None,
    TRUE_NUM_COLUMNS=None,
    COST_COLUMNS=None,
    dnn_out_dir=None,  # Optional: dataset-specific DNN output directory
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
        
        # Track timing even for skipped case
        resources_after = get_resource_usage()
        sampling_time = 0.0  # No actual matching performed
        
        return {
            'n_cases_total': n_cases,
            'n_cases_matched': 0,
            'n_controls': n_controls,
            'M': 0,
            'selected_control_enrolids': controls["ENROLID"].to_numpy(),
            'match_costs': np.array([]),
            'candidate_majority_enrolids': controls["ENROLID"].to_numpy(),
            'case_to_control_map': {},
            'skipped_reason': 'more_cases_than_controls',
            'sampling_time_seconds': sampling_time,
            'sampling_memory_mb': resources_after['memory_mb'] if PSUTIL_AVAILABLE else None,
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
    
    preprocessor_controls = get_preprocessor_with_impute(
        X_train=controls_preprocessed,
        categorical_cols=categorical_cols,
        numeric_cols=num_feats,
        binary_cols=bin_feats,
        verbose=False
    )
    X_majority = preprocessor_controls.fit_transform(controls_preprocessed)
    
    print(f"\n  Preprocessing:")
    print(f"    Features: {len(feature_cols_controls)}")
    print(f"    Preprocessed shape: {X_majority.shape}")
    
    # Precompute or load control-control distances globally
    print(f"\n  Preparing control-control distances (global)...")
    majority_enrolids = controls["ENROLID"].to_numpy()

    # Use dataset-specific DNN directory if provided, otherwise use global default
    if dnn_out_dir is None:
        dnn_out_dir = DNN_OUT_DIR
    
    # Paths used by precompute_leaf_dnn_memmap for the global run
    dnn_matrix_npy = os.path.join(dnn_out_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(dnn_out_dir, "leaf_global_dnn_enrolids.npy")

    # Check if existing files match current ENROLIDs
    should_recompute = False
    if os.path.exists(dnn_matrix_npy) and os.path.exists(dnn_enrolids_npy):
        try:
            existing_dnn_ids = np.load(dnn_enrolids_npy)
            existing_set = set(existing_dnn_ids.astype(int))
            current_set = set(majority_enrolids.astype(int))
            
            if existing_set != current_set:
                print(f"    ⚠️  Existing DNN files have different ENROLIDs!")
                print(f"       Existing: {len(existing_set):,} ENROLIDs")
                print(f"       Current: {len(current_set):,} ENROLIDs")
                print(f"       Mismatch: {len(existing_set - current_set):,} extra, {len(current_set - existing_set):,} missing")
                print(f"       Forcing recomputation...")
                should_recompute = True
            else:
                print(f"    ✓ Found existing global d_nn files with matching ENROLIDs:")
                print(f"      d_nn matrix: {dnn_matrix_npy}")
                print(f"      d_nn enrolids: {dnn_enrolids_npy}")
        except Exception as e:
            print(f"    ⚠️  Error checking existing DNN files: {e}")
            print(f"       Forcing recomputation...")
            should_recompute = True
    else:
        should_recompute = True
    
    if should_recompute:
        print(f"    ⚙️  Computing global control-control distances (this is done once)...")
        dnn_matrix_npy, dnn_enrolids_npy = precompute_leaf_dnn_memmap(
            X_majority_leaf=X_majority,
            majority_enrolids_leaf=majority_enrolids,
            out_dir=dnn_out_dir,
            leaf_id="global",  # Use "global" as leaf_id
            batch_size=750,
        )
        print(f"    ✓ Control-control distances computed and saved")
    
    # Normal case: n_controls >= n_cases
    if not candidate_pool_size:
        M = n_controls//2
    else:
        M = candidate_pool_size
    print(f"\n  K-Center Configuration:")
    print(f"    M (candidate pool size): {M:,} / {n_controls:,} ({M/n_controls*100:.1f}%)")
    print(f"    Cases to match: {n_cases:,}")
    print(f"    Seed method: {seed_method}")
    print(f"    Adaptive pool: {use_adaptive_pool}")
    print(f"    Case weighting: {case_weighting}")
    
    # Run two-stage k-center matching
    print(f"\n  Running two-stage k-center matching (1:{matching_ratio})...")
    
    # Track resources before matching
    resources_before = get_resource_usage()
    sampling_start_time = time.perf_counter()
    
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
        )
        
        sampling_end_time = time.perf_counter()
        sampling_time = sampling_end_time - sampling_start_time
        resources_after = get_resource_usage()
        
        # Extract results
        selected_control_enrolids = out["selected_control_enrolids"]
        all_match_costs = out["match_costs"]
        candidate_majority_enrolids = out["candidate_majority_enrolids"]
        case_to_control_map = out["case_to_control_map"]
        
        print(f"    ✓ Matching complete!")
        print(f"    Cases matched: {n_cases:,}")
        print(f"    Total selected controls: {len(selected_control_enrolids):,} (unique: {len(set(selected_control_enrolids)):,})")
        print(f"    Mean matching cost: {all_match_costs.mean():.4f}")
        print(f"    Sampling time: {format_time(sampling_time)}s")
        if PSUTIL_AVAILABLE:
            memory_delta = resources_after['memory_mb'] - resources_before['memory_mb']
            print(f"    Memory used: {resources_after['memory_mb']:.1f} MB (Δ {memory_delta:+.1f} MB)")
        
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
            'skipped_reason': None,
            'sampling_time_seconds': sampling_time,
            'sampling_memory_mb': resources_after['memory_mb'] if PSUTIL_AVAILABLE else None,
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
    
    from public.model_IAI import finetune_oct, evaluate_binary_oct
    
    # Track resources before training
    resources_before = get_resource_usage()
    oct_start_time = time.perf_counter()
    
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
    
    oct_end_time = time.perf_counter()
    oct_time = oct_end_time - oct_start_time
    resources_after = get_resource_usage()
    
    # Evaluate (use validation set for threshold tuning)
    metrics = evaluate_binary_oct(
        balanced_model, X_test, y_test, preprocessor, feature_names,
        results_dir=results_dir, save_suffix="ckd_best_oct_curated",
        X_val_df=X_val, y_val=y_val  # Use validation set for threshold tuning
    )
    
    print(f"\n✓ Model training complete:")
    print(f"   Best params: {balanced_params}")
    print(f"   OCT training time: {format_time(oct_time)}s")
    if PSUTIL_AVAILABLE:
        memory_delta = resources_after['memory_mb'] - resources_before['memory_mb']
        print(f"   Memory used: {resources_after['memory_mb']:.1f} MB (Δ {memory_delta:+.1f} MB)")
    
    if isinstance(metrics, dict):
        print(f"   AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"   AUC: {metrics.get('auc', 'N/A')}")
        print(f"   PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"   PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        print(f"   Optimal F1: {metrics.get('optimal_f1', 'N/A'):.4f}" if isinstance(metrics.get('optimal_f1'), (int, float)) else f"   Optimal F1: {metrics.get('optimal_f1', 'N/A')}")
        print(f"   Best MCC: {metrics.get('best_mcc', 'N/A'):.4f}" if isinstance(metrics.get('best_mcc'), (int, float)) else f"   Best MCC: {metrics.get('best_mcc', 'N/A')}")
        print(f"   Sensitivity (G-mean threshold): {metrics.get('balanced_recall_gmean', 'N/A'):.4f}" if isinstance(metrics.get('balanced_recall_gmean'), (int, float)) else f"   Sensitivity: {metrics.get('sensitivity_f1', 'N/A')}")
        print(f"   Specificity (G-mean threshold): {metrics.get('balanced_specificity_gmean', 'N/A'):.4f}" if isinstance(metrics.get('balanced_specificity_gmean'), (int, float)) else f"   Specificity: {metrics.get('specificity_f1', 'N/A')}")
    else:
        print(f"   WARNING: metrics is not a dict, type={type(metrics)}")
    
    # Add timing and resource info to metrics
    if isinstance(metrics, dict):
        metrics['oct_training_time_seconds'] = oct_time
        metrics['oct_training_memory_mb'] = resources_after['memory_mb'] if PSUTIL_AVAILABLE else None
        metrics['oct_training_cpu_percent'] = resources_after['cpu_percent'] if PSUTIL_AVAILABLE else None
    
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
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df_og) + [
        'lab_monitoring_adherent', 'nephrology_consult_adherent', 'early_nephrology_referral'
    ]
    STAGE_COLUMNS = [col for col in df_og.columns if "stage" in col.lower()]
    CAT_COLUMNS = df_og.select_dtypes(include=["object", "category"]).columns.tolist()
    TRUE_NUM_COLUMNS = get_true_num_columns(df_og, CAT_COLUMNS) + [
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
    train_ids, test_ids, train_pd, test_pd =train_test_split_enrol(
        df_og, target_col="cost_stratum_2018", test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=TARGET_COL, test_size=0.5, verbose=False,random_state=TRAIN_TEST_SEED
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
        
        # Track total time for this configuration
        config_start_time = time.perf_counter()
        config_resources_start = get_resource_usage()
        
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
            
            config_end_time = time.perf_counter()
            config_total_time = config_end_time - config_start_time
            config_resources_end = get_resource_usage()
            
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
            
            # Add timing metrics
            row['sampling_time_seconds'] = matching_result.get('sampling_time_seconds', None)
            row['oct_training_time_seconds'] = metrics.get('oct_training_time_seconds', None) if isinstance(metrics, dict) else None
            row['total_config_time_seconds'] = config_total_time
            
            # Add resource metrics
            if PSUTIL_AVAILABLE:
                row['sampling_memory_mb'] = matching_result.get('sampling_memory_mb', None)
                row['oct_training_memory_mb'] = metrics.get('oct_training_memory_mb', None) if isinstance(metrics, dict) else None
                row['config_peak_memory_mb'] = config_resources_end['memory_mb']
                row['config_peak_cpu_percent'] = config_resources_end['cpu_percent']
                row['config_memory_delta_mb'] = config_resources_end['memory_mb'] - config_resources_start['memory_mb']
            else:
                row['sampling_memory_mb'] = None
                row['oct_training_memory_mb'] = None
                row['config_peak_memory_mb'] = None
                row['config_peak_cpu_percent'] = None
                row['config_memory_delta_mb'] = None
            
            # Add metrics (handle both dict and non-dict cases)
            if isinstance(metrics, dict):
                # Remove timing/resource keys that we've already added separately
                metrics_clean = {k: v for k, v in metrics.items() 
                               if k not in ['oct_training_time_seconds', 'oct_training_memory_mb', 'oct_training_cpu_percent']}
                row.update(metrics_clean)
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
            print(f"  Total time: {format_time(config_total_time)}s")
            print(f"    - Sampling: {format_time(row.get('sampling_time_seconds', 0))}s")
            print(f"    - OCT training: {format_time(row.get('oct_training_time_seconds', 0))}s")
            if PSUTIL_AVAILABLE and row.get('config_peak_memory_mb'):
                print(f"  Peak memory: {row['config_peak_memory_mb']:.1f} MB")
            
            if isinstance(metrics, dict):
                auc = metrics.get('auc', 0)
                pr_auc = metrics.get('pr_auc', 0)
                opt_f1 = metrics.get('optimal_f1', 0)
                best_mcc = metrics.get('best_mcc', 0)
                sens = metrics.get('balanced_recall_gmean', 0)
                spec = metrics.get('balanced_specificity_gmean', 0)
                print(f"  AUC: {auc:.4f}" if isinstance(auc, (int, float)) else f"  AUC: {auc}")
                print(f"  PR-AUC: {pr_auc:.4f}" if isinstance(pr_auc, (int, float)) else f"  PR-AUC: {pr_auc}")
                print(f"  Optimal F1: {opt_f1:.4f}" if isinstance(opt_f1, (int, float)) else f"  Optimal F1: {opt_f1}")
                print(f"  Best MCC: {best_mcc:.4f}" if isinstance(best_mcc, (int, float)) else f"  Best MCC: {best_mcc}")
                print(f"  Sensitivity: {sens:.4f}" if isinstance(sens, (int, float)) else f"  Sensitivity: {sens}")    
                print(f"  Specificity: {spec:.4f}" if isinstance(spec, (int, float)) else f"  Specificity: {spec}")
            
        except Exception as e:
            config_end_time = time.perf_counter()
            config_total_time = config_end_time - config_start_time
            config_resources_end = get_resource_usage()
            
            print(f"\n✗ ERROR in configuration {config_name}:")
            print(f"  {e}")
            print(f"  Time before error: {format_time(config_total_time)}s")
            import traceback
            traceback.print_exc()
            
            # Log error with timing info
            row = {
                'config_name': config_name,
                'case_weighting': config['case_weighting'],
                'use_adaptive_pool': config['use_adaptive_pool'],
                'seed_method': config['seed_method'],
                'error': str(e),
                'total_config_time_seconds': config_total_time,
            }
            if PSUTIL_AVAILABLE:
                row['config_peak_memory_mb'] = config_resources_end['memory_mb']
                row['config_peak_cpu_percent'] = config_resources_end['cpu_percent']
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
        
        # Calculate and print average times
        sampling_times = results_df['sampling_time_seconds'].dropna()
        oct_training_times = results_df['oct_training_time_seconds'].dropna()
        
        if len(sampling_times) > 0:
            avg_sampling_time = sampling_times.mean()
            print(f"\n⏱️  Average Runtime Across All Configurations:")
            print(f"   Average sampling time: {format_time(avg_sampling_time)}s")
            if len(oct_training_times) > 0:
                avg_oct_time = oct_training_times.mean()
                print(f"   Average OCT training time: {format_time(avg_oct_time)}s")
                total_avg_time = avg_sampling_time + avg_oct_time
                print(f"   Average total time per configuration: {format_time(total_avg_time)}s")
        
        # Sort by AUC (primary metric from evaluate_binary_oct)
        if 'pr_auc' in results_df.columns:
            results_df_sorted = results_df.sort_values('pr_auc', ascending=False)
            print(f"\nTop 5 configurations by PR-AUC:")
            
            # Display relevant metrics
            display_cols = ['config_name']
            for col in ['auc', 'pr_auc', 'optimal_f1', 'best_mcc', 'balanced_recall_gmean', 'balanced_specificity_gmean']:
                if col in results_df.columns:
                    display_cols.append(col)
            
            print(results_df_sorted[display_cols].head())
            
        else:
            print(f"\nAll configurations completed. Available columns: {list(results_df.columns)}")
            print(results_df.head())
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
