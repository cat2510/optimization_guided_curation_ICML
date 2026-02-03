#!/usr/bin/env python3
"""
K-Center Matching Hyperparameter Search (Generic Dataset)
==========================================================
This script performs a grid search over k-center matching hyperparameters:
- Case weighting: None, "boundary", "density_inverse"
- Adaptive pool: True/False
- Seed method: "smart", "centroid", "density", "random"

For each configuration, it:
1. Runs global k-center matching to create undersampled training data
2. Trains an OCT model on the undersampled data
3. Evaluates and saves results

This is a generic template that can be adapted to any tabular dataset with
an imbalanced binary target.

Usage:
    python example_kcenter_hyperparameter_search.py
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

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom modules from public package
from public.model_IAI import (
    finetune_oct, evaluate_binary_oct, train_test_split_enrol,
    get_bin_flag_columns, get_cat_columns, get_true_num_columns,
    get_preprocessor_with_impute
)
from public.two_stage_kcenter_match import two_stage_kcenter_then_match
from public.precompute_distances import (
    get_preprocessor, compute_distances_batched, 
    save_distances_hdf5, precompute_leaf_dnn_memmap
)
from sklearn.impute import SimpleImputer

# ============================================================================
# CONFIGURATION - ADAPT THESE TO YOUR DATASET
# ============================================================================

# Data loading
DATA_PATH = "your_data.parquet"  # Update with your data path
# Alternative: DATA_PATH = "your_data.csv"  # For CSV files

# Column names - UPDATE THESE FOR YOUR DATASET
ID_COL = "patient_id"  # Your unique identifier column
target_col = "high_cost"  # Your binary target column (0/1 or False/True)

# Train/test split
TRAIN_TEST_SEED = 123
TEST_SIZE = 0.3
VAL_SIZE = 0.5  # Fraction of test set to use as validation

# Hyperparameter grid for k-center matching
MATCHING_RATIO = 1  # 1:1 matching (can be changed)
HYPERPARAMETER_GRID = {
    'case_weighting': [None, "boundary"],  # Options: None, "boundary", "density_inverse"
    'use_adaptive_pool': [True, False],
    'seed_method': ["smart", "centroid"],  # Options: "smart", "centroid", "density", "random"
}

# OCT hyperparameters for model training
OCT_DEPTHS = [5, 7, 9]
OCT_MINBUCKETS = [50, 100, 150]
OCT_CPS = [0.00001, 0.0001, 0.001]

# Paths for outputs
BASE_OUTPUT_DIR = "./kcenter_hyperparameter_search_results"
RESULTS_DIR = f"{BASE_OUTPUT_DIR}/seed_{TRAIN_TEST_SEED}_matching_ratio_{MATCHING_RATIO}"
DISTANCES_DIR = "./precomputed_distances"
PN_H5_PATH = os.path.join(DISTANCES_DIR, "distances_majority_minority.h5")
DNN_OUT_DIR = os.path.join(DISTANCES_DIR, f"global_dnn_seed_{TRAIN_TEST_SEED}")

# Columns to exclude (update for your dataset)
# Add any columns that should be excluded from features (e.g., leakage columns)
EXCLUDE_COLUMNS = []  # Example: ["future_info", "target_derived_feature"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_data(data_path):
    """
    Load dataset from file.
    
    Supports Parquet (via pandas or PySpark) and CSV formats.
    """
    print(f"Loading data from: {data_path}")
    
    # Check file extension
    if data_path.endswith('.parquet'):
        # Try pandas first (faster for smaller files)
        try:
            df = pd.read_parquet(data_path)
            print(f"✓ Loaded with pandas: {df.shape}")
        except Exception as e:
            print(f"  Pandas failed, trying PySpark: {e}")
            # Fallback to PySpark for large files
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("DataLoad").getOrCreate()
            df_spark = spark.read.format("parquet").load(data_path)
            df = df_spark.toPandas()
            print(f"✓ Loaded with PySpark: {df.shape}")
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        print(f"✓ Loaded CSV: {df.shape}")
    else:
        raise ValueError(f"Unsupported file format. Use .parquet or .csv")
    
    return df


def prepare_features(df, ID_COL, target_col, exclude_cols=None):
    """
    Prepare feature columns by auto-detecting column types.
    
    Returns:
        tuple: (feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Auto-detect column types
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    
    # Exclude ID, target, and user-specified columns
    exclude_all = [ID_COL, target_col] + exclude_cols
    feature_cols = [c for c in df.columns if c not in exclude_all]
    
    print(f"\nFeature column detection:")
    print(f"  Categorical: {len(CAT_COLUMNS)} columns")
    print(f"  Numeric: {len(TRUE_NUM_COLUMNS)} columns")
    print(f"  Binary flags: {len(BIN_FLAG_COLUMNS)} columns")
    print(f"  Total features: {len(feature_cols)} columns")
    
    return feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS


def run_global_kcenter_matching(
    train_pd,
    target_col,
    ID_COL,
    feature_cols,
    CAT_COLUMNS,
    TRUE_NUM_COLUMNS,
    BIN_FLAG_COLUMNS,
    pn_h5_path,
    dnn_out_dir,
    matching_ratio,
    case_weighting,
    use_adaptive_pool,
    seed_method,
    candidate_pool_size=None,
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
        print(f"     Strategy: Keep ALL minority and ALL majority as-is")
        
        resources_after = get_resource_usage()
        sampling_time = 0.0
        
        return {
            'n_cases_total': n_cases,
            'n_cases_matched': 0,
            'n_controls': n_controls,
            'M': 0,
            'selected_control_enrolids': controls[ID_COL].values,
            'match_costs': np.array([]),
            'candidate_majority_enrolids': controls[ID_COL].values,
            'case_to_control_map': {},
            'skipped_reason': 'more_cases_than_controls',
            'sampling_time_seconds': sampling_time,
            'sampling_memory_mb': resources_after['memory_mb'] if PSUTIL_AVAILABLE else None,
        }
    
    # Preprocess features for distance computation
    print(f"\n  Preprocessing features...")
    cases_prep = cases[feature_cols].copy()
    controls_prep = controls[feature_cols].copy()
    train_prep = pd.concat([cases_prep, controls_prep], axis=0, ignore_index=True)
    
    # Create preprocessor (matching OCT preprocessing)
    preprocessor = get_preprocessor(
        X=train_prep,
        cat_cols=CAT_COLUMNS,
        num_cols=TRUE_NUM_COLUMNS,
        binary_cols=BIN_FLAG_COLUMNS,
        verbose=True
    )
    
    # Transform both cases and controls
    X_minority = preprocessor.fit_transform(cases_prep)
    X_majority = preprocessor.transform(controls_prep)
    
    print(f"    Preprocessed shapes: Minority {X_minority.shape}, Majority {X_majority.shape}")
    
    # Precompute or load minority-majority distances
    print(f"\n  Preparing minority-majority distances...")
    os.makedirs(os.path.dirname(pn_h5_path), exist_ok=True)
    
    should_recompute_pn = False
    if os.path.exists(pn_h5_path):
        try:
            import h5py
            with h5py.File(pn_h5_path, 'r') as f:
                existing_maj = set(f['majority_enrolids'][:].astype(int))
                existing_min = set(f['minority_enrolids'][:].astype(int))
                current_maj = set(controls[ID_COL].values.astype(int))
                current_min = set(cases[ID_COL].values.astype(int))
                
                if existing_maj == current_maj and existing_min == current_min:
                    print(f"    ✓ Found existing minority-majority distances: {pn_h5_path}")
                else:
                    print(f"    ⚠️  Existing distances don't match - recomputing...")
                    should_recompute_pn = True
        except Exception as e:
            print(f"    ⚠️  Error checking existing file: {e}")
            should_recompute_pn = True
    else:
        should_recompute_pn = True
    
    if should_recompute_pn:
        print(f"    ⚙️  Computing minority-majority distances...")
        distances_pn = compute_distances_batched(
            X_majority, X_minority, 
            batch_size=1000, 
            dtype=np.float32
        )
        save_distances_hdf5(
            distances_pn,
            controls[ID_COL].values.astype(np.int64),
            cases[ID_COL].values.astype(np.int64),
            pn_h5_path
        )
        print(f"    ✓ Saved minority-majority distances: {pn_h5_path}")
    
    # Precompute or load control-control distances globally
    print(f"\n  Preparing control-control distances (global)...")
    os.makedirs(dnn_out_dir, exist_ok=True)
    
    dnn_matrix_npy = os.path.join(dnn_out_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(dnn_out_dir, "leaf_global_dnn_enrolids.npy")
    
    should_recompute_dnn = False
    if os.path.exists(dnn_matrix_npy) and os.path.exists(dnn_enrolids_npy):
        try:
            existing_dnn_ids = np.load(dnn_enrolids_npy)
            existing_set = set(existing_dnn_ids.astype(int))
            current_set = set(controls[ID_COL].values.astype(int))
            
            if existing_set == current_set:
                print(f"    ✓ Found existing control-control distances:")
                print(f"      {dnn_matrix_npy}")
            else:
                print(f"    ⚠️  Existing DNN files have different IDs - recomputing...")
                should_recompute_dnn = True
        except Exception as e:
            print(f"    ⚠️  Error checking existing DNN files: {e}")
            should_recompute_dnn = True
    else:
        should_recompute_dnn = True
    
    if should_recompute_dnn:
        print(f"    ⚙️  Computing control-control distances (this may take a while)...")
        dnn_matrix_npy, dnn_enrolids_npy = precompute_leaf_dnn_memmap(
            X_majority_leaf=X_majority,
            majority_enrolids_leaf=controls[ID_COL].values.astype(np.int64),
            out_dir=dnn_out_dir,
            leaf_id="global",
            batch_size=750,
        )
        print(f"    ✓ Control-control distances computed and saved")
    
    # Determine candidate pool size
    if not candidate_pool_size:
        M = n_controls // 2
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
    
    resources_before = get_resource_usage()
    sampling_start_time = time.perf_counter()
    
    try:
        out = two_stage_kcenter_then_match(
            leaf_controls_enrolids=controls[ID_COL].values.astype(np.int64),
            leaf_cases_enrolids=cases[ID_COL].values.astype(np.int64),
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


def build_undersampled_dataset(train_pd, matching_result, target_col, ID_COL, matching_ratio):
    """Build undersampled training dataset from matching results."""
    print(f"\n{'='*80}")
    print("BUILDING UNDERSAMPLED TRAINING DATASET")
    print(f"{'='*80}\n")
    
    # Collect all minority samples (keep ALL)
    all_minority = train_pd[train_pd[target_col] == 1].copy()
    print(f"✓ Collected all minority samples: {len(all_minority):,}")
    
    # Get unique majority samples from matching result
    selected_majority_enrolids = matching_result['selected_control_enrolids']
    if isinstance(selected_majority_enrolids, np.ndarray):
        unique_majority_enrolids = list(set(selected_majority_enrolids))
    else:
        unique_majority_enrolids = list(set(selected_majority_enrolids))
    
    selected_majority = train_pd[
        (train_pd[target_col] == 0) & 
        (train_pd[ID_COL].isin(unique_majority_enrolids))
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
    BIN_FLAG_COLUMNS,
    config_name,
    results_dir,
):
    """Train OCT model and evaluate on test set."""
    print(f"\n{'='*80}")
    print(f"TRAINING OCT MODEL: {config_name}")
    print(f"{'='*80}\n")
    
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
        binary_cols=BIN_FLAG_COLUMNS,
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
        results_dir=results_dir, save_suffix=config_name,
        X_val_df=X_val, y_val=y_val
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
        print(f"   Best MCC: {metrics.get('best_mcc', 'N/A'):.4f}" if isinstance(metrics.get('best_mcc'), (int, float)) else f"   Best MCC: {metrics.get('best_mcc', 'N/A')}")
    
    # Add timing and resource info to metrics
    if isinstance(metrics, dict):
        metrics['oct_training_time_seconds'] = oct_time
        metrics['oct_training_memory_mb'] = resources_after['memory_mb'] if PSUTIL_AVAILABLE else None
    
    return balanced_model, balanced_params, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print("K-CENTER HYPERPARAMETER SEARCH (GENERIC DATASET)")
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
    
    df = load_data(DATA_PATH)
    print(f"✓ Loaded data: {df.shape}")
    print(f"  ID column: {ID_COL}")
    print(f"  Target column: {target_col}")
    
    # Verify required columns exist
    if ID_COL not in df.columns:
        raise ValueError(f"ID column '{ID_COL}' not found in dataset. Available columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    # Check target is binary
    unique_targets = df[target_col].unique()
    if len(unique_targets) > 2:
        raise ValueError(f"Target column '{target_col}' is not binary. Found values: {unique_targets}")
    
    # Prepare features
    feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS = prepare_features(
        df, ID_COL, target_col, EXCLUDE_COLUMNS
    )
    
    # Train/test/val split
    print(f"\n{'='*80}")
    print("SPLITTING DATA")
    print(f"{'='*80}\n")
    
    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=TEST_SIZE, 
        verbose=True, random_state=TRAIN_TEST_SEED
    )
    
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=VAL_SIZE,
        verbose=True, random_state=TRAIN_TEST_SEED
    )
    
    X_test = test_pd[feature_cols]
    y_test = test_pd[target_col]
    X_val = val_pd[feature_cols]
    y_val = val_pd[target_col]
    
    print(f"✓ Train: {train_pd.shape}, Val: {val_pd.shape}, Test: {test_pd.shape}")
    print(f"  Train target distribution:\n{train_pd[target_col].value_counts()}")
    
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
    metrics_master_path = os.path.join(RESULTS_DIR, "metrics_master.csv")
    
    # Loop through all combinations
    for idx, param_combo in enumerate(all_combinations, 1):
        config = dict(zip(param_names, param_combo))
        config_name = f"cw_{config['case_weighting']}_pool_{config['use_adaptive_pool']}_seed_{config['seed_method']}"
        
        print(f"\n{'#'*80}")
        print(f"CONFIGURATION {idx}/{len(all_combinations)}: {config_name}")
        print(f"{'#'*80}")
        
        config_start_time = time.perf_counter()
        config_resources_start = get_resource_usage()
        
        try:
            # Run global k-center matching
            matching_result = run_global_kcenter_matching(
                train_pd=train_pd,
                target_col=target_col,
                ID_COL=ID_COL,
                feature_cols=feature_cols,
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                BIN_FLAG_COLUMNS=BIN_FLAG_COLUMNS,
                pn_h5_path=PN_H5_PATH,
                dnn_out_dir=DNN_OUT_DIR,
                matching_ratio=MATCHING_RATIO,
                case_weighting=config['case_weighting'],
                use_adaptive_pool=config['use_adaptive_pool'],
                seed_method=config['seed_method'],
            )
            
            # Build undersampled dataset
            undersampled_training_data = build_undersampled_dataset(
                train_pd=train_pd,
                matching_result=matching_result,
                target_col=target_col,
                ID_COL=ID_COL,
                matching_ratio=MATCHING_RATIO,
            )
            
            # Save undersampled dataset
            undersample_path = os.path.join(RESULTS_DIR, f"undersampled_{config_name}.csv")
            undersampled_training_data.to_csv(undersample_path, index=False)
            print(f"\n✓ Saved undersampled dataset: {undersample_path}")
            
            # Train and evaluate OCT
            balanced_model, balanced_params, metrics = train_and_evaluate_oct(
                undersampled_training_data=undersampled_training_data,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                feature_cols=feature_cols,
                target_col=target_col,
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                BIN_FLAG_COLUMNS=BIN_FLAG_COLUMNS,
                config_name=config_name,
                results_dir=RESULTS_DIR,
            )
            
            config_end_time = time.perf_counter()
            config_total_time = config_end_time - config_start_time
            config_resources_end = get_resource_usage()
            
            # Collect results
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
                'n_train_minority': (undersampled_training_data[target_col] == 1).sum(),
                'n_train_majority': (undersampled_training_data[target_col] == 0).sum(),
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
            
            # Add metrics
            if isinstance(metrics, dict):
                metrics_clean = {k: v for k, v in metrics.items() 
                               if k not in ['oct_training_time_seconds', 'oct_training_memory_mb']}
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
            if isinstance(metrics, dict):
                print(f"  PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"  PR-AUC: {metrics.get('pr_auc', 'N/A')}")
                print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"  AUC: {metrics.get('auc', 'N/A')}")
            
        except Exception as e:
            config_end_time = time.perf_counter()
            config_total_time = config_end_time - config_start_time
            
            print(f"\n✗ ERROR in configuration {config_name}:")
            print(f"  {e}")
            print(f"  Time before error: {format_time(config_total_time)}s")
            import traceback
            traceback.print_exc()
            
            row = {
                'config_name': config_name,
                'case_weighting': config['case_weighting'],
                'use_adaptive_pool': config['use_adaptive_pool'],
                'seed_method': config['seed_method'],
                'error': str(e),
                'total_config_time_seconds': config_total_time,
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
        
        if 'pr_auc' in results_df.columns:
            results_df_sorted = results_df.sort_values('pr_auc', ascending=False)
            print(f"\nTop 5 configurations by PR-AUC:")
            
            display_cols = ['config_name']
            for col in ['auc', 'pr_auc', 'best_mcc', 'balanced_recall_gmean', 'balanced_specificity_gmean']:
                if col in results_df.columns:
                    display_cols.append(col)
            
            print(results_df_sorted[display_cols].head())
        else:
            print(f"\nAll configurations completed. Available columns: {list(results_df.columns)}")
            print(results_df.head())
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
