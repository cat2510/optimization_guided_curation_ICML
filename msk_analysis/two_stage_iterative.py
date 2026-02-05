# Load MSK data (if not already loaded)
# Uncomment if needed:
import sys,os
import pandas as pd
import numpy as np
# Add parent directory to path to import modules from one level up
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
import importlib
# Import required modules
import h5py
from sklearn.impute import SimpleImputer
import time
import public.precompute_distances
importlib.reload(public.precompute_distances)
from public.precompute_distances import (
    get_preprocessor, compute_distances_batched, 
    save_distances_hdf5, precompute_leaf_dnn_memmap
)

try:
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import two_stage_kcenter_then_match
except ImportError:
    parent_projects_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if parent_projects_dir not in sys.path:
        sys.path.insert(0, parent_projects_dir)
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import two_stage_kcenter_then_match

from public.model_IAI import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataLoad").getOrCreate()
# Load the MSK dataset with enhanced cost features
df_msk_spark = spark.read.format("parquet").load("msk_2017_18_full.parquet")
df_og = df_msk_spark.toPandas()


TRAIN_TEST_SEED = 123

RESULTS_DIR = "./two_stage_kcenter_results_msk_static_pool" #undersampled dataset directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration for k-center matching
MATCHING_RATIO = 1  # 1:1 matching (can be changed)
CASE_WEIGHTING = None  # Options: None, "boundary"
USE_ADAPTIVE_POOL = False  # Options: True, False
SEED_METHOD = "random"  # Options: "smart", "centroid", "density", "random"

# MSK-specific feature column definitions
# Binary flag columns: comorbidity flags, MSK category flags, medication flags
BIN_FLAG_COLUMNS = get_bin_flag_columns(df_og)

# MSK doesn't have stage columns like CKD, but we can identify categorical cost pattern/stability columns
STAGE_COLUMNS = []  # MSK doesn't use stage columns

# Categorical columns
CAT_COLUMNS = get_cat_columns(df_og)
# True numeric columns (excluding binary flags and categorical)
TRUE_NUM_COLUMNS = get_true_num_columns(df_og, CAT_COLUMNS,BIN_FLAG_COLUMNS)


# Cost columns (2017 only - exclude 2018 to prevent leakage)
COST_COLUMNS = [
    col for col in df_og.columns 
    if ("cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower() or 
        "decreasing" in col.lower() or "skewness" in col.lower() or "kurtosis" in col.lower() or
        "cv" in col.lower() or "range" in col.lower())
    and "2018" not in col  # Exclude 2018 columns to prevent leakage
]
# Utilization columns (if any)
UTILIZATION_COLUMNS = [col for col in df_og.columns if "claims" in col.lower() and "2018" not in col]

leftover_cols = [
    c for c in df_og.columns 
    if c not in CAT_COLUMNS and c not in TRUE_NUM_COLUMNS and c not in STAGE_COLUMNS and c not in BIN_FLAG_COLUMNS 
    and c != "ENROLID"
]

if len(leftover_cols) > 0:
    print(f"Number of leftover columns: {len(leftover_cols),leftover_cols[:20]}")


# Create cost stratum from 2018 target (if available)
# Check what target columns are available
target_candidates = [col for col in df_og.columns if "2018" in col and ("top" in col.lower() or "pct" in col.lower() or "cost" in col.lower())]

# Use top_2_pct_cost_2018 as target if available, otherwise create from annual_cost_2018_deflated
if "top_2_pct_cost_2018" in df_og.columns:
    target_col = "top_2_pct_cost_2018"
    print(f"Using {target_col} as target column")
elif "annual_cost_2018_deflated" in df_og.columns:
    # Create binary target from annual_cost_2018_deflated
    # Use 98th percentile as threshold (top 2%)
    threshold = df_og["annual_cost_2018_deflated"].quantile(0.98)
    df_og["top_2_pct_cost_2018"] = (df_og["annual_cost_2018_deflated"] >= threshold).astype(int)
    target_col = "top_2_pct_cost_2018"
    print(f"Created {target_col} using threshold ${threshold:,.2f}")
else:
    raise ValueError("No 2018 target column found. Need either 'top_2_pct_cost_2018' or 'annual_cost_2018_deflated'")

# Exclude all columns containing "2018" from features (to prevent leakage)
# Also exclude ENROLID and target column
exclude_cols = ["ENROLID", target_col] + [col for col in df_og.columns 
if "2018" in col]
feature_cols = [c for c in df_og.columns if c not in exclude_cols]

print(f"\nTotal columns in dataset: {len(df_og.columns)}")
print(f"Feature columns: {len(feature_cols)}")


# Split data into train/test/val (same as multiobjective_bilevel.ipynb)
# Use the target_col defined in cell 0
train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
    df_og,
    target_col=target_col,  # Use target_col from cell 0 (e.g., "top_2_pct_cost_2018")
    test_size=0.3,
    verbose=False,
    random_state=TRAIN_TEST_SEED
)
print(f"Train shape: {train_pd.shape}, Test shape: {test_pd.shape}")
print("Feature cols:", len(feature_cols))

val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
    test_pd, 
    target_col=target_col,
    test_size=0.5,
    verbose=False, 
    random_state=TRAIN_TEST_SEED
)
X_test = test_pd[feature_cols]
y_test = test_pd[target_col]
X_val = val_pd[feature_cols]
y_val = val_pd[target_col]

print(f"Train: {train_pd.shape}, Val: {val_pd.shape}, Test: {test_pd.shape}")

# ============================================================================
# CONFIGURATION: Different precomputed distance directories to test
# ============================================================================

# List of precomputed distance directories to iterate over
# Each directory should contain:
#   - distances_majority_minority.h5
#   - global_dnn_seed_{TRAIN_TEST_SEED}/leaf_global_dnn_matrix.npy
#   - global_dnn_seed_{TRAIN_TEST_SEED}/leaf_global_dnn_enrolids.npy
DISTANCE_DIRS = [
    "./precomputed_distances_msk_medical_only",
    "./precomputed_distances_msk_cost_only",
    "./precomputed_distances_msk_less_cost",
    "./precomputed_distances_msk_with_cost_features",
    "./precomputed_distances_msk_with_target_no_cost",
]

# Filter to only directories that actually exist
DISTANCE_DIRS = [d for d in DISTANCE_DIRS if os.path.exists(d)]

print(f"\n{'='*80}")
print(f"ITERATING OVER {len(DISTANCE_DIRS)} PRECOMPUTED DISTANCE DIRECTORIES")
print(f"{'='*80}")
print(f"Directories to test:")
for i, d in enumerate(DISTANCE_DIRS, 1):
    print(f"  {i}. {d}")
print()

# Separate minority (cases) and majority (controls) - same for all iterations
cases = train_pd[train_pd[target_col] == 1].copy()
controls = train_pd[train_pd[target_col] == 0].copy()

print(f"\nDataset split (same for all iterations):")
print(f"  Cases (minority): {len(cases):,}")
print(f"  Controls (majority): {len(controls):,}")
print(f"  Ratio: {len(controls)/len(cases):.2f}:1")

# Determine candidate pool size M
n_cases = len(cases)
n_controls = len(controls)
M = n_controls // 2  # Use half of controls as candidate pool

# Store results for all configurations
all_results = []

# ============================================================================
# ITERATE OVER DIFFERENT DISTANCE DIRECTORIES
# ============================================================================

for dist_idx, DISTANCES_DIR in enumerate(DISTANCE_DIRS, 1):
    print(f"\n{'#'*80}")
    print(f"ITERATION {dist_idx}/{len(DISTANCE_DIRS)}: {os.path.basename(DISTANCES_DIR)}")
    print(f"{'#'*80}\n")
    
    # Extract feature subset name from directory
    feature_subset_name = os.path.basename(DISTANCES_DIR).replace("precomputed_distances_msk", "").replace("_", " ").strip()
    if not feature_subset_name:
        feature_subset_name = "medical_only"
    
    iteration_start_time = time.perf_counter()
    
    # Paths for this iteration
    PN_H5_PATH = os.path.join(DISTANCES_DIR, "distances_majority_minority.h5")
    DNN_OUT_DIR = os.path.join(DISTANCES_DIR, f"global_dnn_seed_{TRAIN_TEST_SEED}")
    dnn_matrix_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_enrolids.npy")
    
    # Check if required files exist
    if not os.path.exists(PN_H5_PATH):
        print(f"  ⚠️  Skipping: {PN_H5_PATH} not found")
        continue
    
    if not os.path.exists(dnn_matrix_npy) or not os.path.exists(dnn_enrolids_npy):
        print(f"  ⚠️  Skipping: DNN files not found in {DNN_OUT_DIR}")
        continue
    
    print(f"  ✓ Found precomputed distances:")
    print(f"    Minority-Majority: {PN_H5_PATH}")
    print(f"    Majority-Majority: {dnn_matrix_npy}")
    
    try:
        # ====================================================================
        # STEP 1: K-CENTER UNDERSAMPLING
        # ====================================================================
        print(f"\n  {'='*76}")
        print(f"  K-CENTER UNDERSAMPLING")
        print(f"  {'='*76}")
        print(f"  Configuration:")
        print(f"    Matching ratio: 1:{MATCHING_RATIO}")
        print(f"    Case weighting: {CASE_WEIGHTING}")
        print(f"    Adaptive pool: {USE_ADAPTIVE_POOL}")
        print(f"    Seed method: {SEED_METHOD}")
        print(f"    Candidate pool size (M): {M:,}")
        
        matching_start_time = time.perf_counter()
        
        matching_result = two_stage_kcenter_then_match(
            leaf_controls_enrolids=controls['ENROLID'].values.astype(np.int64),
            leaf_cases_enrolids=cases['ENROLID'].values.astype(np.int64),
            leaf_nn_matrix_npy=dnn_matrix_npy,
            leaf_nn_enrolids_npy=dnn_enrolids_npy,
            pn_h5_path=PN_H5_PATH,
            M=M,
            use_adaptive_pool=USE_ADAPTIVE_POOL,
            tau=None,  # Auto-compute from 95th percentile
            plateau_eps=0.01,
            force_nearest_per_case=True,
            force_topm=1,
            assignment_topk_start=None,  # Exact matching
            seed_method=SEED_METHOD,
            matching_ratio=MATCHING_RATIO,
            X_majority_leaf=None,  # Not needed when using precomputed distances
            case_weighting=CASE_WEIGHTING,
        )
        
        matching_end_time = time.perf_counter()
        matching_time = matching_end_time - matching_start_time
        
        # Extract results
        selected_control_enrolids = matching_result["selected_control_enrolids"]
        all_match_costs = matching_result["match_costs"]
        
        print(f"\n  ✓ Matching complete!")
        print(f"    Cases matched: {n_cases:,}")
        print(f"    Total selected controls: {len(selected_control_enrolids):,}")
        print(f"    Unique selected controls: {len(set(selected_control_enrolids)):,}")
        print(f"    Mean matching cost: {all_match_costs.mean():.4f}")
        print(f"    Matching time: {matching_time:.2f}s")
        
        # Build undersampled dataset
        all_minority = train_pd[train_pd[target_col] == 1].copy()
        unique_majority_enrolids = list(set(selected_control_enrolids))
        selected_majority = train_pd[
            (train_pd[target_col] == 0) & 
            (train_pd['ENROLID'].isin(unique_majority_enrolids))
        ].copy()
        
        undersampled_training_data = pd.concat([all_minority, selected_majority], axis=0, ignore_index=True)
        
        print(f"\n  ✓ Undersampled dataset created:")
        print(f"     Total samples: {len(undersampled_training_data):,}")
        print(f"     Class distribution:")
        print(undersampled_training_data[target_col].value_counts().sort_index())
        
        # ====================================================================
        # STEP 2: TRAIN AND EVALUATE OCT
        # ====================================================================
        print(f"\n  {'='*76}")
        print(f"  TRAINING OCT MODEL")
        print(f"  {'='*76}")
        
        training_start_time = time.perf_counter()
        
        balanced_model, balanced_params, _, preprocessor, feature_names = finetune_oct(
            X_train=undersampled_training_data[[col for col in feature_cols]],
            y_train=undersampled_training_data[target_col],
            X_val=X_val,
            y_val=y_val,
            categorical_cols=CAT_COLUMNS,
            numeric_cols=TRUE_NUM_COLUMNS,
            binary_cols=BIN_FLAG_COLUMNS,
            depths=[5, 7],
            minbuckets=[50, 100, 150],
            cps=[0.0001, 0.001]
        )
        
        training_end_time = time.perf_counter()
        training_time = training_end_time - training_start_time
        
        # Evaluate
        evaluation_start_time = time.perf_counter()
        
        # Create results directory for this feature subset
        results_dir = f"msk_balanced_{feature_subset_name.replace(' ', '_')}"
        os.makedirs(results_dir, exist_ok=True)
        
        metrics = evaluate_binary_oct(
            balanced_model, X_test, y_test, preprocessor, feature_names, 
            X_val_df=X_val, y_val=y_val,
            results_dir=results_dir, 
            save_suffix=f"{balanced_params[0]}_{balanced_params[1]}_{balanced_params[2]}"
        )
        
        evaluation_end_time = time.perf_counter()
        evaluation_time = evaluation_end_time - evaluation_start_time
        total_time = time.perf_counter() - iteration_start_time
        
        print(f"\n  ✓ Model training and evaluation complete:")
        print(f"     Best params: depth={balanced_params[0]}, minbucket={balanced_params[1]}, cp={balanced_params[2]}")
        print(f"     Training time: {training_time:.2f}s")
        print(f"     Evaluation time: {evaluation_time:.2f}s")
        print(f"     Total iteration time: {total_time:.2f}s")
        
        if isinstance(metrics, dict):
            print(f"     Test PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"     Test PR-AUC: {metrics.get('pr_auc', 'N/A')}")
            print(f"     Test AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"     Test AUC: {metrics.get('auc', 'N/A')}")
            print(f"     Test Best MCC: {metrics.get('best_mcc', 'N/A'):.4f}" if isinstance(metrics.get('best_mcc'), (int, float)) else f"     Test Best MCC: {metrics.get('best_mcc', 'N/A')}")
        
        # Store results
        result_row = {
            'feature_subset': feature_subset_name,
            'distance_dir': DISTANCES_DIR,
            'n_train_samples': len(undersampled_training_data),
            'n_train_minority': (undersampled_training_data[target_col] == 1).sum(),
            'n_train_majority': (undersampled_training_data[target_col] == 0).sum(),
            'best_depth': balanced_params[0],
            'best_minbucket': balanced_params[1],
            'best_cp': balanced_params[2],
            'matching_time_seconds': matching_time,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': evaluation_time,
            'total_time_seconds': total_time,
        }
        
        # Add metrics
        if isinstance(metrics, dict):
            result_row.update(metrics)
        
        all_results.append(result_row)
        
        # Save undersampled dataset
        config_name = f"cw_{CASE_WEIGHTING}_pool_{USE_ADAPTIVE_POOL}_seed_{SEED_METHOD}"
        undersample_path = os.path.join(RESULTS_DIR, f"{feature_subset_name.replace(' ', '_')}_{config_name}.csv")
        undersampled_training_data.to_csv(undersample_path, index=False)
        print(f"     ✓ Saved undersampled dataset: {undersample_path}")
        
    except Exception as e:
        print(f"\n  ✗ ERROR in iteration:")
        print(f"    {e}")
        import traceback
        traceback.print_exc()
        
        # Store error result
        all_results.append({
            'feature_subset': feature_subset_name,
            'distance_dir': DISTANCES_DIR,
            'error': str(e),
        })
        continue

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print(f"\n{'='*80}")
print("COMPARISON OF ALL FEATURE SUBSETS")
print(f"{'='*80}\n")

if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    results_path = os.path.join(RESULTS_DIR, "feature_subset_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved comparison results to: {results_path}\n")
    
    # Display comparison
    if 'pr_auc' in results_df.columns:
        results_df_sorted = results_df.sort_values('pr_auc', ascending=False)
        
        print("Results sorted by PR-AUC (best to worst):")
        print("="*80)
        
        display_cols = ['feature_subset']
        metric_cols = ['pr_auc', 'auc', 'best_mcc', 'balanced_recall_gmean', 'balanced_specificity_gmean']
        for col in metric_cols:
            if col in results_df.columns:
                display_cols.append(col)
        
        print(results_df_sorted[display_cols].to_string(index=False))
        
        print(f"\n{'='*80}")
        print("BEST FEATURE SUBSET:")
        print(f"{'='*80}")
        best_row = results_df_sorted.iloc[0]
        print(f"  Feature subset: {best_row['feature_subset']}")
        print(f"  Distance directory: {best_row['distance_dir']}")
        if isinstance(best_row.get('pr_auc'), (int, float)):
            print(f"  PR-AUC: {best_row['pr_auc']:.4f}")
        if isinstance(best_row.get('auc'), (int, float)):
            print(f"  AUC: {best_row['auc']:.4f}")
        if isinstance(best_row.get('best_mcc'), (int, float)):
            print(f"  Best MCC: {best_row['best_mcc']:.4f}")
    else:
        print("Available columns:", list(results_df.columns))
        print(results_df.to_string(index=False))
else:
    print("No results to compare. Check for errors above.")

print(f"\n{'='*80}\n")
