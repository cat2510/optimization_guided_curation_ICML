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
BASE_DIR = "./sensitivity_medical_only_features_150_minbucket_kmeanspp"
RESULTS_DIR = os.path.join(BASE_DIR, "results") #undersampled dataset directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration for k-center matching
MATCHING_RATIO = 1  # 1:1 matching (can be changed)
CASE_WEIGHTING = None  # Options: None, "boundary"
USE_ADAPTIVE_POOL = False  # Options: True, False
USE_KMEANSPP = True
SEED_METHOD = "smart"  # Options: "smart", "centroid", "density", "random"

FORCE_NEAREST_PER_CASE = False #last step in two_stage_kcenter_match.py
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
exclude_cols = ["ENROLID", target_col] + [col for col in df_og.columns if "2018" in col] + COST_COLUMNS
feature_cols = [c for c in df_og.columns if c not in exclude_cols]

#TODO try different distance metrics for precompute_distances: 'euclidean' (L2), 'manhattan' (L1), 'chebyshev' (L_infinity)
DISTANCE_METRIC = "euclidean"  # or "chebyshev" for L_infinity; pass to compute_distances_batched / precompute_leaf_dnn_memmap

# Use a single precomputed distance directory
DISTANCES_DIR = "./precomputed_distances_msk_medical_only"  # Change this to your preferred directory
PN_H5_PATH = os.path.join(DISTANCES_DIR, "distances_majority_minority.h5")
DNN_OUT_DIR = os.path.join(DISTANCES_DIR, f"global_dnn_seed_{TRAIN_TEST_SEED}")
dnn_matrix_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_matrix.npy")
dnn_enrolids_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_enrolids.npy")

# Check if required files exist
if not os.path.exists(PN_H5_PATH):
    raise FileNotFoundError(f"Distance file not found: {PN_H5_PATH}")
if not os.path.exists(dnn_matrix_npy) or not os.path.exists(dnn_enrolids_npy):
    raise FileNotFoundError(f"DNN files not found in {DNN_OUT_DIR}")

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
# CONFIGURATION: Pool size sensitivity analysis
# ============================================================================

# Separate minority (cases) and majority (controls)
cases = train_pd[train_pd[target_col] == 1].copy()
controls = train_pd[train_pd[target_col] == 0].copy()

n_cases = len(cases)
n_controls = len(controls)

print(f"Dataset split:")
print(f"  Cases (minority): {n_cases:,}")
print(f"  Controls (majority): {n_controls:,}")
print(f"  Ratio: {n_controls/n_cases:.2f}:1\n")

# Generate pool sizes M to test
# Range from n_cases to n_controls//2 in reasonably separated steps
M_min = n_controls // 8 
M_max = 100000
# Create steps: use approximately 10-15 steps, with larger steps for larger ranges
num_steps = 8
if M_max - M_min < num_steps:
    # If range is small, test every value
    M_values = list(range(M_min, M_max + 1))
else:
    # Create evenly spaced steps (logarithmic might be better, but linear is simpler)
    step_size = max(1, (M_max - M_min) // num_steps)
    M_values = list(range(M_min, M_max + 1, step_size))
    # Ensure we include M_max
    if M_values[-1] != M_max:
        M_values.append(M_max)

print(f"Pool sizes (M) to test: {len(M_values)} values")
print(f"  Range: {M_min:,} to {M_max:,}")
print(f"  Values: {M_values[:5]} ... {M_values[-2:]} (showing first 5 and last 2)")
print(f"  Full list: {M_values}\n")

# Store results for all configurations
all_results = []

# ============================================================================
# ITERATE OVER DIFFERENT POOL SIZES (M) WITH ADAPTIVE_POOL=False
# ============================================================================

print(f"{'='*80}")
print("PHASE 1: Testing fixed pool sizes (adaptive_pool=False)")
print(f"{'='*80}\n")

for m_idx, M in enumerate(M_values, 1):
    print(f"\n{'#'*80}")
    print(f"ITERATION {m_idx}/{len(M_values)}: M = {M:,} (fixed pool)")
    print(f"{'#'*80}\n")
    
    iteration_start_time = time.perf_counter()
    
    try:
        
        # ====================================================================
        # STEP 1: K-CENTER UNDERSAMPLING
        # ====================================================================
        print(f"  {'='*76}")
        print(f"  K-CENTER UNDERSAMPLING")
        print(f"  {'='*76}")
        print(f"  Configuration:")
        print(f"    Matching ratio: 1:{MATCHING_RATIO}")
        print(f"    Case weighting: {CASE_WEIGHTING}")
        print(f"    Adaptive pool: {USE_ADAPTIVE_POOL}")
        print(f"    Use k-means++: {USE_KMEANSPP}")
        print(f"    Seed method: {SEED_METHOD}")
        print(f"    Candidate pool size (M): {M:,} ({M/n_controls*100:.1f}% of controls)")
        
        matching_start_time = time.perf_counter()
        
        matching_result = two_stage_kcenter_then_match(
            leaf_controls_enrolids=controls['ENROLID'].values.astype(np.int64),
            leaf_cases_enrolids=cases['ENROLID'].values.astype(np.int64),
            leaf_nn_matrix_npy=dnn_matrix_npy,
            leaf_nn_enrolids_npy=dnn_enrolids_npy,
            pn_h5_path=PN_H5_PATH,
            M=M,
            use_adaptive_pool=USE_ADAPTIVE_POOL,  # Fixed pool size
            tau=None,  # Not used when adaptive_pool=False
            plateau_eps=0.01,
            force_nearest_per_case=FORCE_NEAREST_PER_CASE,
            force_topm=1,
            assignment_topk_start=None,  # exact matching
            seed_method=SEED_METHOD,
            matching_ratio=MATCHING_RATIO,
            X_majority_leaf=None,  # Not needed when using precomputed distances
            case_weighting=CASE_WEIGHTING,
            use_kmeanspp=USE_KMEANSPP,
        )
        
        matching_end_time = time.perf_counter()
        matching_time = matching_end_time - matching_start_time
        
        # Extract results
        selected_control_enrolids = matching_result["selected_control_enrolids"]
        all_match_costs = matching_result["match_costs"]
        
        print(f"\n  ✓ Matching complete!")
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
        
        # Save undersampled dataset
        config_name = f"cw_{CASE_WEIGHTING}_pool_False_seed_{SEED_METHOD}"
        undersample_path = os.path.join(RESULTS_DIR, f"M{M}_{config_name}.csv")
        undersampled_training_data.to_csv(undersample_path, index=False)
        #undersampled_training_data = pd.read_csv(undersample_path)
        print(f"     ✓ Loaded undersampled dataset: {undersample_path}")
        
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
            minbuckets=[150],
            cps=[0.0001, 0.001, 0.01],
            verbose=False,
            random_seed=TRAIN_TEST_SEED
        )
        
        training_end_time = time.perf_counter()
        training_time = training_end_time - training_start_time
        
        # Evaluate
        evaluation_start_time = time.perf_counter()
        
        # Create results directory
        results_dir = f"{BASE_DIR}/pool_size_M{M}"
        os.makedirs(results_dir, exist_ok=True)
        if isinstance(balanced_params, dict):
            save_suffix = f"M{M}_{balanced_params['depth']}_{balanced_params['minbucket']}_{balanced_params['cp']}"
        else:
            save_suffix = f"M{M}_{balanced_params[0]}_{balanced_params[1]}_{balanced_params[2]}"
        metrics = evaluate_binary_oct(
            balanced_model, X_test, y_test, preprocessor, feature_names, 
            X_val_df=X_val, y_val=y_val, results_dir=results_dir, save_suffix=save_suffix)
        
        
        if isinstance(metrics, dict):
            print(f"     Test PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"     Test PR-AUC: {metrics.get('pr_auc', 'N/A')}")
            print(f"     Test AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"     Test AUC: {metrics.get('auc', 'N/A')}")
            print(f"     Test Best MCC: {metrics.get('best_mcc', 'N/A'):.4f}" if isinstance(metrics.get('best_mcc'), (int, float)) else f"     Test Best MCC: {metrics.get('best_mcc', 'N/A')}")
        if isinstance(balanced_params, dict):
            bd, bm, bcp = balanced_params["depth"], balanced_params["minbucket"], balanced_params["cp"]
        else:
            bd, bm, bcp = balanced_params
        # Store results
        result_row = {
            'M': M,
            'M_pct_of_controls': M / n_controls * 100,
            'adaptive_pool': False,
            'n_train_samples': len(undersampled_training_data),
            'n_train_minority': (undersampled_training_data[target_col] == 1).sum(),
            'n_train_majority': (undersampled_training_data[target_col] == 0).sum(),
            'best_depth': bd,
            'best_minbucket': bm,
            'best_cp': bcp,
            'matching_time_seconds': matching_time,
            'training_time_seconds': training_time,
        }
        
        # Add metrics
        if isinstance(metrics, dict):
            result_row.update(metrics)
        
        all_results.append(result_row)
        
    except Exception as e:
        print(f"\n  ✗ ERROR in iteration:")
        print(f"    {e}")
        import traceback
        traceback.print_exc()
        
        # Store error result
        all_results.append({
            'M': M,
            'M_pct_of_controls': M / n_controls * 100,
            'adaptive_pool': False,
            'error': str(e),
        })
        continue


# ============================================================================
# FINAL COMPARISON
# ============================================================================

print(f"\n{'='*80}")
print("POOL SIZE SENSITIVITY ANALYSIS RESULTS")
print(f"{'='*80}\n")

if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV (append if file exists)
    results_path = os.path.join(RESULTS_DIR, "sensitivity_pool_size.csv")
    write_header = not os.path.exists(results_path)
    results_df.to_csv(results_path, mode='a', index=False, header=write_header)
    print(f"✓ Saved results to: {results_path}\n")
    
    # Display comparison
    if 'pr_auc' in results_df.columns:
        # Sort by M (pool size) for fixed pools, then show adaptive
        results_df_fixed = results_df[results_df['adaptive_pool'] == False].copy()
        results_df_adaptive = results_df[results_df['adaptive_pool'] == True].copy()
        
        if len(results_df_fixed) > 0:
            results_df_fixed = results_df_fixed.sort_values('M')
            
            print("Results for fixed pool sizes (sorted by M):")
            print("="*80)
            
            display_cols = ['M', 'M_pct_of_controls']
            metric_cols = ['pr_auc', 'auc', 'best_mcc', 'balanced_recall_gmean', 'balanced_specificity_gmean']
            for col in metric_cols:
                if col in results_df_fixed.columns:
                    display_cols.append(col)
            
            print(results_df_fixed[display_cols].to_string(index=False))
        
        if len(results_df_adaptive) > 0:
            print(f"\n{'='*80}")
            print("Adaptive pool result:")
            print("="*80)
            adaptive_row = results_df_adaptive.iloc[0]
            print(f"  Stopping pool size (M): {adaptive_row['M']:,} ({adaptive_row['M_pct_of_controls']:.1f}% of controls)")
            if isinstance(adaptive_row.get('pr_auc'), (int, float)):
                print(f"  PR-AUC: {adaptive_row['pr_auc']:.4f}")
            if isinstance(adaptive_row.get('auc'), (int, float)):
                print(f"  AUC: {adaptive_row['auc']:.4f}")
            if isinstance(adaptive_row.get('best_mcc'), (int, float)):
                print(f"  Best MCC: {adaptive_row['best_mcc']:.4f}")
        
        # Find best configuration
        results_df_sorted = results_df.sort_values('pr_auc', ascending=False)
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION:")
        print(f"{'='*80}")
        best_row = results_df_sorted.iloc[0]
        print(f"  Pool size (M): {best_row['M']:,} ({best_row['M_pct_of_controls']:.1f}% of controls)")
        print(f"  Adaptive pool: {best_row['adaptive_pool']}")
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
