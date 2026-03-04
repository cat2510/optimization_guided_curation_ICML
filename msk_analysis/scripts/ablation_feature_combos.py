"""
Ablation runner: 9 combos of (Stage A distance) × (Stage B distance)

Stage A distance controls k-center dispersion via:
  - leaf_nn_matrix_npy / leaf_nn_enrolids_npy (precomputed DNN / NN matrix)

Stage B distance controls minority-majority matching via:
  - distances_majority_minority.h5 (precomputed PN H5)

Distance folders expected:
  ./precomputed_distances_msk_medical_only/
  ./precomputed_distances_msk_all_features/
  ./precomputed_distances_msk_cost_only/

Each folder should contain:
  distances_majority_minority.h5
  global_dnn_seed_{TRAIN_TEST_SEED}/leaf_global_dnn_matrix.npy
  global_dnn_seed_{TRAIN_TEST_SEED}/leaf_global_dnn_enrolids.npy

This script fixes M=50,000 and runs all 9 combos, saving:
  - undersampled training CSVs per combo
  - OCT artifacts per combo
  - a single summary CSV with metrics for all combos
"""

import sys, os, time, traceback
import numpy as np
import pandas as pd

# Add parent directory to path to import modules from one level up
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import importlib
import h5py
from sklearn.impute import SimpleImputer

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
    parent_projects_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if parent_projects_dir not in sys.path:
        sys.path.insert(0, parent_projects_dir)
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import two_stage_kcenter_then_match

from public.model_IAI import *  # includes: train_test_split_enrol, finetune_oct, evaluate_binary_oct, get_*_columns

from pyspark.sql import SparkSession


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
TRAIN_TEST_SEED = 123

# Fixed pool size for this ablation
M_FIXED = 50_000
M_values = [M_FIXED]

# Matching configuration (unchanged)
MATCHING_RATIO = 1              # 1:1 matching
CASE_WEIGHTING = None           # None or "boundary"
USE_ADAPTIVE_POOL = False       # keep False
USE_KMEANSPP = False
SEED_METHOD = "smart"           # "smart", "centroid", "density", "random"
FORCE_NEAREST_PER_CASE = False  # last step in two_stage_kcenter_match.py

# Distance metric type (only relevant if precomputing; here we only load)
DISTANCE_METRIC = "euclidean"

# Ablation sets
ABLATION_FEATURE_SETS = ["medical_only", "with_cost_features", "cost_only"]
DIST_DIR_TEMPLATE = "./precomputed_distances_msk_{distance_features}"

# Output dirs
BASE_DIR = "./ablation_stageA_stageB_distances_M50k_kmeanspp_False"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OCT_DIR = os.path.join(BASE_DIR, "oct_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OCT_DIR, exist_ok=True)


def get_distance_paths(distance_features: str, seed: int):
    dist_dir = DIST_DIR_TEMPLATE.format(distance_features=distance_features)
    pn_h5 = os.path.join(dist_dir, "distances_majority_minority.h5")
    dnn_out = os.path.join(dist_dir, f"global_dnn_seed_{seed}")
    dnn_mat = os.path.join(dnn_out, "leaf_global_dnn_matrix.npy")
    dnn_ids = os.path.join(dnn_out, "leaf_global_dnn_enrolids.npy")
    return dist_dir, pn_h5, dnn_mat, dnn_ids


def ensure_paths_exist(stageA_features: str, stageB_features: str, pn_h5: str, dnn_mat: str, dnn_ids: str):
    if not os.path.exists(pn_h5):
        raise FileNotFoundError(f"[StageB={stageB_features}] PN distance file not found: {pn_h5}")
    if not os.path.exists(dnn_mat):
        raise FileNotFoundError(f"[StageA={stageA_features}] DNN matrix file not found: {dnn_mat}")
    if not os.path.exists(dnn_ids):
        raise FileNotFoundError(f"[StageA={stageA_features}] DNN enrolids file not found: {dnn_ids}")


# =============================================================================
# LOAD DATA
# =============================================================================
spark = SparkSession.builder.appName("DataLoad").getOrCreate()
df_msk_spark = spark.read.format("parquet").load("msk_2017_18_full.parquet")
df_og = df_msk_spark.toPandas()

# Feature columns setup (unchanged from your script)
BIN_FLAG_COLUMNS = get_bin_flag_columns(df_og)
STAGE_COLUMNS = []
CAT_COLUMNS = get_cat_columns(df_og)
TRUE_NUM_COLUMNS = get_true_num_columns(df_og, CAT_COLUMNS, BIN_FLAG_COLUMNS)

# Cost columns (2017 only - exclude 2018 to prevent leakage)
COST_COLUMNS = [
    col for col in df_og.columns
    if ("cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower() or
        "decreasing" in col.lower() or "skewness" in col.lower() or "kurtosis" in col.lower() or
        "cv" in col.lower() or "range" in col.lower())
    and "2018" not in col
]
UTILIZATION_COLUMNS = [col for col in df_og.columns if "claims" in col.lower() and "2018" not in col]

leftover_cols = [
    c for c in df_og.columns
    if c not in CAT_COLUMNS and c not in TRUE_NUM_COLUMNS and c not in STAGE_COLUMNS and c not in BIN_FLAG_COLUMNS
    and c != "ENROLID"
]
if len(leftover_cols) > 0:
    print(f"Number of leftover columns: {len(leftover_cols)}; example: {leftover_cols[:20]}")

# Target selection
if "top_2_pct_cost_2018" in df_og.columns:
    target_col = "top_2_pct_cost_2018"
    print(f"Using {target_col} as target column")
elif "annual_cost_2018_deflated" in df_og.columns:
    threshold = df_og["annual_cost_2018_deflated"].quantile(0.98)
    df_og["top_2_pct_cost_2018"] = (df_og["annual_cost_2018_deflated"] >= threshold).astype(int)
    target_col = "top_2_pct_cost_2018"
    print(f"Created {target_col} using threshold ${threshold:,.2f}")
else:
    raise ValueError("No 2018 target column found. Need 'top_2_pct_cost_2018' or 'annual_cost_2018_deflated'.")

exclude_cols = ["ENROLID", target_col] + [col for col in df_og.columns if "2018" in col]
feature_cols = [c for c in df_og.columns if c not in exclude_cols]

# Split train/val/test (unchanged)
train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
    df_og,
    target_col=target_col,
    test_size=0.3,
    verbose=False,
    random_state=TRAIN_TEST_SEED
)
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

# Separate minority/majority in TRAIN
cases = train_pd[train_pd[target_col] == 1].copy()
controls = train_pd[train_pd[target_col] == 0].copy()
n_cases = len(cases)
n_controls = len(controls)
print(f"\nDataset split (train):")
print(f"  Cases (minority):   {n_cases:,}")
print(f"  Controls (majority):{n_controls:,}")
print(f"  Ratio: {n_controls/n_cases:.2f}:1")
print(f"Fixed pool size M: {M_FIXED:,} ({M_FIXED/n_controls*100:.1f}% of controls)")

# =============================================================================
# RUN 9-COMBO ABLATION
# =============================================================================
all_results = []
combo_idx = 0
run_start_global = time.perf_counter()

for stageA_features in ABLATION_FEATURE_SETS:
    for stageB_features in ABLATION_FEATURE_SETS:
        combo_idx += 1
        combo_tag = f"A_{stageA_features}__B_{stageB_features}"
        print("\n" + "=" * 100)
        print(f"ABLATION COMBO {combo_idx}/9: {combo_tag}")
        print("=" * 100)

        # Stage A: DNN nn matrix paths
        _, _, dnn_matrix_npy, dnn_enrolids_npy = get_distance_paths(stageA_features, TRAIN_TEST_SEED)
        # Stage B: PN h5 path
        _, PN_H5_PATH, _, _ = get_distance_paths(stageB_features, TRAIN_TEST_SEED)

        # Validate existence
        ensure_paths_exist(stageA_features, stageB_features, PN_H5_PATH, dnn_matrix_npy, dnn_enrolids_npy)

        # Per-combo outputs
        combo_results_dir = os.path.join(RESULTS_DIR, combo_tag)
        os.makedirs(combo_results_dir, exist_ok=True)

        for M in M_values:
            iteration_start_time = time.perf_counter()

            try:
                print(f"\n--- Running M={M:,} | StageA={stageA_features} | StageB={stageB_features} ---")
                print(f"StageA DNN: {dnn_matrix_npy}")
                print(f"StageB PN : {PN_H5_PATH}")

                # =========================
                # STEP 1: TWO-STAGE SAMPLING
                # =========================
                matching_start_time = time.perf_counter()

                matching_result = two_stage_kcenter_then_match(
                    leaf_controls_enrolids=controls["ENROLID"].values.astype(np.int64),
                    leaf_cases_enrolids=cases["ENROLID"].values.astype(np.int64),
                    leaf_nn_matrix_npy=dnn_matrix_npy,
                    leaf_nn_enrolids_npy=dnn_enrolids_npy,
                    pn_h5_path=PN_H5_PATH,
                    M=M,
                    use_adaptive_pool=USE_ADAPTIVE_POOL,
                    tau=None,
                    plateau_eps=0.01,
                    force_nearest_per_case=FORCE_NEAREST_PER_CASE,
                    force_topm=1,
                    assignment_topk_start=None,  # exact matching
                    seed_method=SEED_METHOD,
                    matching_ratio=MATCHING_RATIO,
                    X_majority_leaf=None,
                    case_weighting=CASE_WEIGHTING,
                    use_kmeanspp=USE_KMEANSPP,
                )

                matching_time = time.perf_counter() - matching_start_time

                selected_control_enrolids = matching_result["selected_control_enrolids"]
                all_match_costs = matching_result["match_costs"]

                print(f"✓ Matching complete | mean cost={all_match_costs.mean():.4f} | time={matching_time:.2f}s")

                # Build undersampled dataset (all minority + unique selected majority)
                all_minority = train_pd[train_pd[target_col] == 1].copy()
                unique_majority_enrolids = list(set(selected_control_enrolids))
                selected_majority = train_pd[
                    (train_pd[target_col] == 0) &
                    (train_pd["ENROLID"].isin(unique_majority_enrolids))
                ].copy()

                undersampled_training_data = pd.concat([all_minority, selected_majority], axis=0, ignore_index=True)

                print(f"✓ Undersampled dataset: {len(undersampled_training_data):,} "
                      f"(minority={int((undersampled_training_data[target_col]==1).sum()):,}, "
                      f"majority={int((undersampled_training_data[target_col]==0).sum()):,})")

                # Save dataset
                config_name = f"{combo_tag}__cw_{CASE_WEIGHTING}_pool_False_seed_{SEED_METHOD}"
                undersample_path = os.path.join(combo_results_dir, f"M{M}_{config_name}.csv")
                undersampled_training_data.to_csv(undersample_path, index=False)
                print(f"✓ Saved undersampled CSV: {undersample_path}")

                # =========================
                # STEP 2: TRAIN OCT
                # =========================
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

                training_time = time.perf_counter() - training_start_time

                # =========================
                # STEP 3: EVALUATE OCT
                # =========================
                results_dir = os.path.join(OCT_DIR, combo_tag, f"M{M}")
                os.makedirs(results_dir, exist_ok=True)

                if isinstance(balanced_params, dict):
                    bd, bm, bcp = balanced_params["depth"], balanced_params["minbucket"], balanced_params["cp"]
                    save_suffix = f"M{M}_{bd}_{bm}_{bcp}"
                else:
                    bd, bm, bcp = balanced_params
                    save_suffix = f"M{M}_{bd}_{bm}_{bcp}"

                metrics = evaluate_binary_oct(
                    balanced_model,
                    X_test, y_test,
                    preprocessor, feature_names,
                    X_val_df=X_val, y_val=y_val,
                    results_dir=results_dir,
                    save_suffix=save_suffix
                )

                # =========================
                # LOG RESULTS
                # =========================
                total_time = time.perf_counter() - iteration_start_time

                result_row = {
                    "combo_tag": combo_tag,
                    "stageA_features": stageA_features,
                    "stageB_features": stageB_features,
                    "M": M,
                    "M_pct_of_controls": M / n_controls * 100,
                    "adaptive_pool": USE_ADAPTIVE_POOL,
                    "matching_ratio": MATCHING_RATIO,
                    "case_weighting": CASE_WEIGHTING,
                    "seed_method": SEED_METHOD,
                    "use_kmeanspp": USE_KMEANSPP,
                    "force_nearest_per_case": FORCE_NEAREST_PER_CASE,
                    "n_cases_train": n_cases,
                    "n_controls_train": n_controls,
                    "n_train_samples": len(undersampled_training_data),
                    "n_train_minority": int((undersampled_training_data[target_col] == 1).sum()),
                    "n_train_majority": int((undersampled_training_data[target_col] == 0).sum()),
                    "mean_match_cost": float(np.mean(all_match_costs)) if hasattr(all_match_costs, "__len__") else None,
                    "q50_match_cost": float(np.quantile(all_match_costs, 0.50)) if hasattr(all_match_costs, "__len__") else None,
                    "q90_match_cost": float(np.quantile(all_match_costs, 0.90)) if hasattr(all_match_costs, "__len__") else None,
                    "best_depth": bd,
                    "best_minbucket": bm,
                    "best_cp": bcp,
                    "matching_time_s": matching_time,
                    "train_time_s": training_time,
                    "total_time_s": total_time,
                    "undersample_csv": undersample_path,
                    "oct_results_dir": results_dir,
                }

                if isinstance(metrics, dict):
                    result_row.update(metrics)

                all_results.append(result_row)

                # Quick print
                if isinstance(metrics, dict):
                    pr = metrics.get("pr_auc", None)
                    auc = metrics.get("auc", None)
                    mcc = metrics.get("best_mcc", None)
                    if isinstance(pr, (int, float)) and isinstance(auc, (int, float)) and isinstance(mcc, (int, float)):
                        print(f"✓ Metrics | PR-AUC={pr:.4f} | AUC={auc:.4f} | best_MCC={mcc:.4f}")
                    else:
                        print(f"✓ Metrics keys: {list(metrics.keys())}")

            except Exception as e:
                print("\n✗ ERROR in combo run:")
                print(f"  combo={combo_tag}, M={M:,}")
                print(f"  {e}")
                traceback.print_exc()

                all_results.append({
                    "combo_tag": combo_tag,
                    "stageA_features": stageA_features,
                    "stageB_features": stageB_features,
                    "M": M,
                    "error": str(e),
                })
                continue

# =============================================================================
# SAVE SUMMARY
# =============================================================================
elapsed_global = time.perf_counter() - run_start_global
print("\n" + "=" * 100)
print(f"ABLATION COMPLETE in {elapsed_global/60:.2f} minutes. Saving summary CSV...")
print("=" * 100)

results_df = pd.DataFrame(all_results)
summary_path = os.path.join(RESULTS_DIR, "ablation_stageA_stageB_M50k.csv")
results_df.to_csv(summary_path, index=False)
print(f"✓ Saved summary: {summary_path}")

# Display best by PR-AUC if available
if "pr_auc" in results_df.columns:
    results_ok = results_df[~results_df.get("pr_auc").isna()].copy()
    if len(results_ok) > 0:
        results_ok = results_ok.sort_values("pr_auc", ascending=False)
        best = results_ok.iloc[0]
        print("\nBEST BY PR-AUC:")
        print(f"  combo_tag: {best.get('combo_tag')}")
        print(f"  StageA: {best.get('stageA_features')} | StageB: {best.get('stageB_features')}")
        print(f"  PR-AUC: {best.get('pr_auc')}")
        if "auc" in best:
            print(f"  AUC: {best.get('auc')}")
        if "best_mcc" in best:
            print(f"  best_mcc: {best.get('best_mcc')}")
else:
    print("No pr_auc column found in results. Columns are:")
    print(list(results_df.columns))

print("\nDone.\n")