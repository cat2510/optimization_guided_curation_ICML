#!/usr/bin/env python3
"""
Train OCT on Existing Undersampled Datasets
============================================
This script loads pre-generated undersampled datasets and trains OCT models,
logging ALL metrics returned by evaluate_binary_oct().

Much faster than the full hyperparameter search since k-center matching is skipped.

Cohort data: set env OCT_TRAIN_PARQUET, or place 0917_2017_18_with_2017_cost.parquet next to
this script or cwd. That path may be a single .parquet file or a Spark-style parquet *folder*.
"""

import os
import re
import sys
import glob
import pandas as pd
from datetime import datetime

# Project root (so `import public.model_IAI` works when cwd is not my_projects)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from public.model_IAI import (
    finetune_oct,
    evaluate_binary_oct,
    get_bin_flag_columns,
    get_true_num_columns,
    train_test_split_enrol,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (resolved relative to this script so runs work from any cwd)
# Set to either: (1) a directory containing undersampled_*.csv, or (2) one undersampled .csv file
UNDERSAMPLED_PATH = "/Users/cat2510/my_projects/kcenter_hyperparams_search/kcenter_hyperparameter_search_results_global_seed_123_matching_ratio_1/undersampled_cw_None_pool_False_seed_random.csv"
RESULTS_DIR = os.path.join(
    _SCRIPT_DIR, "kcenter_hyperparameter_search_results", "oct_training_results"
)


def resolve_undersampled_csvs(path):
    """Return sorted list of CSV paths; single file or glob undersampled_*.csv in a directory."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p) and p.lower().endswith(".csv"):
        return [p]
    if os.path.isdir(p):
        return sorted(glob.glob(os.path.join(p, "undersampled_*.csv")))
    return []


PARQUET_FILENAME = "0917_2017_18_with_2017_cost.parquet"


def _path_is_readable_parquet_dataset(p):
    """
    True if pandas can read this path with read_parquet.

    Spark exports are often a *directory* named something.parquet/ containing part-*.parquet;
    os.path.isfile is False for those, but pd.read_parquet(path) works on the folder.
    """
    if not p or not os.path.exists(p):
        return False
    if os.path.isfile(p):
        return p.lower().endswith(".parquet")
    if os.path.isdir(p):
        # Folder named like a single parquet file (common Spark layout)
        if os.path.basename(p).lower().endswith(".parquet"):
            return True
        # Generic dataset folder with parquet fragments
        try:
            for name in os.listdir(p):
                if name.endswith(".parquet") and os.path.isfile(os.path.join(p, name)):
                    return True
        except OSError:
            return False
    return False


def resolve_parquet_path():
    """
    Find the main cohort parquet (same convention as competing_methods.ipynb: cwd-relative name).

    Order: OCT_TRAIN_PARQUET env (absolute path) -> next to this script -> current working directory.

    Accepts either a single .parquet file or a parquet *dataset directory* (e.g. Spark output).
    """
    env_path = os.environ.get("OCT_TRAIN_PARQUET", "").strip()
    candidates = []
    if env_path:
        candidates.append(("OCT_TRAIN_PARQUET", os.path.abspath(os.path.expanduser(env_path))))
    candidates.append(("script_dir", os.path.join(_SCRIPT_DIR, PARQUET_FILENAME)))
    candidates.append(("cwd", os.path.join(os.getcwd(), PARQUET_FILENAME)))
    for label, p in candidates:
        if p and _path_is_readable_parquet_dataset(p):
            return p, candidates, label
    return None, candidates, None

# OCT hyperparameters for model training
OCT_DEPTHS = [7]
OCT_MINBUCKETS = [25]
OCT_CPS = [0.00005, 0.0001, 0.001,0.01]

# Target column
TARGET_COL = "highcost_gt_200000"

# Same seed as competing_methods.ipynb (both splits must use it for reproducible val/test)
TRAIN_TEST_SEED = 123

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print("TRAIN OCT ON EXISTING UNDERSAMPLED DATASETS")
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
    
    parquet_path, _cand_list, source_label = resolve_parquet_path()
    if not parquet_path:
        print(f"ERROR: No readable parquet dataset at: {PARQUET_FILENAME}")
        print("  (Expect a single .parquet file or a Spark-style directory named *.parquet/)")
        print("  Tried (in order):")
        for label, p in _cand_list:
            exists = os.path.exists(p) if p else False
            kind = (
                "dir"
                if exists and os.path.isdir(p)
                else "file"
                if exists and os.path.isfile(p)
                else "missing"
            )
            print(f"    - [{label}] {p}  ({kind})")
        print("  Set OCT_TRAIN_PARQUET to the full path of your cohort file or parquet folder.")
        return
    df_og = pd.read_parquet(parquet_path)
    print(f"✓ Loaded original data ({source_label}): {parquet_path}")
    print(f"  Shape: {df_og.shape}")
    
    # Column groups (same as competing_methods.ipynb)
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df_og)
    CAT_COLUMNS = df_og.select_dtypes(include=["object", "category"]).columns.tolist()
    TRUE_NUM_COLUMNS = get_true_num_columns(df_og, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    
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
    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df_og,
        target_col="cost_stratum_2018",
        test_size=0.3,
        verbose=False,
        random_state=TRAIN_TEST_SEED,
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd,
        target_col=TARGET_COL,
        test_size=0.5,
        verbose=False,
        random_state=TRAIN_TEST_SEED,
    )
    
    X_test = test_pd[feature_cols]
    y_test = test_pd[TARGET_COL]
    X_val = val_pd[feature_cols]
    y_val = val_pd[TARGET_COL]
    
    print(f"✓ Train: {train_pd.shape}, Val: {val_pd.shape}, Test: {test_pd.shape}")
    
    # ========================================================================
    # FIND ALL UNDERSAMPLED DATASETS
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINDING UNDERSAMPLED DATASETS")
    print(f"{'='*80}\n")

    undersampled_files = resolve_undersampled_csvs(UNDERSAMPLED_PATH)

    if not undersampled_files:
        print(f"ERROR: No undersampled CSVs found for UNDERSAMPLED_PATH={UNDERSAMPLED_PATH!r}")
        print("  Use a path to a .csv file, or a directory containing undersampled_*.csv")
        return

    print(f"Found {len(undersampled_files)} undersampled dataset(s):")
    for f in undersampled_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # ========================================================================
    # TRAIN OCT ON EACH DATASET
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING OCT MODELS")
    print(f"{'='*80}\n")

    all_results = []
    metrics_master_path = f"{RESULTS_DIR}/metrics_master.csv"

    for idx, undersampled_path in enumerate(undersampled_files, 1):
        basename = os.path.basename(undersampled_path)
        config_name = basename.replace("undersampled_", "").replace(".csv", "")

        # Parse config from filename: cw_{X}_pool_{Y}_seed_{Z}
        parts = config_name.split("_")
        try:
            cw_idx = parts.index("cw")
            pool_idx = parts.index("pool")
            seed_idx = parts.index("seed")

            case_weighting = "_".join(parts[cw_idx + 1 : pool_idx])
            use_adaptive_pool = parts[pool_idx + 1]
            seed_method = "_".join(parts[seed_idx + 1 :])

            if case_weighting.lower() == "none":
                case_weighting = None
            use_adaptive_pool = use_adaptive_pool.lower() == "true"

        except (ValueError, IndexError):
            print(f"  ⚠️ WARNING: Could not parse config from filename: {basename}")
            case_weighting = None
            use_adaptive_pool = None
            seed_method = None

        print(f"\n{'#'*80}")
        print(f"CONFIGURATION {idx}/{len(undersampled_files)}: {config_name}")
        print(f"{'#'*80}")
        print(f"  case_weighting: {case_weighting}")
        print(f"  use_adaptive_pool: {use_adaptive_pool}")
        print(f"  seed_method: {seed_method}")
        print(f"  file: {basename}\n")

        try:
            print(f"  Loading undersampled dataset...")
            undersampled_training_data = pd.read_csv(undersampled_path)

            missing_for_train = [c for c in feature_cols if c not in undersampled_training_data.columns]
            missing_target = TARGET_COL not in undersampled_training_data.columns
            if missing_target:
                raise KeyError(f"CSV missing target column {TARGET_COL!r}")
            if missing_for_train:
                raise KeyError(
                    f"CSV missing {len(missing_for_train)} feature column(s) vs parquet feature set "
                    f"(e.g. {missing_for_train[:5]})"
                )

            n_samples = len(undersampled_training_data)
            n_minority = (undersampled_training_data[TARGET_COL] == 1).sum()
            n_majority = (undersampled_training_data[TARGET_COL] == 0).sum()

            print(f"    ✓ Loaded {n_samples:,} samples")
            print(f"      Minority: {n_minority:,}")
            print(f"      Majority: {n_majority:,}")
            if n_minority > 0:
                print(f"      Ratio: {n_majority/n_minority:.2f}:1")
            else:
                print("      Ratio: n/a (no positive class in training CSV)")

            print(f"\n  Training OCT model...")
            balanced_model, balanced_params, _, preprocessor, feature_names = finetune_oct(
                X_train=undersampled_training_data[feature_cols],
                y_train=undersampled_training_data[TARGET_COL],
                X_val=X_val,
                y_val=y_val,
                categorical_cols=CAT_COLUMNS,
                numeric_cols=TRUE_NUM_COLUMNS,
                binary_cols=BIN_FLAG_COLUMNS,
                depths=OCT_DEPTHS,
                minbuckets=OCT_MINBUCKETS,
                cps=OCT_CPS,
            )

            print(f"  Evaluating on test set...")
            safe_suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", config_name)[:120]
            metrics = evaluate_binary_oct(
                balanced_model,
                X_test,
                y_test,
                preprocessor,
                feature_names,
                results_dir=RESULTS_DIR,
                save_suffix=safe_suffix,
                X_val_df=X_val,
                y_val=y_val,
            )

            if isinstance(balanced_params, dict):
                hc = balanced_params.get("hyperplane_config")
                params_dict = {
                    "best_variant": balanced_params.get("variant"),
                    "best_depth": balanced_params.get("depth"),
                    "best_minbucket": balanced_params.get("minbucket"),
                    "best_cp": balanced_params.get("cp"),
                    "best_tuning_time_seconds": balanced_params.get("tuning_time_seconds"),
                    "best_fit_time_seconds": balanced_params.get("best_fit_time_seconds"),
                    "best_hyperplane_config": repr(hc) if hc is not None else None,
                }
            elif isinstance(balanced_params, tuple) and len(balanced_params) == 3:
                params_dict = {
                    "best_depth": balanced_params[0],
                    "best_minbucket": balanced_params[1],
                    "best_cp": balanced_params[2],
                }
            else:
                params_dict = {"best_params": str(balanced_params)}

            row = {
                "config_name": config_name,
                "case_weighting": case_weighting,
                "use_adaptive_pool": use_adaptive_pool,
                "seed_method": seed_method,
                "n_train_samples": n_samples,
                "n_train_minority": n_minority,
                "n_train_majority": n_majority,
                **params_dict,
            }

            if isinstance(metrics, dict):
                row.update(metrics)
            else:
                row["metrics_error"] = str(metrics)

            all_results.append(row)

            pd.DataFrame([row]).to_csv(
                metrics_master_path,
                mode="a",
                header=not os.path.exists(metrics_master_path),
                index=False,
            )

            print(f"\n  ✓ Configuration {idx}/{len(undersampled_files)} complete")
            print(f"    Best params: {balanced_params}")
            if isinstance(metrics, dict):
                print(f"    AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"    AUC: {metrics.get('auc', 'N/A')}")
                print(f"    PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"    PR-AUC: {metrics.get('pr_auc', 'N/A')}")
                print(f"    Optimal F1: {metrics.get('optimal_f1', 'N/A'):.4f}" if isinstance(metrics.get('optimal_f1'), (int, float)) else f"    Optimal F1: {metrics.get('optimal_f1', 'N/A')}")
                print(f"    Best MCC: {metrics.get('best_mcc', 'N/A'):.4f}" if isinstance(metrics.get('best_mcc'), (int, float)) else f"    Best MCC: {metrics.get('best_mcc', 'N/A')}")
                print(f"    Sensitivity (G-mean threshold): {metrics.get('balanced_recall_gmean', 'N/A'):.4f}" if isinstance(metrics.get('balanced_recall_gmean'), (int, float)) else f"    Sensitivity: {metrics.get('sensitivity_f1', 'N/A')}")
                print(f"    Specificity (G-mean threshold): {metrics.get('balanced_specificity_gmean', 'N/A'):.4f}" if isinstance(metrics.get('balanced_specificity_gmean'), (int, float)) else f"    Specificity: {metrics.get('specificity_f1', 'N/A')}")

        except Exception as e:
            print(f"\n  ✗ ERROR in configuration {config_name}:")
            print(f"    {e}")
            import traceback

            traceback.print_exc()

            row = {
                "config_name": config_name,
                "case_weighting": case_weighting,
                "use_adaptive_pool": use_adaptive_pool,
                "seed_method": seed_method,
                "error": str(e),
            }
            all_results.append(row)
            continue

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("OCT TRAINING COMPLETE")
    print(f"\nTotal configurations: {len(all_results)}")
    print(f"Results saved to: {metrics_master_path}")
    
    # Show best configurations by AUC
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        if 'auc' in results_df.columns:
            # Sort by AUC (descending)
            results_df_sorted = results_df.sort_values('auc', ascending=False)
            
            print(f"\n{'='*80}")
            print("TOP 10 CONFIGURATIONS BY AUC")
            print(f"{'='*80}\n")
            
            # Display relevant columns
            display_cols = ['config_name']
            for col in ['auc', 'pr_auc', 'optimal_f1', 'best_mcc', 'balanced_recall_gmean', 'balanced_specificity_gmean']:
                if col in results_df.columns:
                    display_cols.append(col)
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)
            
            print(results_df_sorted[display_cols].head(10).to_string(index=False))
            
              
            # Save sorted results
            sorted_path = f"{RESULTS_DIR}/metrics_sorted_by_auc.csv"
            results_df_sorted.to_csv(sorted_path, index=False)
            print(f"\n✓ Sorted results saved to: {sorted_path}")
            
        else:
            print(f"\n⚠️ No AUC column found. Available columns: {list(results_df.columns)}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
