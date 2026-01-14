#!/usr/bin/env python3
"""
Train OCT on Existing Undersampled Datasets
============================================
This script loads pre-generated undersampled datasets and trains OCT models,
logging ALL metrics returned by evaluate_binary_oct().

Much faster than the full hyperparameter search since k-center matching is skipped.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
import model_pipeline
import model_IAI
from model_IAI import finetune_oct, evaluate_binary_oct

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
UNDERSAMPLED_DIR = "./kcenter_hyperparameter_search_results"
RESULTS_DIR = "./oct_training_results"

# OCT hyperparameters for model training
OCT_DEPTHS = [7, 9]
OCT_MINBUCKETS = [50, 100, 120, 150]
OCT_CPS = [0.00001, 0.0001, 0.001, 0.01]

# Target column
TARGET_COL = "highcost_gt_200000"

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
    
    # Load Spark data for test/val sets
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("OCTTraining").getOrCreate()
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
    # FIND ALL UNDERSAMPLED DATASETS
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINDING UNDERSAMPLED DATASETS")
    print(f"{'='*80}\n")
    
    # Find all undersampled_*.csv files
    pattern = os.path.join(UNDERSAMPLED_DIR, "undersampled_*.csv")
    undersampled_files = sorted(glob.glob(pattern))
    
    if not undersampled_files:
        print(f"ERROR: No undersampled datasets found in {UNDERSAMPLED_DIR}/")
        print(f"Expected files matching pattern: undersampled_*.csv")
        return
    
    print(f"Found {len(undersampled_files)} undersampled datasets:")
    for f in undersampled_files:
        basename = os.path.basename(f)
        print(f"  - {basename}")
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
        # Extract config name from filename
        basename = os.path.basename(undersampled_path)
        config_name = basename.replace('undersampled_', '').replace('.csv', '')
        
        # Parse config from filename
        # Format: cw_{X}_pool_{Y}_seed_{Z}
        parts = config_name.split('_')
        try:
            cw_idx = parts.index('cw')
            pool_idx = parts.index('pool')
            seed_idx = parts.index('seed')
            
            case_weighting = '_'.join(parts[cw_idx+1:pool_idx])
            use_adaptive_pool = parts[pool_idx+1]
            seed_method = '_'.join(parts[seed_idx+1:])
            
            # Convert to proper types
            if case_weighting.lower() == 'none':
                case_weighting = None
            use_adaptive_pool = (use_adaptive_pool.lower() == 'true')
            
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
            # Load undersampled dataset
            print(f"  Loading undersampled dataset...")
            undersampled_training_data = pd.read_csv(undersampled_path)
            
            n_samples = len(undersampled_training_data)
            n_minority = (undersampled_training_data[TARGET_COL] == 1).sum()
            n_majority = (undersampled_training_data[TARGET_COL] == 0).sum()
            
            print(f"    ✓ Loaded {n_samples:,} samples")
            print(f"      Minority: {n_minority:,}")
            print(f"      Majority: {n_majority:,}")
            print(f"      Ratio: {n_majority/n_minority:.2f}:1")
            
            # Train OCT
            print(f"\n  Training OCT model...")
            balanced_model, balanced_params, _, preprocessor, feature_names = finetune_oct(
                X_train=undersampled_training_data[feature_cols],
                y_train=undersampled_training_data[TARGET_COL],
                X_val=X_val,
                y_val=y_val,
                categorical_cols=CAT_COLUMNS,
                numeric_cols=TRUE_NUM_COLUMNS,
                depths=OCT_DEPTHS,
                minbuckets=OCT_MINBUCKETS,
                cps=OCT_CPS,
            )
            
            # Evaluate
            print(f"  Evaluating on test set...")
            metrics = evaluate_binary_oct(
                balanced_model, X_test, y_test, preprocessor, feature_names,
                results_dir=RESULTS_DIR, ratio=1.0
            )
            
            # Handle balanced_params (tuple of depth, minbucket, cp)
            if isinstance(balanced_params, tuple) and len(balanced_params) == 3:
                params_dict = {
                    'best_depth': balanced_params[0],
                    'best_minbucket': balanced_params[1],
                    'best_cp': balanced_params[2],
                }
            else:
                params_dict = {'best_params': str(balanced_params)}
            
            # Collect ALL metrics (no filtering)
            row = {
                'config_name': config_name,
                'case_weighting': case_weighting,
                'use_adaptive_pool': use_adaptive_pool,
                'seed_method': seed_method,
                'n_train_samples': n_samples,
                'n_train_minority': n_minority,
                'n_train_majority': n_majority,
                **params_dict,
            }
            
            # Add ALL metrics from evaluate_binary_oct
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
            
            # Print summary
            print(f"\n  ✓ Configuration {idx}/{len(undersampled_files)} complete")
            print(f"    Best params: {balanced_params}")
            if isinstance(metrics, dict):
                print(f"    AUC: {metrics.get('auc', 'N/A'):.4f}" if isinstance(metrics.get('auc'), (int, float)) else f"    AUC: {metrics.get('auc', 'N/A')}")
                print(f"    PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), (int, float)) else f"    PR-AUC: {metrics.get('pr_auc', 'N/A')}")
                print(f"    Optimal F1: {metrics.get('optimal_f1', 'N/A'):.4f}" if isinstance(metrics.get('optimal_f1'), (int, float)) else f"    Optimal F1: {metrics.get('optimal_f1', 'N/A')}")
                print(f"    Sensitivity (F1): {metrics.get('sensitivity_f1', 'N/A'):.4f}" if isinstance(metrics.get('sensitivity_f1'), (int, float)) else f"    Sensitivity (F1): {metrics.get('sensitivity_f1', 'N/A')}")
                print(f"    Specificity (F1): {metrics.get('specificity_f1', 'N/A'):.4f}" if isinstance(metrics.get('specificity_f1'), (int, float)) else f"    Specificity (F1): {metrics.get('specificity_f1', 'N/A')}")
            
        except Exception as e:
            print(f"\n  ✗ ERROR in configuration {config_name}:")
            print(f"    {e}")
            import traceback
            traceback.print_exc()
            
            # Log error
            row = {
                'config_name': config_name,
                'case_weighting': case_weighting,
                'use_adaptive_pool': use_adaptive_pool,
                'seed_method': seed_method,
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
    print("OCT TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
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
            for col in ['auc', 'pr_auc', 'optimal_f1', 'sensitivity_f1', 'specificity_f1', 
                       'balanced_recall_gmean', 'balanced_specificity_gmean']:
                if col in results_df.columns:
                    display_cols.append(col)
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)
            
            print(results_df_sorted[display_cols].head(10).to_string(index=False))
            
            # Also show by optimal F1 if available
            if 'balanced_recall_gmean' in results_df.columns:
                print(f"\n{'='*80}")
                print("TOP 10 CONFIGURATIONS BY BALANCED RECALL G-MEAN")
                print(f"{'='*80}\n")
                results_df_sorted_f1 = results_df.sort_values('balanced_recall_gmean', ascending=False)
                print(results_df_sorted_f1[display_cols].head(10).to_string(index=False))
            
            # Save sorted results
            sorted_path = f"{RESULTS_DIR}/metrics_sorted_by_auc.csv"
            results_df_sorted.to_csv(sorted_path, index=False)
            print(f"\n✓ Sorted results saved to: {sorted_path}")
            
        else:
            print(f"\n⚠️ No AUC column found. Available columns: {list(results_df.columns)}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
