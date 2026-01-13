#!/usr/bin/env python3
"""
K-Center Matching Hyperparameter Search
========================================
This script performs a grid search over k-center matching hyperparameters:
- Case weighting: None, "boundary", "uncertainty", "density_inverse"
- Adaptive pool: True/False
- Seed method: "smart", "centroid", "density", "random"

For each configuration, it:
1. Runs per-leaf k-center matching to create undersampled training data
2. Trains an OCT model on the undersampled data
3. Evaluates and saves results
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
MODEL_PATH = "saved_models/oct_stratifier_model.pkl"
DF_WITH_LEAVES_PATH = "saved_models/df_with_leaves.csv"
PN_H5_PATH = "./precomputed_distances/distances_majority_minority.h5"
RESULTS_DIR = "./kcenter_hyperparameter_search_results"
DNN_OUT_DIR = "./precomputed_distances/leaf_dnn_oct_stratifier"

# Target column
TARGET_COL = "highcost_gt_200000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_oct_probabilities(model, preprocessor, feature_names, leaf_cases, feature_cols):
    """Get OCT predicted probabilities for uncertainty weighting."""
    X_cases = leaf_cases[feature_cols].copy()
    X_cases_transformed = preprocessor.transform(X_cases)
    X_cases_df = pd.DataFrame(X_cases_transformed, columns=feature_names)
    predicted_probs = model.predict_proba(X_cases_df)
    return predicted_probs


def run_kcenter_matching_for_config(
    df_with_leaves,
    target_col,
    feature_cols,
    pn_h5_path,
    matching_ratio,
    case_weighting,
    use_adaptive_pool,
    seed_method,
    model=None,
    preprocessor=None,
    feature_names=None,
    CAT_COLUMNS=None,
    TRUE_NUM_COLUMNS=None,
    COST_COLUMNS=None,
):
    """
    Run k-center matching for all leaves with specified hyperparameters.
    
    Returns:
        dict: all_leaf_results dictionary with matching results for each leaf
    """
    print(f"\n{'='*80}")
    print(f"CONFIGURATION:")
    print(f"  case_weighting: {case_weighting}")
    print(f"  use_adaptive_pool: {use_adaptive_pool}")
    print(f"  seed_method: {seed_method}")
    print(f"  matching_ratio: 1:{matching_ratio}")
    print(f"{'='*80}\n")
    
    # Get unique leaves
    unique_leaves = sorted(df_with_leaves['leaf_assignment'].unique())
    print(f"Total leaves to process: {len(unique_leaves)}\n")
    
    # Store results for all leaves
    all_leaf_results = {}
    
    for leaf_id in unique_leaves:
        print(f"\n{'='*80}")
        print(f"PROCESSING LEAF {int(leaf_id)}")
        print(f"{'='*80}")
        
        # Extract data for this leaf
        leaf_df = df_with_leaves[df_with_leaves['leaf_assignment'] == leaf_id].copy()
        leaf_cases = leaf_df[leaf_df[target_col] == 1].copy()
        leaf_controls = leaf_df[leaf_df[target_col] == 0].copy()
        
        n_cases = len(leaf_cases)
        n_controls = len(leaf_controls)
        
        print(f"\nLeaf {int(leaf_id)} Statistics:")
        print(f"  Cases (minority): {n_cases:,}")
        print(f"  Controls (majority): {n_controls:,}")
        print(f"  Ratio: {n_controls/n_cases if n_cases > 0 else float('inf'):.2f}:1")
        
        if n_cases == 0:
            print(f"  ⚠️ No cases in leaf {int(leaf_id)}, skipping...")
            continue
        
        if n_controls == 0:
            print(f"  ⚠️ No controls in leaf {int(leaf_id)}, skipping...")
            continue
        
        # Preprocess controls to get feature matrix
        exclude_cols_leaf = ["cost_stratum_2018"] + COST_COLUMNS + ["leaf_assignment", "predicted_cost_stratum"]
        drop_cols_leaf = ['ENROLID', target_col] + exclude_cols_leaf
        feature_cols_leaf = [c for c in leaf_controls.columns if c not in drop_cols_leaf]
        
        numeric_cols_leaf = leaf_controls[feature_cols_leaf].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols_leaf = leaf_controls[feature_cols_leaf].select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Impute and preprocess
        leaf_controls_preprocessed = leaf_controls[feature_cols_leaf].copy()
        if numeric_cols_leaf:
            imputer = SimpleImputer(strategy='median')
            leaf_controls_preprocessed[numeric_cols_leaf] = imputer.fit_transform(leaf_controls_preprocessed[numeric_cols_leaf])
        if categorical_cols_leaf:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            leaf_controls_preprocessed[categorical_cols_leaf] = cat_imputer.fit_transform(leaf_controls_preprocessed[categorical_cols_leaf])
        
        bin_feats_leaf = get_bin_flag_columns(leaf_controls_preprocessed)
        num_feats_leaf = [c for c in numeric_cols_leaf if c not in bin_feats_leaf]
        
        preprocessor_leaf = get_preprocessor(
            df=leaf_controls_preprocessed,
            categorical_cols=categorical_cols_leaf,
            numeric_cols=num_feats_leaf,
            verbose=False
        )
        X_majority_leaf = preprocessor_leaf.fit_transform(leaf_controls_preprocessed)
        
        print(f"\n  Preprocessing:")
        print(f"    Features: {len(feature_cols_leaf)}")
        print(f"    Preprocessed shape: {X_majority_leaf.shape}")
        
        # Precompute control-control distances for this leaf
        print(f"\n  Loading/precomputing control-control distances...")
        majority_enrolids_leaf = leaf_controls["ENROLID"].to_numpy()
        
        dnn_matrix_npy, dnn_enrolids_npy = precompute_leaf_dnn_memmap(
            X_majority_leaf=X_majority_leaf,
            majority_enrolids_leaf=majority_enrolids_leaf,
            out_dir=DNN_OUT_DIR,
            leaf_id=str(int(leaf_id)),
            batch_size=750,
        )
        print(f"    ✓ Control-control distances ready")
        
        # Handle edge case: More cases than controls
        if n_cases > n_controls:
            print(f"\n  ⚠️ WARNING: More cases ({n_cases:,}) than controls ({n_controls:,}) in leaf {int(leaf_id)}")
            print(f"     Strategy: SKIP k-center matching - keep ALL minority and ALL majority as-is")
            
            all_leaf_results[leaf_id] = {
                'n_cases_total': n_cases,
                'n_cases_matched': 0,
                'n_controls': n_controls,
                'M': 0,
                'selected_control_enrolids': leaf_controls["ENROLID"].to_numpy(),
                'match_costs': np.array([]),
                'candidate_majority_enrolids': leaf_controls["ENROLID"].to_numpy(),
                'case_to_control_map': {},
                'skipped_reason': 'more_cases_than_controls'
            }
            
            print(f"    ✓ Leaf will use ALL {n_controls:,} controls (no matching)")
            continue
        
        # Normal case: n_controls >= n_cases
        M_leaf = max(n_cases * matching_ratio, min(8000, n_controls))
        
        print(f"\n  K-Center Configuration:")
        print(f"    M (candidate pool size): {M_leaf:,} / {n_controls:,} ({M_leaf/n_controls*100:.1f}%)")
        print(f"    Cases to match: {n_cases:,}")
        print(f"    Seed method: {seed_method}")
        print(f"    Adaptive pool: {use_adaptive_pool}")
        print(f"    Case weighting: {case_weighting}")
        
        # Get OCT probabilities if needed for uncertainty weighting
        predicted_probs = None
        if case_weighting == "uncertainty" and model is not None:
            print(f"\n  Computing OCT probabilities for uncertainty weighting...")
            predicted_probs = get_oct_probabilities(
                model, preprocessor, feature_names, leaf_cases, feature_cols
            )
            print(f"    ✓ Got probabilities for {len(predicted_probs)} cases")
        
        # Run two-stage k-center matching
        print(f"\n  Running two-stage k-center matching (1:{matching_ratio})...")
        try:
            out = two_stage_kcenter_then_match(
                leaf_controls_enrolids=leaf_controls["ENROLID"].to_numpy(),
                leaf_cases_enrolids=leaf_cases["ENROLID"].to_numpy(),
                leaf_nn_matrix_npy=dnn_matrix_npy,
                leaf_nn_enrolids_npy=dnn_enrolids_npy,
                pn_h5_path=pn_h5_path,
                M=M_leaf,
                use_adaptive_pool=use_adaptive_pool,
                tau=None,  # Auto-compute from 95th percentile
                plateau_eps=0.01,
                force_nearest_per_case=True,
                force_topm=1,
                assignment_topk_start=None,  # Exact matching
                seed_method=seed_method,
                matching_ratio=matching_ratio,
                X_majority_leaf=X_majority_leaf,
                case_weighting=case_weighting,
                predicted_probs=predicted_probs,
            )
            
            # Extract results
            selected_control_enrolids = out["selected_control_enrolids"]
            all_match_costs = out["match_costs"]
            candidate_majority_enrolids = out["candidate_majority_enrolids"]
            case_to_control_map = out["case_to_control_map"]
            
            # Store results
            all_leaf_results[leaf_id] = {
                'n_cases_total': n_cases,
                'n_cases_matched': n_cases,
                'n_controls': n_controls,
                'M': M_leaf,
                'matching_ratio': matching_ratio,
                'selected_control_enrolids': selected_control_enrolids,
                'match_costs': all_match_costs,
                'candidate_majority_enrolids': candidate_majority_enrolids,
                'case_to_control_map': case_to_control_map,
                'skipped_reason': None
            }
            
            print(f"    ✓ Matching complete!")
            print(f"    Cases matched: {n_cases:,}")
            print(f"    Total selected controls: {len(selected_control_enrolids):,} (unique: {len(set(selected_control_enrolids)):,})")
            print(f"    Mean matching cost: {all_match_costs.mean():.4f}")
            
        except Exception as e:
            print(f"    ✗ ERROR in leaf {int(leaf_id)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n\n{'='*80}")
    print("K-CENTER MATCHING COMPLETE FOR ALL LEAVES")
    print(f"{'='*80}")
    print(f"\nSuccessfully processed {len(all_leaf_results)}/{len(unique_leaves)} leaves")
    
    return all_leaf_results


def build_undersampled_dataset(df_with_leaves, all_leaf_results, target_col, matching_ratio):
    """Build undersampled training dataset from matching results."""
    print(f"\n{'='*80}")
    print("BUILDING UNDERSAMPLED TRAINING DATASET")
    print(f"{'='*80}\n")
    
    # Collect all minority samples (keep ALL)
    all_minority = df_with_leaves[df_with_leaves[target_col] == 1].copy()
    print(f"✓ Collected all minority samples: {len(all_minority):,}")
    
    # Collect majority samples from each leaf
    selected_majority_enrolids = []
    
    for leaf_id, result in all_leaf_results.items():
        if result.get('skipped_reason') == 'more_cases_than_controls':
            # For imbalanced leaves, keep all controls
            selected_majority_enrolids.extend(result['selected_control_enrolids'].tolist())
        else:
            # For matched leaves, use the selected controls
            selected_majority_enrolids.extend(result['selected_control_enrolids'].tolist())
    
    # Handle control reuse in 1:k matching
    if matching_ratio > 1:
        print(f"\n⚠️ 1:{matching_ratio} matching: Controls may be reused")
        print(f"   Total selections: {len(selected_majority_enrolids):,}")
        print(f"   Unique controls: {len(set(selected_majority_enrolids)):,}")
        print(f"   Reused controls: {len(selected_majority_enrolids) - len(set(selected_majority_enrolids)):,}")
    
    # Get unique majority samples
    unique_majority_enrolids = list(set(selected_majority_enrolids))
    selected_majority = df_with_leaves[
        (df_with_leaves[target_col] == 0) & 
        (df_with_leaves['ENROLID'].isin(unique_majority_enrolids))
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
    print(f"   Test accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   Test recall: {metrics.get('recall', 0):.4f}")
    print(f"   Test precision: {metrics.get('precision', 0):.4f}")
    print(f"   Test F1: {metrics.get('f1', 0):.4f}")
    
    return balanced_model, balanced_params, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print("K-CENTER HYPERPARAMETER SEARCH")
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
    spark = SparkSession.builder.appName("KCenterSearch").getOrCreate()
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
    
    # Load OCT stratifier model
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        preprocessor = saved_data['preprocessor']
        feature_names = saved_data['feature_names']
    print(f"✓ Loaded OCT stratifier from: {MODEL_PATH}")
    
    # Load dataframe with leaves
    df_with_leaves = pd.read_csv(DF_WITH_LEAVES_PATH)
    print(f"✓ Loaded df_with_leaves from: {DF_WITH_LEAVES_PATH}")
    print(f"  Shape: {df_with_leaves.shape}")
    print(f"  Unique leaves: {df_with_leaves['leaf_assignment'].nunique()}")
    
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
            # Run k-center matching
            all_leaf_results = run_kcenter_matching_for_config(
                df_with_leaves=df_with_leaves,
                target_col=TARGET_COL,
                feature_cols=feature_cols,
                pn_h5_path=PN_H5_PATH,
                matching_ratio=MATCHING_RATIO,
                case_weighting=config['case_weighting'],
                use_adaptive_pool=config['use_adaptive_pool'],
                seed_method=config['seed_method'],
                model=model,
                preprocessor=preprocessor,
                feature_names=feature_names,
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                COST_COLUMNS=COST_COLUMNS,
            )
            
            # Build undersampled dataset
            undersampled_training_data = build_undersampled_dataset(
                df_with_leaves=df_with_leaves,
                all_leaf_results=all_leaf_results,
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
            row = {
                'config_name': config_name,
                'case_weighting': config['case_weighting'],
                'use_adaptive_pool': config['use_adaptive_pool'],
                'seed_method': config['seed_method'],
                'matching_ratio': MATCHING_RATIO,
                'n_train_samples': len(undersampled_training_data),
                'n_train_minority': (undersampled_training_data[TARGET_COL] == 1).sum(),
                'n_train_majority': (undersampled_training_data[TARGET_COL] == 0).sum(),
                **metrics,
                **balanced_params,
            }
            
            all_results.append(row)
            
            # Save incrementally
            pd.DataFrame([row]).to_csv(
                metrics_master_path,
                mode='a',
                header=not os.path.exists(metrics_master_path),
                index=False,
            )
            
            print(f"\n✓ Configuration {idx}/{len(all_combinations)} complete")
            print(f"  Test Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Test Recall: {metrics.get('recall', 0):.4f}")
            print(f"  Test Precision: {metrics.get('precision', 0):.4f}")
            
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
        if 'accuracy' in results_df.columns:
            results_df_sorted = results_df.sort_values('accuracy', ascending=False)
            print(f"\nTop 5 configurations by test accuracy:")
            print(results_df_sorted[['config_name', 'accuracy', 'recall', 'precision', 'f1']].head())
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
