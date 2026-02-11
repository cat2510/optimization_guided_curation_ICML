#!/usr/bin/env python3
"""
Recompute sensitivity_pool_size.csv from existing prediction files
and analyze split features across different pool sizes.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import glob

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)

from public.model_IAI import train_test_split_enrol

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to original data (needed to get y_test)
DATA_PATH = "msk_2017_18_full.parquet"
TRAIN_TEST_SEED = 123
target_col = "top_2_pct_cost_2018"

# Results directory
base_dir = "./sensitivity_quota_cfg_pool_size_all_cost_features_150_minbucket"
RESULTS_DIR = os.path.join(base_dir, "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "sensitivity_pool_size.csv")

# ============================================================================
# LOAD TEST SET LABELS
# ============================================================================

print("Loading data to get test set labels...")
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DataLoad").getOrCreate()
df_msk_spark = spark.read.format("parquet").load(DATA_PATH)
df_og = df_msk_spark.toPandas()

# Create target if needed
if target_col not in df_og.columns and "annual_cost_2018_deflated" in df_og.columns:
    threshold = df_og["annual_cost_2018_deflated"].quantile(0.98)
    df_og[target_col] = (df_og["annual_cost_2018_deflated"] >= threshold).astype(int)
    print(f"Created {target_col} using threshold ${threshold:,.2f}")

# Split data to get test set (must match original split)
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

y_test = test_pd[target_col].values
print(f"✓ Loaded test set: {len(y_test):,} samples")
print(f"  Test target distribution: {pd.Series(y_test).value_counts().to_dict()}")

# ============================================================================
# FIND ALL PREDICTION FILES
# ============================================================================

print(f"\n{'='*80}")
print("FINDING PREDICTION FILES")
print(f"{'='*80}\n")

# Find all directories matching pattern (under base_dir)
pattern = "pool_size_M*"
matching_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("pool_size_M")]

print(f"Found {len(matching_dirs)} directories matching pattern")

all_results = []
all_splits_data = []

for dir_name in sorted(matching_dirs):
    # Extract M from directory name
    match = re.search(r'M(\d+)', dir_name)
    if not match:
        print(f"  ⚠️  Could not extract M from {dir_name}, skipping")
        continue
    
    M = int(match.group(1))
    print(f"\n  Processing {dir_name} (M={M:,})")
    
    # Find prediction file (under base_dir)
    pred_dir = os.path.join(base_dir, dir_name, "predictions")
    if not os.path.exists(pred_dir):
        print(f"    ⚠️  No predictions directory found, skipping")
        continue
    
    pred_files = glob.glob(os.path.join(pred_dir, "oct_predictions_*.csv"))
    if len(pred_files) == 0:
        print(f"    ⚠️  No prediction files found, skipping")
        continue
    
    if len(pred_files) > 1:
        print(f"    ⚠️  Multiple prediction files found, using first: {pred_files[0]}")
    
    pred_file = pred_files[0]
    print(f"    ✓ Found prediction file: {os.path.basename(pred_file)}")
    
    # Load predictions
    try:
        # Read CSV - don't specify dtype upfront to avoid issues with malformed values
        pred_df = pd.read_csv(pred_file, na_values=['', 'NA', 'N/A', 'nan', 'NaN'])
        print(f"    ✓ Loaded predictions: {len(pred_df):,} rows")
        
        # Check if we have predicted_proba column
        if "predicted_proba" not in pred_df.columns:
            print(f"    ⚠️  No 'predicted_proba' column found, skipping metrics")
            continue
        
        # Convert predicted_proba to float, handling any string issues (including trailing periods)
        # First, try to clean the column if it's a string
        if pred_df["predicted_proba"].dtype == 'object':
            # Remove trailing periods and whitespace
            pred_df["predicted_proba"] = pred_df["predicted_proba"].astype(str).str.rstrip('.').str.strip()
        
        y_proba = pd.to_numeric(pred_df["predicted_proba"], errors='coerce').values
        
        # Check for NaN values
        if np.isnan(y_proba).any():
            n_nan = np.isnan(y_proba).sum()
            print(f"    ⚠️  Found {n_nan} NaN values in predicted_proba, removing them")
            valid_mask = ~np.isnan(y_proba)
            y_proba = y_proba[valid_mask]
            if len(y_proba) == 0:
                print(f"    ✗ No valid predictions after removing NaN, skipping")
                continue
        
        # Align with test set (prediction files might have different order)
        # If ENROLID is present, use it to align
        if "ENROLID" in pred_df.columns:
            # Create mapping from ENROLID to index in test set
            test_enrolid_to_idx = {eid: idx for idx, eid in enumerate(test_pd["ENROLID"].values)}
            pred_enrolids = pred_df["ENROLID"].values
            
            # Get indices in test set for each prediction
            aligned_indices = [test_enrolid_to_idx.get(eid, -1) for eid in pred_enrolids]
            valid_mask = np.array([idx >= 0 for idx in aligned_indices])
            
            if valid_mask.sum() < len(y_test) * 0.9:
                print(f"    ⚠️  Only {valid_mask.sum()}/{len(y_test)} predictions align with test set, skipping")
                continue
            
            y_proba_aligned = np.full(len(y_test), np.nan)
            y_proba_aligned[np.array(aligned_indices)[valid_mask]] = y_proba[valid_mask]
            y_test_aligned = y_test
            
            # Use only valid predictions
            valid_idx = ~np.isnan(y_proba_aligned)
            y_proba_final = y_proba_aligned[valid_idx]
            y_test_final = y_test_aligned[valid_idx]
            
            print(f"    ✓ Aligned {valid_idx.sum():,}/{len(y_test):,} predictions with test set")
        else:
            # Assume same order (risky but might work)
            if len(pred_df) != len(y_test):
                print(f"    ⚠️  Prediction length ({len(pred_df)}) != test length ({len(y_test)}), skipping")
                continue
            y_proba_final = y_proba
            y_test_final = y_test
            print(f"    ✓ Using predictions in order (no ENROLID alignment)")
        
        # Compute metrics
        auc = roc_auc_score(y_test_final, y_proba_final)
        pr_auc = average_precision_score(y_test_final, y_proba_final)
        
        # Get best MCC threshold
        from public.model_IAI import best_mcc_threshold
        mcc_result = best_mcc_threshold(y_test_final, y_proba_final)
        best_mcc = mcc_result["mcc"]
        best_mcc_threshold = mcc_result["threshold"]
        
        # Get balanced threshold (G-mean)
        from public.model_IAI import best_balanced_threshold
        balanced_result = best_balanced_threshold(y_test_final, y_proba_final)
        balanced_recall_gmean = balanced_result["gmean_opt"]["recall"]
        balanced_specificity_gmean = balanced_result["gmean_opt"]["specificity"]
        
        # Extract model parameters from filename or directory
        # Try to extract from filename: oct_predictions_M{M}_{depth}_{minbucket}_{cp}.csv
        filename = os.path.basename(pred_file)
        param_match = re.search(r'M\d+_(\d+)_(\d+)_([\d.]+)', filename)
        if param_match:
            try:
                depth = int(param_match.group(1))
                minbucket = int(param_match.group(2))
                # Handle potential trailing periods or other issues
                cp_str = param_match.group(3).rstrip('.')
                cp = float(cp_str)
            except (ValueError, AttributeError) as e:
                print(f"    ⚠️  Could not parse parameters from filename: {e}")
                depth = minbucket = cp = None
        else:
            depth = minbucket = cp = None
        
        result_row = {
            'M': M,
            'M_pct_of_controls': None,  # Will compute if we know n_controls
            'adaptive_pool': False,  # Assume False for now
            'best_depth': depth,
            'best_minbucket': minbucket,
            'best_cp': cp,
            'auc': auc,
            'pr_auc': pr_auc,
            'best_mcc': best_mcc,
            'best_mcc_threshold': best_mcc_threshold,
            'balanced_recall_gmean': balanced_recall_gmean,
            'balanced_specificity_gmean': balanced_specificity_gmean,
        }
        
        all_results.append(result_row)
        print(f"    ✓ Computed metrics: PR-AUC={pr_auc:.4f}, AUC={auc:.4f}, MCC={best_mcc:.4f}")
        
    except Exception as e:
        print(f"    ✗ Error processing {dir_name}: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # Load split file (under base_dir)
    split_files = glob.glob(os.path.join(base_dir, dir_name, "oct_tree_*_splits.csv"))
    if len(split_files) > 0:
        split_file = split_files[0]
        try:
            splits_df = pd.read_csv(split_file)
            splits_df['M'] = M
            all_splits_data.append(splits_df)
            print(f"    ✓ Loaded splits: {len(splits_df)} nodes")
        except Exception as e:
            print(f"    ⚠️  Error loading splits: {e}")

# ============================================================================
# COMPUTE M_PCT_OF_CONTROLS
# ============================================================================

if all_results:
    # Estimate n_controls from M values (M_max should be around n_controls//2)
    M_max = max([r['M'] for r in all_results])
    estimated_n_controls = M_max * 2
    
    for result in all_results:
        if result['M'] is not None:
            result['M_pct_of_controls'] = (result['M'] / estimated_n_controls * 100) if estimated_n_controls > 0 else None

# ============================================================================
# SAVE RESULTS CSV
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}\n")

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('M')
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved results to: {OUTPUT_CSV}")
    print(f"  Total configurations: {len(results_df)}")
    print(f"\nResults summary:")
    print(results_df[['M', 'M_pct_of_controls', 'pr_auc', 'auc', 'best_mcc']].to_string(index=False))
else:
    print("⚠️  No results to save!")

# ============================================================================
# CREATE PLOTS
# ============================================================================

if len(all_results) > 0:
    print(f"\n{'='*80}")
    print("CREATING PLOTS")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(all_results).sort_values('M')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: PR-AUC vs M
    axes[0].plot(results_df['M'], results_df['pr_auc'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Pool Size (M)', fontsize=12)
    axes[0].set_ylabel('PR-AUC', fontsize=12)
    axes[0].set_title('PR-AUC vs Pool Size', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='plain', axis='x')
    
    # Add value labels
    for _, row in results_df.iterrows():
        axes[0].annotate(f'{row["pr_auc"]:.3f}', 
                        (row['M'], row['pr_auc']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', fontsize=8)
    
    # Plot 2: AUC vs M
    axes[1].plot(results_df['M'], results_df['auc'], 's-', color='orange', linewidth=2, markersize=8)
    axes[1].set_xlabel('Pool Size (M)', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('AUC vs Pool Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='plain', axis='x')
    
    # Add value labels
    for _, row in results_df.iterrows():
        axes[1].annotate(f'{row["auc"]:.3f}', 
                        (row['M'], row['auc']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "sensitivity_pool_size_extended.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plots to: {plot_path}")
    plt.close()

# ============================================================================
# ANALYZE SPLIT FEATURES
# ============================================================================

if len(all_splits_data) > 0:
    print(f"\n{'='*80}")
    print("ANALYZING SPLIT FEATURES")
    print(f"{'='*80}\n")
    
    all_splits = pd.concat(all_splits_data, ignore_index=True)
    
    # Count unique features used across all trees
    unique_features = all_splits['feature'].dropna().unique()
    feature_counts = all_splits['feature'].value_counts()
    
    print(f"Total unique features used across all trees: {len(unique_features)}")
    print(f"Total split nodes across all trees: {len(all_splits)}")
    print(f"\nTop 20 most frequently used features:")
    print(feature_counts.head(20).to_string())
    
    # Feature usage by M
    print(f"\n{'='*80}")
    print("FEATURE USAGE BY POOL SIZE (M)")
    print(f"{'='*80}")
    for M in sorted(all_splits['M'].unique()):
        splits_M = all_splits[all_splits['M'] == M]
        unique_features_M = splits_M['feature'].dropna().nunique()
        print(f"  M={M:,}: {unique_features_M} unique features, {len(splits_M)} splits")
    
    # Analyze leaf numbers
    # Method 1: From splits file (internal nodes)
    # In a binary tree: n_internal_nodes splits -> n_internal_nodes + 1 leaves
    
    # Method 2: From prediction files (actual leaf assignments)
    # Count unique leaf_assignment values to get actual number of leaves
    
    leaf_stats = []
    for M in sorted(all_splits['M'].unique()):
        splits_M = all_splits[all_splits['M'] == M]
        max_node_id = splits_M['node_id'].max()
        min_node_id = splits_M['node_id'].min()
        n_splits = len(splits_M)
        
        # Try to get actual leaf count from prediction file
        dir_name = f"pool_size_M{M}"
        pred_dir = os.path.join(base_dir, dir_name, "predictions")
        pred_files = glob.glob(os.path.join(pred_dir, "oct_predictions_*.csv"))
        
        n_leaves_from_splits = n_splits + 1  # Binary tree property
        n_leaves_actual = None
        
        if len(pred_files) > 0:
            try:
                pred_df = pd.read_csv(pred_files[0])
                if "leaf_assignment" in pred_df.columns:
                    n_leaves_actual = pred_df["leaf_assignment"].nunique()
            except:
                pass
        
        leaf_stats.append({
            'M': M,
            'max_node_id': max_node_id,
            'min_node_id': min_node_id,
            'n_splits': n_splits,
            'n_leaves_from_splits': n_leaves_from_splits,
            'n_leaves_actual': n_leaves_actual,
        })
    
    leaf_stats_df = pd.DataFrame(leaf_stats)
    
    print(f"\n{'='*80}")
    print("LEAF NUMBER STATISTICS")
    print(f"{'='*80}")
    print(leaf_stats_df.to_string(index=False))
    
    # Use actual leaf counts if available, otherwise use estimated
    if leaf_stats_df['n_leaves_actual'].notna().any():
        leaf_col = 'n_leaves_actual'
        print(f"\nUsing actual leaf counts from prediction files:")
    else:
        leaf_col = 'n_leaves_from_splits'
        print(f"\nUsing estimated leaf counts (n_splits + 1):")
    
    valid_leaves = leaf_stats_df[leaf_col].dropna()
    if len(valid_leaves) > 0:
        print(f"  Min leaves across all M: {valid_leaves.min()}")
        print(f"  Max leaves across all M: {valid_leaves.max()}")
        print(f"  Mean leaves: {valid_leaves.mean():.1f}")
        print(f"  Median leaves: {valid_leaves.median():.1f}")
    
    print(f"\nSplit statistics:")
    print(f"  Min splits: {leaf_stats_df['n_splits'].min()}")
    print(f"  Max splits: {leaf_stats_df['n_splits'].max()}")
    print(f"  Mean splits: {leaf_stats_df['n_splits'].mean():.1f}")
    
    # Save feature analysis
    feature_analysis_path = os.path.join(RESULTS_DIR, "feature_analysis.csv")
    feature_counts.to_frame('count').to_csv(feature_analysis_path)
    print(f"\n✓ Saved feature counts to: {feature_analysis_path}")
    
    leaf_stats_path = os.path.join(RESULTS_DIR, "leaf_stats.csv")
    leaf_stats_df.to_csv(leaf_stats_path, index=False)
    print(f"✓ Saved leaf statistics to: {leaf_stats_path}")
else:
    print(f"\n⚠️  No split files found for analysis")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}\n")
