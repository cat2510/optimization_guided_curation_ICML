# Quick OCT Training on Existing Undersampled Datasets

## Overview

`train_oct_on_undersampled.py` trains OCT models on pre-generated undersampled datasets, **skipping the time-consuming k-center matching step**.

## Usage

```bash
cd /Users/cat2510/my_projects
python train_oct_on_undersampled.py
```

## What It Does

1. ✅ **Finds all undersampled datasets** in `kcenter_hyperparameter_search_results/`
2. ✅ **Loads each dataset** (skips k-center matching)
3. ✅ **Trains OCT model** with hyperparameter tuning
4. ✅ **Evaluates on test set** 
5. ✅ **Logs ALL 14 metrics** from `evaluate_binary_oct()`
6. ✅ **Ranks by AUC** (primary metric)

## Speed Comparison

| Script | Time | What It Does |
|--------|------|--------------|
| `kcenter_hyperparameter_search.py` | ~5-16 hours | K-center matching + OCT training |
| `train_oct_on_undersampled.py` | ~1-3 hours | OCT training only (uses existing datasets) |

**Use this script** when you already have undersampled datasets and just want to:
- Re-train with different OCT hyperparameters
- Re-evaluate with different metrics
- Quickly compare configurations

## Output

All results saved to `./oct_training_results/`:

### 1. `metrics_master.csv`
Complete results table with ALL metrics for each configuration:

**Configuration columns:**
- `config_name`: e.g., `cw_boundary_pool_True_seed_smart`
- `case_weighting`: None, boundary, uncertainty, density_inverse
- `use_adaptive_pool`: True/False
- `seed_method`: smart, centroid, density, random
- `n_train_samples`, `n_train_minority`, `n_train_majority`

**OCT hyperparameters:**
- `best_depth`, `best_minbucket`, `best_cp`

**ALL 14 metrics from `evaluate_binary_oct()`:**
- `auc` - Area Under ROC Curve ⭐ (primary ranking metric)
- `pr_auc` - Precision-Recall AUC
- `optimal_f1` - Best F1 score
- `f1_threshold` - Threshold for optimal F1
- `sensitivity_f1` - Recall at F1 threshold
- `specificity_f1` - Specificity at F1 threshold
- `sensitivity_default` - Recall at default threshold
- `specificity_default` - Specificity at default threshold
- `balanced_threshold_gmean` - G-mean optimal threshold
- `balanced_recall_gmean` - Recall at G-mean threshold
- `balanced_specificity_gmean` - Specificity at G-mean threshold
- `balanced_threshold_minside` - Min-side optimal threshold
- `balanced_recall_minside` - Recall at min-side threshold
- `balanced_specificity_minside` - Specificity at min-side threshold

### 2. `metrics_sorted_by_auc.csv`
Same as above, but sorted by AUC (highest first) for easy viewing.

### 3. Console Output
Prints:
- **Top 10 configurations by AUC**
- **Top 10 configurations by Optimal F1**

## Example Output

```
================================================================================
TOP 10 CONFIGURATIONS BY AUC
================================================================================

config_name                              auc     pr_auc  optimal_f1  sensitivity_f1  specificity_f1
cw_boundary_pool_True_seed_smart        0.8234  0.7891  0.7456      0.8123          0.7234
cw_uncertainty_pool_True_seed_smart     0.8201  0.7823  0.7398      0.8089          0.7189
cw_None_pool_True_seed_smart            0.8156  0.7756  0.7312      0.7998          0.7123
...

================================================================================
TOP 10 CONFIGURATIONS BY OPTIMAL F1
================================================================================

config_name                              auc     pr_auc  optimal_f1  sensitivity_f1  specificity_f1
cw_boundary_pool_True_seed_smart        0.8234  0.7891  0.7456      0.8123          0.7234
cw_uncertainty_pool_True_seed_centroid  0.8189  0.7845  0.7423      0.8056          0.7201
...
```

## When to Use Each Script

### Use `kcenter_hyperparameter_search.py` when:
- ❌ You DON'T have undersampled datasets yet
- ✅ You want to test new k-center matching configurations
- ✅ You want to change: case_weighting, adaptive_pool, seed_method

### Use `train_oct_on_undersampled.py` when:
- ✅ You ALREADY have undersampled datasets
- ✅ You just want to re-train OCT models (faster)
- ✅ You want to change only OCT hyperparameters (depths, minbuckets, cps)
- ✅ You want to re-evaluate with different metrics

## Analysis Tips

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('oct_training_results/metrics_sorted_by_auc.csv')

# Compare case weighting methods
results.groupby('case_weighting')[['auc', 'optimal_f1', 'sensitivity_f1']].mean().sort_values('auc', ascending=False)

# Compare adaptive vs fixed pool
results.groupby('use_adaptive_pool')[['auc', 'optimal_f1', 'sensitivity_f1']].mean()

# Compare seed methods
results.groupby('seed_method')[['auc', 'optimal_f1', 'sensitivity_f1']].mean().sort_values('auc', ascending=False)

# Find best overall configuration
best = results.iloc[0]
print(f"Best config: {best['config_name']}")
print(f"  AUC: {best['auc']:.4f}")
print(f"  Optimal F1: {best['optimal_f1']:.4f}")
print(f"  Sensitivity: {best['sensitivity_f1']:.4f}")
print(f"  Specificity: {best['specificity_f1']:.4f}")

# Heatmap: Case weighting vs Seed method
pivot = results.pivot_table(
    values='auc',
    index='case_weighting',
    columns='seed_method',
    aggfunc='mean'
)
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.75, vmax=0.85)
plt.title('AUC: Case Weighting vs Seed Method')
plt.tight_layout()
plt.savefig('auc_heatmap.png', dpi=300)
```

## Troubleshooting

### No datasets found
```
ERROR: No undersampled datasets found in kcenter_hyperparameter_search_results/
```
**Solution**: Run `kcenter_hyperparameter_search.py` first to generate datasets, or check the `UNDERSAMPLED_DIR` path.

### Column mismatch errors
**Solution**: Make sure the undersampled CSVs have the same columns as the original training data.

### Memory issues
**Solution**: Process datasets one at a time (already done), or reduce OCT hyperparameter grid.
