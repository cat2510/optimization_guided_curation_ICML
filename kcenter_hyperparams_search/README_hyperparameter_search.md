# K-Center Hyperparameter Search

## Overview

`kcenter_hyperparameter_search.py` performs a comprehensive grid search over k-center matching hyperparameters to find the optimal configuration for creating undersampled training datasets.

## Hyperparameters Tested

The script tests all combinations of:

1. **Case Weighting** (4 options):
   - `None`: Uniform weights (all cases treated equally)
   - `"boundary"`: Weight by inverse distance to nearest majority sample
   - `"density_inverse"`: Weight by inverse local density

2. **Adaptive Pool** (2 options):
   - `True`: Use adaptive pool size with stopping criteria (may select < M)
   - `False`: Use fixed k-center with exactly M candidates

3. **Seed Method** (4 options):
   - `"smart"`: Minority-biased (argmin mean distance to cases) - **RECOMMENDED**
   - `"centroid"`: Closest to majority centroid
   - `"density"`: Highest local density
   - `"random"`: Random initialization

**Total configurations**: 4 × 2 × 4 = **32 configurations**

## Workflow

For each configuration:
1. **Run per-leaf k-center matching** to select diverse controls
2. **Build undersampled dataset** with all minority + selected majority samples
3. **Train OCT model** on the undersampled data
4. **Evaluate on test set** and save metrics

## Usage

### Basic Usage

```bash
cd /Users/
python kcenter_hyperparameter_search.py
```

### Requirements

- Pre-computed distance files:
  - `./precomputed_distances/distances_majority_minority.h5`
  - Control-control distances will be computed per-leaf if not cached
- Original Parquet dataset

### Configuration

Edit these constants in the script to customize:

```python
# Matching ratio (1:k)
MATCHING_RATIO = 1  # Can change to 2, 3, etc. for 1:k matching

# Hyperparameter grid
HYPERPARAMETER_GRID = {
    'case_weighting': [None, "boundary", "density_inverse"],
    'use_adaptive_pool': [True, False],
    'seed_method': ["smart", "centroid", "density", "random"],
}

# OCT hyperparameters
OCT_DEPTHS = [7, 9]
OCT_MINBUCKETS = [50, 100, 120, 150]
OCT_CPS = [0.00001, 0.0001, 0.001, 0.01]
```

## Output

### Files Created

All results are saved to `./kcenter_hyperparameter_search_results/`:

1. **`metrics_master.csv`**: Master table with all configurations and their metrics
   - Configuration details (case_weighting, use_adaptive_pool, seed_method)
   - Training data statistics (n_train_samples, n_train_minority, n_train_majority)
   - Test metrics (accuracy, recall, precision, F1, AUC, etc.)
   - Best OCT hyperparameters

2. **`undersampled_cw_{X}_pool_{Y}_seed_{Z}.csv`**: Undersampled datasets for each config
   - Can be reused for further analysis without re-running matching

3. **Model outputs**: OCT visualizations and predictions per configuration

### Example Output Structure

```
kcenter_hyperparameter_search_results/
├── metrics_master.csv
├── undersampled_cw_None_pool_True_seed_smart.csv
├── undersampled_cw_None_pool_True_seed_centroid.csv
├── undersampled_cw_None_pool_True_seed_density.csv
├── undersampled_cw_None_pool_True_seed_random.csv
├── undersampled_cw_boundary_pool_True_seed_smart.csv
├── ... (32 total configurations)
```

## Interpreting Results

### Best Configuration

The script prints a ranked table at the end:

```
Top 5 configurations by test accuracy:
                                  config_name  accuracy  recall  precision    f1
0    cw_boundary_pool_True_seed_smart        0.9234   0.8567     0.7891  0.8213
1    cw_uncertainty_pool_True_seed_smart     0.9201   0.8489     0.7823  0.8142
...
```

### Key Metrics

- **Accuracy**: Overall correctness
- **Recall**: Sensitivity to minority class (high-cost patients)
- **Precision**: Positive predictive value
- **F1**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Comparison Across Dimensions

Load `metrics_master.csv` to compare:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv('kcenter_hyperparameter_search_results/metrics_master.csv')

# Compare case weighting methods
results.groupby('case_weighting')[['accuracy', 'recall', 'f1']].mean()

# Compare adaptive vs fixed pool
results.groupby('use_adaptive_pool')[['accuracy', 'recall', 'f1']].mean()

# Compare seed methods
results.groupby('seed_method')[['accuracy', 'recall', 'f1']].mean()

# Heatmap of interactions
pivot = results.pivot_table(
    values='f1',
    index='case_weighting',
    columns='seed_method',
    aggfunc='mean'
)
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis')
plt.title('F1 Score: Case Weighting vs Seed Method')
plt.tight_layout()
plt.savefig('f1_heatmap.png', dpi=300)
```

## Tips

1. **Start with subset**: Test a few configs first by modifying `HYPERPARAMETER_GRID`
2. **Monitor progress**: Results are saved incrementally to `metrics_master.csv`
3. **Reuse datasets**: Once undersampled CSVs are created, you can train additional models without re-running matching
4. **Parallel execution**: For faster results, manually split configs into multiple scripts

## Troubleshooting

### Memory Issues

If you run out of memory:
- Reduce batch size in `precompute_leaf_dnn_memmap()` (line 106)
- Process leaves sequentially (already done)
- Close Spark session after loading data
