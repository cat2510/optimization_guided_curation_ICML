# Requirements for Running Two-Stage K-Center Undersampling Pipeline

This document outlines all requirements needed to run the two-stage k-center undersampling pipeline for imbalanced binary classification on tabular datasets. This pipeline performs intelligent undersampling of the majority class using a two-stage approach: k-center selection followed by optimal bipartite matching.

## Python Package Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

### Core Dependencies

The pipeline requires the following Python packages (see `requirements.txt` for versions):

1. **Data Science Libraries**
   - `pandas` - Data manipulation and DataFrame operations
   - `numpy` - Numerical computing and array operations

2. **Machine Learning**
   - `scikit-learn` - Preprocessing, metrics, and utilities
   - `interpretableai` - Optimal Classification Trees (OCT) model training (optional, for model evaluation)

3. **Big Data Processing**
   - `pyspark` - For loading large Parquet data files (optional, can use pandas directly for smaller datasets)

4. **Data Storage**
   - `h5py` - HDF5 file format for storing precomputed distance matrices
   - `pyarrow` - Parquet file support (if using Parquet format)

5. **Optimization**
   - `ortools` - Google OR-Tools for min-cost flow matching in the two-stage k-center algorithm

6. **Utilities**
   - `tqdm` - Progress bars for long-running operations
   - `matplotlib` - Plotting and visualization (optional)
   - `psutil` - Optional resource tracking (gracefully handles if missing)

## Project Structure Requirements

The pipeline expects the following directory structure:

```
project_root/
├── public/
    ├── requirements.txt
│   ├── __init__.py (or ensure public is a package)
│   ├── model_IAI.py
│   ├── precompute_distances.py
│   └── two_stage_kcenter_match.py
└── your_analysis_directory/
    ├── your_notebook.ipynb  # Your analysis notebook
    └── your_data.parquet     # Your dataset (or CSV, etc.)
```

## Data Requirements

1. **Input Data Format**: 
   - Supported formats: Parquet (recommended for large datasets), CSV, or any pandas-readable format
   - For Parquet files, PySpark can be used for efficient loading
   - For CSV or other formats, pandas can be used directly

2. **Required Data Columns**:
   - **ID Column**: A unique identifier column (e.g., `ID`, `patient_id`, `ENROLID`, etc.)
     - Used for tracking samples across train/test splits
     - Must be unique per row
   - **Target Column**: Binary target variable (0/1 or False/True)
     - Class 1 = minority class (cases)
     - Class 0 = majority class (controls)
     - Must be imbalanced (minority class should be < 50% of data)
   - **Feature Columns**: Any combination of:
     - **Categorical columns**: String or integer categorical features (will be one-hot encoded)
     - **Numeric columns**: Continuous numeric features (will be standardized)
     - **Binary flag columns**: Binary 0/1 features (will be passed through without scaling)
     - **Stage/ordinal columns**: Optional ordinal categorical features (if applicable)

3. **Data Preprocessing**:
   - The pipeline automatically identifies column types using helper functions:
     - `get_bin_flag_columns()` - Identifies binary flag columns
     - `get_cat_columns()` - Identifies categorical columns
     - `get_true_num_columns()` - Identifies true numeric columns
   - Missing values are handled automatically via imputation
   - Feature leakage prevention: Exclude any columns that contain future information or are derived from the target

## Module Dependencies

The pipeline imports from the `public` package. These modules provide the core functionality:

1. **`public.model_IAI`**: 
   - **Column identification**: `get_bin_flag_columns()`, `get_cat_columns()`, `get_true_num_columns()`
   - **Data splitting**: `train_test_split_enrol()` - Stratified train/test split by ID
   - **Preprocessing**: `get_preprocessor_with_impute()` - Creates preprocessing pipeline with imputation
   - **Model training** (optional): `finetune_oct()`, `evaluate_binary_oct()` - For training Optimal Classification Trees
   - **Utilities**: `format_time()` - Time formatting helper

2. **`public.precompute_distances`**:
   - **Preprocessing**: `get_preprocessor()` - Creates preprocessor matching model pipeline
   - **Distance computation**: `compute_distances_batched()` - Computes pairwise distances in batches
   - **Storage**: `save_distances_hdf5()` - Saves distance matrices to HDF5 format
   - **Majority-majority distances**: `precompute_leaf_dnn_memmap()` - Precomputes control-control distances for k-center

3. **`public.two_stage_kcenter_match`**:
   - **Main function**: `two_stage_kcenter_then_match()` - Performs two-stage k-center undersampling:
     - Stage 1: Farthest-first k-center selection to identify diverse candidate controls
     - Stage 2: Optimal bipartite matching to assign cases to controls

## System Requirements

- **Python**: 3.8+ (tested with Python 3.11)
- **Memory**: The pipeline processes datasets and computes distance matrices. Recommended:
  - At least 8GB RAM for small-medium datasets (< 100K samples)
  - 16GB+ RAM for large datasets (> 100K samples)
  - More memory may be needed for distance matrix computations on very large datasets
- **Disk Space**: 
  - Precomputed distance matrices can be large (several GB for large datasets)
  - Output directories will be created automatically (e.g., `precomputed_distances/`, `two_stage_kcenter_results/`, etc.)
- **Processing Time**: 
  - Distance computation: O(n²) complexity, can take minutes to hours depending on dataset size
  - K-center matching: Typically faster, depends on candidate pool size M

## Setup Instructions

1. **Clone the repository** (or ensure the `public` directory is accessible)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Load your tabular dataset (Parquet, CSV, or other pandas-readable format)
   - Ensure your dataset has:
     - A unique ID column
     - A binary target column (0/1 or False/True)
     - Feature columns (categorical, numeric, binary flags)
   - Update the data loading code in your notebook to point to your dataset

4. **Configure your analysis**:
   - Set your ID column name (e.g., `ID_COL = "patient_id"`)
   - Set your target column name (e.g., `target_col = "high_cost"`)
   - Identify or let the pipeline auto-detect:
     - Categorical columns
     - Numeric columns  
     - Binary flag columns
   - Exclude any columns that could cause data leakage (e.g., future information, target-derived features)

5. **Run the pipeline**:
   - Execute the preprocessing and undersampling steps
   - The pipeline will:
     1. Precompute distance matrices (minority-majority and majority-majority)
     2. Run two-stage k-center matching to create balanced training set
     3. Optionally train a model on the balanced data

## Usage Example

Here's a minimal example of how to use the pipeline:

```python
import pandas as pd
import numpy as np
import sys
import os

# Add public directory to path
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
from public.model_IAI import *
from public.precompute_distances import *
from public.two_stage_kcenter_match import two_stage_kcenter_then_match

# Load your dataset
df = pd.read_parquet("your_data.parquet")  # or pd.read_csv("your_data.csv")

# Configure your analysis
ID_COL = "patient_id"  # Your unique ID column
target_col = "high_cost"  # Your binary target column

# Auto-detect column types
BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
CAT_COLUMNS = get_cat_columns(df)
TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

# Exclude ID, target, and any leakage columns
exclude_cols = [ID_COL, target_col] + [col for col in df.columns if "future" in col.lower()]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Split data
train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
    df, target_col=target_col, test_size=0.3, random_state=42
)

# Separate cases and controls
cases = train_pd[train_pd[target_col] == 1].copy()
controls = train_pd[train_pd[target_col] == 0].copy()

# Preprocess features
preprocessor = get_preprocessor(
    X=train_pd[feature_cols],
    cat_cols=CAT_COLUMNS,
    num_cols=TRUE_NUM_COLUMNS,
    binary_cols=BIN_FLAG_COLUMNS
)
X_minority = preprocessor.fit_transform(cases[feature_cols])
X_majority = preprocessor.transform(controls[feature_cols])

# Precompute distances (see notebook for full example)
# ... distance computation code ...

# Run two-stage k-center matching
M = len(controls) // 2  # Candidate pool size
matching_result = two_stage_kcenter_then_match(
    leaf_controls_enrolids=controls[ID_COL].values.astype(np.int64),
    leaf_cases_enrolids=cases[ID_COL].values.astype(np.int64),
    # ... other parameters ...
)

# Build balanced dataset
selected_control_ids = matching_result["selected_control_enrolids"]
balanced_train = pd.concat([
    cases,
    controls[controls[ID_COL].isin(selected_control_ids)]
], ignore_index=True)

# ============================================================================
# Train and Evaluate OCT Models: Imbalanced vs. Balanced
# ============================================================================

from public.model_IAI import finetune_oct, evaluate_binary_oct
import time

# Prepare validation and test sets (if not already split)
# Assuming you have X_val, y_val, X_test, y_test from earlier train/test split
# If not, split the test set:
# val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
#     test_pd, target_col=target_col, test_size=0.5, random_state=42
# )
# X_val = val_pd[feature_cols]
# y_val = val_pd[target_col]
# X_test = test_pd[feature_cols]
# y_test = test_pd[target_col]

print("="*80)
print("TRAINING OCT ON IMBALANCED DATASET")
print("="*80)
imbalanced_start = time.perf_counter()

# Train OCT on imbalanced (original) training data
imbalanced_model, imbalanced_params, _, imbalanced_preprocessor, imbalanced_feature_names = finetune_oct(
    X_train=train_pd[feature_cols],
    y_train=train_pd[target_col],
    X_val=X_val,
    y_val=y_val,
    categorical_cols=CAT_COLUMNS,
    numeric_cols=TRUE_NUM_COLUMNS,
    binary_cols=BIN_FLAG_COLUMNS,
    depths=[5, 7, 9],  # Adjust based on your needs
    minbuckets=[50, 100, 150],
    cps=[0.00001, 0.0001, 0.001]
)

# Evaluate imbalanced model
imbalanced_metrics = evaluate_binary_oct(
    imbalanced_model, 
    X_test, 
    y_test, 
    imbalanced_preprocessor, 
    imbalanced_feature_names,
    X_val_df=X_val, 
    y_val=y_val,
    results_dir="results_imbalanced/",
    save_suffix=f"{imbalanced_params[0]}_{imbalanced_params[1]}_{imbalanced_params[2]}"
)

imbalanced_time = time.perf_counter() - imbalanced_start

print(f"\n✓ Imbalanced model training complete ({imbalanced_time:.2f}s)")
print(f"  Best params: depth={imbalanced_params[0]}, minbucket={imbalanced_params[1]}, cp={imbalanced_params[2]}")
print(f"  Test PR-AUC: {imbalanced_metrics['pr_auc']:.4f}")
print(f"  Test AUC: {imbalanced_metrics['auc']:.4f}")

print("\n" + "="*80)
print("TRAINING OCT ON BALANCED DATASET (K-CENTER UNDERSAMPLED)")
print("="*80)
balanced_start = time.perf_counter()

# Train OCT on balanced (undersampled) training data
balanced_model, balanced_params, _, balanced_preprocessor, balanced_feature_names = finetune_oct(
    X_train=balanced_train[feature_cols],
    y_train=balanced_train[target_col],
    X_val=X_val,
    y_val=y_val,
    categorical_cols=CAT_COLUMNS,
    numeric_cols=TRUE_NUM_COLUMNS,
    binary_cols=BIN_FLAG_COLUMNS,
    depths=[5, 7, 9],  # Adjust based on your needs
    minbuckets=[50, 100, 150],
    cps=[0.00001, 0.0001, 0.001]
)

# Evaluate balanced model
balanced_metrics = evaluate_binary_oct(
    balanced_model, 
    X_test, 
    y_test, 
    balanced_preprocessor, 
    balanced_feature_names,
    X_val_df=X_val, 
    y_val=y_val,
    results_dir="results_balanced/",
    save_suffix=f"{balanced_params[0]}_{balanced_params[1]}_{balanced_params[2]}"
)

balanced_time = time.perf_counter() - balanced_start

print(f"\n✓ Balanced model training complete ({balanced_time:.2f}s)")
print(f"  Best params: depth={balanced_params[0]}, minbucket={balanced_params[1]}, cp={balanced_params[2]}")
print(f"  Test PR-AUC: {balanced_metrics['pr_auc']:.4f}")
print(f"  Test AUC: {balanced_metrics['auc']:.4f}")

# ============================================================================
# Compare Results
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: IMBALANCED vs. BALANCED")
print("="*80)
print(f"\nTraining Data:")
print(f"  Imbalanced: {len(train_pd):,} samples ({train_pd[target_col].sum():,} cases, {len(train_pd) - train_pd[target_col].sum():,} controls)")
print(f"  Balanced:    {len(balanced_train):,} samples ({balanced_train[target_col].sum():,} cases, {len(balanced_train) - balanced_train[target_col].sum():,} controls)")

print(f"\nTest Set Performance:")
print(f"  Metric                    Imbalanced    Balanced      Improvement")
print(f"  {'-'*60}")
print(f"  PR-AUC (Primary)          {imbalanced_metrics['pr_auc']:.4f}        {balanced_metrics['pr_auc']:.4f}        {balanced_metrics['pr_auc'] - imbalanced_metrics['pr_auc']:+.4f}")
print(f"  ROC-AUC                   {imbalanced_metrics['auc']:.4f}        {balanced_metrics['auc']:.4f}        {balanced_metrics['auc'] - imbalanced_metrics['auc']:+.4f}")
print(f"  Best MCC                  {imbalanced_metrics['best_mcc']:.4f}        {balanced_metrics['best_mcc']:.4f}        {balanced_metrics['best_mcc'] - imbalanced_metrics['best_mcc']:+.4f}")
print(f"  Recall @ Best MCC         {imbalanced_metrics['recall_mcc']:.4f}        {balanced_metrics['recall_mcc']:.4f}        {balanced_metrics['recall_mcc'] - imbalanced_metrics['recall_mcc']:+.4f}")
print(f"  Precision @ Best MCC      {imbalanced_metrics['precision_mcc']:.4f}        {balanced_metrics['precision_mcc']:.4f}        {balanced_metrics['precision_mcc'] - imbalanced_metrics['precision_mcc']:+.4f}")
print(f"  Balanced Recall (G-mean)  {imbalanced_metrics['balanced_recall_gmean']:.4f}        {balanced_metrics['balanced_recall_gmean']:.4f}        {balanced_metrics['balanced_recall_gmean'] - imbalanced_metrics['balanced_recall_gmean']:+.4f}")

print(f"\nTraining Time:")
print(f"  Imbalanced: {imbalanced_time:.2f}s")
print(f"  Balanced:   {balanced_time:.2f}s")
```

## Troubleshooting

- **Import Errors**: 
  - Ensure the `public` directory is in the Python path
  - Adjust `sys.path` modifications in your notebook if needed
  - Verify all modules in `public/` are present

- **PySpark Warnings**: 
  - Warnings about native libraries are normal and can be ignored
  - If not using PySpark, you can load data directly with pandas

- **Memory Issues**: 
  - For large datasets, reduce batch sizes in `compute_distances_batched()` (e.g., `batch_size=500`)
  - Consider using a subset of data for initial testing
  - Use `dtype=np.float32` for distance matrices to save memory

- **Data Issues**: 
  - Ensure ID column is unique per row
  - Verify target column is binary (0/1 or False/True)
  - Check for data leakage: exclude any columns derived from the target or containing future information
  - Handle missing values appropriately (pipeline includes imputation, but check for excessive missingness)

- **Matching Issues**:
  - If matching fails, try increasing candidate pool size `M`
  - Adjust `tau` threshold for adaptive pool if needed
  - Check that distance matrices were computed correctly
