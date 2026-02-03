# Load MSK data (if not already loaded)
# Uncomment if needed:
import sys,os,time
# Add parent directory to path to import modules from one level up
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
import importlib
import model_IAI
importlib.reload(model_IAI)
from model_IAI import *
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DataLoad").getOrCreate()
# Load the MSK dataset with enhanced cost features
df_msk_spark = spark.read.format("parquet").load("msk_2017_18_full.parquet")
df_og = df_msk_spark.toPandas()

# MSK-specific feature column definitions
# Binary flag columns: comorbidity flags, MSK category flags, medication flags
BIN_FLAG_COLUMNS = model_IAI.get_bin_flag_columns(df_og)

# MSK doesn't have stage columns like CKD, but we can identify categorical cost pattern/stability columns
STAGE_COLUMNS = []  # MSK doesn't use stage columns

# Categorical columns
CAT_COLUMNS = df_og.select_dtypes(include=["object","category","string"]).columns.tolist()

# True numeric columns (excluding binary flags and categorical)
TRUE_NUM_COLUMNS = model_IAI.get_true_num_columns(df_og, CAT_COLUMNS,BIN_FLAG_COLUMNS)

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

print("categorical cols: ", CAT_COLUMNS[:10], f"... ({len(CAT_COLUMNS)} total)")
print("stage cols: ", STAGE_COLUMNS)
print("cost cols (sample): ", COST_COLUMNS[:10], f"... ({len(COST_COLUMNS)} total)")

leftover_cols = [
    c for c in df_og.columns 
    if c not in CAT_COLUMNS and c not in TRUE_NUM_COLUMNS and c not in STAGE_COLUMNS and c not in BIN_FLAG_COLUMNS 
    and c != "ENROLID"
]

print(f"Number of leftover columns: {len(leftover_cols)}")
if len(leftover_cols) > 0:
    print("Leftover columns:", leftover_cols[:20])

# Create cost stratum from 2018 target (if available)
# Check what target columns are available
target_candidates = [col for col in df_og.columns if "2018" in col and ("top" in col.lower() or "pct" in col.lower() or "cost" in col.lower())]
print(f"\nAvailable 2018 target columns: {target_candidates}")

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
exclude_cols = ["ENROLID", target_col] + [col for col in df_og.columns if "2018" in col]

feature_cols = [c for c in df_og.columns if c not in exclude_cols]
# Check for high correlation with target (optional - can be slow for large datasets)
if len(feature_cols) < 500:  # Only do this for reasonable number of features
    numeric_cols = df_og[feature_cols + [target_col]].select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        corrs = df_og[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        # Columns to drop (very high correlation, likely leakage)
        high_corr_cols = corrs[corrs > 0.95].index.tolist()
        # Remove the target column itself, if present
        high_corr_cols = [col for col in high_corr_cols if col != target_col]
        # Final filtered feature set
        feature_cols = [col for col in feature_cols if col not in high_corr_cols]
        

TRAIN_TEST_SEED = 123

# Split data into train/test/val (same as multiobjective_bilevel.ipynb)
# Use the target_col defined in cell 0
train_ids, test_ids, train_pd, test_pd = model_IAI.train_test_split_enrol(
    df_og,
    target_col=target_col,  # Use target_col from cell 0 (e.g., "top_2_pct_cost_2018")
    test_size=0.3,
    verbose=False,
    random_state=TRAIN_TEST_SEED
)
print(f"Train shape: {train_pd.shape}, Test shape: {test_pd.shape}")
print("Feature cols:", len(feature_cols))

val_ids, test_ids, val_pd, test_pd = model_IAI.train_test_split_enrol(
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

ratio_values = [1.0]

print(train_pd.shape)

# Track resources before training
training_start_time = time.perf_counter()

vanilla_model, best_params, _, preprocessor, feature_names = finetune_oct(
    X_train=train_pd[[col for col in feature_cols]],
    y_train=train_pd[target_col],
    X_val=X_val,
    y_val=y_val,
    categorical_cols=CAT_COLUMNS,
    numeric_cols=TRUE_NUM_COLUMNS,
    depths=[5, 7],
    minbuckets = [100, 200, 300],
    cps = [0.00001, 0.0001, 0.001, 0.01]
)

training_end_time = time.perf_counter()
training_time = training_end_time - training_start_time

# Evaluate
evaluation_start_time = time.perf_counter()
metrics = evaluate_binary_oct(
    vanilla_model, X_test, y_test, preprocessor, feature_names, X_val_df=X_val, y_val=y_val,
    results_dir = "msk_vanilla_oct/", save_suffix="best_vanilla"
)
evaluation_end_time = time.perf_counter()
evaluation_time = evaluation_end_time - evaluation_start_time
total_time = time.perf_counter() - training_start_time

print(f"\n{'='*80}")
print("MODEL TRAINING COMPLETE")
print(f"{'='*80}")
print(f"\n✅ Trained OCT on K-Center undersampled data:")
print(f"   - Training samples: {len(train_pd):,}")
print(f"\n⏱️  Runtime:")
print(f"   - OCT training time: {format_time(training_time)}s")
print(f"   - Evaluation time: {format_time(evaluation_time)}s")
print(f"   - Total time: {format_time(total_time)}s")
print(f"   - Best parameters: {best_params}")
print(f"   - Metrics: {metrics}")
