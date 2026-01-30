import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from precompute_distances import compute_distances_batched, save_distances_hdf5
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def prepare_dataset_for_kcenter(X, y, dataset_name):
    """
    Prepare a dataset for k-center matching by:
    1. Creating ENROLID column
    2. Converting target from -1/1 to 0/1 (1 = minority/cases, 0 = majority/controls)
    3. Combining features and target into a single dataframe
    
    This version handles object types and different target formats.
    """
    # Create dataframe with features
    df = X.copy()
    
    # Add ENROLID (using index + 1 as IDs)
    df['ENROLID'] = range(1, len(df) + 1)
    
    # Convert target to numeric if it's not already
    y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Check unique values to determine format
    unique_vals = y_numeric.dropna().unique()
    
    # Convert target based on format:
    # - If values are -1 and 1: convert -1 -> 0, 1 -> 1
    # - If values are already 0 and 1: keep as is
    if set(unique_vals).issubset({-1, 1}):
        # Format is -1/1, convert -1 -> 0, 1 -> 1
        df['target'] = ((y_numeric == 1).astype(int))
    elif set(unique_vals).issubset({0, 1}):
        # Format is already 0/1
        df['target'] = y_numeric.astype(int)
    else:
        # Find the minority class (smaller count) and map it to 1
        value_counts = y_numeric.value_counts()
        minority_value = value_counts.idxmin()
        df['target'] = (y_numeric == minority_value).astype(int)
        print(f"  Note: Mapped minority class value {minority_value} to 1")
    
    # Reorder columns to put ENROLID and target first
    cols = ['ENROLID', 'target'] + [c for c in df.columns if c not in ['ENROLID', 'target']]
    df = df[cols]
    
    print(f"\n=== {dataset_name.upper()} DATASET PREPARED ===")
    print(f"Shape: {df.shape}")
    print(f"Original target type: {type(y)}")
    print(f"Original target unique values: {sorted(unique_vals)}")
    print(f"Target distribution:")
    print(df['target'].value_counts().sort_index())
    print(f"Class imbalance ratio: {df['target'].value_counts().min() / df['target'].value_counts().max():.4f}")
    
    return df

def setup_feature_columns(df, target_col='target', drop_time=True, log1p_amount=True):
    """
    Setup feature columns, categorical columns, and numeric columns.
    Optionally:
      - drop Time from features
      - replace Amount with Amount_log = log1p(Amount)
    Returns: df_out, feature_cols, cat_cols, true_num_cols, bin_cols
    """
    df_out = df.copy()

    # --- Amount log transform (create new column; drop raw Amount from features) ---
    if log1p_amount and "Amount" in df_out.columns:
        # Guard: Amount should be nonnegative in this dataset; clip just in case
        amt = pd.to_numeric(df_out["Amount"], errors="coerce").fillna(0.0)
        amt = np.clip(amt, a_min=0.0, a_max=None)
        df_out["Amount_log"] = np.log1p(amt)

    # --- Feature columns ---
    exclude = {"ENROLID", target_col}
    if drop_time:
        exclude.add("Time")

    # If we created Amount_log, exclude raw Amount and use Amount_log instead
    if log1p_amount and "Amount" in df_out.columns:
        exclude.add("Amount")

    feature_cols = [c for c in df_out.columns if c not in exclude]

    # Ensure Amount_log is included (and is numeric)
    if log1p_amount and "Amount_log" in df_out.columns and "Amount_log" not in feature_cols:
        feature_cols.append("Amount_log")

    # --- Column type discovery (on feature_cols) ---
    cat_cols = df_out[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df_out[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Binary detection
    bin_cols = []
    for col in numeric_cols:
        u = df_out[col].dropna().unique()
        if len(u) <= 2 and set(u).issubset({0, 1, 0.0, 1.0}):
            bin_cols.append(col)

    true_num_cols = [c for c in numeric_cols if c not in bin_cols]

    print(f"\n=== COLUMN SETUP ===")
    print(f"Total features: {len(feature_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Binary columns: {len(bin_cols)}")
    print(f"True numeric columns: {len(true_num_cols)}")
    if log1p_amount and "Amount" in df_out.columns:
        print("Using Amount_log = log1p(Amount) and excluding raw Amount.")
    if drop_time and "Time" in df_out.columns:
        print("Dropping Time from features.")

    return df_out, feature_cols, cat_cols, true_num_cols, bin_cols

def create_train_test_split(df, target_col='target', test_size=0.3, val_size=0.5, random_state=123):
    """
    Create train/val/test splits similar to the main script.
    Returns: train_df, val_df, test_df
    """
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_col], 
        random_state=random_state
    )
    
    # Second split: val vs test (from temp_df)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=temp_df[target_col],
        random_state=random_state
    )
    
    print(f"\n=== DATA SPLITS ===")
    print(f"Train: {len(train_df):,} samples")
    print(f"  - Minority: {(train_df[target_col] == 1).sum():,}")
    print(f"  - Majority: {(train_df[target_col] == 0).sum():,}")
    print(f"Val: {len(val_df):,} samples")
    print(f"  - Minority: {(val_df[target_col] == 1).sum():,}")
    print(f"  - Majority: {(val_df[target_col] == 0).sum():,}")
    print(f"Test: {len(test_df):,} samples")
    print(f"  - Minority: {(test_df[target_col] == 1).sum():,}")
    print(f"  - Majority: {(test_df[target_col] == 0).sum():,}")
    
    return train_df, val_df, test_df


# Precompute case-control distances (required for k-center matching)
# This follows the pattern from kcenter_hyperparameter_search_global.py
def get_preprocessor_with_impute(categorical_cols, numeric_cols, verbose=True):
    if verbose:
        print("→ Building preprocessor w/ imputation:")
        print(f"   • Cat: impute(most_frequent) + OHE on: {categorical_cols}")
        print(f"   • Num: impute(median) + scale on: {numeric_cols}")
   
    transformers = []

    if categorical_cols:
        cat_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    if numeric_cols:
        num_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    
    return ColumnTransformer(transformers=transformers, remainder="drop")


def precompute_case_control_distances(
    train_df, target_col, feature_cols,
    cat_columns, true_num_columns,
    dataset_name, seed=123
):
    # Save to H5
    output_dir = "./precomputed_distances"
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"distances_{dataset_name}_seed_{seed}.h5")
    
    if os.path.exists(h5_path):
        print(f"  Distance matrix already exists at: {h5_path}")
        return h5_path, None, None
    
    covered = set([c for c in cat_columns if c in feature_cols] + [c for c in true_num_columns if c in feature_cols])
    dropped = [c for c in feature_cols if c not in covered]
    assert len(dropped) == 0, f"These feature_cols are being dropped by preprocessor: {dropped}"

    cases = train_df[train_df[target_col] == 1].copy()
    controls = train_df[train_df[target_col] == 0].copy()

    X_cases = cases[feature_cols].copy()
    X_controls = controls[feature_cols].copy()

    print(f"Cases (minority): {len(cases):,}")
    print(f"Controls (majority): {len(controls):,}")

    # Fit preprocessing on controls only
    preprocessor = get_preprocessor_with_impute(
        categorical_cols=[c for c in cat_columns if c in X_controls.columns],
        numeric_cols=[c for c in true_num_columns if c in X_controls.columns],
        verbose=True
    )
    X_controls_processed = preprocessor.fit_transform(X_controls)
    X_cases_processed = preprocessor.transform(X_cases)

    # (Optional) sanity checks
    import numpy as np
    assert np.isfinite(X_controls_processed).all()
    assert np.isfinite(X_cases_processed).all()

    distances = compute_distances_batched(
        X_controls_processed, X_cases_processed,
        batch_size=1000, dtype=np.float32
    )
    print(f"  Distance matrix shape: {distances.shape}")
    print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"  Distance mean: {distances.mean():.3f}")
    
    
    majority_enrolids = controls["ENROLID"].to_numpy()
    minority_enrolids = cases["ENROLID"].to_numpy()
    
    save_distances_hdf5(
        distances,
        majority_enrolids,
        minority_enrolids,
        h5_path,
        compression='gzip'
    )
    
    print(f"\n✓ Saved distances to: {h5_path}")
    
    return h5_path, X_controls_processed, majority_enrolids

