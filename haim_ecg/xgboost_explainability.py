"""
XGBoost Multiclass Model Explainability with SHAP
==================================================

This script generates explainability visualizations for a trained multiclass XGBoost model:
1. Per-class SHAP beeswarm plots (4 panels)
2. Per-class SHAP bar plots (top 10 features)
3. Dependence plots for top ECG features
4. Stability check with bootstrap sampling

Usage:
    python xgboost_explainability.py --model_path model.pkl --X_test_path X_test.parquet --output_dir ./shap_results
    python xgboost_explainability.py --model_path model.pkl --X_test_path X_test.pkl --output_dir ./shap_results
    
    # With translation file (auto-detects column_name_translations.csv if in same directory):
    python xgboost_explainability.py --model_path model.pkl --X_test_path X_test.pkl --translation_file column_name_translations.csv
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from sklearn.metrics import jaccard_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Class label mappings for LVEF categories
CLASS_LABELS = {
    0: "Below 30%",
    1: "30-40%",
    2: "40-50%",
    3: "Above 50%"
}


def load_translation_mapping(translation_file=None, default_paths=None):
    """
    Load column name translation mapping from CSV file.
    
    Parameters:
    -----------
    translation_file : str, optional
        Path to CSV file with 'original_name' and 'translated_name' columns.
        If None, will check default paths.
    default_paths : list, optional
        List of default paths to check if translation_file is None
    
    Returns:
    --------
    translation_dict : dict
        Dictionary mapping original names to translated names
    """
    # If no file specified, check default paths
    if translation_file is None:
        if default_paths is None:
            # Default: check current directory and script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_paths = [
                'column_name_translations.csv',  # Current directory
                os.path.join(script_dir, 'column_name_translations.csv')  # Script directory
            ]
        
        for default_path in default_paths:
            if os.path.exists(default_path):
                translation_file = default_path
                print(f"Found translation file at: {translation_file}")
                break
    
    if translation_file is None or not os.path.exists(translation_file):
        return None
    
    print(f"Loading translation mapping from {translation_file}...")
    try:
        # Use python engine for more robust CSV parsing (handles unquoted commas and quoted fields better)
        # The python engine properly handles doubled quotes (CSV standard: "" inside quoted fields)
        try:
            translation_df = pd.read_csv(
                translation_file, 
                engine='python', 
                quotechar='"',
                doublequote=True,  # Handle doubled quotes ("" inside quoted fields)
                skipinitialspace=True
            )
        except Exception as e1:
            # If that fails, try with more lenient settings
            try:
                translation_df = pd.read_csv(
                    translation_file,
                    engine='python',
                    quotechar='"',
                    doublequote=True,
                    on_bad_lines='skip',
                    sep=',',
                    skipinitialspace=True
                )
            except Exception as e2:
                # Last resort: try with C engine (faster but less forgiving)
                translation_df = pd.read_csv(
                    translation_file,
                    engine='c',
                    quotechar='"',
                    doublequote=True
                )
        print(f"  Read CSV: {translation_df.shape[0]} rows, {translation_df.shape[1]} columns")
        print(f"  Columns found: {list(translation_df.columns)}")
        
        # Check for required columns
        if 'original_name' not in translation_df.columns or 'translated_name' not in translation_df.columns:
            print(f"  ⚠️  CSV must have 'original_name' and 'translated_name' columns")
            print(f"  Available columns: {list(translation_df.columns)}")
            return None
        
        # Remove any rows with missing values
        initial_count = len(translation_df)
        translation_df = translation_df.dropna(subset=['original_name', 'translated_name'])
        if len(translation_df) < initial_count:
            print(f"  ⚠️  Removed {initial_count - len(translation_df)} rows with missing values")
        
        # Debug: Check if ar_coefficient row is in the dataframe
        ar_coeff_rows = translation_df[translation_df['original_name'].astype(str).str.contains('ar_coefficient', na=False)]
        if len(ar_coeff_rows) > 0:
            print(f"  ✓ Debug: Found {len(ar_coeff_rows)} ar_coefficient row(s) in CSV:")
            for idx, row in ar_coeff_rows.iterrows():
                print(f"     Row {idx}: '{row['original_name']}' -> '{row['translated_name']}'")
        else:
            print(f"  ⚠️  Debug: No ar_coefficient rows found in CSV dataframe")
        
        # Create dictionary - strip whitespace from keys and values to avoid matching issues
        translation_dict = {}
        skipped_count = 0
        for idx, (orig, trans) in enumerate(zip(translation_df['original_name'], translation_df['translated_name'])):
            # Strip whitespace and handle NaN values
            if pd.notna(orig) and pd.notna(trans):
                orig_clean = str(orig).strip()
                trans_clean = str(trans).strip()
                # Remove any non-printable characters that might cause issues
                orig_clean = ''.join(char for char in orig_clean if char.isprintable() or char in ['_', '-', '.', ':', '(', ')', '[', ']', '"', "'"])
                translation_dict[orig_clean] = trans_clean
            else:
                skipped_count += 1
                if skipped_count <= 3:  # Show first few skipped rows
                    print(f"  ⚠️  Skipped row {idx}: orig={orig}, trans={trans}")
        
        if skipped_count > 0:
            print(f"  ⚠️  Skipped {skipped_count} rows due to missing values")
        
        print(f"  ✓ Loaded {len(translation_dict)} translations")
        
        # Debug: Check if the problematic feature is in the dictionary
        debug_feature = "tsfresh_dim_0__ar_coefficient__coeff_1__k_10_leadV3"
        if debug_feature in translation_dict:
            print(f"  ✓ Debug: Found '{debug_feature}' in translation dict")
        else:
            # Try to find similar keys
            similar_keys = [k for k in translation_dict.keys() if 'ar_coefficient' in k and 'coeff_1' in k and 'leadV3' in k]
            if similar_keys:
                print(f"  ⚠️  Debug: '{debug_feature}' not found, but found similar: {similar_keys}")
            else:
                print(f"  ⚠️  Debug: '{debug_feature}' not found in translation dict")
                # Show first few keys for debugging
                sample_keys = list(translation_dict.keys())[:5]
                print(f"     Sample keys in dict: {sample_keys}")
        
        # Show sample translations
        if len(translation_dict) > 0:
            sample_keys = list(translation_dict.keys())[:3]
            print(f"  Sample translations:")
            for key in sample_keys:
                print(f"    '{key}' -> '{translation_dict[key]}'")
        
        return translation_dict
    except Exception as e:
        print(f"  ✗ Error loading translation file: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return None


def load_model_and_data(model_path, X_test_path, y_test_path=None):
    """
    Load trained XGBoost model and test data.
    
    Parameters:
    -----------
    model_path : str
        Path to saved XGBoost model (pickle file)
    X_test_path : str
        Path to test features (parquet or CSV)
    y_test_path : str, optional
        Path to test labels (parquet or CSV)
    
    Returns:
    --------
    model : XGBClassifier
        Trained XGBoost model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series, optional
        Test labels
    """
    print("Loading model and data...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded model from {model_path}")
    
    # Load test data (support parquet, CSV, and pickle formats)
    if X_test_path.endswith('.parquet'):
        X_test = pd.read_parquet(X_test_path)
    elif X_test_path.endswith('.pkl') or X_test_path.endswith('.pickle'):
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        # Handle case where pickle contains a dict or other structure
        if isinstance(X_test, dict):
            # Try common keys
            for key in ['X_test', 'test', 'X', 'data']:
                if key in X_test:
                    X_test = X_test[key]
                    break
        # Ensure it's a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
    else:
        X_test = pd.read_csv(X_test_path)
    print(f"✓ Loaded X_test: {X_test.shape}")
    
    y_test = None
    if y_test_path:
        if y_test_path.endswith('.parquet'):
            y_test = pd.read_parquet(y_test_path)
        elif y_test_path.endswith('.pkl') or y_test_path.endswith('.pickle'):
            with open(y_test_path, 'rb') as f:
                y_test = pickle.load(f)
            # Handle case where pickle contains a dict or other structure
            if isinstance(y_test, dict):
                # Try common keys
                for key in ['y_test', 'test', 'y', 'target', 'labels']:
                    if key in y_test:
                        y_test = y_test[key]
                        break
            # Ensure it's a Series or convert from array
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.iloc[:, 0]  # Take first column
            elif isinstance(y_test, np.ndarray):
                y_test = pd.Series(y_test.flatten())
        else:
            y_test = pd.read_csv(y_test_path)
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]  # Take first column
        print(f"✓ Loaded y_test: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
    
    return model, X_test, y_test


def compute_shap_values(model, X_test, n_samples=None, random_state=42):
    """
    Compute SHAP values for the test set.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X_test : pd.DataFrame
        Test features (preferred over training set for explainability)
    n_samples : int, optional
        Number of samples to use for SHAP computation (for speed)
    random_state : int
        Random seed for sampling
    
    Returns:
    --------
    shap_values : np.ndarray
        SHAP values (n_samples, n_features, n_classes)
    explainer : shap.TreeExplainer
        SHAP explainer object
    """
    print("\nComputing SHAP values on TEST SET (recommended for explainability)...")
    
    # Sample if needed
    if n_samples and n_samples < len(X_test):
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test.iloc[sample_idx].copy()
        print(f"  Using {n_samples} samples for SHAP computation")
    else:
        X_sample = X_test.copy()
        print(f"  Using all {len(X_test)} samples")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multiclass: shap_values is a list of arrays (one per class)
    if isinstance(shap_values, list):
        print(f"  ✓ Computed SHAP values for {len(shap_values)} classes")
        print(f"    Shape per class: {shap_values[0].shape}")
    else:
        print(f"  ✓ Computed SHAP values: {shap_values.shape}")
    
    return shap_values, explainer, X_sample


def plot_beeswarm_per_class(shap_values, X_sample, feature_names, output_dir, n_classes=4, class_labels=None, figsize_individual=(14, 10), figsize_combined=(24, 18)):
    """
    Create per-class SHAP beeswarm plots.
    
    Parameters:
    -----------
    shap_values : list of np.ndarray
        SHAP values for each class
    X_sample : pd.DataFrame
        Sample features used for SHAP
    feature_names : list
        Feature names
    output_dir : str
        Output directory for plots
    n_classes : int
        Number of classes
    class_labels : dict, optional
        Dictionary mapping class indices to label strings (e.g., {0: "Below 30%", ...})
    figsize_individual : tuple, optional
        Figure size for individual plots (width, height) in inches. Default: (14, 10)
    figsize_combined : tuple, optional
        Figure size for combined plot (width, height) in inches. Default: (24, 18)
    """
    print("\nCreating per-class SHAP beeswarm plots...")
    
    # Use class labels if provided, otherwise use default
    if class_labels is None:
        class_labels = CLASS_LABELS
    
    # Create individual plots for each class (SHAP doesn't support ax parameter with subplots)
    for class_idx in range(n_classes):
        # Get SHAP values for this class
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[class_idx]
        else:
            shap_vals_class = shap_values[:, :, class_idx]
        
        # Create SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=shap_vals_class,
            base_values=np.zeros(len(X_sample)),  # Will be adjusted by SHAP
            data=X_sample.values,
            feature_names=feature_names
        )
        
        # Create figure for this class
        plt.figure(figsize=figsize_individual)
        
        # Plot beeswarm (without ax parameter)
        # Note: plot_size parameter can be used but conflicts with ax parameter
        shap.plots.beeswarm(
            shap_explanation,
            show=False,
            max_display=20,  # Top 20 features
            plot_size=None  # Use figure size instead
        )
        class_label = class_labels.get(class_idx, f'Class {class_idx}')
        plt.title(f'{class_label} - SHAP Beeswarm Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save individual plot
        output_path = os.path.join(output_dir, f'shap_beeswarm_class_{class_idx}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")
    
    # Also create a combined figure with subplots using manual plotting
    print("  Creating combined beeswarm plot...")
    fig, axes = plt.subplots(2, 2, figsize=figsize_combined)
    axes = axes.flatten()
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        # Get SHAP values for this class
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[class_idx]
        else:
            shap_vals_class = shap_values[:, :, class_idx]
        
        # Compute mean absolute SHAP for top features
        mean_abs_shap = np.abs(shap_vals_class).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_shap = shap_vals_class[:, top_indices]
        
        # Create a simplified beeswarm-like plot manually
        y_pos = np.arange(len(top_features))
        for i, feature_idx in enumerate(top_indices):
            feature_shap = shap_vals_class[:, feature_idx]
            # Sample points for visualization (if too many)
            if len(feature_shap) > 100:
                sample_idx = np.random.choice(len(feature_shap), 100, replace=False)
                feature_shap = feature_shap[sample_idx]
            
            # Scatter plot with jitter
            jitter = np.random.normal(0, 0.1, len(feature_shap))
            colors = ['red' if x < 0 else 'blue' for x in feature_shap]
            ax.scatter(feature_shap, i + jitter, alpha=0.3, s=10, c=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=10)  # Increased from 8 for better readability
        ax.set_xlabel('SHAP value', fontsize=12)  # Increased from 10
        class_label = class_labels.get(class_idx, f'Class {class_idx}')
        ax.set_title(f'{class_label} - Top 20 Features', fontsize=14, fontweight='bold')  # Increased from 12
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    combined_output_path = os.path.join(output_dir, 'shap_beeswarm_per_class_combined.png')
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {combined_output_path}")
    plt.close()


def plot_bar_per_class(shap_values, X_sample, feature_names, output_dir, n_classes=4, top_k=10, class_labels=None):
    """
    Create per-class SHAP bar plots (top features by mean |SHAP|).
    
    Parameters:
    -----------
    shap_values : list of np.ndarray
        SHAP values for each class
    X_sample : pd.DataFrame
        Sample features used for SHAP
    feature_names : list
        Feature names
    output_dir : str
        Output directory for plots
    n_classes : int
        Number of classes
    top_k : int
        Number of top features to show
    class_labels : dict, optional
        Dictionary mapping class indices to label strings (e.g., {0: "Below 30%", ...})
    """
    print("\nCreating per-class SHAP bar plots...")
    
    # Use class labels if provided, otherwise use default
    if class_labels is None:
        class_labels = CLASS_LABELS
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        # Get SHAP values for this class
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[class_idx]
        else:
            shap_vals_class = shap_values[:, :, class_idx]
        
        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals_class).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        # Create bar plot
        bars = ax.barh(range(len(top_features)), top_values, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        class_label = class_labels.get(class_idx, f'Class {class_idx}')
        ax.set_title(f'{class_label} - Top {top_k} Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'shap_bar_per_class.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_dependence_plots(shap_values, X_sample, feature_names, output_dir, n_classes=4, n_features=2):
    """
    Create dependence plots for top ECG-derived features.
    
    Parameters:
    -----------
    shap_values : list of np.ndarray
        SHAP values for each class
    X_sample : pd.DataFrame
        Sample features used for SHAP
    feature_names : list
        Feature names
    output_dir : str
        Output directory for plots
    n_classes : int
        Number of classes
    n_features : int
        Number of top ECG features to plot
    """
    print("\nCreating dependence plots for top ECG features...")
    
    # Identify ECG features (contain 'lead', 'ecg', 'static_lead', or 'tsfresh')
    ecg_feature_mask = np.array([
        any(keyword in name.lower() for keyword in ['lead', 'ecg', 'static_lead', 'tsfresh'])
        for name in feature_names
    ])
    ecg_indices = np.where(ecg_feature_mask)[0]
    ecg_feature_names = [feature_names[i] for i in ecg_indices]
    
    if len(ecg_indices) == 0:
        print("  ⚠️  No ECG features found. Using top features overall.")
        ecg_indices = np.arange(len(feature_names))
        ecg_feature_names = feature_names
    
    # Compute mean absolute SHAP across all classes to find top features
    if isinstance(shap_values, list):
        mean_abs_shap_all = np.mean([np.abs(shap_vals).mean(axis=0) for shap_vals in shap_values], axis=0)
    else:
        mean_abs_shap_all = np.abs(shap_values).mean(axis=(0, 2))
    
    # Get top ECG features
    ecg_mean_shap = mean_abs_shap_all[ecg_indices]
    top_ecg_indices = ecg_indices[np.argsort(ecg_mean_shap)[-n_features:][::-1]]
    top_ecg_names = [feature_names[i] for i in top_ecg_indices]
    
    print(f"  Top {n_features} ECG features: {top_ecg_names}")
    
    # Create dependence plots for each top feature
    fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
    if n_features == 1:
        axes = [axes]
    
    for idx, (feature_idx, feature_name) in enumerate(zip(top_ecg_indices, top_ecg_names)):
        ax = axes[idx]
        
        # Aggregate SHAP values across classes (weighted by class frequency or simple mean)
        if isinstance(shap_values, list):
            # Average across classes
            shap_vals_agg = np.mean([shap_vals[:, feature_idx] for shap_vals in shap_values], axis=0)
        else:
            shap_vals_agg = shap_values[:, feature_idx, :].mean(axis=1)
        
        # Create scatter plot
        feature_values = X_sample.iloc[:, feature_idx].values
        
        # Create bins for smoother visualization
        if len(np.unique(feature_values)) > 50:
            # Use hexbin for continuous features
            hb = ax.hexbin(feature_values, shap_vals_agg, gridsize=30, cmap='YlOrRd', mincnt=1)
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('Mean SHAP value', fontsize=10)
            ax.set_title(f'Dependence Plot: {feature_name}', fontsize=11, fontweight='bold')
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            # Use scatter for discrete features
            scatter = ax.scatter(feature_values, shap_vals_agg, alpha=0.5, s=20, c=shap_vals_agg, 
                                cmap='RdBu_r', vmin=-np.abs(shap_vals_agg).max(), 
                                vmax=np.abs(shap_vals_agg).max())
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('Mean SHAP value', fontsize=10)
            ax.set_title(f'Dependence Plot: {feature_name}', fontsize=11, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='SHAP value')
        
        # Add trend line
        z = np.polyfit(feature_values, shap_vals_agg, 1)
        p = np.poly1d(z)
        sorted_idx = np.argsort(feature_values)
        ax.plot(feature_values[sorted_idx], p(feature_values[sorted_idx]), 
               "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'shap_dependence_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def extract_important_features(shap_values, X_sample, feature_names, output_dir, n_classes=4, top_k=20):
    """
    Extract all important features that appear in SHAP analyses.
    This helps create a comprehensive translation map.
    
    Parameters:
    -----------
    shap_values : list of np.ndarray
        SHAP values for each class
    X_sample : pd.DataFrame
        Sample features used for SHAP
    feature_names : list
        Feature names (translated or original)
    output_dir : str
        Output directory for results
    n_classes : int
        Number of classes
    top_k : int
        Number of top features to extract per analysis
    
    Returns:
    --------
    important_features : set
        Set of important feature names
    """
    important_features = set()
    
    print("\nExtracting important features from SHAP analyses...")
    
    # 1. Extract from per-class bar plots (top features by mean |SHAP|)
    print("  Extracting from per-class bar plots...")
    for class_idx in range(n_classes):
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[class_idx]
        else:
            shap_vals_class = shap_values[:, :, class_idx]
        
        mean_abs_shap = np.abs(shap_vals_class).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        important_features.update(top_features)
        class_label = CLASS_LABELS.get(class_idx, f'Class {class_idx}')
        print(f"    {class_label}: {len(top_features)} features")
    
    # 2. Extract from beeswarm plots (top features overall)
    print("  Extracting from beeswarm plots...")
    if isinstance(shap_values, list):
        # Average across classes
        mean_abs_shap_all = np.mean([np.abs(shap_vals).mean(axis=0) for shap_vals in shap_values], axis=0)
    else:
        mean_abs_shap_all = np.abs(shap_values).mean(axis=(0, 2))
    
    top_indices_all = np.argsort(mean_abs_shap_all)[-top_k:][::-1]
    top_features_all = [feature_names[i] for i in top_indices_all]
    important_features.update(top_features_all)
    print(f"    Overall top {len(top_features_all)} features")
    
    # 3. Extract ECG features for dependence plots
    print("  Extracting ECG features for dependence plots...")
    ecg_feature_mask = np.array([
        any(keyword in name.lower() for keyword in ['lead', 'ecg', 'static_lead', 'tsfresh'])
        for name in feature_names
    ])
    if np.any(ecg_feature_mask):
        ecg_indices = np.where(ecg_feature_mask)[0]
        ecg_mean_shap = mean_abs_shap_all[ecg_indices]
        top_ecg_indices = ecg_indices[np.argsort(ecg_mean_shap)[-top_k:][::-1]]
        top_ecg_features = [feature_names[i] for i in top_ecg_indices]
        important_features.update(top_ecg_features)
        print(f"    Top {len(top_ecg_features)} ECG features")
    
    # 4. Extract from stability check (most stable features)
    print("  Extracting from stability check...")
    # This will be done in the stability_check function and added here
    
    print(f"  ✓ Total unique important features: {len(important_features)}")
    return important_features


def save_important_features_for_translation(important_features, original_feature_names, output_dir):
    """
    Save important features to a CSV file for creating translation map.
    
    Parameters:
    -----------
    important_features : set or list
        Important feature names (may be translated)
    original_feature_names : list
        Original feature names from the model
    output_dir : str
        Output directory for results
    """
    print("\nSaving important features for translation mapping...")
    
    # Create mapping from translated names back to original names
    # If feature_names were translated, we need to map them back
    # For now, assume important_features might be translated names
    # We'll create a file with both original and any translated names found
    
    # Create DataFrame
    features_df = pd.DataFrame({
        'original_name': sorted(important_features),
        'translated_name': [''] * len(important_features)  # Empty for user to fill
    })
    
    # If we have original feature names, try to match
    # (This handles the case where important_features are translated names)
    if original_feature_names:
        # Create reverse lookup if needed
        # For now, just save what we have
        pass
    
    output_path = os.path.join(output_dir, 'important_features_for_translation.csv')
    features_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(features_df)} features to: {output_path}")
    print(f"     Fill in the 'translated_name' column with human-readable names")
    print(f"     Then merge with your full translation map")
    
    return output_path


def stability_check(shap_values, X_sample, feature_names, output_dir, n_bootstrap=20, top_k=10, random_state=42):
    """
    Perform stability check: compute top-k feature overlap across bootstrap samples.
    
    Parameters:
    -----------
    shap_values : list of np.ndarray
        SHAP values for each class
    X_sample : pd.DataFrame
        Sample features used for SHAP
    feature_names : list
        Feature names
    output_dir : str
        Output directory for results
    n_bootstrap : int
        Number of bootstrap samples
    top_k : int
        Number of top features to consider
    random_state : int
        Random seed
    
    Returns:
    --------
    results : dict
        Stability check results
    """
    print(f"\nPerforming stability check ({n_bootstrap} bootstrap samples)...")
    
    np.random.seed(random_state)
    n_samples = len(X_sample)
    
    # Aggregate SHAP values across classes
    if isinstance(shap_values, list):
        # Average across classes
        shap_vals_agg = np.mean([np.abs(shap_vals) for shap_vals in shap_values], axis=0)
    else:
        shap_vals_agg = np.abs(shap_values).mean(axis=2)
    
    # Compute mean absolute SHAP per feature
    mean_abs_shap = shap_vals_agg.mean(axis=0)
    
    # Bootstrap sampling
    top_features_list = []
    for i in range(n_bootstrap):
        # Sample with replacement
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_shap = shap_vals_agg[bootstrap_idx]
        
        # Compute mean absolute SHAP for bootstrap sample
        bootstrap_mean_shap = bootstrap_shap.mean(axis=0)
        
        # Get top-k features
        top_indices = np.argsort(bootstrap_mean_shap)[-top_k:][::-1]
        top_features = tuple(sorted([feature_names[idx] for idx in top_indices]))
        top_features_list.append(top_features)
    
    # Compute Jaccard overlap
    # Compare each pair of bootstrap samples
    jaccard_scores = []
    for i in range(n_bootstrap):
        for j in range(i + 1, n_bootstrap):
            set1 = set(top_features_list[i])
            set2 = set(top_features_list[j])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)
    
    mean_jaccard = np.mean(jaccard_scores)
    std_jaccard = np.std(jaccard_scores)
    
    # Find most stable features (appear in most bootstrap samples)
    all_features = [f for features in top_features_list for f in features]
    feature_counts = Counter(all_features)
    most_common = feature_counts.most_common(top_k)
    
    # Get overall top features (from full dataset)
    overall_top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
    overall_top_features = [feature_names[idx] for idx in overall_top_indices]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Jaccard score distribution
    ax1.hist(jaccard_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(mean_jaccard, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_jaccard:.3f}')
    ax1.set_xlabel('Jaccard Similarity', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Top-10 Feature Overlap Distribution\n(Bootstrap Pairs)', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Most stable features
    stable_features = [f[0] for f in most_common]
    stable_counts = [f[1] for f in most_common]
    stable_pct = [c / n_bootstrap * 100 for c in stable_counts]
    
    bars = ax2.barh(range(len(stable_features)), stable_pct, color='coral')
    ax2.set_yticks(range(len(stable_features)))
    ax2.set_yticklabels(stable_features, fontsize=9)
    ax2.set_xlabel('Appearance Frequency (%)', fontsize=10)
    ax2.set_title(f'Most Stable Top-{top_k} Features\n(across {n_bootstrap} bootstrap samples)', 
                  fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, stable_pct)):
        ax2.text(pct, i, f' {pct:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'stability_check.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # Save results to text file
    results_text = f"""
Stability Check Results
=======================

Bootstrap Configuration:
  - Number of bootstrap samples: {n_bootstrap}
  - Top-k features per sample: {top_k}
  - Total comparisons: {len(jaccard_scores)}

Jaccard Similarity:
  - Mean: {mean_jaccard:.4f}
  - Std: {std_jaccard:.4f}
  - Min: {np.min(jaccard_scores):.4f}
  - Max: {np.max(jaccard_scores):.4f}

Overall Top-{top_k} Features (from full dataset):
{chr(10).join(f'  {i+1}. {feat}' for i, feat in enumerate(overall_top_features))}

Most Stable Features (appear in most bootstrap samples):
{chr(10).join(f'  {feat}: {count}/{n_bootstrap} ({count/n_bootstrap*100:.1f}%)' for feat, count in most_common)}
"""
    
    results_path = os.path.join(output_dir, 'stability_check_results.txt')
    with open(results_path, 'w') as f:
        f.write(results_text)
    print(f"  ✓ Saved: {results_path}")
    
    print(f"\n  Stability Summary:")
    print(f"    Mean Jaccard similarity: {mean_jaccard:.4f} ± {std_jaccard:.4f}")
    print(f"    Range: [{np.min(jaccard_scores):.4f}, {np.max(jaccard_scores):.4f}]")
    
    return {
        'mean_jaccard': mean_jaccard,
        'std_jaccard': std_jaccard,
        'jaccard_scores': jaccard_scores,
        'overall_top_features': overall_top_features,
        'most_stable_features': most_common
    }


def extract_important_features(shap_values, X_sample, feature_names, output_dir, n_classes=4, top_k=20):
    """
    Extract all important features that appear in SHAP analyses.
    This helps create a comprehensive translation map.
    
    Parameters:
    -----------
    shap_values : list of np.ndarray
        SHAP values for each class
    X_sample : pd.DataFrame
        Sample features used for SHAP
    feature_names : list
        Feature names (should be original names, not translated)
    output_dir : str
        Output directory for results
    n_classes : int
        Number of classes
    top_k : int
        Number of top features to extract per analysis
    
    Returns:
    --------
    important_features : set
        Set of important feature names
    """
    important_features = set()
    
    print("\nExtracting important features from SHAP analyses...")
    
    # 1. Extract from per-class bar plots (top features by mean |SHAP|)
    print("  Extracting from per-class bar plots...")
    for class_idx in range(n_classes):
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[class_idx]
        else:
            shap_vals_class = shap_values[:, :, class_idx]
        
        mean_abs_shap = np.abs(shap_vals_class).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        important_features.update(top_features)
        class_label = CLASS_LABELS.get(class_idx, f'Class {class_idx}')
        print(f"    {class_label}: {len(top_features)} features")
    
    # 2. Extract from beeswarm plots (top features overall)
    print("  Extracting from beeswarm plots...")
    if isinstance(shap_values, list):
        # Average across classes
        mean_abs_shap_all = np.mean([np.abs(shap_vals).mean(axis=0) for shap_vals in shap_values], axis=0)
    else:
        mean_abs_shap_all = np.abs(shap_values).mean(axis=(0, 2))
    
    top_indices_all = np.argsort(mean_abs_shap_all)[-top_k:][::-1]
    top_features_all = [feature_names[i] for i in top_indices_all]
    important_features.update(top_features_all)
    print(f"    Overall top {len(top_features_all)} features")
    
    # 3. Extract ECG features for dependence plots
    print("  Extracting ECG features for dependence plots...")
    ecg_feature_mask = np.array([
        any(keyword in name.lower() for keyword in ['lead', 'ecg', 'static_lead', 'tsfresh'])
        for name in feature_names
    ])
    if np.any(ecg_feature_mask):
        ecg_indices = np.where(ecg_feature_mask)[0]
        ecg_mean_shap = mean_abs_shap_all[ecg_indices]
        top_ecg_indices = ecg_indices[np.argsort(ecg_mean_shap)[-top_k:][::-1]]
        top_ecg_features = [feature_names[i] for i in top_ecg_indices]
        important_features.update(top_ecg_features)
        print(f"    Top {len(top_ecg_features)} ECG features")
    
    print(f"  ✓ Total unique important features: {len(important_features)}")
    return important_features


def save_important_features_for_translation(important_features, original_feature_names, output_dir):
    """
    Save important features to a CSV file for creating translation map.
    
    Parameters:
    -----------
    important_features : set or list
        Important feature names (original names from model)
    original_feature_names : list
        Original feature names from the model (for reference)
    output_dir : str
        Output directory for results
    
    Returns:
    --------
    output_path : str
        Path to saved CSV file
    """
    print("\nSaving important features for translation mapping...")
    
    # Convert to sorted list
    features_list = sorted(list(important_features))
    
    # Create DataFrame
    features_df = pd.DataFrame({
        'original_name': features_list,
        'translated_name': [''] * len(features_list)  # Empty for user to fill
    })
    
    output_path = os.path.join(output_dir, 'important_features_for_translation.csv')
    features_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(features_df)} features to: {output_path}")
    print(f"     Fill in the 'translated_name' column with human-readable names")
    print(f"     Then merge with your full translation map")
    
    # Also save a summary
    summary_path = os.path.join(output_dir, 'important_features_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Important Features Extracted from SHAP Analysis\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total unique features: {len(features_list)}\n\n")
        f.write(f"Features (sorted alphabetically):\n")
        for i, feat in enumerate(features_list, 1):
            f.write(f"  {i}. {feat}\n")
    print(f"  ✓ Saved summary to: {summary_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate SHAP explainability visualizations for XGBoost model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained XGBoost model (pickle file)')
    parser.add_argument('--X_test_path', type=str, required=True,
                       help='Path to test features (parquet, CSV, or pickle). Note: Test set is recommended over training set for SHAP analysis.')
    parser.add_argument('--y_test_path', type=str, default=None,
                       help='Path to test labels (optional, parquet, CSV, or pickle)')
    parser.add_argument('--translation_file', type=str, default=None,
                       help='Path to CSV file with column name translations (original_name, translated_name). '
                            'If not provided, will automatically look for column_name_translations.csv in current directory.')
    parser.add_argument('--output_dir', type=str, default='./shap_results',
                       help='Output directory for plots and results')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples to use for SHAP (default: all)')
    parser.add_argument('--n_classes', type=int, default=4,
                       help='Number of classes in the model')
    parser.add_argument('--n_bootstrap', type=int, default=20,
                       help='Number of bootstrap samples for stability check')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top features for bar plots and stability check')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--beeswarm_width', type=float, default=14,
                       help='Width of individual beeswarm plots in inches (default: 14)')
    parser.add_argument('--beeswarm_height', type=float, default=10,
                       help='Height of individual beeswarm plots in inches (default: 10)')
    parser.add_argument('--beeswarm_combined_width', type=float, default=24,
                       help='Width of combined beeswarm plot in inches (default: 24)')
    parser.add_argument('--beeswarm_combined_height', type=float, default=18,
                       help='Height of combined beeswarm plot in inches (default: 18)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(
        args.model_path, 
        args.X_test_path, 
        args.y_test_path
    )
    
    # Compute SHAP values
    shap_values, explainer, X_sample = compute_shap_values(
        model, 
        X_test, 
        n_samples=args.n_samples,
        random_state=args.random_state
    )
    
    # Get feature names (original names from model)
    original_feature_names = list(X_sample.columns)
    
    # Load and apply translation mapping (auto-detect if not provided)
    translation_dict = load_translation_mapping(args.translation_file)
    
    # Apply translations to feature names
    if translation_dict:
        feature_names = []
        debug_feature = "tsfresh_dim_0__ar_coefficient__coeff_1__k_10_leadV3"
        for name in original_feature_names:
            # Strip whitespace from original name for matching
            name_clean = str(name).strip()
            # Remove any non-printable characters that might cause issues
            name_clean = ''.join(char for char in name_clean if char.isprintable() or char in ['_', '-', '.', ':', '(', ')', '[', ']', '"', "'"])
            
            # Debug specific feature - check if this is the problematic feature
            if 'ar_coefficient' in name_clean and 'coeff_1' in name_clean and 'leadV3' in name_clean:
                print(f"\n  🔍 Debug: Found ar_coefficient feature in model: '{name_clean}'")
                print(f"     Length: {len(name_clean)}")
                print(f"     In translation dict: {name_clean in translation_dict}")
                if name_clean in translation_dict:
                    print(f"     ✓ Translation found: '{translation_dict[name_clean]}'")
                else:
                    # Try to find exact or similar keys
                    exact_match = [k for k in translation_dict.keys() if k == name_clean]
                    similar = [k for k in translation_dict.keys() if 'ar_coefficient' in k and 'coeff_1' in k and 'leadV3' in k]
                    if exact_match:
                        print(f"     Found exact match: {exact_match}")
                    elif similar:
                        print(f"     Found similar keys: {similar}")
                        print(f"     Trying to match with first similar key...")
                        # Try using the first similar key
                        if similar:
                            translated = translation_dict[similar[0]]
                            print(f"     Using translation from similar key: '{translated}'")
                            feature_names.append(translated)
                            continue
                    else:
                        print(f"     ⚠️  No match found in translation dict")
                        # Show what keys are actually in the dict for comparison
                        ar_keys = [k for k in translation_dict.keys() if 'ar_coefficient' in k]
                        if ar_keys:
                            print(f"     Available ar_coefficient keys: {ar_keys[:3]}")
            
            translated = translation_dict.get(name_clean, name)  # Use translated name if available, else original
            feature_names.append(translated)
        translated_count = sum(1 for orig, trans in zip(original_feature_names, feature_names) if orig != trans)
        print(f"\n✓ Applied translations to {translated_count}/{len(feature_names)} feature names")
        
        # Show which features were translated and which weren't (for debugging)
        if translated_count > 0:
            print(f"\n  Sample translated features:")
            translated_examples = [(orig, trans) for orig, trans in zip(original_feature_names, feature_names) if orig != trans][:5]
            for orig, trans in translated_examples:
                print(f"    '{orig}' -> '{trans}'")
        
        # Check if there are features in the model that aren't in the translation file
        missing_translations = [name for name in original_feature_names if str(name).strip() not in translation_dict]
        if missing_translations:
            print(f"\n  ⚠️  {len(missing_translations)} features in model are not in translation file (will use original names)")
            if len(missing_translations) <= 10:
                print(f"     Missing: {missing_translations}")
            else:
                print(f"     Missing (first 10): {missing_translations[:10]}")
    else:
        feature_names = original_feature_names
        print(f"\n⚠️  No translation file provided - using original feature names")
    
    # Generate visualizations (using translated names and class labels)
    plot_beeswarm_per_class(
        shap_values, X_sample, feature_names, args.output_dir, args.n_classes, 
        class_labels=CLASS_LABELS,
        figsize_individual=(args.beeswarm_width, args.beeswarm_height),
        figsize_combined=(args.beeswarm_combined_width, args.beeswarm_combined_height)
    )
    plot_bar_per_class(shap_values, X_sample, feature_names, args.output_dir, args.n_classes, args.top_k, class_labels=CLASS_LABELS)
    plot_dependence_plots(shap_values, X_sample, feature_names, args.output_dir, args.n_classes, n_features=2)
    
    # Stability check (using translated names)
    stability_results = stability_check(
        shap_values, 
        X_sample, 
        feature_names, 
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        top_k=args.top_k,
        random_state=args.random_state
    )
    
    # Extract important features for translation mapping
    # Use original feature names (not translated) for the mapping
    print("\n" + "="*60)
    print("EXTRACTING IMPORTANT FEATURES FOR TRANSLATION MAPPING")
    print("="*60)
    
    important_features_set = extract_important_features(
        shap_values,
        X_sample,
        original_feature_names,  # Use original names for mapping
        args.output_dir,
        args.n_classes,
        top_k=max(args.top_k, 20)  # Get more features for comprehensive mapping
    )
    
    # Add features from stability check
    # Note: stability_results uses feature_names (which might be translated)
    # We need to map them back to original names using indices
    if stability_results:
        # Recompute top features using indices to get original names
        if isinstance(shap_values, list):
            mean_abs_shap_all = np.mean([np.abs(shap_vals).mean(axis=0) for shap_vals in shap_values], axis=0)
        else:
            mean_abs_shap_all = np.abs(shap_values).mean(axis=(0, 2))
        
        # Get overall top features using indices (to ensure original names)
        overall_top_indices = np.argsort(mean_abs_shap_all)[-args.top_k:][::-1]
        overall_top_original = [original_feature_names[i] for i in overall_top_indices]
        important_features_set.update(overall_top_original)
        print(f"  Added {len(overall_top_original)} features from stability check (overall top)")
        
        # For most stable features, map translated names back to original
        if 'most_stable_features' in stability_results:
            stable_feat_translated = [feat[0] for feat in stability_results['most_stable_features']]
            # Create mapping from translated to original if translation was applied
            if translation_dict:
                # Reverse lookup: translated -> original
                reverse_translation = {v: k for k, v in translation_dict.items()}
                for trans_feat in stable_feat_translated:
                    original_feat = reverse_translation.get(trans_feat, trans_feat)
                    important_features_set.add(original_feat)
            else:
                # No translation, names are already original
                important_features_set.update(stable_feat_translated)
            print(f"  Added {len(stable_feat_translated)} features from stability check (most stable)")
    
    # Save important features for translation
    translation_file_path = save_important_features_for_translation(
        important_features_set,
        original_feature_names,
        args.output_dir
    )
    
    print(f"\n{'='*60}")
    print("✓ All explainability analyses complete!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"  Important features for translation: {translation_file_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
