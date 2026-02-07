#!/usr/bin/env python3
"""
Train XGBoost student model with OCT distillation.

Usage:
    python train_student.py --distill --alpha 0.3 --teacher oct --early_stop_metric pr_auc
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from oct_distillation import (
    OCTTeacher, train_student_distilled, compute_minority_metrics, check_calibration
)
from model_nonIAI_utils import (
    get_preprocessor_with_impute, train_test_split_enrol,
    get_bin_flag_columns, get_true_num_columns
)

# Try to import IAI utilities
try:
    from public.model_IAI import get_cat_columns
except ImportError:
    def get_cat_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from parquet or CSV."""
    if data_path.endswith('.parquet'):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("DataLoad").getOrCreate()
            df = spark.read.format("parquet").load(data_path).toPandas()
        except:
            df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    return df


def prepare_features(df: pd.DataFrame, target_col: str, exclude_cols: list = None) -> tuple:
    """Prepare feature column lists."""
    if exclude_cols is None:
        exclude_cols = []
    
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    
    # Remove excluded columns
    feature_cols = [c for c in df.columns 
                   if c not in ['ENROLID', target_col] + exclude_cols]
    
    CAT_COLUMNS = [c for c in CAT_COLUMNS if c in feature_cols]
    TRUE_NUM_COLUMNS = [c for c in TRUE_NUM_COLUMNS if c in feature_cols]
    BIN_FLAG_COLUMNS = [c for c in BIN_FLAG_COLUMNS if c in feature_cols]
    
    return feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS


def main():
    parser = argparse.ArgumentParser(
        description='Train XGBoost student with OCT distillation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, default='0917_2017_18_with_2017_cost.parquet',
                       help='Path to dataset (parquet or CSV)')
    parser.add_argument('--target_col', type=str, default='highcost_gt_200000',
                       help='Target column name')
    parser.add_argument('--exclude_cols', type=str, nargs='+', default=None,
                       help='Columns to exclude from features')
    
    # Teacher (OCT) arguments
    parser.add_argument('--teacher', type=str, default='oct', choices=['oct'],
                       help='Teacher model type')
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Path to saved OCT model (pickle or IAI format)')
    parser.add_argument('--teacher_splits', type=str,
                       default='two_stage_kcenter_results_global/oct_tree_ckd_best_curated_prauc_splits.csv',
                       help='Path to OCT splits CSV')
    
    # Distillation arguments
    parser.add_argument('--distill', action='store_true',
                       help='Enable distillation')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Distillation strength (0=no distillation, 1=only teacher)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature scaling (not used in current implementation)')
    
    # Training arguments
    parser.add_argument('--n_estimators', type=int, default=500,
                       help='Maximum number of trees')
    parser.add_argument('--max_depth', type=int, default=6,
                       help='Tree depth')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--scale_pos_weight', type=float, default=None,
                       help='XGBoost scale_pos_weight (auto if None)')
    parser.add_argument('--early_stop_metric', type=str, default='pr_auc',
                       choices=['pr_auc', 'recall', 'auc', 'logloss'],
                       help='Metric for early stopping')
    parser.add_argument('--early_stop_rounds', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='student_distillation_results',
                       help='Output directory for results')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print progress')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)
    
    print(f"\n{'='*80}")
    print("XGBOOST STUDENT TRAINING WITH OCT DISTILLATION")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print(f"{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}\n")
    
    df = load_data(args.data)
    print(f"✓ Loaded data: {df.shape}")
    
    # Verify target column exists
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found. Available: {list(df.columns)[:10]}...")
    
    # Check target distribution
    target_counts = df[args.target_col].value_counts().sort_index()
    print(f"Target distribution ({args.target_col}):")
    print(target_counts)
    print(f"Minority class: {target_counts.idxmin()} ({target_counts.min()} samples)")
    
    # Prepare features
    feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS = prepare_features(
        df, args.target_col, args.exclude_cols
    )
    print(f"\nFeatures: {len(feature_cols)} total")
    print(f"  Categorical: {len(CAT_COLUMNS)}")
    print(f"  Numeric: {len(TRUE_NUM_COLUMNS)}")
    print(f"  Binary flags: {len(BIN_FLAG_COLUMNS)}")
    
    # ========================================================================
    # SPLIT DATA
    # ========================================================================
    print(f"\n{'='*80}")
    print("SPLITTING DATA")
    print(f"{'='*80}\n")
    
    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=args.target_col, test_size=0.3,
        verbose=args.verbose, random_state=args.random_state
    )
    
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=args.target_col, test_size=0.5,
        verbose=args.verbose, random_state=args.random_state
    )
    
    X_train = train_pd[feature_cols]
    y_train = train_pd[args.target_col].values
    X_val = val_pd[feature_cols]
    y_val = val_pd[args.target_col].values
    X_test = test_pd[feature_cols]
    y_test = test_pd[args.target_col].values
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    print(f"\n{'='*80}")
    print("PREPROCESSING")
    print(f"{'='*80}\n")
    
    preprocessor = get_preprocessor_with_impute(
        X_train, CAT_COLUMNS, TRUE_NUM_COLUMNS, binary_cols=BIN_FLAG_COLUMNS, verbose=args.verbose
    )
    preprocessor.fit(X_train)
    
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            if hasattr(transformer, 'named_steps'):
                ohe = transformer.named_steps.get('ohe')
                if ohe:
                    feature_names.extend(ohe.get_feature_names_out(columns))
        elif name in ['num', 'binary']:
            feature_names.extend(columns)
    
    print(f"Processed features: {len(feature_names)}")
    
    # ========================================================================
    # LOAD TEACHER (OCT)
    # ========================================================================
    teacher_proba_train = None
    teacher_proba_val = None
    
    if args.distill:
        print(f"\n{'='*80}")
        print("LOADING OCT TEACHER")
        print(f"{'='*80}\n")
        
        teacher = OCTTeacher(
            model_path=args.teacher_model,
            splits_csv=args.teacher_splits,
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            feature_names=feature_names
        )
        
        teacher_proba_train = teacher.predict_proba(X_train)
        teacher_proba_val = teacher.predict_proba(X_val)
        
        print(f"✓ Teacher probabilities computed")
        print(f"  Train: mean P(class=1) = {teacher_proba_train[:, 1].mean():.4f}")
        print(f"  Val: mean P(class=1) = {teacher_proba_val[:, 1].mean():.4f}")
    
    # ========================================================================
    # TRAIN STUDENT
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING STUDENT MODEL")
    print(f"{'='*80}\n")
    
    if args.distill:
        student_model, val_metrics = train_student_distilled(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            teacher_proba=teacher_proba_train,
            preprocessor=preprocessor,
            alpha=args.alpha,
            temperature=args.temperature,
            scale_pos_weight=args.scale_pos_weight,
            early_stop_metric=args.early_stop_metric,
            early_stop_rounds=args.early_stop_rounds,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            verbose=args.verbose
        )
    else:
        # Standard XGBoost training (no distillation)
        import xgboost as xgb
        
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = args.scale_pos_weight or (n_neg / n_pos if n_pos > 0 else 1.0)
        
        student_model = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=args.random_state,
            eval_metric='logloss'
        )
        
        student_model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            early_stopping_rounds=args.early_stop_rounds,
            verbose=args.verbose
        )
        
        y_val_pred_proba = student_model.predict_proba(X_val_processed)[:, 1]
        val_metrics = compute_minority_metrics(y_val, y_val_pred_proba, verbose=args.verbose)
    
    # ========================================================================
    # EVALUATE ON TEST SET
    # ========================================================================
    print(f"\n{'='*80}")
    print("TEST SET EVALUATION")
    print(f"{'='*80}\n")
    
    X_test_processed = preprocessor.transform(X_test)
    y_test_pred_proba = student_model.predict_proba(X_test_processed)[:, 1]
    test_metrics = compute_minority_metrics(y_test, y_test_pred_proba, verbose=args.verbose)
    
    # Calibration check
    calibration_info = check_calibration(y_test, y_test_pred_proba)
    print(f"\nCalibration (ECE): {calibration_info['ece']:.4f}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save model
    if args.save_model:
        model_path = f"{args.output_dir}/models/student_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': student_model,
                'preprocessor': preprocessor,
                'feature_names': feature_names,
                'feature_cols': feature_cols,
                'CAT_COLUMNS': CAT_COLUMNS,
                'TRUE_NUM_COLUMNS': TRUE_NUM_COLUMNS,
                'BIN_FLAG_COLUMNS': BIN_FLAG_COLUMNS
            }, f)
        print(f"✓ Saved model to {model_path}")
    
    # Save metrics
    metrics_dict = {
        'config': {
            'distill': args.distill,
            'alpha': args.alpha if args.distill else None,
            'early_stop_metric': args.early_stop_metric,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'scale_pos_weight': args.scale_pos_weight,
            'random_state': args.random_state
        },
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'calibration': calibration_info
    }
    
    metrics_path = f"{args.output_dir}/metrics/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
    
    # Save CSV summary
    summary_df = pd.DataFrame({
        'split': ['val', 'test'],
        'auc': [val_metrics['auc'], test_metrics['auc']],
        'pr_auc': [val_metrics['pr_auc'], test_metrics['pr_auc']],
        'recall_mcc': [val_metrics['mcc_optimal']['recall'], test_metrics['mcc_optimal']['recall']],
        'precision_mcc': [val_metrics['mcc_optimal']['precision'], test_metrics['mcc_optimal']['precision']],
        'mcc': [val_metrics['mcc_optimal']['mcc'], test_metrics['mcc_optimal']['mcc']],
        'recall_gmean': [val_metrics['gmean_optimal']['recall'], test_metrics['gmean_optimal']['recall']],
        'gmean': [val_metrics['gmean_optimal']['gmean'], test_metrics['gmean_optimal']['gmean']]
    })
    summary_path = f"{args.output_dir}/metrics/summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary to {summary_path}")
    
    # Feature importance
    if hasattr(student_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': student_model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_path = f"{args.output_dir}/metrics/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Saved feature importance to {importance_path}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Test minority recall (MCC): {test_metrics['mcc_optimal']['recall']:.4f}")


if __name__ == '__main__':
    main()
