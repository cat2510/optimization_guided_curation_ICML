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

# Paths: script dir (distillation_with_OCT) and repo root (one level up)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)   # oct_distillation
sys.path.insert(0, _REPO_ROOT)    # model_nonIAI_utils, public

from oct_distillation import (
    OCTTeacher, train_student_distilled, compute_minority_metrics, check_calibration,
    extract_oct_rule_features,
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


def _resolve_path(path: str, must_exist: bool = False) -> str:
    """If path is relative and not found in cwd, try repo root (one level up from script)."""
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    root_path = os.path.join(_REPO_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path if not must_exist else root_path


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
    cutoff_columns = [col for col in df.columns if col.startswith('highcost_gt_')]
    exclude_cols = exclude_cols + cutoff_columns
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
    parser.add_argument('--exclude_cols', type=str, nargs='+', default=['annual_cost_2017', 'annual_cost_2018_deflated', "ENROLID", "cost_stratum_2018"],
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
    parser.add_argument('--distill', action='store_true', default=True,
                       help='Enable rule-based distillation')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Run both baseline (no distill) and distilled; save comparison table')
    parser.add_argument('--use_rule_features', action='store_true', default=True,
                       help='Add OCT rule features (leaf assignment, rule indicators) to XGBoost')
    parser.add_argument('--use_sample_weights', action='store_true', default=True,
                       help='Use rule-based sample weighting')
    parser.add_argument('--weight_strategy', type=str, default='confidence',
                       choices=['confidence', 'agreement', 'minority_boost'],
                       help='Sample weighting strategy: confidence (rule confidence), agreement (OCT-label agreement), minority_boost (boost minority in confident leaves)')
    parser.add_argument('--weight_min', type=float, default=1.0,
                       help='Min sample weight (default 1.0 = boost-only; no downweighting)')
    parser.add_argument('--weight_max', type=float, default=2.0,
                       help='Max sample weight for confident leaves')
    parser.add_argument('--confidence_exponent', type=float, default=1.0,
                       help='Use confidence**exponent for weights; >1 strengthens high-confidence samples (default 1.0)')
    parser.add_argument('--rule_feature_scale', type=float, default=1.0,
                       help='Scale rule features by this factor; >1 makes OCT rules count more in splits (default 1.0)')
    parser.add_argument('--no_rule_confidence_feature', action='store_true',
                       help='Do not add oct_rule_confidence as a feature (only leaf id and rule indicators); sample weights still use confidence if enabled')
    
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

    # Resolve paths relative to repo root when running from distillation_with_OCT
    args.data = _resolve_path(args.data)
    args.teacher_splits = _resolve_path(args.teacher_splits)
    if args.teacher_model:
        args.teacher_model = _resolve_path(args.teacher_model)
    
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
    # LOAD TEACHER (OCT) — when using distillation or compare
    # ========================================================================
    teacher = None
    if args.distill or args.compare:
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
        print(f"✓ OCT teacher loaded")
        if teacher.splits_df is not None:
            print(f"  Tree splits: {len(teacher.splits_df)} nodes")
        if teacher.leaf_probs:
            print(f"  Leaf probabilities computed for {len(teacher.leaf_probs)} leaves")

    def _run_baseline():
        """Train XGBoost without distillation; return model, val_metrics, test_metrics, calibration.
        Uses the same hyperparameters as the distilled path for a fair comparison.
        """
        import xgboost as xgb
        X_train_p = preprocessor.transform(X_train)
        X_val_p = preprocessor.transform(X_val)
        X_test_p = preprocessor.transform(X_test)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = args.scale_pos_weight or (n_neg / n_pos if n_pos > 0 else 1.0)
        # Match distilled path: same objective, regularization, and tree_method for fair comparison
        model = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=args.random_state,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method='hist',
            early_stopping_rounds=args.early_stop_rounds,
        )
        model.fit(
            X_train_p, y_train,
            eval_set=[(X_val_p, y_val)],
            verbose=args.verbose
        )
        y_val_p = model.predict_proba(X_val_p)[:, 1]
        y_test_p = model.predict_proba(X_test_p)[:, 1]
        val_m = compute_minority_metrics(y_val, y_val_p, verbose=False)
        test_m = compute_minority_metrics(y_test, y_test_p, verbose=False)
        cal = check_calibration(y_test, y_test_p)
        return model, val_m, test_m, cal, feature_names

    def _run_distilled():
        """Train XGBoost with distillation; return model, val_metrics, test_metrics, calibration, feature_names (enriched)."""
        model, val_m = train_student_distilled(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            teacher=teacher, preprocessor=preprocessor,
            use_rule_features=args.use_rule_features,
            use_sample_weights=args.use_sample_weights,
            weight_strategy=args.weight_strategy,
            weight_min=args.weight_min,
            weight_max=args.weight_max,
            confidence_exponent=args.confidence_exponent,
            rule_feature_scale=args.rule_feature_scale,
            include_rule_confidence_feature=not args.no_rule_confidence_feature,
            scale_pos_weight=args.scale_pos_weight,
            early_stop_metric=args.early_stop_metric,
            early_stop_rounds=args.early_stop_rounds,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            verbose=args.verbose
        )
        X_test_p = preprocessor.transform(X_test)
        if args.use_rule_features:
            rule_test, _ = extract_oct_rule_features(
                teacher, X_test, include_leaf_assignment=True,
                include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
            )
            rtest = rule_test.values.astype(np.float64) * args.rule_feature_scale
            X_test_p = np.hstack([X_test_p, rtest])
        y_test_p = model.predict_proba(X_test_p)[:, 1]
        test_m = compute_minority_metrics(y_test, y_test_p, verbose=False)
        cal = check_calibration(y_test, y_test_p)
        # Enriched feature names (base + rule features from train_student_distilled internals)
        rule_train, _ = extract_oct_rule_features(
            teacher, X_train, include_leaf_assignment=True,
            include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
        )
        enriched_names = feature_names + list(rule_train.columns)
        return model, val_m, test_m, cal, enriched_names

    if args.compare:
        # ========== COMPARE: run baseline then distilled ==========
        print(f"\n{'='*80}")
        print("RUN 1/2: BASELINE (no distillation)")
        print(f"{'='*80}\n")
        baseline_model, baseline_val, baseline_test, baseline_cal, _ = _run_baseline()
        print(f"Baseline test PR-AUC: {baseline_test['pr_auc']:.4f}")

        print(f"\n{'='*80}")
        print("RUN 2/2: WITH DISTILLATION")
        print(f"{'='*80}\n")
        distilled_model, distilled_val, distilled_test, distilled_cal, enriched_feature_names = _run_distilled()
        print(f"Distilled test PR-AUC: {distilled_test['pr_auc']:.4f}")

        # Build comparison table (same metrics as summary.csv)
        rows = [
            {'metric': 'auc', 'baseline_val': baseline_val['auc'], 'baseline_test': baseline_test['auc'],
             'distilled_val': distilled_val['auc'], 'distilled_test': distilled_test['auc'],
             'delta_val': distilled_val['auc'] - baseline_val['auc'], 'delta_test': distilled_test['auc'] - baseline_test['auc']},
            {'metric': 'pr_auc', 'baseline_val': baseline_val['pr_auc'], 'baseline_test': baseline_test['pr_auc'],
             'distilled_val': distilled_val['pr_auc'], 'distilled_test': distilled_test['pr_auc'],
             'delta_val': distilled_val['pr_auc'] - baseline_val['pr_auc'], 'delta_test': distilled_test['pr_auc'] - baseline_test['pr_auc']},
            {'metric': 'recall_mcc', 'baseline_val': baseline_val['mcc_optimal']['recall'], 'baseline_test': baseline_test['mcc_optimal']['recall'],
             'distilled_val': distilled_val['mcc_optimal']['recall'], 'distilled_test': distilled_test['mcc_optimal']['recall'],
             'delta_val': distilled_val['mcc_optimal']['recall'] - baseline_val['mcc_optimal']['recall'],
             'delta_test': distilled_test['mcc_optimal']['recall'] - baseline_test['mcc_optimal']['recall']},
            {'metric': 'precision_mcc', 'baseline_val': baseline_val['mcc_optimal']['precision'], 'baseline_test': baseline_test['mcc_optimal']['precision'],
             'distilled_val': distilled_val['mcc_optimal']['precision'], 'distilled_test': distilled_test['mcc_optimal']['precision'],
             'delta_val': distilled_val['mcc_optimal']['precision'] - baseline_val['mcc_optimal']['precision'],
             'delta_test': distilled_test['mcc_optimal']['precision'] - baseline_test['mcc_optimal']['precision']},
            {'metric': 'mcc', 'baseline_val': baseline_val['mcc_optimal']['mcc'], 'baseline_test': baseline_test['mcc_optimal']['mcc'],
             'distilled_val': distilled_val['mcc_optimal']['mcc'], 'distilled_test': distilled_test['mcc_optimal']['mcc'],
             'delta_val': distilled_val['mcc_optimal']['mcc'] - baseline_val['mcc_optimal']['mcc'],
             'delta_test': distilled_test['mcc_optimal']['mcc'] - baseline_test['mcc_optimal']['mcc']},
            {'metric': 'recall_gmean', 'baseline_val': baseline_val['gmean_optimal']['recall'], 'baseline_test': baseline_test['gmean_optimal']['recall'],
             'distilled_val': distilled_val['gmean_optimal']['recall'], 'distilled_test': distilled_test['gmean_optimal']['recall'],
             'delta_val': distilled_val['gmean_optimal']['recall'] - baseline_val['gmean_optimal']['recall'],
             'delta_test': distilled_test['gmean_optimal']['recall'] - baseline_test['gmean_optimal']['recall']},
            {'metric': 'gmean', 'baseline_val': baseline_val['gmean_optimal']['gmean'], 'baseline_test': baseline_test['gmean_optimal']['gmean'],
             'distilled_val': distilled_val['gmean_optimal']['gmean'], 'distilled_test': distilled_test['gmean_optimal']['gmean'],
             'delta_val': distilled_val['gmean_optimal']['gmean'] - baseline_val['gmean_optimal']['gmean'],
             'delta_test': distilled_test['gmean_optimal']['gmean'] - baseline_test['gmean_optimal']['gmean']},
        ]
        comparison_df = pd.DataFrame(rows)
        os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)
        comparison_path = f"{args.output_dir}/metrics/comparison_baseline_vs_distilled.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ Saved comparison to {comparison_path}")

        print(f"\n{'='*80}")
        print("COMPARISON: Baseline vs Distilled (test set)")
        print(f"{'='*80}\n")
        print(comparison_df.to_string(index=False))
        print()

        if args.save_model:
            for name, model, fnames in [
                ('student_baseline.pkl', baseline_model, feature_names),
                ('student_distilled.pkl', distilled_model, enriched_feature_names),
            ]:
                path = f"{args.output_dir}/models/{name}"
                with open(path, 'wb') as f:
                    pickle.dump({
                        'model': model, 'preprocessor': preprocessor, 'feature_names': fnames,
                        'feature_cols': feature_cols, 'CAT_COLUMNS': CAT_COLUMNS,
                        'TRUE_NUM_COLUMNS': TRUE_NUM_COLUMNS, 'BIN_FLAG_COLUMNS': BIN_FLAG_COLUMNS,
                        'distill': (name == 'student_distilled.pkl'),
                    }, f)
                print(f"✓ Saved {path}")

        for label, val_m, test_m, cal in [
            ('baseline', baseline_val, baseline_test, baseline_cal),
            ('distilled', distilled_val, distilled_test, distilled_cal),
        ]:
            path = f"{args.output_dir}/metrics/metrics_{label}.json"
            with open(path, 'w') as f:
                json.dump({'val_metrics': val_m, 'test_metrics': test_m, 'calibration': cal}, f, indent=2)
            print(f"✓ Saved {path}")

        print(f"\n{'='*80}")
        print("COMPLETE (compare mode)")
        print(f"{'='*80}")
        print(f"Results in {args.output_dir}/")
        print(f"  comparison_baseline_vs_distilled.csv — side-by-side metrics")
        print(f"  metrics_baseline.json / metrics_distilled.json")
        print(f"  student_baseline.pkl / student_distilled.pkl")
        return

    # ========================================================================
    # SINGLE RUN: TRAIN STUDENT
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
            teacher=teacher,
            preprocessor=preprocessor,
            use_rule_features=args.use_rule_features,
            use_sample_weights=args.use_sample_weights,
            weight_strategy=args.weight_strategy,
            weight_min=args.weight_min,
            weight_max=args.weight_max,
            confidence_exponent=args.confidence_exponent,
            rule_feature_scale=args.rule_feature_scale,
            include_rule_confidence_feature=not args.no_rule_confidence_feature,
            scale_pos_weight=args.scale_pos_weight,
            early_stop_metric=args.early_stop_metric,
            early_stop_rounds=args.early_stop_rounds,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            verbose=args.verbose
        )
        X_test_processed = preprocessor.transform(X_test)
        if teacher is not None and args.use_rule_features:
            rule_features_test, _ = extract_oct_rule_features(
                teacher, X_test, include_leaf_assignment=True,
                include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
            )
            rtest = rule_features_test.values.astype(np.float64) * args.rule_feature_scale
            X_test_processed = np.hstack([X_test_processed, rtest])
        y_test_pred_proba = student_model.predict_proba(X_test_processed)[:, 1]
        test_metrics = compute_minority_metrics(y_test, y_test_pred_proba, verbose=args.verbose)
        calibration_info = check_calibration(y_test, y_test_pred_proba)
        print(f"\nCalibration (ECE): {calibration_info['ece']:.4f}")
    else:
        student_model, val_metrics, test_metrics, calibration_info, _ = _run_baseline()
        print(f"\nCalibration (ECE): {calibration_info['ece']:.4f}")

    # ========================================================================
    # SAVE RESULTS (single run)
    # ========================================================================
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    if args.save_model:
        model_path = f"{args.output_dir}/models/student_model.pkl"
        fnames = feature_names
        if args.distill and args.use_rule_features:
            rule_train, _ = extract_oct_rule_features(
                teacher, X_train, include_leaf_assignment=True,
                include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
            )
            fnames = feature_names + list(rule_train.columns)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': student_model,
                'preprocessor': preprocessor,
                'feature_names': fnames,
                'feature_cols': feature_cols,
                'CAT_COLUMNS': CAT_COLUMNS,
                'TRUE_NUM_COLUMNS': TRUE_NUM_COLUMNS,
                'BIN_FLAG_COLUMNS': BIN_FLAG_COLUMNS,
            }, f)
        print(f"✓ Saved model to {model_path}")

    metrics_dict = {
        'config': {
            'distill': args.distill,
            'use_rule_features': args.use_rule_features if args.distill else None,
            'use_sample_weights': args.use_sample_weights if args.distill else None,
            'weight_strategy': args.weight_strategy if args.distill else None,
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
    os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")

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

    if hasattr(student_model, 'feature_importances_'):
        fnames = feature_names
        if args.distill and args.use_rule_features:
            rule_train, _ = extract_oct_rule_features(
                teacher, X_train, include_leaf_assignment=True,
                include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
            )
            fnames = feature_names + list(rule_train.columns)
        importance_df = pd.DataFrame({
            'feature': fnames,
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
