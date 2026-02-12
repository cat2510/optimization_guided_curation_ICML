#!/usr/bin/env python3
"""
Train student (XGBoost or RF) with OCT distillation on MSK data.

Uses msk_analysis/msk_2017_18_full.parquet and the same data loading and
preprocessing pipeline as msk_analysis/train_vanilla_oct.py and
msk_analysis/two_stage_iterative.py: target top_2_pct_cost_2018 (or created from
98th percentile of annual_cost_2018_deflated), exclude all 2018 columns from
features, optional high-correlation drop, train/val/test split 0.3 then 0.5.

Usage:
    python train_student_msk.py --model rf --compare --teacher_splits /path/to/splits.csv
    python train_student_msk.py --data msk_analysis/msk_2017_18_full.parquet --distill
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
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

from oct_distillation import (
    OCTTeacher, train_student_distilled, train_student_distilled_rf,
    compute_minority_metrics, check_calibration,
    extract_oct_rule_features,
)
from model_nonIAI_utils import (
    get_preprocessor_with_impute, train_test_split_enrol,
    get_bin_flag_columns, get_true_num_columns
)

try:
    from public.model_IAI import get_cat_columns
except ImportError:
    def get_cat_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()


def _get_booster(model):
    if hasattr(model, 'get_booster'):
        return model.get_booster()
    return getattr(model, '_booster', None)


def _compute_shap_top10(model, X_val_matrix, feature_names, output_path=None, max_samples=500, random_state=42):
    try:
        import shap
    except ImportError:
        print("  (SHAP not installed; pip install shap to enable SHAP analysis)")
        return None
    X_val_matrix = np.asarray(X_val_matrix, dtype=np.float64)
    n = X_val_matrix.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sampled = X_val_matrix[idx]
    else:
        X_sampled = X_val_matrix
    booster = _get_booster(model)
    if booster is not None:
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_sampled)
    else:
        if not hasattr(model, 'predict_proba') or not hasattr(model, 'fit'):
            return None
        try:
            explainer = shap.TreeExplainer(model, X_sampled)
            shap_vals = explainer.shap_values(X_sampled)
        except Exception:
            return None
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    n_feat = len(feature_names)
    if len(mean_abs) != n_feat:
        mean_abs = mean_abs[:n_feat] if len(mean_abs) >= n_feat else np.pad(mean_abs, (0, n_feat - len(mean_abs)))
    order = np.argsort(mean_abs)[::-1]
    top10 = [(feature_names[i], float(mean_abs[i])) for i in order[:10]]
    df_out = pd.DataFrame(top10, columns=['feature', 'mean_abs_shap'])
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df_out.to_csv(output_path, index=False)
    return df_out


def _resolve_path(path: str, must_exist: bool = False) -> str:
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    root_path = os.path.join(_REPO_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path if not must_exist else root_path


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from parquet or CSV (same as train_student.py)."""
    if data_path.endswith('.parquet'):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("DataLoad").getOrCreate()
            df = spark.read.format("parquet").load(data_path).toPandas()
        except Exception:
            df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    return df


def ensure_msk_target(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Ensure target column exists. If not, create top_2_pct_cost_2018 from
    annual_cost_2018_deflated (98th percentile), matching train_vanilla_oct / two_stage_iterative.
    Returns (df, target_col).
    """
    if target_col in df.columns:
        return df, target_col
    if "annual_cost_2018_deflated" not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found and 'annual_cost_2018_deflated' not in df. "
            f"Available 2018 target-like: {[c for c in df.columns if '2018' in c and ('top' in c.lower() or 'pct' in c.lower() or 'cost' in c.lower())]}"
        )
    threshold = df["annual_cost_2018_deflated"].quantile(0.98)
    df = df.copy()
    df["top_2_pct_cost_2018"] = (df["annual_cost_2018_deflated"] >= threshold).astype(int)
    print(f"Created top_2_pct_cost_2018 using 98th percentile threshold ${threshold:,.2f}")
    return df, "top_2_pct_cost_2018"


def prepare_features_msk(
    df: pd.DataFrame,
    target_col: str,
    drop_high_corr: bool = True,
    verbose: bool = True
) -> tuple:
    """
    MSK feature preparation: exclude ENROLID, target, and all columns containing '2018';
    optional drop of features with >0.95 correlation with target (leakage guard).
    Returns (feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS).
    """
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if drop_high_corr and len(feature_cols) < 500:
        numeric_cols = df[feature_cols + [target_col]].select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) > 0 and target_col in df.columns:
            corrs = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            high_corr_cols = corrs[corrs > 0.95].index.tolist()
            high_corr_cols = [c for c in high_corr_cols if c != target_col]
            feature_cols = [c for c in feature_cols if c not in high_corr_cols]
            if verbose and high_corr_cols:
                print(f"  Dropped {len(high_corr_cols)} high-correlation (leakage) columns")

    BIN_FLAG_COLUMNS = [c for c in get_bin_flag_columns(df) if c in feature_cols]
    CAT_COLUMNS = [c for c in get_cat_columns(df) if c in feature_cols]
    TRUE_NUM_COLUMNS = [c for c in get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS) if c in feature_cols]

    return feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS


def main():
    parser = argparse.ArgumentParser(
        description='Train student (XGBoost or RF) with OCT distillation on MSK data (msk_2017_18_full.parquet pipeline)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data: MSK defaults
    parser.add_argument('--data', type=str, default='msk_analysis/msk_2017_18_full.parquet',
                       help='Path to dataset (parquet or CSV)')
    parser.add_argument('--target_col', type=str, default='top_2_pct_cost_2018',
                       help='Target column (created from annual_cost_2018_deflated 98th pct if missing)')
    parser.add_argument('--no_drop_high_corr', action='store_true',
                       help='Do not drop features with >0.95 correlation with target')

    # Model choice
    parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'rf'],
                       help='Student model: xgb or rf')
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Path to saved OCT model (pickle or IAI format)')
    parser.add_argument('--teacher_splits', type=str,
                       default='two_stage_kcenter_results_global/oct_tree_ratio_1.00_splits.csv',
                       help='Path to OCT splits CSV')

    # Distillation
    parser.add_argument('--distill', action='store_true', default=True,
                       help='Enable rule-based distillation')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Run baseline and distilled; save comparison table')
    parser.add_argument('--use_rule_features', action='store_true', default=True,
                       help='Add OCT rule features')
    parser.add_argument('--use_sample_weights', action='store_true', default=True,
                       help='Use rule-based sample weighting')
    parser.add_argument('--weight_strategy', type=str, default='confidence',
                       choices=['confidence', 'agreement', 'minority_boost'])
    parser.add_argument('--weight_min', type=float, default=1.0)
    parser.add_argument('--weight_max', type=float, default=2.0)
    parser.add_argument('--confidence_exponent', type=float, default=1.0)
    parser.add_argument('--rule_feature_scale', type=float, default=1.0)
    parser.add_argument('--no_rule_confidence_feature', action='store_true')

    # Training
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--scale_pos_weight', type=float, default=None)
    parser.add_argument('--early_stop_metric', type=str, default='pr_auc',
                       choices=['pr_auc', 'recall', 'auc', 'logloss'])
    parser.add_argument('--early_stop_rounds', type=int, default=50)
    parser.add_argument('--random_state', type=int, default=123,
                       help='Random seed (123 matches MSK train_vanilla_oct / two_stage_iterative)')

    # Output
    parser.add_argument('--output_dir', type=str, default='student_distillation_results_msk',
                       help='Output directory for results')
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args()

    args.data = _resolve_path(args.data)
    args.teacher_splits = _resolve_path(args.teacher_splits)
    if args.teacher_model:
        args.teacher_model = _resolve_path(args.teacher_model)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/metrics", exist_ok=True)

    print(f"\n{'='*80}")
    print("MSK STUDENT TRAINING WITH OCT DISTILLATION")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---------- LOAD DATA (MSK pipeline) ----------
    print(f"{'='*80}")
    print("LOADING DATA (MSK pipeline)")
    print(f"{'='*80}\n")

    df = load_data(args.data)
    print(f"✓ Loaded data: {df.shape}")

    df, target_col = ensure_msk_target(df, args.target_col)
    target_counts = df[target_col].value_counts().sort_index()
    print(f"Target distribution ({target_col}):")
    print(target_counts)
    print(f"Minority class: {target_counts.idxmin()} ({target_counts.min()} samples)")

    feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS = prepare_features_msk(
        df, target_col, drop_high_corr=not args.no_drop_high_corr, verbose=args.verbose
    )
    print(f"\nFeatures: {len(feature_cols)} total")
    print(f"  Categorical: {len(CAT_COLUMNS)}, Numeric: {len(TRUE_NUM_COLUMNS)}, Binary flags: {len(BIN_FLAG_COLUMNS)}")

    # ---------- SPLIT (same as MSK: 0.3 test, then half of test = val) ----------
    print(f"\n{'='*80}")
    print("SPLITTING DATA")
    print(f"{'='*80}\n")

    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3,
        verbose=args.verbose, random_state=args.random_state
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5,
        verbose=args.verbose, random_state=args.random_state
    )

    X_train = train_pd[feature_cols]
    y_train = train_pd[target_col].values
    X_val = val_pd[feature_cols]
    y_val = val_pd[target_col].values
    X_test = test_pd[feature_cols]
    y_test = test_pd[target_col].values

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # ---------- PREPROCESSING ----------
    print(f"\n{'='*80}")
    print("PREPROCESSING")
    print(f"{'='*80}\n")

    preprocessor = get_preprocessor_with_impute(
        X_train, CAT_COLUMNS, TRUE_NUM_COLUMNS, binary_cols=BIN_FLAG_COLUMNS, verbose=args.verbose
    )
    preprocessor.fit(X_train)

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

    # ---------- LOAD TEACHER ----------
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
        X_train_p = preprocessor.transform(X_train)
        X_val_p = preprocessor.transform(X_val)
        X_test_p = preprocessor.transform(X_test)
        if args.model == 'xgb':
            import xgboost as xgb
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos_weight = args.scale_pos_weight or (n_neg / n_pos if n_pos > 0 else 1.0)
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
            model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=args.verbose)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.random_state,
                #class_weight='balanced',
                n_jobs=-1,
            )
            model.fit(X_train_p, y_train)
        y_val_p = model.predict_proba(X_val_p)[:, 1]
        y_test_p = model.predict_proba(X_test_p)[:, 1]
        val_m = compute_minority_metrics(y_val, y_val_p, verbose=False)
        test_m = compute_minority_metrics(y_test, y_test_p, verbose=False)
        cal = check_calibration(y_test, y_test_p)
        return model, val_m, test_m, cal, feature_names

    def _run_distilled():
        if args.model == 'rf':
            model, val_m = train_student_distilled_rf(
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
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.random_state,
                verbose=args.verbose
            )
        else:
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
        rule_train, _ = extract_oct_rule_features(
            teacher, X_train, include_leaf_assignment=True,
            include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
        )
        enriched_names = feature_names + list(rule_train.columns)
        return model, val_m, test_m, cal, enriched_names

    if args.compare:
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
        comparison_path = f"{args.output_dir}/metrics/comparison_baseline_vs_distilled.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ Saved comparison to {comparison_path}")
        print("\nCOMPARISON: Baseline vs Distilled (test set)")
        print(comparison_df.to_string(index=False))

        if args.save_model:
            for name, model, fnames in [
                ('student_baseline.pkl', baseline_model, feature_names),
                ('student_distilled.pkl', distilled_model, enriched_feature_names),
            ]:
                path = f"{args.output_dir}/models/{name}"
                with open(path, 'wb') as f:
                    pickle.dump({
                        'model': model, 'model_type': args.model, 'preprocessor': preprocessor, 'feature_names': fnames,
                        'feature_cols': feature_cols, 'CAT_COLUMNS': CAT_COLUMNS,
                        'TRUE_NUM_COLUMNS': TRUE_NUM_COLUMNS, 'BIN_FLAG_COLUMNS': BIN_FLAG_COLUMNS,
                        'distill': (name == 'student_distilled.pkl'), 'target_col': target_col,
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
        X_val_p = preprocessor.transform(X_val)
        shap_baseline = _compute_shap_top10(baseline_model, X_val_p, feature_names,
            output_path=f"{args.output_dir}/metrics/shap_top10_baseline.csv")
        if shap_baseline is not None:
            print(f"✓ Saved SHAP top 10 (baseline)")
            print(shap_baseline.to_string(index=False))
        X_val_distilled = preprocessor.transform(X_val)
        if teacher is not None and args.use_rule_features:
            rule_val, _ = extract_oct_rule_features(
                teacher, X_val, include_leaf_assignment=True,
                include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
            )
            rval = rule_val.values.astype(np.float64) * args.rule_feature_scale
            X_val_distilled = np.hstack([X_val_distilled, rval])
        shap_distilled = _compute_shap_top10(
            distilled_model, X_val_distilled, enriched_feature_names,
            output_path=f"{args.output_dir}/metrics/shap_top10_distilled.csv")
        if shap_distilled is not None:
            print(f"✓ Saved SHAP top 10 (distilled)")
            print(shap_distilled.to_string(index=False))
        print(f"\nCOMPLETE (compare mode). Results in {args.output_dir}/")
        return

    # ---------- SINGLE RUN ----------
    print(f"\n{'='*80}")
    print("TRAINING STUDENT MODEL")
    print(f"{'='*80}\n")

    if args.distill:
        if args.model == 'rf':
            student_model, val_metrics = train_student_distilled_rf(
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
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.random_state,
                verbose=args.verbose
            )
        else:
            student_model, val_metrics = train_student_distilled(
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

    # ---------- SAVE RESULTS ----------
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    if args.save_model:
        fnames = feature_names
        if args.distill and args.use_rule_features:
            rule_train, _ = extract_oct_rule_features(
                teacher, X_train, include_leaf_assignment=True,
                include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
            )
            fnames = feature_names + list(rule_train.columns)
        with open(f"{args.output_dir}/models/student_model.pkl", 'wb') as f:
            pickle.dump({
                'model': student_model, 'model_type': args.model,
                'preprocessor': preprocessor, 'feature_names': fnames,
                'feature_cols': feature_cols, 'CAT_COLUMNS': CAT_COLUMNS,
                'TRUE_NUM_COLUMNS': TRUE_NUM_COLUMNS, 'BIN_FLAG_COLUMNS': BIN_FLAG_COLUMNS,
                'target_col': target_col,
            }, f)
        print(f"✓ Saved model to {args.output_dir}/models/student_model.pkl")

    metrics_dict = {
        'config': {
            'model': args.model, 'distill': args.distill,
            'use_rule_features': args.use_rule_features if args.distill else None,
            'use_sample_weights': args.use_sample_weights if args.distill else None,
            'weight_strategy': args.weight_strategy if args.distill else None,
            'early_stop_metric': args.early_stop_metric,
            'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
            'learning_rate': args.learning_rate, 'scale_pos_weight': args.scale_pos_weight,
            'random_state': args.random_state, 'target_col': target_col,
        },
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'calibration': calibration_info
    }
    with open(f"{args.output_dir}/metrics/metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Saved metrics to {args.output_dir}/metrics/metrics.json")

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
    summary_df.to_csv(f"{args.output_dir}/metrics/summary.csv", index=False)
    print(f"✓ Saved summary to {args.output_dir}/metrics/summary.csv")

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
        importance_df.to_csv(f"{args.output_dir}/metrics/feature_importance.csv", index=False)
        print(f"✓ Saved feature importance to {args.output_dir}/metrics/feature_importance.csv")

    X_val_for_shap = preprocessor.transform(X_val)
    if args.distill and args.use_rule_features and teacher is not None:
        rule_val, _ = extract_oct_rule_features(
            teacher, X_val, include_leaf_assignment=True,
            include_rule_indicators=True, include_rule_confidence=not args.no_rule_confidence_feature
        )
        rval = rule_val.values.astype(np.float64) * args.rule_feature_scale
        X_val_for_shap = np.hstack([X_val_for_shap, rval])
        fnames_shap = feature_names + list(rule_val.columns)
    else:
        fnames_shap = feature_names
    shap_top10 = _compute_shap_top10(
        student_model, X_val_for_shap, fnames_shap,
        output_path=f"{args.output_dir}/metrics/shap_top10.csv"
    )
    if shap_top10 is not None:
        print(f"✓ Saved SHAP top 10 to {args.output_dir}/metrics/shap_top10.csv")
        print("\nSHAP top 10 features (mean |SHAP| on validation):")
        print(shap_top10.to_string(index=False))

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Test minority recall (MCC): {test_metrics['mcc_optimal']['recall']:.4f}")


if __name__ == '__main__':
    main()
