#!/usr/bin/env python3
"""
vanilla_oct_multi_cohort.py
===========================

Train and evaluate a *vanilla* OCT (no undersampling) on multiple cohorts.

For each cohort code:
  - load engineered feature parquet
  - define target: top_2_pct_cost_<outcome_year> (or create from annual_cost_<outcome_year>_deflated @98th pct)
  - drop all columns containing outcome_year to prevent leakage
  - split train/val/test by ENROLID (same helper as MSK)
  - train OCT with a small grid:
        depths=[7], minbuckets=[200], cps=[0.0001, 0.001]
  - evaluate on test using evaluate_binary_oct
  - also compute two operating-point metrics that match your downstream focus:
        (A) recall/spec/gmean at val-best-gmean threshold
        (B) recall/spec/gmean at val-max-recall subject to specificity >= 0.60 threshold

Outputs:
  output_root/<CODE>/vanilla/
  output_root/summary_vanilla_multi_cohort.csv
"""

import os
import sys
import re
import time
import math
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# repo imports (mirrors your other scripts)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import importlib
import public.model_IAI
importlib.reload(public.model_IAI)
from public.model_IAI import (
    train_test_split_enrol,
    finetune_oct,
    evaluate_binary_oct,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)

from pyspark.sql import SparkSession

# optional sklearn for threshold selection
try:
    from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", type=str,
                   default= "F32, E78, E66, E11",
                   help="Comma-separated cohort codes")
    p.add_argument("--features_dir", type=str,
                   default="/Users/cat2510/my_projects/misc_conditions/misc_conditions_features_with_meds",
                   help="Directory containing <CODE>_features_<baseline>_<outcome>([_with_meds]).parquet")
    p.add_argument("--baseline_year", type=int, default=2017)
    p.add_argument("--outcome_year", type=int, default=2018)
    p.add_argument("--train_test_seed", type=int, default=123)
    p.add_argument("--output_root", type=str, default="/Users/cat2510/scratch/oct_vanilla_big_cohorts")

    # OCT grid (small, as requested)
    p.add_argument("--depths", nargs="+", type=int, default=[7])
    p.add_argument("--minbuckets", nargs="+", type=int, default=[200])
    p.add_argument("--cps", nargs="+", type=float, default=[0.0001, 0.001])

    # thresholding metric
    p.add_argument("--spec_floor", type=float, default=0.60)

    # spark
    p.add_argument("--spark_app", type=str, default="VanillaOCTMultiCohort")

    return p.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def resolve_feature_path(features_dir: Path, code: str, baseline_year: int, outcome_year: int) -> Path:
    candidates = [
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_100feat.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No features parquet found for {code}. Tried: {candidates}")


# Target/feature logic imported from curated (handles outcome_year as int or list for I25/I50)
from curated_vs_random_1to1_multi_cohort_oct import pick_target_and_features


# ----------------------------
# Per-cohort runner
# ----------------------------
def run_one_cohort(code: str, args) -> dict:
    print(f"\n{'='*90}\nCOHORT: {code}\n{'='*90}")

    feat_path = resolve_feature_path(Path(args.features_dir), code, args.baseline_year, args.outcome_year)
    print(f"  Features: {feat_path}")
    df = pd.read_parquet(feat_path)


    outcome_years = [2018, 2019] if code in ("I25", "I50") else args.outcome_year
    target_col, feature_cols = pick_target_and_features(df, args.baseline_year, outcome_years)

    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        # keep only 250,000 rows stratified on target_col
        df, _ = train_test_split(
            df, train_size=250_000, stratify=df[target_col], random_state=args.train_test_seed
        )
        print(f"  Sampled to {len(df):,} rows (stratified on {target_col})")
    else:
        print(f"  Using {len(df):,} rows")

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    # split
    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df,
        target_col=target_col,
        test_size=0.3,
        verbose=False,
        random_state=args.train_test_seed,
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd,
        target_col=target_col,
        test_size=0.5,
        verbose=False,
        random_state=args.train_test_seed,
    )

    X_val, y_val = val_pd[feature_cols], val_pd[target_col]
    X_test, y_test = test_pd[feature_cols], test_pd[target_col]

    nP = int((train_pd[target_col] == 1).sum())
    nN = int((train_pd[target_col] == 0).sum())
    print(f"  Split sizes: train={train_pd.shape}, val={val_pd.shape}, test={test_pd.shape}")
    print(f"  Train prevalence: positives={nP:,}, controls={nN:,}, ratio={nN/max(nP,1):.2f}:1")

    out_dir = Path(args.output_root) / code / "vanilla_100feat"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    # train
    t0 = time.perf_counter()
    model, params, grid_df, preprocessor, feat_names = finetune_oct(
        X_train=train_pd[feature_cols],
        y_train=train_pd[target_col],
        X_val=X_val,
        y_val=y_val,
        categorical_cols=CAT_COLUMNS,
        numeric_cols=TRUE_NUM_COLUMNS,
        binary_cols=BIN_FLAG_COLUMNS,
        depths=args.depths,
        minbuckets=args.minbuckets,
        cps=args.cps,
        verbose=False,
        random_seed=args.train_test_seed,
    )
    train_time = time.perf_counter() - t0

    # evaluate (your existing function writes artifacts)
    eval_t0 = time.perf_counter()
    save_sfx = f"vanilla_{params['depth']}_{params['minbucket']}_{params['cp']}"
    metrics_std = evaluate_binary_oct(
        model,
        X_test,
        y_test,
        preprocessor,
        feat_names,
        X_val_df=X_val,
        y_val=y_val,
        results_dir=str(out_dir),
        save_suffix=save_sfx,
    )
    eval_time = time.perf_counter() - eval_t0

    row = {
        "code": code,
        "method": "vanilla_100feat",
        "baseline_year": args.baseline_year,
        "outcome_year": args.outcome_year,
        "train_test_seed": args.train_test_seed,
        "n_train": int(len(train_pd)),
        "n_train_pos": nP,
        "n_train_neg": nN,
        "best_depth": params['depth'],
        "best_minbucket": params['minbucket'],
        "best_cp": params['cp'],
        "training_time_s": float(train_time),
        "eval_time_s": float(eval_time),
        "run_dir": str(out_dir),
    }

    # merge std metrics (AUC/PR-AUC etc. if present) + threshold metrics
    for k, v in metrics_std.items():
        # keep only JSON-safe scalars
        if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
            row[f"test_{k}"] = float(v) if v is not None else np.nan

    
    return row


def main():
    args = parse_args()
    codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Codes: {codes}")
    print(f"Features dir: {args.features_dir}")
    print(f"Output root: {args.output_root}")
    print(f"OCT grid: depths={args.depths}, minbuckets={args.minbuckets}, cps={args.cps}")
    print(f"Spec floor: {args.spec_floor}")
    print()

    rows = []
    for code in codes:
        try:
            rows.append(run_one_cohort(code, args))
        except Exception as e:
            print(f"\nERROR in cohort {code}: {e}")
            traceback.print_exc()
            rows.append({"code": code, "method": "vanilla_100feat", "error": str(e)})

    df_out = pd.DataFrame(rows)
    out_csv = out_root / "summary_vanilla_multi_cohort.csv"
    file_exists = os.path.isfile(out_csv)
    df_out.to_csv(out_csv, mode='a', index=False, header=not file_exists)
    print(f"\nSaved summary: {out_csv} (rows={len(df_out)})")

    # quick view
    show_cols = [
        "code",
        "n_train_pos",
        "n_train_neg",
        "test_recall_at_best_gmean",
        "test_specificity_at_best_gmean",
        "test_recall_at_specfloor",
        "test_specificity_at_specfloor",
        "test_pr_auc",
        "test_auc",
        "best_cp",
    ]
    avail = [c for c in show_cols if c in df_out.columns]
    if avail:
        print("\n" + df_out[avail].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
