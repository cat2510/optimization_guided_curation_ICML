#!/usr/bin/env python3
"""
vanilla_oct_multi_cohort_flex_target.py
=======================================

Variant of vanilla_oct_multi_cohort.py with flexible target selection.

Target can be top 5%, 2%, 1%, or 0.5% cost (computed from annual_cost_<year>_deflated
using adaptive percentiles). Default: top 2%.

For each cohort code:
  - load engineered feature parquet
  - define target via --target_pct (5, 2, 1, 0.5) or explicit target_col
  - drop all columns containing outcome_year to prevent leakage
  - split train/val/test by ENROLID (same helper as MSK)
  - train OCT with a small grid
  - evaluate on test

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


# Supported target percentiles: 5%, 2%, 1%, 0.5%
VALID_TARGET_PCTS = (5, 2, 1, 0.5)
DEFAULT_TARGET_PCT = 2


def _parse_target_pct_from_col(col: str) -> float | None:
    """Parse top_X_pct_cost_ or top_0_5_pct_cost_ style names. Returns pct or None."""
    m = re.match(r"top_(\d+(?:_\d+)?)_pct_cost_", col)
    if not m:
        return None
    s = m.group(1).replace("_", ".")
    try:
        return float(s)
    except ValueError:
        return None


def pick_target_and_features(
    df: pd.DataFrame,
    baseline_year: int,
    outcome_year: int | list[int],
    target_col: str | None = None,
    target_pct: float = DEFAULT_TARGET_PCT,
    feature_regex: str = "",
) -> tuple[str, list[str]]:
    """
    Build or use target column for cost-based outcome. Supports flexible percentiles.

    Args:
        df: Cohort dataframe.
        baseline_year: Baseline year (unused here, kept for API compatibility).
        outcome_year: Single int or list (e.g. [2018, 2019] for I25/I50).
        target_col: Explicit target column name. If None, derived from target_pct.
        target_pct: Percentile for top-cost definition. One of 5, 2, 1, 0.5.
                   Column name: top_5_pct_cost_, top_2_pct_cost_, top_1_pct_cost_, top_0_5_pct_cost_
        feature_regex: Optional regex to filter feature columns.

    Returns:
        (target_col, feature_cols)

    If target_col is not in df, it is computed from annual_cost_{year}_deflated
    using adaptive quantile (e.g. 0.98 for 2%, 0.995 for 0.5%).
    """
    years = [outcome_year] if isinstance(outcome_year, int) else outcome_year
    label_suffix = "_".join(str(y) for y in years)

    if target_col is None:
        if target_pct not in VALID_TARGET_PCTS:
            raise ValueError(
                f"target_pct must be one of {VALID_TARGET_PCTS}, got {target_pct}"
            )
        pct_str = "0_5" if target_pct == 0.5 else str(int(target_pct))
        target_col = f"top_{pct_str}_pct_cost_{label_suffix}"
        pct_for_quantile = target_pct
    else:
        pct_for_quantile = _parse_target_pct_from_col(target_col)
        if pct_for_quantile is None:
            pct_for_quantile = target_pct  # fallback when col exists, or for custom names

    quantile = 1.0 - (pct_for_quantile / 100.0)  # 5%->0.95, 2%->0.98, 1%->0.99, 0.5%->0.995

    if target_col not in df.columns:
        pct_str = "0_5" if pct_for_quantile == 0.5 else str(int(pct_for_quantile))
        alias = f"top_{pct_str}_pct_cost_{years[0]}" if years else None
        if alias and alias in df.columns:
            target_col = alias
        else:
            annual_col = f"annual_cost_{years[0]}_deflated"
            if annual_col in df.columns:
                thr = float(df[annual_col].quantile(quantile))
                df[target_col] = (df[annual_col] >= thr).astype(int)
                print(f"  Created {target_col} from {annual_col} @{quantile*100:.1f}th pct = {thr:,.2f}")
            else:
                raise ValueError(
                    f"Need '{target_col}' or '{annual_col}' in the cohort features parquet."
                )

    def contains_outcome_year(col: str) -> bool:
        return any(str(y) in col for y in years)

    exclude = ["ENROLID", target_col] + [c for c in df.columns if contains_outcome_year(c)]
    feature_cols = [c for c in df.columns if c not in exclude]

    if feature_regex:
        pat = re.compile(feature_regex)
        feature_cols = [c for c in feature_cols if pat.search(c)]

    if not feature_cols:
        raise ValueError("No feature columns left after exclusions/regex.")
    print(f"  Number of feature columns: {len(feature_cols)}")
    print(f" TARGET COLUMN: {target_col}")
    return target_col, feature_cols


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", type=str,
                   default= "F32, I25",
                   help="Comma-separated cohort codes")
    p.add_argument("--features_dir", type=str,
                   default="/Users/cat2510/my_projects/misc_conditions/misc_conditions_features_with_meds",
                   help="Directory containing <CODE>_features_<baseline>_<outcome>([_with_meds]).parquet")
    p.add_argument("--baseline_year", type=int, default=2017)
    p.add_argument("--outcome_year", type=int, default=2018)
    p.add_argument("--target_pct", type=float, default=DEFAULT_TARGET_PCT,
                   choices=VALID_TARGET_PCTS,
                   help=f"Top-cost percentile: one of {VALID_TARGET_PCTS}. Default: {DEFAULT_TARGET_PCT}%%")
    p.add_argument("--target_col", type=str, default=None,
                   help="Override: explicit target column (e.g. top_1_pct_cost_2018). If set, ignores --target_pct.")
    p.add_argument("--train_test_seed", type=int, default=123)
    p.add_argument("--output_root", type=str, default="/Users/cat2510/scratch/oct_vanilla_big_cohorts")

    # OCT grid (small, as requested)
    p.add_argument("--depths", nargs="+", type=int, default=[7])
    p.add_argument("--minbuckets", nargs="+", type=int, default=[100,150]) # [200] used to be default
    p.add_argument("--cps", nargs="+", type=float, default=[0.0001, 0.001])

    # thresholding metric
    p.add_argument("--spec_floor", type=float, default=0.60)

    # spark
    p.add_argument("--spark_app", type=str, default="VanillaOCTMultiCohortFlexTarget")

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


def _merge_annual_cost_from_with_meds(
    df: pd.DataFrame,
    feat_path: Path,
    features_dir: Path,
    code: str,
    baseline_year: int,
    outcome_year: int,
) -> pd.DataFrame:
    """
    If df was loaded from a 100feat parquet, annual_cost_{year}_deflated may be missing.
    Merge it from the corresponding _with_meds parquet on ENROLID.
    """
    if "_100feat.parquet" not in str(feat_path):
        return df

    years = [outcome_year] if isinstance(outcome_year, int) else outcome_year
    year = years[0]
    annual_col = f"annual_cost_{year}_deflated"

    if annual_col in df.columns:
        return df

    with_meds_path = features_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.parquet"
    if not with_meds_path.exists():
        raise FileNotFoundError(
            f"100feat parquet lacks '{annual_col}'. Need {with_meds_path.name} to merge it."
        )

    src = pd.read_parquet(with_meds_path, columns=["ENROLID", annual_col])
    df = df.merge(src, on="ENROLID", how="left")
    print(f"  Merged {annual_col} from {with_meds_path.name}")
    return df


# ----------------------------
# Per-cohort runner
# ----------------------------
def run_one_cohort(code: str, args) -> dict:
    print(f"\n{'='*90}\nCOHORT: {code}\n{'='*90}")

    feat_path = resolve_feature_path(Path(args.features_dir), code, args.baseline_year, args.outcome_year)
    print(f"  Features: {feat_path}")
    df = pd.read_parquet(feat_path)

    outcome_years = [2018, 2019] if code in ("I25", "I50") else args.outcome_year

    # 100feat parquets lack annual_cost_*_deflated; merge from _with_meds if needed
    df = _merge_annual_cost_from_with_meds(
        df, feat_path, Path(args.features_dir), code, args.baseline_year, args.outcome_year
    )

    target_col, feature_cols = pick_target_and_features(
        df,
        args.baseline_year,
        outcome_years,
        target_col=args.target_col,
        target_pct=args.target_pct,
    )

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

    out_dir = Path(args.output_root) / code / f"vanilla_{args.target_pct}_pct_cost"
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
        "method": f"vanilla_{args.target_pct}_pct_cost",
        "target_col": target_col,
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
    print(f"Target: {args.target_col or f'top_{args.target_pct}_pct_cost (adaptive)'}")
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
        "target_col",
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
