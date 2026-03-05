#!/usr/bin/env python3
"""
curated_vs_random_1to1_multi_cohort_oct.py
==========================================

For each cohort code:
  - load engineered feature parquet (already built)
  - split train/val/test by ENROLID
  - CURATED 1:1: two-stage k-center + exact matching_ratio=1
  - RANDOM  1:1: random undersampling of controls
  - Train+tune OCT (IAI) on each undersample
  - Evaluate with:
      (A) threshold maximizing G-mean on validation
      (B) threshold maximizing recall subject to specificity >= 0.60 on validation

Outputs:
  output_root/<CODE>/{curated,random_s<seed>}/
  output_root/summary_all_cohorts.csv
"""

import os
import sys
import re
import json
import time
import math
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo import path (same pattern you used)
# ---------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import importlib

# Matching + distances
import public.two_stage_kcenter_match
importlib.reload(public.two_stage_kcenter_match)
from public.two_stage_kcenter_match import two_stage_kcenter_then_match

# IAI OCT + feature helpers / splits
from public.model_IAI import *  # assumes finetune_oct, evaluate_binary_oct, train_test_split_enrol, etc.

from pyspark.sql import SparkSession

# Optional sklearn metrics for robust thresholding
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        roc_curve,
        confusion_matrix,
    )
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--codes", type=str,
                   default="C50,C61,E10,E11,E66,E78,F32,I25,I50",
                   help="Comma-separated cohort codes")

    p.add_argument("--features_dir", type=str,
                   default="/Users/cat2510/my_projects/misc_conditions/misc_conditions_features_with_meds",
                   help="Directory with <CODE>_features_<baseline>_<outcome>([_with_meds]).parquet")

    p.add_argument("--baseline_year", type=int, default=2017)
    p.add_argument("--outcome_year", type=int, default=2018)

    p.add_argument("--distances_dir_template", type=str,
                   default="/Users/cat2510/my_projects/misc_conditions/precomputed_distances_{code}_100_features",
                   help="Template for cohort-specific distance dir, e.g. ./precomputed_distances_{code}")

    p.add_argument("--train_test_seed", type=int, default=123)

    p.add_argument("--random_seeds", nargs="+", type=int, default=list(range(5)),
                   help="Seeds for random 1:1 baseline")

    p.add_argument("--M_pool", type=int, default=50000) # or 80000 for E11 Mar 5
    p.add_argument("--seed_method", type=str, default="smart")

    # OCT grid
    p.add_argument("--depths", nargs="+", type=int, default=[7])
    p.add_argument("--minbuckets", nargs="+", type=int, default=[100])
    p.add_argument("--cps", nargs="+", type=float, default=[0.0001, 0.001, 0.01])

    # Evaluation
    p.add_argument("--spec_floor", type=float, default=0.60,
                   help="Specificity floor for 'max recall subject to spec>=floor' thresholding on val")

    # Output
    p.add_argument("--output_root", type=str, default="/Users/cat2510/my_projects/misc_conditions/oct_curated_vs_random_1to1_multi")
    p.add_argument("--spark_app", type=str, default="CuratedVsRandom1to1MultiCohort")

    return p.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def resolve_feature_path(features_dir: Path, code: str, baseline_year: int, outcome_year: int) -> Path:
    """
    Try common naming patterns (with/without meds augmentation).
    """
    candidates = [
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_100feat.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No features parquet found for {code}. Tried: {candidates}")


def resolve_distance_files(dist_dir: Path, seed: int) -> tuple[str, str, str]:
    """
    Expected layout (per cohort):
      dist_dir/distances_majority_minority.h5
      dist_dir/global_dnn_seed_<seed>/leaf_global_dnn_matrix.npy
      dist_dir/global_dnn_seed_<seed>/leaf_global_dnn_enrolids.npy
    """
    pn_h5 = dist_dir / "distances_majority_minority.h5"
    dnn_dir = dist_dir / f"global_dnn_seed_{seed}"
    dnn_mat = dnn_dir / "leaf_global_dnn_matrix.npy"
    dnn_ids = dnn_dir / "leaf_global_dnn_enrolids.npy"

    for f in [pn_h5, dnn_mat, dnn_ids]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required distance file: {f}")

    return str(pn_h5), str(dnn_mat), str(dnn_ids)


def pick_target_and_features(df: pd.DataFrame, baseline_year: int, outcome_year: int) -> tuple[str, list[str]]:
    """
    Use top_2_pct_cost_<outcome_year> if present, otherwise construct from annual_cost_<outcome_year>_deflated.
    Exclude columns containing outcome_year to prevent leakage.
    """
    target_col = f"top_2_pct_cost_{outcome_year}"
    annual_col = f"annual_cost_{outcome_year}_deflated"

    if target_col not in df.columns:
        if annual_col in df.columns:
            thr = float(df[annual_col].quantile(0.98))
            df[target_col] = (df[annual_col] >= thr).astype(int)
            print(f"  Created {target_col} from {annual_col} @98th pct = {thr:,.2f}")
        else:
            raise ValueError(
                f"Need either '{target_col}' or '{annual_col}' in the cohort features parquet."
            )

    # leakage guard: drop outcome year columns
    exclude = ["ENROLID", target_col] + [c for c in df.columns if str(outcome_year) in c]
    feature_cols = [c for c in df.columns if c not in exclude]

    return target_col, feature_cols


def _predict_scores(model, X_df, preprocessor, feature_names) -> np.ndarray:
    Xp = preprocessor.transform(X_df)
    Xp_df = pd.DataFrame(Xp, columns=feature_names)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xp_df)
        if isinstance(proba, pd.DataFrame):
            proba = proba.values
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        return proba.reshape(-1).astype(float)

    # Fallback: decision/predict (less ideal)
    pred = model.predict(Xp_df)
    return np.asarray(pred).astype(float)


def _rates_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    y_pred = (scores >= thr).astype(int)

    # confusion_matrix: [[tn, fp],[fn,tp]]
    if _HAS_SK:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    gmean = math.sqrt(max(recall * specificity, 0.0))

    return {
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "recall": float(recall),
        "specificity": float(specificity),
        "gmean": float(gmean),
    }


def pick_thresholds_from_val(y_val: np.ndarray, s_val: np.ndarray, spec_floor: float) -> dict:
    """
    Returns two thresholds chosen on validation scores:
      - thr_gmean: maximize sqrt(tpr * (1-fpr)) on val
      - thr_recall_spec: maximize recall subject to specificity >= spec_floor on val
    """
    # thresholds from ROC curve (sklearn includes +inf sentinel; we filter finite)
    if _HAS_SK:
        fpr, tpr, thr = roc_curve(y_val, s_val)
        thr = np.asarray(thr)
        finite = np.isfinite(thr)
        fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
        spec = 1.0 - fpr
        gmean = np.sqrt(np.maximum(tpr * spec, 0.0))

        # A) best gmean
        i_g = int(np.nanargmax(gmean)) if len(gmean) else 0
        thr_gmean = float(thr[i_g])

        # B) max recall subject to spec>=floor
        ok = np.where(spec >= spec_floor)[0]
        if len(ok):
            # among feasible, pick max recall; tie-break by gmean
            best = ok[np.lexsort((gmean[ok], tpr[ok]))][-1]
            thr_recall_spec = float(thr[best])
        else:
            # fallback: closest-to-floor by spec, then max recall
            best = int(np.nanargmax(spec)) if len(spec) else i_g
            thr_recall_spec = float(thr[best])

        return {
            "thr_gmean": thr_gmean,
            "thr_recall_spec": thr_recall_spec,
        }

    # No sklearn: use unique score thresholds (slower but fine)
    thr = np.unique(s_val)[::-1]
    best_g = (-1.0, None)
    best_r = (-1.0, None)

    for t in thr:
        rates = _rates_at_threshold(y_val, s_val, float(t))
        if rates["gmean"] > best_g[0]:
            best_g = (rates["gmean"], float(t))
        if rates["specificity"] >= spec_floor and rates["recall"] > best_r[0]:
            best_r = (rates["recall"], float(t))

    thr_gmean = best_g[1] if best_g[1] is not None else float(np.median(s_val))
    thr_recall_spec = best_r[1] if best_r[1] is not None else thr_gmean
    return {"thr_gmean": float(thr_gmean), "thr_recall_spec": float(thr_recall_spec)}


def evaluate_oct_custom(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
    feat_names: list[str],
    spec_floor: float,
) -> dict:
    """
    Adds the two metrics you care about:
      - recall/spec at val-best-gmean threshold
      - recall/spec at val-max-recall subject to spec>=floor threshold
    Also returns AUC / PR-AUC on test if sklearn is present.
    """
    yv = np.asarray(y_val).astype(int)
    yt = np.asarray(y_test).astype(int)

    s_val = _predict_scores(model, X_val, preprocessor, feat_names)
    s_tst = _predict_scores(model, X_test, preprocessor, feat_names)

    thrs = pick_thresholds_from_val(yv, s_val, spec_floor=spec_floor)

    out = {}

    # Apply thresholds on TEST
    r_g = _rates_at_threshold(yt, s_tst, thrs["thr_gmean"])
    r_s = _rates_at_threshold(yt, s_tst, thrs["thr_recall_spec"])

    out.update({
        "val_thr_best_gmean": float(thrs["thr_gmean"]),
        "test_recall_at_best_gmean": r_g["recall"],
        "test_specificity_at_best_gmean": r_g["specificity"],
        "test_gmean_at_best_gmean": r_g["gmean"],

        "val_thr_max_recall_specfloor": float(thrs["thr_recall_spec"]),
        "spec_floor": float(spec_floor),
        "test_recall_at_specfloor": r_s["recall"],
        "test_specificity_at_specfloor": r_s["specificity"],
        "test_gmean_at_specfloor": r_s["gmean"],
    })

    if _HAS_SK:
        out["test_auc"] = float(roc_auc_score(yt, s_tst)) if len(np.unique(yt)) == 2 else float("nan")
        out["test_pr_auc"] = float(average_precision_score(yt, s_tst)) if len(np.unique(yt)) == 2 else float("nan")

    return out


def compute_num_leaves(model, X_df, preprocessor, feature_names):
    X_proc = preprocessor.transform(X_df)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
    leaves = model.apply(X_proc_df)
    return int(len(pd.unique(leaves)))


# ----------------------------
# Per-cohort runner
# ----------------------------
def run_one_cohort(code: str, spark: SparkSession, args) -> list[dict]:
    print(f"\n{'='*90}")
    print(f"COHORT: {code}")
    print(f"{'='*90}")

    features_dir = Path(args.features_dir)
    feat_path = resolve_feature_path(features_dir, code, args.baseline_year, args.outcome_year)
    print(f"  Features: {feat_path}")

    # Load features parquet
    df_spark = spark.read.format("parquet").load(str(feat_path))
    df = df_spark.toPandas()

    target_col, feature_cols = pick_target_and_features(df, args.baseline_year, args.outcome_year)
    print(f"  Target: {target_col}")
    print(f"  #features: {len(feature_cols)}")

    # Column types for OCT preprocessor (your helpers)
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    # Train/val/test split (ENROLID-aware)
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

    print(f"  Split sizes: train={train_pd.shape}, val={val_pd.shape}, test={test_pd.shape}")

    cases = train_pd[train_pd[target_col] == 1].copy()
    controls = train_pd[train_pd[target_col] == 0].copy()
    nP = len(cases)
    nC_avail = len(controls)

    if nP == 0:
        raise ValueError(f"{code}: no positives in training split (target={target_col}).")

    if nC_avail < nP:
        raise ValueError(f"{code}: not enough controls for 1:1 (controls={nC_avail}, cases={nP}).")

    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)

    print(f"  Train positives nP={nP:,}, controls={nC_avail:,}, orig ratio={nC_avail/nP:.2f}:1")
    print(f"  Using 1:1 => nC={nP:,}")

    # Distances
    dist_dir = Path(args.distances_dir_template.format(code=code))
    pn_h5, dnn_mat, dnn_ids = resolve_distance_files(dist_dir, seed=args.train_test_seed)
    print(f"  Distances dir: {dist_dir}")

    out_root = Path(args.output_root) / f"{code}_100feat"
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    # ----------------------------
    # CURATED 1:1
    # ----------------------------
    curated_dir = out_root / "curated_1to1"
    curated_dir.mkdir(parents=True, exist_ok=True)
    (curated_dir / "predictions").mkdir(exist_ok=True)

    print(f"\n  {'='*72}\n  CURATED 1:1 (k-center + matching)\n  {'='*72}")
    try:
        t0 = time.perf_counter()

        match_t0 = time.perf_counter()
        matching_result = two_stage_kcenter_then_match(
            leaf_controls_enrolids=control_enrolids.copy(),
            leaf_cases_enrolids=case_enrolids.copy(),
            leaf_nn_matrix_npy=dnn_mat,
            leaf_nn_enrolids_npy=dnn_ids,
            pn_h5_path=pn_h5,
            M=args.M_pool,
            use_adaptive_pool=False,
            force_nearest_per_case=False,
            force_topm=1,
            assignment_topk_start=None,  # exact matching
            seed_method=args.seed_method,
            matching_ratio=1,
            case_weighting=None,
            quota_cfg=None,
        )
        matching_time = time.perf_counter() - match_t0

        selected_ctrl = list(set(matching_result["selected_control_enrolids"]))
        if len(selected_ctrl) != nP:
            raise RuntimeError(f"Expected {nP} matched controls, got {len(selected_ctrl)}")

        mean_match_cost = float(np.asarray(matching_result["match_costs"]).mean())

        all_minority = cases.copy()
        selected_majority = controls[controls["ENROLID"].isin(selected_ctrl)].copy()

        undersampled = pd.concat([all_minority, selected_majority], axis=0, ignore_index=True)

        us_path = curated_dir / "undersampled_curated_1to1.csv"
        undersampled.to_csv(us_path, index=False)

        # Train OCT
        train_t0 = time.perf_counter()
        model, params, grid_df, preprocessor, feat_names = finetune_oct(
            X_train=undersampled[feature_cols],
            y_train=undersampled[target_col],
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
        train_time = time.perf_counter() - train_t0
        val_pr_auc = float(grid_df.iloc[0]["pr_auc"]) if "pr_auc" in grid_df.columns else float("nan")

        # Keep your existing evaluator (for plots/preds), but also compute the two metrics explicitly
        save_sfx = f"curated_1to1_{params['depth']}_{params['minbucket']}_{params['cp']}"
        _ = evaluate_binary_oct(
            model, X_test, y_test, preprocessor, feat_names,
            X_val_df=X_val, y_val=y_val,
            results_dir=str(curated_dir),
            save_suffix=save_sfx,
        )

        # Save model JSON for deployment (HTML is saved by evaluate_binary_oct)
        if hasattr(model, "write_json"):
            model.write_json(str(curated_dir / f"oct_model_{save_sfx}.json"))

        extra = evaluate_oct_custom(
            model, X_val, y_val, X_test, y_test,
            preprocessor, feat_names,
            spec_floor=args.spec_floor,
        )

        num_leaves = compute_num_leaves(model, X_test, preprocessor, feat_names)
        total_time = time.perf_counter() - t0

        row = {
            "code": code,
            "method": "curated",
            "seed": np.nan,
            "nP": nP,
            "nC": nP,
            "M_pool": args.M_pool,
            "matching_mean_cost": mean_match_cost,
            "best_depth": params["depth"],
            "best_minbucket": params["minbucket"],
            "best_cp": params["cp"],
            "val_pr_auc": val_pr_auc,
            "num_leaves": num_leaves,
            "matching_time_s": matching_time,
            "training_time_s": train_time,
            "total_time_s": total_time,
            "undersample_path": str(us_path),
            "run_dir": str(curated_dir),
        }
        row.update(extra)
        results.append(row)

        print(
            f"  CURATED DONE | "
            f"test_recall@Gmean={row['test_recall_at_best_gmean']:.3f}, "
            f"test_spec@Gmean={row['test_specificity_at_best_gmean']:.3f} | "
            f"test_recall@spec>=0.6={row['test_recall_at_specfloor']:.3f}, "
            f"test_spec@spec>=0.6={row['test_specificity_at_specfloor']:.3f} | "
            f"match_cost={mean_match_cost:.4f}, leaves={num_leaves}"
        )

    except Exception as e:
        print(f"\n  ERROR (curated {code}): {e}")
        traceback.print_exc()
        results.append({"code": code, "method": "curated", "seed": np.nan, "error": str(e)})

    # ----------------------------
    # RANDOM 1:1 (per seed)
    # ----------------------------
    for seed in args.random_seeds:
        rnd_dir = out_root / f"random_1to1_s{seed}"
        rnd_dir.mkdir(parents=True, exist_ok=True)
        (rnd_dir / "predictions").mkdir(exist_ok=True)

        print(f"\n  {'='*72}\n  RANDOM 1:1 (seed={seed})\n  {'='*72}")
        try:
            t0 = time.perf_counter()

            rng = np.random.RandomState(seed)
            sampled_idx = rng.choice(len(control_enrolids), size=nP, replace=False)
            sampled_ctrl = set(control_enrolids[sampled_idx])

            all_minority = cases.copy()
            sampled_majority = controls[controls["ENROLID"].isin(sampled_ctrl)].copy()
            if len(sampled_majority) != nP:
                raise RuntimeError(f"Expected {nP} sampled controls, got {len(sampled_majority)}")

            undersampled = pd.concat([all_minority, sampled_majority], axis=0, ignore_index=True)

            us_path = rnd_dir / f"undersampled_random_1to1_s{seed}.csv"
            undersampled.to_csv(us_path, index=False)

            # Train OCT
            train_t0 = time.perf_counter()
            model, params, grid_df, preprocessor, feat_names = finetune_oct(
                X_train=undersampled[feature_cols],
                y_train=undersampled[target_col],
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
            train_time = time.perf_counter() - train_t0
            val_pr_auc = float(grid_df.iloc[0]["pr_auc"]) if "pr_auc" in grid_df.columns else float("nan")

            save_sfx = f"random_1to1_s{seed}_{params['depth']}_{params['minbucket']}_{params['cp']}"
            _ = evaluate_binary_oct(
                model, X_test, y_test, preprocessor, feat_names,
                X_val_df=X_val, y_val=y_val,
                results_dir=str(rnd_dir),
                save_suffix=save_sfx,
            )

            # Save model JSON for deployment (HTML is saved by evaluate_binary_oct)
            if hasattr(model, "write_json"):
                model.write_json(str(rnd_dir / f"oct_model_{save_sfx}.json"))

            extra = evaluate_oct_custom(
                model, X_val, y_val, X_test, y_test,
                preprocessor, feat_names,
                spec_floor=args.spec_floor,
            )

            num_leaves = compute_num_leaves(model, X_test, preprocessor, feat_names)
            total_time = time.perf_counter() - t0

            row = {
                "code": code,
                "method": "random",
                "seed": seed,
                "nP": nP,
                "nC": nP,
                "M_pool": np.nan,
                "matching_mean_cost": np.nan,
                "best_depth": params["depth"],
                "best_minbucket": params["minbucket"],
                "best_cp": params["cp"],
                "val_pr_auc": val_pr_auc,
                "num_leaves": num_leaves,
                "matching_time_s": np.nan,
                "training_time_s": train_time,
                "total_time_s": total_time,
                "undersample_path": str(us_path),
                "run_dir": str(rnd_dir),
            }
            row.update(extra)
            results.append(row)

            print(
                f"  RANDOM DONE | "
                f"test_recall@Gmean={row['test_recall_at_best_gmean']:.3f}, "
                f"test_spec@Gmean={row['test_specificity_at_best_gmean']:.3f} | "
                f"test_recall@spec>=0.6={row['test_recall_at_specfloor']:.3f}, "
                f"test_spec@spec>=0.6={row['test_specificity_at_specfloor']:.3f} | "
                f"leaves={num_leaves}"
            )

        except Exception as e:
            print(f"\n  ERROR (random {code} seed={seed}): {e}")
            traceback.print_exc()
            results.append({"code": code, "method": "random", "seed": seed, "error": str(e)})

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
    print(f"Codes: {codes}")
    print(f"Features dir: {args.features_dir}")
    print(f"Distances template: {args.distances_dir_template}")
    print(f"Output root: {args.output_root}")
    print(f"OCT grid: depths={args.depths} minbuckets={args.minbuckets} cps={args.cps}")
    print(f"Spec floor for constrained threshold: {args.spec_floor}")
    print()

    spark = SparkSession.builder.appName(args.spark_app).getOrCreate()

    all_rows = []
    for code in codes:
        try:
            all_rows.extend(run_one_cohort(code, spark, args))
        except Exception as e:
            print(f"\nFATAL ERROR in cohort {code}: {e}")
            traceback.print_exc()
            all_rows.append({"code": code, "method": "cohort", "error": str(e)})

    spark.stop()

    # Save summary CSV
    if all_rows:
        df_res = pd.DataFrame(all_rows)
        out_csv = output_root / "summary_all_cohorts.csv"

        # Check if file exists to determine if header is needed
        file_exists = os.path.isfile(out_csv)
        df_res.to_csv(out_csv, mode='a', index=False, header=not file_exists)
        print(f"\nSaved summary: {out_csv} (added rows={len(df_res)})")

        # Quick aggregated print: random mean/std vs curated per cohort
        if "method" in df_res.columns:
            print(f"\n{'='*90}\nAGGREGATED (random mean±std) vs curated per cohort\n{'='*90}")
            for code in sorted(df_res["code"].dropna().unique()):
                sub = df_res[df_res["code"] == code]
                cur = sub[sub["method"] == "curated"]
                rnd = sub[sub["method"] == "random"]

                def ms(col):
                    return (float(rnd[col].mean()), float(rnd[col].std())) if (col in rnd.columns and len(rnd)) else (float("nan"), float("nan"))

                if len(cur):
                    c_r_g = cur["test_recall_at_best_gmean"].iloc[0] if "test_recall_at_best_gmean" in cur.columns else np.nan
                    c_s_g = cur["test_specificity_at_best_gmean"].iloc[0] if "test_specificity_at_best_gmean" in cur.columns else np.nan
                    c_r_s = cur["test_recall_at_specfloor"].iloc[0] if "test_recall_at_specfloor" in cur.columns else np.nan
                    c_s_s = cur["test_specificity_at_specfloor"].iloc[0] if "test_specificity_at_specfloor" in cur.columns else np.nan
                else:
                    c_r_g = c_s_g = c_r_s = c_s_s = np.nan

                r_r_g_m, r_r_g_sd = ms("test_recall_at_best_gmean")
                r_s_g_m, r_s_g_sd = ms("test_specificity_at_best_gmean")
                r_r_s_m, r_r_s_sd = ms("test_recall_at_specfloor")
                r_s_s_m, r_s_s_sd = ms("test_specificity_at_specfloor")

                print(
                    f"{code} | "
                    f"CUR: recall@G={c_r_g:.3f}, spec@G={c_s_g:.3f}, "
                    f"recall@spec={c_r_s:.3f}, spec@spec={c_s_s:.3f} || "
                    f"RND: recall@G={r_r_g_m:.3f}±{r_r_g_sd:.3f}, spec@G={r_s_g_m:.3f}±{r_s_g_sd:.3f}, "
                    f"recall@spec={r_r_s_m:.3f}±{r_r_s_sd:.3f}, spec@spec={r_s_s_m:.3f}±{r_s_s_sd:.3f}"
                )

    else:
        print("No results collected.")

    print("\nDone.")


if __name__ == "__main__":
    main()
