#!/usr/bin/env python3
"""
curated_vs_random_1to1_multi_cohort_oct_flex_target.py
======================================================

Variant of curated_vs_random_1to1_multi_cohort_oct.py with flexible target selection,
aligned with vanilla_oct_multi_cohort_flex_target and precompute_distances_multi_cohort_flex_target.

- Flexible target: --target_pct (5, 2, 1, 0.5) or --target_col
- Optional random 1:1 baseline via --run_random
- Uses model_IAI.evaluate_binary_oct for threshold metrics (removes redundant logic)
- Huge-cohort sampling: 250k rows when len(df) > 500k (aligned with vanilla)
- Output layout: output_root/{code}_top{target_suffix}pct/curated_M{M}_{seed_method}/

Requires precomputed distances from precompute_distances_multi_cohort_flex_target.py
with matching target_pct/target_col and distances_dir_template.
"""

import os
import sys
import re
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import importlib
import public.two_stage_kcenter_match
importlib.reload(public.two_stage_kcenter_match)
from public.two_stage_kcenter_match import two_stage_kcenter_then_match

from public.model_IAI import (
    finetune_oct,
    evaluate_binary_oct,
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)

from vanilla_oct_multi_cohort_flex_target import (
    pick_target_and_features,
    _merge_annual_cost_from_with_meds,
    VALID_TARGET_PCTS,
    DEFAULT_TARGET_PCT,
)

from pyspark.sql import SparkSession


def _target_suffix_from_args(args) -> str:
    """Derive target_suffix for distances/output dirs: '2', '1', '5', '0_5'."""
    if args.target_col:
        m = re.match(r"top_(\d+(?:_\d+)?)_pct_cost_", args.target_col)
        if m:
            return m.group(1)
        return "custom"
    return "0_5" if args.target_pct == 0.5 else str(int(args.target_pct))


def compute_num_leaves(model, X_df, preprocessor, feature_names):
    X_proc = preprocessor.transform(X_df)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
    leaves = model.apply(X_proc_df)
    return int(len(pd.unique(leaves)))


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

    p.add_argument("--target_pct", type=float, default=DEFAULT_TARGET_PCT,
                   choices=VALID_TARGET_PCTS,
                   help=f"Top-cost percentile: one of {VALID_TARGET_PCTS}. Default: {DEFAULT_TARGET_PCT}%%")
    p.add_argument("--target_col", type=str, default=None,
                   help="Override: explicit target column. If set, ignores --target_pct.")

    p.add_argument("--distances_dir_template", type=str,
                   default="/Users/cat2510/scratch/precomputed_distances_{code}_top{target_suffix}pct", 
                   #default="/Users/cat2510/scratch/precomputed_distances_{code}_with_cost_features",
                   help="Template for distance dir. Use {code} and {target_suffix}. Must match precompute_distances_multi_cohort_flex_target.")

    p.add_argument("--run_random", action="store_true", default=False,
                   help="Also run random 1:1 undersampling + OCT baseline (per --random_seeds)")

    p.add_argument("--train_test_seed", type=int, default=123)

    p.add_argument("--random_seeds", nargs="+", type=int, default=list(range(5)),
                   help="Seeds for random 1:1 baseline (used only when --run_random)")

    p.add_argument("--M_pool", type=int, default=80000)
    p.add_argument("--seed_method", type=str, default="random")

    p.add_argument("--feature_regex", type=str, default="",
                   help="Optional regex to filter feature columns (applied after leakage exclusion).")

    p.add_argument("--depths", nargs="+", type=int, default=[7])
    p.add_argument("--minbuckets", nargs="+", type=int, default=[50, 100])
    p.add_argument("--cps", nargs="+", type=float, default=[0.0001, 0.01, 0.001])

    p.add_argument("--spec_floor", type=float, default=0.60,
                   help="Specificity floor (model_IAI uses 0.60; kept for doc).")

    p.add_argument("--output_root", type=str, default="/Users/cat2510/scratch/oct_curated_vs_random_flex_target")
    p.add_argument("--spark_app", type=str, default="CuratedVsRandom1to1FlexTarget")

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


def resolve_distance_files(dist_dir: Path, seed: int) -> tuple[str, str, str]:
    pn_h5 = dist_dir / "distances_majority_minority.h5"
    dnn_dir = dist_dir / f"global_dnn_seed_{seed}"
    dnn_mat = dnn_dir / "leaf_global_dnn_matrix.npy"
    dnn_ids = dnn_dir / "leaf_global_dnn_enrolids.npy"

    for f in [pn_h5, dnn_mat, dnn_ids]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required distance file: {f}")

    return str(pn_h5), str(dnn_mat), str(dnn_ids)


# ----------------------------
# Per-cohort runner
# ----------------------------
def run_one_cohort(code: str, spark: SparkSession, args) -> list[dict]:
    print(f"\n{'='*90}\nCOHORT: {code}\n{'='*90}")

    features_dir = Path(args.features_dir)
    feat_path = resolve_feature_path(features_dir, code, args.baseline_year, args.outcome_year)
    print(f"  Features: {feat_path}")

    df = pd.read_parquet(str(feat_path))
    outcome_years = [2018, 2019] if code in ("I25", "I50") else args.outcome_year

    df = _merge_annual_cost_from_with_meds(
        df, feat_path, features_dir, code, args.baseline_year, args.outcome_year
    )

    target_col, feature_cols = pick_target_and_features(
        df,
        args.baseline_year,
        outcome_years,
        target_col=args.target_col,
        target_pct=args.target_pct,
        feature_regex=args.feature_regex,
    )
    print(f"  Target: {target_col}")
    print(f"  #features: {len(feature_cols)}")

    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df, train_size=250_000, stratify=df[target_col], random_state=args.train_test_seed
        )
        print(f"  Sampled to {len(df):,} rows (stratified on {target_col})")
    else:
        print(f"  Using {len(df):,} rows")

    target_suffix = _target_suffix_from_args(args)
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.train_test_seed
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=args.train_test_seed
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

    dist_dir = Path(args.distances_dir_template.format(code=code, target_suffix=target_suffix))
    pn_h5, dnn_mat, dnn_ids = resolve_distance_files(dist_dir, seed=args.train_test_seed)
    print(f"  Distances dir: {dist_dir}")

    out_root = Path(args.output_root) / f"{code}_top{target_suffix}pct"
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    # ----------------------------
    # CURATED 1:1
    # ----------------------------
    curated_dir = out_root / f"curated_M{args.M_pool}_{args.seed_method}"
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
            assignment_topk_start=None,
            seed_method=args.seed_method,
            matching_ratio=1,
            case_weighting=None,
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

        save_sfx = f"curated_top{target_suffix}pct_M{args.M_pool}_{args.seed_method}_{params['depth']}_{params['minbucket']}_{params['cp']}"
        metrics_std = evaluate_binary_oct(
            model, X_test, y_test, preprocessor, feat_names,
            X_val_df=X_val, y_val=y_val,
            results_dir=str(curated_dir),
            save_suffix=save_sfx,
        )

        if hasattr(model, "write_json"):
            model.write_json(str(curated_dir / f"oct_model_{save_sfx}.json"))

        extra = {
            "test_recall_at_best_gmean": metrics_std["balanced_recall_gmean"],
            "test_specificity_at_best_gmean": metrics_std["balanced_specificity_gmean"],
            "test_recall_at_specfloor": metrics_std["recall_at_specificity_0.6"],
            "test_specificity_at_specfloor": metrics_std["achieved_specificity_0.6"],
        }
        if "auc" in metrics_std:
            extra["test_auc"] = metrics_std["auc"]
        if "pr_auc" in metrics_std:
            extra["test_pr_auc"] = metrics_std["pr_auc"]

        num_leaves = compute_num_leaves(model, X_test, preprocessor, feat_names)
        total_time = time.perf_counter() - t0

        row = {
            "code": code,
            "method": "curated",
            "target_col": target_col,
            "target_suffix": target_suffix,
            "seed": np.nan,
            "nP": nP,
            "nC": nP,
            "M_pool": args.M_pool,
            "seed_method": args.seed_method,
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
        results.append({
            "code": code, "method": "curated", "target_col": target_col, "target_suffix": target_suffix,
            "seed": np.nan, "error": str(e)
        })

    # ----------------------------
    # RANDOM 1:1 (per seed, optional)
    # ----------------------------
    if args.run_random:
        for seed in args.random_seeds:
            rnd_dir = out_root / f"random_s{seed}"
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

                us_path = rnd_dir / f"undersampled_random_s{seed}.csv"
                undersampled.to_csv(us_path, index=False)

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

                save_sfx = f"random_top{target_suffix}pct_s{seed}_{params['depth']}_{params['minbucket']}_{params['cp']}"
                metrics_std = evaluate_binary_oct(
                    model, X_test, y_test, preprocessor, feat_names,
                    X_val_df=X_val, y_val=y_val,
                    results_dir=str(rnd_dir),
                    save_suffix=save_sfx,
                )

                if hasattr(model, "write_json"):
                    model.write_json(str(rnd_dir / f"oct_model_{save_sfx}.json"))

                extra = {
                    "test_recall_at_best_gmean": metrics_std["balanced_recall_gmean"],
                    "test_specificity_at_best_gmean": metrics_std["balanced_specificity_gmean"],
                    "test_recall_at_specfloor": metrics_std["recall_at_specificity_0.6"],
                    "test_specificity_at_specfloor": metrics_std["achieved_specificity_0.6"],
                }
                if "auc" in metrics_std:
                    extra["test_auc"] = metrics_std["auc"]
                if "pr_auc" in metrics_std:
                    extra["test_pr_auc"] = metrics_std["pr_auc"]

                num_leaves = compute_num_leaves(model, X_test, preprocessor, feat_names)
                total_time = time.perf_counter() - t0

                row = {
                    "code": code,
                    "method": "random",
                    "target_col": target_col,
                    "target_suffix": target_suffix,
                    "seed": seed,
                    "nP": nP,
                    "nC": nP,
                    "M_pool": np.nan,
                    "seed_method": np.nan,
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
                results.append({
                    "code": code, "method": "random", "target_col": target_col, "target_suffix": target_suffix,
                    "seed": seed, "error": str(e)
                })

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
    target_suffix = _target_suffix_from_args(args)

    print(f"Codes: {codes}")
    print(f"Features dir: {args.features_dir}")
    print(f"Target: {args.target_col or f'top {args.target_pct}%'} (target_suffix={target_suffix})")
    print(f"Distances template: {args.distances_dir_template}")
    print(f"Output root: {args.output_root}")
    print(f"Run random baseline: {args.run_random}")
    print(f"OCT grid: depths={args.depths} minbuckets={args.minbuckets} cps={args.cps}")
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

    if all_rows:
        df_res = pd.DataFrame(all_rows)
        out_csv = output_root / "summary_curated_vs_random_flex_target.csv"
        file_exists = os.path.isfile(out_csv)
        if file_exists and out_csv.stat().st_size > 0:
            with open(out_csv, "rb+") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    f.write(b"\n")
        df_res.to_csv(out_csv, mode="a", index=False, header=not file_exists)
        print(f"\nSaved summary: {out_csv} (added rows={len(df_res)})")

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
