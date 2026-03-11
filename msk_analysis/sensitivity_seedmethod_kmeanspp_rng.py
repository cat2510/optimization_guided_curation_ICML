#!/usr/bin/env python3
"""
sensitivity_seedmethod_kmeanspp_rng.py
======================================

Grid-sensitivity experiment:
  - seed_method in {"smart", "centroid"}
  - kmeans++ RNG seed in {0..K-1}  (controls D^2 sampling trajectory)
  - candidate pool size M in a list

Pipeline per config:
  Stage A: candidate pool of size M via k-means++ (D^2 sampling) on d_nn
  Stage B: 1:1 matching from candidate pool using d_pn
  Train OCT on (N minority + N matched majority), evaluate on fixed test

Outputs:
  results CSV with metrics + timings, one row per (seed_method, kmeanspp_seed, M)

Usage:
  python sensitivity_seedmethod_kmeanspp_rng.py \
    --parquet_path msk_2017_18_full.parquet \
    --distances_dir ./precomputed_distances_msk_medical_only \
    --M_values 20000,40000,60000,80000\
    --kmeanspp_seeds 0,1,2,3 \
    --outdir ./sens_seedmethod_rng
"""

from __future__ import annotations

import os, sys, time, argparse, traceback
from typing import List, Optional

import numpy as np
import pandas as pd
# Use pd.read_parquet (not Spark) for deterministic row order; Spark may shuffle and change train/test split.

# Path setup to import project modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import importlib

import public.two_stage_kcenter_match
importlib.reload(public.two_stage_kcenter_match)
from public.two_stage_kcenter_match import two_stage_kcenter_then_match

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    finetune_oct,
    evaluate_binary_oct,
)

TRAIN_TEST_SEED = 123

OCT_DEPTHS = [5, 7]
OCT_MINBUCKETS = [150]
OCT_CPS = [0.0001, 0.001, 0.01]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--distances_dir", type=str, default="./precomputed_distances_msk_medical_only")
    p.add_argument("--outdir", type=str, default="./sens_seedmethod_rng")
    p.add_argument("--feature_set", type=str, choices=["medical_only", "all_cost", "less_cost"], default="all_cost")

    p.add_argument("--M_values", type=str, default="20000,40000,60000,80000")
    p.add_argument("--kmeanspp_seeds", type=str, default="0,1,2,3")
    p.add_argument("--seed_methods", type=str, default="smart,centroid")

    p.add_argument("--use_kmeanspp", action="store_true", default=True)
    p.add_argument("--force_nearest_per_case", action="store_true", default=False)

    p.add_argument("--resume", action="store_true", help="Skip rows already present in output CSV")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "sensitivity_seedmethod_kmeanspp_rng.csv")

    M_values = [int(x) for x in args.M_values.split(",") if x.strip()]
    kpp_seeds = [int(x) for x in args.kmeanspp_seeds.split(",") if x.strip()]
    seed_methods = [x.strip() for x in args.seed_methods.split(",") if x.strip()]

    # Load data
    df = pd.read_parquet(args.parquet_path)

    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    COST_COLUMNS = [
        col for col in df.columns
        if ("cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower()
            or "decreasing" in col.lower() or "skewness" in col.lower() or "kurtosis" in col.lower()
            or "cv" in col.lower() or "range" in col.lower())
        and "2018" not in col
    ]
    AUXILIARY_COST_COLUMNS = [col for col in df.columns if col.startswith("comorbidity_only") or col.startswith("msk_procedure")]

    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    if args.feature_set == "medical_only":
        exclude_cols += COST_COLUMNS
    elif args.feature_set == "less_cost":
        exclude_cols += AUXILIARY_COST_COLUMNS
    # else all_cost: keep cost cols

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Split (fixed)
    _, _, train_pd, test_pd = train_test_split_enrol(df, target_col=target_col, test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED)
    _, _, val_pd, test_pd = train_test_split_enrol(test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=TRAIN_TEST_SEED)

    cases = train_pd[train_pd[target_col] == 1].copy()
    controls = train_pd[train_pd[target_col] == 0].copy()
    n_cases = len(cases)
    n_controls = len(controls)

    print(f"Train/Val/Test: {len(train_pd):,} / {len(val_pd):,} / {len(test_pd):,}")
    print(f"Cases N={n_cases:,} | Controls={n_controls:,} | Feature cols={len(feature_cols):,}")

    # Distances
    PN_H5_PATH = os.path.join(args.distances_dir, "distances_majority_minority.h5")
    DNN_OUT_DIR = os.path.join(args.distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}")
    dnn_matrix_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_enrolids.npy")
    for p in [PN_H5_PATH, dnn_matrix_npy, dnn_enrolids_npy]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Precompute X_majority_leaf for centroid (aligned to dnn_ids)
    dnn_ids = np.load(dnn_enrolids_npy)
    # Choose numeric columns for centroid
    numeric_feature_cols = [c for c in feature_cols if c in df.select_dtypes(include="number").columns]
    if not numeric_feature_cols:
        numeric_feature_cols = TRUE_NUM_COLUMNS + BIN_FLAG_COLUMNS

    controls_by_id = controls.set_index("ENROLID")
    X_majority_leaf = (
        controls_by_id.reindex(dnn_ids)[numeric_feature_cols]
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    # sanity: ensure dnn_ids are in train controls
    missing_rows = np.isnan(controls_by_id.reindex(dnn_ids)[numeric_feature_cols].to_numpy()).all(axis=1).sum()
    if missing_rows > 0:
        raise RuntimeError(f"{missing_rows} controls in dnn_ids are missing from controls frame (alignment issue).")

    # Resume skip logic
    done = set()
    if args.resume and os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        for _, r in prev.iterrows():
            done.add((str(r["seed_method"]), int(r["kmeanspp_seed"]), int(r["M"])))

    all_rows = []

    def append_row(row: dict):
        nonlocal all_rows
        all_rows.append(row)
        # stream append (safe for interruptions)
        df_row = pd.DataFrame([row])
        header = (not os.path.exists(results_path)) or (os.path.getsize(results_path) == 0)
        df_row.to_csv(results_path, mode="a", index=False, header=header)

    # Loop grid
    for seed_method in seed_methods:
        for kpp_seed in kpp_seeds:
            for M in M_values:
                key = (seed_method, kpp_seed, M)
                if key in done:
                    print(f"[SKIP] seed_method={seed_method} kpp_seed={kpp_seed} M={M}")
                    continue

                print("\n" + "="*90)
                print(f"seed_method={seed_method} | kmeanspp_seed={kpp_seed} | M={M:,}")
                print("="*90)

                t0 = time.perf_counter()
                try:
                    # Call two-stage pipeline
                    match_t0 = time.perf_counter()
                    matching_result = two_stage_kcenter_then_match(
                        leaf_controls_enrolids=controls["ENROLID"].values.astype(np.int64),
                        leaf_cases_enrolids=cases["ENROLID"].values.astype(np.int64),
                        leaf_nn_matrix_npy=dnn_matrix_npy,
                        leaf_nn_enrolids_npy=dnn_enrolids_npy,
                        pn_h5_path=PN_H5_PATH,
                        M=int(M),
                        use_adaptive_pool=False,
                        tau=None,
                        plateau_eps=0.01,
                        force_nearest_per_case=bool(args.force_nearest_per_case),
                        force_topm=1,
                        assignment_topk_start=None,  # exact matching
                        seed_method=seed_method,
                        random_state=int(kpp_seed),  # <-- assumed to seed k-means++ RNG too
                        matching_ratio=1,
                        X_majority_leaf=(X_majority_leaf if seed_method == "centroid" else None),
                        case_weighting=None,
                        use_kmeanspp=bool(args.use_kmeanspp),
                    )
                    match_time = time.perf_counter() - match_t0

                    selected_control_enrolids = np.asarray(matching_result["selected_control_enrolids"], dtype=np.int64)
                    match_costs = np.asarray(matching_result["match_costs"], dtype=np.float32)

                    # Ensure 1:1 dataset truly has N unique controls
                    uniq_ctrl = np.unique(selected_control_enrolids)
                    if len(uniq_ctrl) != n_cases:
                        raise AssertionError(f"Expected {n_cases} unique matched controls, got {len(uniq_ctrl)}")

                    # Build balanced training set
                    selected_majority = controls[controls["ENROLID"].isin(uniq_ctrl)].copy()
                    train_bal = pd.concat([cases, selected_majority], ignore_index=True)

                    # Train OCT
                    train_t0 = time.perf_counter()
                    model, params, _, preprocessor, feat_names = finetune_oct(
                        X_train=train_bal[feature_cols],
                        y_train=train_bal[target_col],
                        X_val=val_pd[feature_cols],
                        y_val=val_pd[target_col],
                        categorical_cols=CAT_COLUMNS,
                        numeric_cols=TRUE_NUM_COLUMNS,
                        binary_cols=BIN_FLAG_COLUMNS,
                        depths=OCT_DEPTHS,
                        minbuckets=OCT_MINBUCKETS,
                        cps=OCT_CPS,
                        tree_kind="oct",
                        verbose=False,
                        random_seed=TRAIN_TEST_SEED,
                    )
                    train_time = time.perf_counter() - train_t0

                    # Evaluate
                    eval_t0 = time.perf_counter()
                    metrics = evaluate_binary_oct(
                        model,
                        test_pd[feature_cols],
                        test_pd[target_col],
                        preprocessor,
                        feat_names,
                        X_val_df=val_pd[feature_cols],
                        y_val=val_pd[target_col],
                        results_dir=args.outdir,
                        save_suffix=f"sens_{seed_method}_kpp{kpp_seed}_M{M}",
                    )
                    eval_time = time.perf_counter() - eval_t0
                    total_time = time.perf_counter() - t0

                    # Params unpack
                    if isinstance(params, dict):
                        bd, bm, bcp = params.get("depth"), params.get("minbucket"), params.get("cp")
                    else:
                        bd, bm, bcp = params[0], params[1], params[2]

                    row = {
                        "seed_method": seed_method,
                        "kmeanspp_seed": int(kpp_seed),
                        "M": int(M),
                        "n_cases": int(n_cases),
                        "n_controls_train": int(len(uniq_ctrl)),
                        "match_cost_mean": float(np.mean(match_costs)),
                        "match_cost_q50": float(np.quantile(match_costs, 0.5)),
                        "match_cost_q90": float(np.quantile(match_costs, 0.9)),
                        "matching_time_s": float(match_time),
                        "train_time_s": float(train_time),
                        "eval_time_s": float(eval_time),
                        "total_time_s": float(total_time),
                        "best_depth": bd,
                        "best_minbucket": bm,
                        "best_cp": bcp,
                    }
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            if isinstance(v, (int, float, np.floating, np.integer)):
                                row[k] = float(v)

                    append_row(row)
                    print(f"✓ PR-AUC={row.get('pr_auc', np.nan):.4f} | AUC={row.get('auc', np.nan):.4f} | match_q90={row['match_cost_q90']:.4f}")

                except Exception as e:
                    traceback.print_exc()
                    append_row({
                        "seed_method": seed_method,
                        "kmeanspp_seed": int(kpp_seed),
                        "M": int(M),
                        "error": str(e),
                    })
                    continue

    print(f"\nDone. Results at: {results_path}")


if __name__ == "__main__":
    main()