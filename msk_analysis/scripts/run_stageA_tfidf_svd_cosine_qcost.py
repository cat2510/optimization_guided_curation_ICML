#!/usr/bin/env python
"""
run_stageA_tfidf_svd_cosine_qcost.py
====================================
Stage A representation pipeline: TF-IDF+SVD cosine + quantile-transformed cost.

Precomputes DNN for tfidf_svd_cosine_qcost in ./precomputed_distances_msk_za_tfidf_svd_cosine_qcost/
Uses Gower P-N for sampler seed selection (Stage A only).

Compares: Full majority, Random 1N, Random 10N, Stage A 1N, Stage A 10N (tfidf_svd_cosine_qcost).
Boxplot of annual_cost_2018_deflated (log scale).

Usage
-----
  cd msk_analysis
  python scripts/run_stageA_tfidf_svd_cosine_qcost.py [--parquet_path msk_2017_18_full.parquet]
"""

from __future__ import annotations

import sys
import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

import importlib
import experiments_compare_random_vs_curation
importlib.reload(experiments_compare_random_vs_curation)
from experiments_compare_random_vs_curation import (
    sample_random_controls,
    sample_stageA_dispersed_controls,
)

import exp6_distance_metric_ablation
importlib.reload(exp6_distance_metric_ablation)
from exp6_distance_metric_ablation import ensure_distances_for_metric

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)

# Use pd.read_parquet (not Spark) for deterministic row order; Spark may shuffle and change train/test split.
TRAIN_TEST_SEED = 123
DISTANCES_DIR = "./precomputed_distances_msk_za_tfidf_svd_cosine_qcost"
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description="Stage A TF-IDF+SVD+QCost: cost boxplot")
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--outdir", type=str, default="./stageA_tfidf_svd_cosine_qcost")
    p.add_argument("--distances_dir", type=str, default=DISTANCES_DIR)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip precompute; use existing tfidf DNN + Gower P-N if present. Fails if DNN missing.",
    )
    return p.parse_args()


def run_random_sampling(
    control_enrolids: np.ndarray,
    controls: pd.DataFrame,
    k: int,
    seed: int,
) -> pd.DataFrame:
    ids = sample_random_controls(control_enrolids, k, seed)
    return controls[controls["ENROLID"].isin(ids)].copy()


def run_stageA_sampling(
    control_enrolids: np.ndarray,
    controls: pd.DataFrame,
    cases: pd.DataFrame,
    dnn_matrix_path: str,
    dnn_enrolids_path: str,
    pn_h5_path: str,
    k: int,
    seed: int,
) -> pd.DataFrame:
    ids, _ = sample_stageA_dispersed_controls(
        leaf_controls_enrolids=control_enrolids,
        leaf_nn_matrix_npy=dnn_matrix_path,
        leaf_nn_enrolids_npy=dnn_enrolids_path,
        pn_h5_path=pn_h5_path,
        leaf_cases_enrolids=cases["ENROLID"].values.astype(np.int64),
        k=k,
        seed_method="random",
        seed=seed,
        M_pool=len(control_enrolids),
        use_kmeanspp=False,
        X_majority_leaf=None,
        verbose=False,
    )
    return controls[controls["ENROLID"].isin(ids)].copy()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.distances_dir, exist_ok=True)

    target_col = "top_2_pct_cost_2018"
    cost_col = "annual_cost_2018_deflated"

    print("=" * 80)
    print("Stage A TF-IDF+SVD+QCost: Cost Boxplot")
    print("=" * 80)
    print(f"  Parquet: {args.parquet_path}")
    print(f"  Distances: {args.distances_dir}")
    print()

    df = pd.read_parquet(args.parquet_path)

    if target_col not in df.columns and cost_col in df.columns:
        thresh = df[cost_col].quantile(0.98)
        df[target_col] = (df[cost_col] >= thresh).astype(int)

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=TRAIN_TEST_SEED
    )

    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]
    N = len(cases)
    n_controls = len(controls)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)

    print(f"Train: {len(train_pd):,}  N (cases): {N:,}  n_controls: {n_controls:,}")
    print()

    # Precompute: Gower P-N + tfidf_svd_cosine_qcost DNN (or resume if --resume and DNN exists)
    dnn_dir = os.path.join(args.distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_tfidf_svd_cosine_qcost")
    dnn_matrix_path = os.path.join(dnn_dir, "leaf_global_dnn_matrix.h5")
    dnn_enrolids_path = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")
    pn_h5 = os.path.join(args.distances_dir, "distances_majority_minority_gower.h5")

    if args.resume and os.path.exists(dnn_matrix_path) and os.path.exists(dnn_enrolids_path) and os.path.exists(pn_h5):
        print("Resume: using existing Gower P-N + tfidf_svd_cosine_qcost DNN")
        dnn_mat, dnn_ids = dnn_matrix_path, dnn_enrolids_path
    else:
        print("Precomputing distances: tfidf_svd_cosine_qcost")
        pn_h5, dnn_mat, dnn_ids = ensure_distances_for_metric(
            "tfidf_svd_cosine_qcost",
            cases,
            controls,
            feature_cols,
            CAT_COLUMNS,
            TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS,
            args.distances_dir,
            use_hdf5_dnn=True,
        )
    print()

    # Build undersampled sets and collect cost data (top_2_pct_cost_2018 == 0 for controls)
    plot_data: List[pd.DataFrame] = []

    full_maj = controls[[cost_col]].copy()
    full_maj["Method"] = "Full majority"
    plot_data.append(full_maj)

    rnd_1n = run_random_sampling(control_enrolids, controls, N, SEED)
    rnd_1n = rnd_1n[[cost_col]].copy()
    rnd_1n["Method"] = "Random 1N"
    plot_data.append(rnd_1n)

    k10 = min(10 * N, n_controls)
    rnd_10n = run_random_sampling(control_enrolids, controls, k10, SEED)
    rnd_10n = rnd_10n[[cost_col]].copy()
    rnd_10n["Method"] = "Random 10N"
    plot_data.append(rnd_10n)

    stage_1n = run_stageA_sampling(
        control_enrolids, controls, cases, dnn_mat, dnn_ids, pn_h5, k=N, seed=SEED
    )
    stage_1n = stage_1n[[cost_col]].copy()
    stage_1n["Method"] = "Stage A 1N (ZA_tfidf_svd_cosine_qcost)"
    plot_data.append(stage_1n)

    stage_10n = run_stageA_sampling(
        control_enrolids, controls, cases, dnn_mat, dnn_ids, pn_h5, k=k10, seed=SEED
    )
    stage_10n = stage_10n[[cost_col]].copy()
    stage_10n["Method"] = "Stage A 10N (ZA_tfidf_svd_cosine_qcost)"
    plot_data.append(stage_10n)

    full_df = pd.concat(plot_data, ignore_index=True)
    method_order = [
        "Full majority",
        "Random 1N",
        "Random 10N",
        "Stage A 1N (ZA_tfidf_svd_cosine_qcost)",
        "Stage A 10N (ZA_tfidf_svd_cosine_qcost)",
    ]
    full_df["Method"] = pd.Categorical(full_df["Method"], categories=method_order, ordered=True)
    full_df = full_df.sort_values("Method")

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.boxplot(data=full_df, x="Method", y=cost_col)
    plt.yscale("log")
    plt.title(
        "Stage A TF-IDF+SVD+QCost: Majority Cost Distribution\n(top_2_pct_cost_2018 == 0)",
        fontsize=14,
    )
    plt.ylabel("Annual Cost 2018 (Log Scale $)")
    plt.xlabel("Sampling Method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(args.outdir, "stageA_tfidf_svd_cosine_qcost_cost_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Box plot saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
