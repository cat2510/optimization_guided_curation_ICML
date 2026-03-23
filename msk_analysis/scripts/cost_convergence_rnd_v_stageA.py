#!/usr/bin/env python
"""
cost_convergence_rnd_v_stageA.py
================================
Compare box plots of annual_cost_2018_deflated for undersampled majority sets
(target_col = top_2_pct_cost_2018 == 0).

Precomputes d_pn and global_dnn distances to /Users/cat2510/scratch/precomputed_distances_msk_with_meds
for: Gower, Manhattan, Cosine, Euclidean.

Creates undersampled majority sets:
  1. 1N purely random
  2. 10*N purely random
  3. 1N purely Stage A (k-center max dispersion) per metric
  4. 10*N purely Stage A per metric
  5. n_controls Stage A (Gower only, sanity check)

Box plot order: full majority, random sets, stage A sets.

Usage
-----
  cd msk_analysis
  python scripts/cost_convergence_rnd_v_stageA.py [--parquet_path msk_2017_18_full.parquet]
  python scripts/cost_convergence_rnd_v_stageA.py --use_hdf5_dnn   # save D-N-N as HDF5 (~3-5x smaller)
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path setup
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

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from ensure_distances_for_metric import ensure_distances_for_metric

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)

# Use pd.read_parquet (not Spark) for deterministic row order; Spark may shuffle and change train/test split.
TRAIN_TEST_SEED = 123
DISTANCES_DIR = os.environ.get("SCRATCH_DISTANCES_DIR", "/Users/cat2510/scratch/precomputed_distances_msk")
METRICS = ["gower", "manhattan", "cosine", "euclidean"]
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description="Cost convergence: random vs Stage A undersampling")
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--outdir", type=str, default="./cost_convergence_rnd_v_stageA")
    p.add_argument("--distances_dir", type=str, default=DISTANCES_DIR)
    p.add_argument(
        "--use_hdf5_dnn",
        action="store_true",
        help="Save D-N-N as HDF5 compressed (~3-5x smaller) for scratch disk space",
    )
    return p.parse_args()


def run_random_sampling(
    control_enrolids: np.ndarray,
    controls: pd.DataFrame,
    k: int,
    seed: int,
) -> pd.DataFrame:
    """Select k controls at random. Returns DataFrame of selected controls."""
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
    """Select k controls via farthest-first k-center (use_kmeanspp=False). Returns DataFrame."""
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
        use_kmeanspp=False,  # use farthest_first_kcenter_indices
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
    print("Cost Convergence: Random vs Stage A Undersampling")
    print("=" * 80)
    print(f"  Parquet: {args.parquet_path}")
    print(f"  Distances: {args.distances_dir}")
    print(f"  D-N-N format: {'HDF5 (compressed)' if args.use_hdf5_dnn else 'NPY (uncompressed)'}")
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
    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)

    print(f"Train: {len(train_pd):,}  N (cases): {N:,}  n_controls: {n_controls:,}")
    print()

    # ---------------------------------------------------------------------------
    # Precompute distances (d_pn, global_dnn) for Gower, Manhattan, Cosine, Euclidean
    # ---------------------------------------------------------------------------
    dist_paths: dict[str, Tuple[str, str, str]] = {}
    for metric in METRICS:
        print(f"Precomputing distances: {metric}")
        try:
            pn_h5, dnn_mat, dnn_ids = ensure_distances_for_metric(
                metric,
                cases,
                controls,
                feature_cols,
                CAT_COLUMNS,
                TRUE_NUM_COLUMNS,
                BIN_FLAG_COLUMNS,
                args.distances_dir,
                use_hdf5_dnn=args.use_hdf5_dnn,
            )
            dist_paths[metric] = (pn_h5, dnn_mat, dnn_ids)
        except Exception as e:
            print(f"  ERROR {metric}: {e}")
            raise

    # ---------------------------------------------------------------------------
    # Build undersampled majority sets and collect cost data
    # ---------------------------------------------------------------------------
    plot_data: List[pd.DataFrame] = []

    # 1. Full majority (reference)
    full_maj = controls[[cost_col]].copy()
    full_maj["Method"] = "Full majority"
    plot_data.append(full_maj)

    # 2. 1N random
    rnd_1n = run_random_sampling(control_enrolids, controls, N, SEED)
    rnd_1n = rnd_1n[[cost_col]].copy()
    rnd_1n["Method"] = "Random 1N"
    plot_data.append(rnd_1n)
    print(f"Random 1N DONE: {len(rnd_1n)} controls selected")
    # 3. 10*N random
    k10 = min(10 * N, n_controls)
    rnd_10n = run_random_sampling(control_enrolids, controls, k10, SEED)
    rnd_10n = rnd_10n[[cost_col]].copy()
    rnd_10n["Method"] = "Random 10N"
    plot_data.append(rnd_10n)
    print(f"Random 10N DONE: {len(rnd_10n)} controls selected")
    # 4. 1N Stage A per metric
    for metric in METRICS:
        pn_h5, dnn_mat, dnn_ids = dist_paths[metric]
        stage1n = run_stageA_sampling(
            control_enrolids,
            controls,
            cases,
            dnn_mat,
            dnn_ids,
            pn_h5,
            k=N,
            seed=SEED,
        )
        stage1n = stage1n[[cost_col]].copy()
        stage1n["Method"] = f"Stage A 1N ({metric})"
        plot_data.append(stage1n)
    print(f"Stage A 1N DONE: {len(stage1n)} controls selected")
    save_path = os.path.join(args.outdir, "stage1n_data.csv")
    stage1n.to_csv(save_path, index=False)
    print(f"Stage A 1N data saved: {save_path}")
    # 5. 10*N Stage A per metric
    for metric in METRICS:
        pn_h5, dnn_mat, dnn_ids = dist_paths[metric]
        stage10n = run_stageA_sampling(
            control_enrolids,
            controls,
            cases,
            dnn_mat,
            dnn_ids,
            pn_h5,
            k=k10,
            seed=SEED,
        )
        stage10n = stage10n[[cost_col]].copy()
        stage10n["Method"] = f"Stage A 10N ({metric})"
        plot_data.append(stage10n)
    print(f"Stage A 10N DONE: {len(stage10n)} controls selected")
    save_path = os.path.join(args.outdir, "stage10n_data.csv")
    stage10n.to_csv(save_path, index=False)
    print(f"Stage A 10N data saved: {save_path}")
    # 6. n_controls Stage A (Gower only)
    pn_h5, dnn_mat, dnn_ids = dist_paths["gower"]
    stage_full = run_stageA_sampling(
        control_enrolids,
        controls,
        cases,
        dnn_mat,
        dnn_ids,
        pn_h5,
        k=n_controls,
        seed=SEED,
    )
    stage_full = stage_full[[cost_col]].copy()
    stage_full["Method"] = "Stage A n_controls (gower)"
    plot_data.append(stage_full)
    print(f"Stage A n_controls DONE: {len(stage_full)} controls selected")
    # ---------------------------------------------------------------------------
    # Box plot: full majority, random sets, stage A sets
    # ---------------------------------------------------------------------------
    full_df = pd.concat(plot_data, ignore_index=True)

    method_order = [
        "Full majority",
        "Random 1N",
        "Random 10N",
        "Stage A 1N (gower)",
        "Stage A 1N (manhattan)",
        "Stage A 1N (cosine)",
        "Stage A 1N (euclidean)",
        "Stage A 10N (gower)",
        "Stage A 10N (manhattan)",
        "Stage A 10N (cosine)",
        "Stage A 10N (euclidean)",
        "Stage A n_controls (gower)",
    ]
    # Use actual method names (capitalization may differ)
    method_order = [m for m in method_order if m in full_df["Method"].unique()]
    full_df["Method"] = pd.Categorical(full_df["Method"], categories=method_order, ordered=True)
    full_df = full_df.sort_values("Method")

    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    ax = sns.boxplot(data=full_df, x="Method", y=cost_col)
    plt.yscale("log")
    plt.title(
        "Majority Class Cost Distribution: Random vs Stage A Undersampling\n(top_2_pct_cost_2018 == 0)",
        fontsize=14,
    )
    plt.ylabel("Annual Cost 2018 (Log Scale $)")
    plt.xlabel("Sampling Method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(args.outdir, "cost_convergence_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Box plot saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
