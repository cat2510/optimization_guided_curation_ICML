#!/usr/bin/env python
"""
run_1to1_sampling_gower.py
==========================
Load **precomputed** Gower P-N and D-N-N from ``distances_dir`` (same layout as
``public.precompute_gower_distances`` / misc_conditions precompute), then run:

  * **RND 1-1**: N cases + N random controls
  * **Ours 1-1**: N cases + N controls from two-stage k-center + match (Gower)

Does **not** compute distances; run ``precompute_gower_distances.py`` or
``precompute_distances_multi_cohort*.py`` first.

Usage
-----
  cd msk_analysis
  python scripts/run_1to1_sampling_gower.py \\
    --distances_dir /path/to/precomputed_distances_gower \\
    --parquet_path msk_2017_18_full.parquet \\
    --outdir ./1to1_gower_results
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from public.dnn_matrix_storage import (
    dnn_enrolids_npy_path,
    dnn_matrix_storage_exists,
    ensure_dnn_matrix_npy,
)
from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)
from msk_analysis.experiments_compare_random_vs_curation import train_and_evaluate_oct
from msk_analysis.scripts.gower_1to1_sampling import run_ours_1to1_sampling, run_rnd_1to1_sampling

# Must match the seed used when precomputing (global_dnn_seed_{seed}_gower)
DEFAULT_TRAIN_TEST_SEED = 123


def load_precomputed_gower_paths(distances_dir: str, train_test_seed: int) -> Tuple[str, str, str]:
    """
    Resolve P-N HDF5, D-N-N .npy (memmap), and enrolids .npy under distances_dir.
    Raises FileNotFoundError if anything is missing.
    """
    d = os.path.abspath(distances_dir)
    pn_h5 = os.path.join(d, "distances_majority_minority_gower.h5")
    dnn_dir = os.path.join(d, f"global_dnn_seed_{train_test_seed}_gower")
    enrol = dnn_enrolids_npy_path(dnn_dir)

    missing = []
    if not os.path.isfile(pn_h5):
        missing.append(pn_h5)
    if not dnn_matrix_storage_exists(dnn_dir):
        missing.append(os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy"))
    if not os.path.isfile(enrol):
        missing.append(enrol)
    if missing:
        raise FileNotFoundError(
            "Precomputed Gower distances not found. Expected:\n"
            f"  - {pn_h5}\n"
            f"  - {dnn_dir}/leaf_global_dnn_matrix.npy\n"
            f"  - {enrol}\n"
            f"Missing: {missing}\n"
            "Run public.precompute_gower_distances or misc_conditions precompute first."
        )
    dnn_npy = ensure_dnn_matrix_npy(dnn_dir)
    return pn_h5, dnn_npy, enrol


def plot_rnd_vs_ours(
    results_dir: str,
    target_col: str = "annual_cost_2018_deflated",
    filter_col: str = "top_2_pct_cost_2018",
) -> str:
    files = list(Path(results_dir).glob("rnd_1to1_s*.csv")) + list(
        Path(results_dir).glob("ours_1to1_s*.csv")
    )
    if not files:
        print(f"No rnd_1to1 / ours_1to1 CSVs in {results_dir}")
        return ""
    rows = []
    for f in files:
        name = f.name.lower()
        method = "RND 1-1" if "rnd_1to1" in name else "Ours 1-1 (Gower curated)"
        try:
            df = pd.read_csv(f, usecols=[target_col, filter_col])
            maj = df[df[filter_col] == 0].copy()
            maj["Method"] = method
            rows.append(maj)
        except Exception as e:
            print(f"Skip {f.name}: {e}")
    if not rows:
        return ""
    full = pd.concat(rows, ignore_index=True)
    order = ["RND 1-1", "Ours 1-1 (Gower curated)"]
    full["Method"] = pd.Categorical(full["Method"], categories=order, ordered=True)
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.boxplot(data=full, x="Method", y=target_col, hue="Method", legend=False)
    plt.yscale("log")
    plt.title("Majority class cost: random 1-1 vs Gower curated 1-1")
    plt.ylabel(target_col)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(results_dir, "rnd_vs_ours_1to1_cost.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Plot: {out}")
    return out


def median_majority_cost(
    csv_path: str,
    target_col: str = "annual_cost_2018_deflated",
    filter_col: str = "top_2_pct_cost_2018",
) -> float:
    df = pd.read_csv(csv_path, usecols=[target_col, filter_col])
    maj = df[df[filter_col] == 0][target_col]
    return float(maj.median())


def parse_args():
    p = argparse.ArgumentParser(
        description="1:1 random vs Gower-curated sampling (precomputed distances only)"
    )
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    p.add_argument(
        "--rnd_seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for random 1-1 sampling (defaults to --seeds)",
    )
    p.add_argument(
        "--ours_seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for ours 1-1 sampling / seed_method randomness (defaults to --seeds)",
    )
    p.add_argument("--outdir", type=str, default="./1to1_gower_sampling")
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument(
        "--distances_dir",
        type=str,
        default="../../scratch/msk_analysis/precomputed_distances_gower",
        help="Dir with distances_majority_minority_gower.h5 and global_dnn_seed_<train_test_seed>_gower/",
    )
    p.add_argument(
        "--train_test_seed",
        type=int,
        default=DEFAULT_TRAIN_TEST_SEED,
        help="Seed in folder name global_dnn_seed_{train_test_seed}_gower (must match precompute)",
    )
    p.add_argument("--oct_seed", type=int, default=DEFAULT_TRAIN_TEST_SEED)
    p.add_argument("--feature_set", type=str, default="all_cost",
                   choices=["medical_only", "all_cost", "less_cost"])
    p.add_argument("--M_pool", type=int, default=None)
    p.add_argument("--seed_method", type=str, default="random", choices=["random", "smart","centroid","density"])
    p.add_argument("--skip_plot", action="store_true")
    p.add_argument("--skip_oct", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def parse_seed_list(seed_str: str) -> List[int]:
    return [int(s.strip()) for s in seed_str.split(",") if s.strip()]


def summarize_results(all_rows: List[Dict], results_dir: str) -> None:
    if not all_rows:
        print("No OCT rows to summarize.")
        return

    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    df_out = pd.DataFrame(all_rows)
    df_out = df_out.sort_values(["experiment", "seed"]).reset_index(drop=True)
    df_out.to_csv(summary_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {summary_path}")

    agg_cols = [
        "pr_auc",
        "auc",
        "best_mcc",
        "balanced_recall_gmean",
        "balanced_specificity_gmean",
        "optimal_f1",
    ]
    agg_cols = [c for c in agg_cols if c in df_out.columns]
    if agg_cols:
        agg = df_out.groupby(["experiment"]).agg(
            {c: ["mean", "std"] for c in agg_cols}
        ).round(4)
        print("\nAggregated (mean ± std):")
        print(agg)
        agg.to_csv(os.path.join(results_dir, "experiment_summary_aggregated.csv"))


def main():
    args = parse_args()
    base_seeds = parse_seed_list(args.seeds)
    rnd_seeds = parse_seed_list(args.rnd_seeds) if args.rnd_seeds else base_seeds
    ours_seeds = parse_seed_list(args.ours_seeds) if args.ours_seeds else base_seeds
    results_dir = os.path.join(args.outdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("1:1 sampling — random vs Gower curated (precomputed distances)")
    print("=" * 72)
    print(f"  distances_dir: {args.distances_dir}")
    print(f"  fixed train/test seed: {args.train_test_seed}")
    print(f"  rnd_seeds: {rnd_seeds}")
    print(f"  ours_seeds: {ours_seeds}")

    pn_h5, dnn_mat, dnn_ids_path = load_precomputed_gower_paths(
        args.distances_dir, args.train_test_seed
    )
    print(f"  P-N: {pn_h5}")
    print(f"  D-N-N: {dnn_mat}")
    print(f"  Enrolids: {dnn_ids_path}")

    df = pd.read_parquet(args.parquet_path)
    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    bin_cols = get_bin_flag_columns(df)
    cat_cols = get_cat_columns(df)
    num_cols = get_true_num_columns(df, cat_cols, bin_cols)
    cost_cols = [
        c for c in df.columns
        if ("cost" in c.lower() or "quarterly" in c.lower() or "increasing" in c.lower()
            or "decreasing" in c.lower() or "skewness" in c.lower() or "kurtosis" in c.lower()
            or "cv" in c.lower() or "range" in c.lower())
        and "2018" not in c
    ]
    aux = [c for c in df.columns if c.startswith("comorbidity_only") or c.startswith("msk_procedure")]
    exclude = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    if args.feature_set == "medical_only":
        exclude = exclude + cost_cols
    elif args.feature_set == "less_cost":
        exclude = exclude + aux
    feature_cols = [c for c in df.columns if c not in exclude]

    _, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False,
        random_state=args.train_test_seed,
    )
    _, _, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False,
        random_state=args.train_test_seed,
    )

    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]
    n_cases = len(cases)
    n_ctrl = len(controls)
    case_ids = cases["ENROLID"].values.astype(np.int64)
    ctrl_ids = controls["ENROLID"].values.astype(np.int64)
    m_pool = args.M_pool if args.M_pool is not None else n_ctrl // 2

    print(f"  Train {len(train_pd):,} | N cases {n_cases:,} | controls {n_ctrl:,} | M_pool {m_pool:,}")

    medians = {}

    print("\n--- RND 1-1 ---")
    for seed in rnd_seeds:
        path = os.path.join(results_dir, f"rnd_1to1_s{seed}.csv")
        if args.resume and os.path.isfile(path):
            print(f"  seed {seed}: cached")
        else:
            run_rnd_1to1_sampling(ctrl_ids, cases, controls, n_cases, seed).to_csv(
                path, index=False
            )
            print(f"  seed {seed}: wrote {path}")
        medians[f"rnd_s{seed}"] = median_majority_cost(path)

    print("\n--- Ours 1-1 (Gower k-center + match) ---")
    for seed in ours_seeds:
        path = os.path.join(results_dir, f"ours_1to1_s{seed}.csv")
        if args.resume and os.path.isfile(path):
            print(f"  seed {seed}: cached")
        else:
            run_ours_1to1_sampling(
                ctrl_ids, case_ids, controls, cases,
                dnn_mat, dnn_ids_path, pn_h5, n_cases, m_pool, seed, seed_method=args.seed_method,
            ).to_csv(path, index=False)
            print(f"  seed {seed}: wrote {path}")
        medians[f"ours_s{seed}"] = median_majority_cost(path)

    print(f"\n  Majority-cost medians: {medians}")

    if not args.skip_plot:
        plot_rnd_vs_ours(results_dir)

    if not args.skip_oct:
        all_rows: List[Dict] = []
        experiment_to_seeds = {
            "rnd_1to1": rnd_seeds,
            "ours_1to1": ours_seeds,
        }
        experiment_labels = {
            "rnd_1to1": "RND 1-1",
            "ours_1to1": "Ours 1-1 Gower",
        }

        for base, seed_list in experiment_to_seeds.items():
            label = experiment_labels[base]
            for seed in seed_list:
                csv_path = os.path.join(results_dir, f"{base}_s{seed}.csv")
                if not os.path.isfile(csv_path):
                    print(f"  OCT skip: {csv_path} missing")
                    continue
                train_df = pd.read_csv(csv_path)
                print(f"\nOCT: {label} (seed={seed}) …")
                try:
                    m = train_and_evaluate_oct(
                        train_df,
                        val_pd,
                        test_pd,
                        feature_cols,
                        target_col,
                        cat_cols,
                        num_cols,
                        bin_cols,
                        results_dir,
                        f"{base}_s{seed}",
                        args.oct_seed,
                    )
                    print(f"  PR-AUC {m.get('pr_auc', 0):.4f}  AUC {m.get('auc', 0):.4f}")
                    row = {"experiment": base, "seed": seed}
                    row.update(m)
                    all_rows.append(row)
                except Exception as e:
                    print(f"  OCT error: {e}")
                    traceback.print_exc()

        summarize_results(all_rows, results_dir)

    print("\nDone.")


if __name__ == "__main__":
    """python3 scripts/run_1to1_sampling_gower.py \
  --outdir ./1to1_gower_sampling \
  --train_test_seed 123 \
  --rnd_seeds 0,1,2,3,4 \
  --ours_seeds 0,1,2,3,4"""
    main()
