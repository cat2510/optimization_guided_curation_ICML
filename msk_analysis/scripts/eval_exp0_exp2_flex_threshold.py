#!/usr/bin/env python
"""
eval_exp0_exp2_flex_threshold.py
================================
Evaluate exp0, exp2, exp3, and exp4 models (from exp_0_to_4_gower_v2 results) across
varying cost thresholds at evaluation time. Models are trained on top_2_pct_cost_2018;
we re-evaluate using top 2%, 5%, 10%, and 15% as the binary target.

No sampling or training. Loads existing prediction CSVs only.

Usage
-----
  cd msk_analysis
  python scripts/eval_exp0_exp2_flex_threshold.py
  python scripts/eval_exp0_exp2_flex_threshold.py --seeds 115,116,117 --target_pcts 2,5,10,15
  python scripts/eval_exp0_exp2_flex_threshold.py --results_dir /path/to/exp_0_to_4_gower_v2/results
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))  # msk_analysis
project_root = os.path.abspath(os.path.join(parent_dir, ".."))  # my_projects
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from public.dnn_matrix_storage import dnn_enrolids_npy_path, dnn_matrix_storage_exists
from public.model_IAI import best_mcc_threshold, best_balanced_threshold, recall_at_specificity

TRAIN_TEST_SEED = 123
DISTANCES_DIR = "/Users/cat2510/scratch/msk_analysis/precomputed_distances_gower"
DEFAULT_RESULTS_DIR = "/Users/cat2510/scratch/msk_analysis/exp_0_to_4_gower_v2/results"
COST_COL = "annual_cost_2018_deflated"
DEFAULT_AGGREGATED_CSV = "/Users/cat2510/my_projects/msk_analysis/eval_threshold_results/results/eval_threshold_aggregated.csv"

# Short labels for experiments in plots
EXP_LABELS = {
    ("exp0_rnd", "random"): "Exp0 (random)",
    ("exp2", "stageB"): "Exp2 (stageB)",
    ("exp3", "mix"): "Exp3 (mix)",
    ("exp4", "matched_1N_plus_1N_random"): "Exp4 (1N+1N rnd)",
}


def target_column_for_pct(pct: int | float) -> str:
    """e.g. 2 -> top_2_pct_cost_2018, 0.5 -> top_0_5_pct_cost_2018, 15 -> top_15_pct_cost_2018."""
    if pct == int(pct):
        return f"top_{int(pct)}_pct_cost_2018"
    return f"top_{str(pct).replace('.', '_')}_pct_cost_2018"


def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_leaves: float = np.nan,
) -> dict:
    """Compute binary classification metrics from (y_true, y_proba). Mirrors load_metrics_from_predictions."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    if len(y_true_arr) != len(y_proba_arr):
        raise ValueError(f"length mismatch: y_true={len(y_true_arr)}, y_proba={len(y_proba_arr)}")

    auc = roc_auc_score(y_true_arr, y_proba_arr)
    pr_auc = average_precision_score(y_true_arr, y_proba_arr)
    mcc_res = best_mcc_threshold(y_true_arr, y_proba_arr)
    best_mcc = mcc_res["mcc"]
    y_pred_mcc = mcc_res["y_pred"]

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_mcc).ravel()
    recall_mcc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_mcc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    balanced = best_balanced_threshold(y_true_arr, y_proba_arr)
    gmean_recall = balanced["gmean_opt"]["recall"]
    gmean_specificity = balanced["gmean_opt"]["specificity"]

    prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_proba_arr)
    f1_scores = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-10)
    optimal_f1 = float(f1_scores.max())

    recall_at_spec_06, achieved_spec_06, threshold_spec_06 = recall_at_specificity(
        y_true_arr, y_proba_arr, target_specificity=0.60
    )

    return {
        "best_depth": np.nan,
        "best_minbucket": np.nan,
        "best_cp": np.nan,
        "num_leaves": num_leaves,
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": best_mcc,
        "recall_mcc": recall_mcc,
        "specificity_mcc": specificity_mcc,
        "optimal_f1": optimal_f1,
        "balanced_recall_gmean": gmean_recall,
        "balanced_specificity_gmean": gmean_specificity,
        "recall_at_specificity_0.6": float(recall_at_spec_06),
        "achieved_specificity_0.6": float(achieved_spec_06),
        "threshold_specificity_0.6": float(threshold_spec_06),
    }


def get_test_pd(
    parquet_path: str,
    distances_dir: str,
    cost_col: str = COST_COL,
) -> pd.DataFrame:
    """Replicate train/val/test split to obtain test_pd. No sampling or training."""
    df = pd.read_parquet(parquet_path)
    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and cost_col in df.columns:
        thresh = df[cost_col].quantile(0.98)
        df[target_col] = (df[cost_col] >= thresh).astype(int)

    pn_h5 = os.path.join(distances_dir, "distances_majority_minority_gower.h5")
    dnn_dir = os.path.join(distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_gower")
    dnn_enrolids_path = dnn_enrolids_npy_path(dnn_dir)

    if not os.path.isfile(pn_h5) or not dnn_matrix_storage_exists(dnn_dir) or not os.path.isfile(dnn_enrolids_path):
        raise FileNotFoundError(
            f"Missing precomputed enrolids under {distances_dir!r}: "
            f"need {pn_h5}, {dnn_dir}/leaf_global_dnn_matrix.npy, {dnn_enrolids_path}"
        )

    ctrl_ids = np.load(dnn_enrolids_path)
    with h5py.File(pn_h5, "r") as f:
        case_ids = f["minority_enrolids"][:]
    train_ids_set = set(int(e) for e in ctrl_ids) | set(int(e) for e in case_ids)

    remainder_pd = df[~df["ENROLID"].isin(train_ids_set)].copy()
    _, test_ids = train_test_split(
        remainder_pd["ENROLID"],
        test_size=0.5,
        random_state=TRAIN_TEST_SEED,
        stratify=remainder_pd[target_col],
    )
    test_pd = remainder_pd[remainder_pd["ENROLID"].isin(test_ids)].reset_index(drop=True)
    return test_pd


def ensure_target_columns(df: pd.DataFrame, target_pcts: List[float], cost_col: str) -> None:
    """Ensure top_X_pct_cost_2018 columns exist. Derive from cost_col if missing."""
    for pct in target_pcts:
        col = target_column_for_pct(pct)
        if col in df.columns:
            continue
        if cost_col not in df.columns:
            raise ValueError(f"Need {cost_col!r} to derive {col}")
        thresh = df[cost_col].quantile(1.0 - pct / 100.0)
        df[col] = (df[cost_col] >= thresh).astype(int)


def discover_seeds_from_results(results_dir: str, experiments: List[Tuple[str, str]]) -> List[int]:
    """Get seeds present in experiment_summary.csv for the given experiments, or from prediction files."""
    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    if os.path.isfile(summary_path):
        try:
            sm = pd.read_csv(summary_path)
            exp_names = {e[0] for e in experiments}
            sub = sm[sm["experiment"].isin(exp_names)]
            if len(sub) > 0:
                return sorted(sub["seed"].unique().tolist())
        except Exception:
            pass

    preds_dir = os.path.join(results_dir, "predictions")
    if not os.path.isdir(preds_dir):
        return []
    seeds = set()
    for exp_name, variant in experiments:
        prefix = f"oct_predictions_{exp_name}_{variant}_s"
        for f in os.listdir(preds_dir):
            if f.startswith(prefix) and f.endswith(".csv"):
                try:
                    seed = int(f[len(prefix) : -4])
                    seeds.add(seed)
                except ValueError:
                    pass
    return sorted(seeds)


def plot_eval_threshold_results(
    aggregated_path: str = DEFAULT_AGGREGATED_CSV,
    out_path: str | None = None,
) -> str:
    """
    Plot PR-AUC and AUC vs eval_target_pct from eval_threshold_aggregated.csv,
    with mean ± std as confidence bands. One line per experiment.
    """
    df = pd.read_csv(aggregated_path, header=[0, 1], index_col=[0, 1, 2])
    df = df.reset_index()
    # Flatten MultiIndex columns for easier access
    df.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [
        ("pr_auc", "PR-AUC", axes[0]),
        ("auc", "AUC", axes[1]),
    ]

    colors = plt.cm.tab10.colors
    for idx, ((exp, variant), label) in enumerate(EXP_LABELS.items()):
        sub = df[(df["experiment"] == exp) & (df["variant"] == variant)]
        if sub.empty:
            continue
        sub = sub.sort_values("eval_target_pct")
        x_sub = sub["eval_target_pct"].values
        color = colors[idx % len(colors)]

        for metric, title, ax in metrics:
            mean_col, std_col = f"{metric}_mean", f"{metric}_std"
            if mean_col not in sub.columns:
                continue
            y_mean = sub[mean_col].values
            y_std = sub[std_col].values if std_col in sub.columns else np.zeros_like(y_mean)
            ax.plot(x_sub, y_mean, "-o", label=label, color=color, markersize=4)
            ax.fill_between(x_sub, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

    for metric, title, ax in metrics:
        ax.set_xlabel("Eval target percentile (top X%)")
        ax.set_ylabel(title)
        ax.set_title(title + " vs cost threshold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(aggregated_path), "eval_threshold_pr_auc_auc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate exp0/exp2 predictions across cost thresholds (2%, 5%, 10%, 15%)"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Results dir with predictions/ and experiment_summary.csv",
    )
    p.add_argument("--distances_dir", type=str, default=DISTANCES_DIR)
    p.add_argument(
        "--parquet_path",
        type=str,
        default="msk_2017_18_full.parquet",
        help="Parquet path (must match run_experiments; default from exp_0_to_4_gower_v2 config)",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="auto",
        help="Comma-separated seeds or 'auto' to discover from results",
    )
    p.add_argument("--experiments", type=str, default="0,2,3,4",
                   help="Comma-separated experiment indices (0=random, 2=stageB, 3=mix, 4=matched_1N+1N_random)")
    p.add_argument("--target_pcts", type=str, default="2,5,10,15",
                   help="Comma-separated percentiles, e.g. 0.5,2,5,10,15")
    p.add_argument("--outdir", type=str, default="./eval_threshold_results")
    p.add_argument("--plot", action="store_true", help="Generate PR-AUC and AUC vs eval_target_pct plot after eval")
    p.add_argument("--plot_only", action="store_true", help="Only generate plot from existing eval_threshold_aggregated.csv (skip eval)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        agg_path = os.path.join(args.outdir, "results", "eval_threshold_aggregated.csv")
        if not os.path.isfile(agg_path):
            agg_path = DEFAULT_AGGREGATED_CSV
        if os.path.isfile(agg_path):
            plot_eval_threshold_results(aggregated_path=agg_path, out_path=os.path.join(os.path.dirname(agg_path), "eval_threshold_pr_auc_auc.png"))
        else:
            print(f"Not found: {agg_path}. Run eval first or set --outdir.")
        return

    experiments_config = {
        0: ("exp0_rnd", "random"),
        2: ("exp2", "stageB"),
        3: ("exp3", "mix"),
        4: ("exp4", "matched_1N_plus_1N_random"),
    }
    experiments_to_run = [int(x.strip()) for x in args.experiments.split(",")]
    experiments = [experiments_config[e] for e in experiments_to_run if e in experiments_config]
    if not experiments:
        raise ValueError(f"No valid experiments; use --experiments 0,2,3,4 (got {args.experiments})")

    target_pcts = [float(x.strip()) for x in args.target_pcts.split(",")]
    target_cols = [target_column_for_pct(p) for p in target_pcts]

    if args.seeds.strip().lower() == "auto":
        seeds = discover_seeds_from_results(args.results_dir, experiments)
        if not seeds:
            raise ValueError(
                "Could not discover seeds. Set --seeds explicitly or ensure "
                f"experiment_summary.csv or prediction files exist in {args.results_dir}"
            )
        print(f"Discovered seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''} ({len(seeds)} total)")
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    parquet_path = args.parquet_path
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.join(parent_dir, parquet_path)
    if not (os.path.isfile(parquet_path) or os.path.isdir(parquet_path)):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    outdir = os.path.join(args.outdir, "results")
    os.makedirs(outdir, exist_ok=True)
    preds_dir = os.path.join(args.results_dir, "predictions")

    print("=" * 72)
    print("Eval exp0/2/3/4 across cost thresholds (no sampling, no training)")
    print("=" * 72)
    print(f"  results_dir: {args.results_dir}")
    print(f"  seeds: {len(seeds)} seeds")
    print(f"  target_pcts: {target_pcts} -> {target_cols}")

    test_pd = get_test_pd(parquet_path, args.distances_dir)
    ensure_target_columns(test_pd, target_pcts, COST_COL)
    print(f"  test set: {len(test_pd):,} rows")

    all_rows: List[dict] = []
    for exp_name, variant in experiments:
        for seed in seeds:
            pred_file = os.path.join(preds_dir, f"oct_predictions_{exp_name}_{variant}_s{seed}.csv")
            if not os.path.isfile(pred_file):
                print(f"  Skip (missing): {os.path.basename(pred_file)}")
                continue

            pred_df = pd.read_csv(pred_file)
            if "predicted_proba" not in pred_df.columns:
                print(f"  Skip (no predicted_proba): {os.path.basename(pred_file)}")
                continue

            merged = pred_df[["ENROLID", "predicted_proba"]].merge(
                test_pd[["ENROLID"] + target_cols],
                on="ENROLID",
                how="inner",
            )
            if len(merged) != len(test_pd):
                print(
                    f"  Warning: merge size {len(merged)} != test size {len(test_pd)} for {exp_name} s{seed}"
                )

            num_leaves = (
                int(pred_df["leaf_assignment"].nunique())
                if "leaf_assignment" in pred_df.columns
                else np.nan
            )

            for pct, target_col in zip(target_pcts, target_cols):
                y_true = merged[target_col].values
                y_proba = merged["predicted_proba"].values
                m = compute_metrics_from_arrays(y_true, y_proba, num_leaves=num_leaves)
                row = {
                    "experiment": exp_name,
                    "variant": variant,
                    "seed": seed,
                    "eval_target_pct": pct,
                    **m,
                }
                all_rows.append(row)

    if not all_rows:
        print("No rows to save.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out = df_out.sort_values(["experiment", "variant", "seed", "eval_target_pct"]).reset_index(
        drop=True
    )
    summary_path = os.path.join(outdir, "eval_threshold_summary.csv")
    df_out.to_csv(summary_path, mode="a", header=not os.path.isfile(summary_path), index=False)
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
        agg = (
            df_out.groupby(["experiment", "variant", "eval_target_pct"])
            .agg({c: ["mean", "std"] for c in agg_cols})
            .round(4)
        )
        print("\nAggregated (mean ± std) by experiment, variant, eval_target_pct:")
        print(agg)
        agg_path = os.path.join(outdir, "eval_threshold_aggregated.csv")
        agg.to_csv(agg_path)
        print(f"Saved aggregated to {agg_path}")

    if args.plot:
        agg_path = os.path.join(outdir, "eval_threshold_aggregated.csv")
        if os.path.isfile(agg_path):
            plot_eval_threshold_results(aggregated_path=agg_path, out_path=os.path.join(outdir, "eval_threshold_pr_auc_auc.png"))
        else:
            print("Skip plot: eval_threshold_aggregated.csv not found. Run eval first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
