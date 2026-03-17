#!/usr/bin/env python
"""
recover_from_cache_exp3_exp4.py
===============================
Recover experiment 3 and 4 results by re-running train/eval using cached matched_1N IDs.
Uses fixed OCT hyperparams (depth=7, minbucket=150, cp=0.0001) from experiments_compare_random_vs_curation.

Shared: matched_1N from cache (same for both exp3 and exp4).
Exp3: match (cache) + disp_ids (sample_stageA_on_restricted_pool) = 2N mix.
Exp4: match (cache) + 1N random = 2N total.

Usage:
  cd msk_analysis
  python scripts/recover_from_cache_exp3_exp4.py --results_dir /Users/cat2510/scratch/exp_0_to_4_gower/results
  python scripts/recover_from_cache_exp3_exp4.py --results_dir ... --resume
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import traceback
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

import importlib
import experiments_compare_random_vs_curation
importlib.reload(experiments_compare_random_vs_curation)
from experiments_compare_random_vs_curation import (
    sample_random_controls,
    sample_stageA_on_restricted_pool,
    load_metrics_from_predictions,
    train_and_evaluate_oct,
)

from public.model_IAI import (
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)

TRAIN_TEST_SEED = 123


def discover_cached_seeds(cache_dir: str, cache_prefix: str) -> List[int]:
    """Find all seeds that have cached matched_1N files for the given prefix."""
    pattern = os.path.join(cache_dir, f"{cache_prefix}_s*.npy")
    files = glob.glob(pattern)
    seeds = []
    pat = re.compile(rf"{re.escape(cache_prefix)}_s(\d+)\.npy$")
    for f in files:
        basename = os.path.basename(f)
        m = pat.search(basename)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def main():
    p = argparse.ArgumentParser(description="Recover exp3 & exp4 from cached matched_1N")
    p.add_argument("--results_dir", type=str, default="/Users/cat2510/scratch/exp_0_to_4_gower/results")
    p.add_argument("--resume", action="store_true", help="Skip seeds that already have prediction files")
    args = p.parse_args()

    results_dir = args.results_dir
    preds_dir = os.path.join(results_dir, "predictions")
    cache_dir = os.path.join(results_dir, "cache_ids")
    config_path = os.path.join(results_dir, "config.json")

    if not os.path.isdir(cache_dir):
        print(f"Cache dir not found: {cache_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    parquet_path = config.get("parquet_path", "msk_2017_18_no_meds.parquet")
    stageA_seed_method = config.get("stageA_seed_method", "random")
    M_pool = config["M_pool"]
    use_kmeanspp = config.get("use_kmeanspp", False)

    pn_h5 = config["pn_h5"]
    dnn_matrix = config["dnn_matrix"]
    dnn_enrolids_path = config["dnn_enrolids"]

    cache_prefix = f"matched_1N_seedmethod_{stageA_seed_method}_Mpool_{M_pool}"
    cached_seeds = discover_cached_seeds(cache_dir, cache_prefix)
    if not cached_seeds:
        print(f"No cached seeds found for {cache_prefix}")
        sys.exit(0)

    print(f"Found {len(cached_seeds)} cached seeds: {cached_seeds[0]}-{cached_seeds[-1]}")
    os.makedirs(preds_dir, exist_ok=True)

    target_col = "top_2_pct_cost_2018"
    cost_col = "annual_cost_2018_deflated"

    df = pd.read_parquet(parquet_path)
    if target_col not in df.columns and cost_col in df.columns:
        thresh = df[cost_col].quantile(0.98)
        df[target_col] = (df[cost_col] >= thresh).astype(int)

    import h5py
    ctrl_ids = np.load(dnn_enrolids_path)
    with h5py.File(pn_h5, "r") as f:
        case_ids = f["minority_enrolids"][:]
    train_ids_set = set(int(e) for e in ctrl_ids) | set(int(e) for e in case_ids)
    train_pd = df[df["ENROLID"].isin(train_ids_set)].copy()
    remainder_pd = df[~df["ENROLID"].isin(train_ids_set)].copy()

    val_ids, test_ids = train_test_split(
        remainder_pd["ENROLID"],
        test_size=0.5,
        random_state=TRAIN_TEST_SEED,
        stratify=remainder_pd[target_col],
    )
    val_pd = remainder_pd[remainder_pd["ENROLID"].isin(val_ids)].reset_index(drop=True)
    test_pd = remainder_pd[remainder_pd["ENROLID"].isin(test_ids)].reset_index(drop=True)

    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]
    N = len(cases)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)
    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    K = 2 * N

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    dnn_ids = np.load(dnn_enrolids_path)
    id_to_pos = {int(e): i for i, e in enumerate(dnn_ids)}

    X_majority_leaf = None
    if stageA_seed_method == "centroid":
        numeric_feature_cols = [c for c in feature_cols if c in df.select_dtypes(include="number").columns]
        if not numeric_feature_cols:
            numeric_feature_cols = TRUE_NUM_COLUMNS + BIN_FLAG_COLUMNS
        controls_by_id = controls.set_index("ENROLID")
        try:
            X_majority_leaf = (
                controls_by_id.reindex(dnn_ids)[numeric_feature_cols]
                .fillna(0)
                .values.astype(np.float64)
            )
        except Exception:
            X_majority_leaf = None

    def _pred_path(exp_name: str, variant: str, seed: int) -> str:
        return os.path.join(preds_dir, f"oct_predictions_{exp_name}_{variant}_s{seed}.csv")

    def load_matched_1N(seed: int) -> np.ndarray:
        path = os.path.join(cache_dir, f"{cache_prefix}_s{seed}.npy")
        return np.load(path)

    def get_random_perm_excluding(seed: int, exclude_ids: np.ndarray) -> np.ndarray:
        exclude = set(int(x) for x in exclude_ids)
        remaining = np.array([e for e in control_enrolids if int(e) not in exclude], dtype=np.int64)
        rng = np.random.RandomState(seed + 202603)
        return rng.permutation(remaining)

    all_rows: List[dict] = []

    for seed in cached_seeds:
        match_ids = load_matched_1N(seed)

        # --- Exp 3: match (cache) + disp_ids (stageA on restricted pool) = 2N ---
        print("\n" + "#" * 80)
        print("EXPERIMENT 3 (recover): Stage B 1:1 + extra dispersed (2N)")
        print(f"  seed={seed}  [matched from cache, disp from stageA]")
        pred_mix = _pred_path("exp3", "mix", seed)
        try:
            if args.resume and os.path.exists(pred_mix):
                m = load_metrics_from_predictions(pred_mix, test_pd[target_col])
                all_rows.append({"experiment": "exp3", "variant": "mix", "seed": seed, "n_cases": N, "n_controls": K, **m})
                print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
            else:
                disp_ids = sample_stageA_on_restricted_pool(
                    control_enrolids, match_ids, dnn_matrix, dnn_enrolids_path, pn_h5,
                    case_enrolids, N, stageA_seed_method, seed,
                    X_majority_leaf, id_to_pos, verbose=False, use_kmeanspp=use_kmeanspp,
                )
                mix_ids = np.unique(np.concatenate([match_ids, disp_ids]))[:K]
                if len(mix_ids) < K:
                    remaining = np.array([e for e in control_enrolids if int(e) not in set(int(x) for x in mix_ids)])
                    extra = sample_random_controls(remaining, K - len(mix_ids), seed + 9999)
                    mix_ids = np.concatenate([mix_ids, extra])[:K]
                mix_train = pd.concat([cases, controls[controls["ENROLID"].isin(mix_ids)]], ignore_index=True)
                m = train_and_evaluate_oct(
                    mix_train, val_pd, test_pd, feature_cols, target_col,
                    CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                    results_dir, f"exp3_mix_s{seed}", random_seed=TRAIN_TEST_SEED,
                    save_predictions=True,
                )
                all_rows.append({"experiment": "exp3", "variant": "mix", "seed": seed, "n_cases": N, "n_controls": len(mix_ids), **m})
                print(f"    PR-AUC={m.get('pr_auc', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_rows.append({"experiment": "exp3", "variant": "error", "seed": seed, "error": str(e)})

        # --- Exp 4: match (cache) + 1N random = 2N ---
        print("\n" + "#" * 80)
        print("EXPERIMENT 4 (recover): 1N matched + 1N random (2N)")
        print(f"  seed={seed}  [matched from cache, random re-run]")
        variant = "matched_1N_plus_1N_random"
        pred_path = _pred_path("exp4", variant, seed)
        total_controls = 2 * N
        try:
            if args.resume and os.path.exists(pred_path):
                m = load_metrics_from_predictions(pred_path, test_pd[target_col])
                all_rows.append({"experiment": "exp4", "variant": variant, "seed": seed, "n_cases": N, "n_controls": total_controls, **m})
                print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
            else:
                perm = get_random_perm_excluding(seed, match_ids)
                extra_ids = perm[:N].astype(np.int64)
                ctrl_ids_exp4 = np.unique(np.concatenate([match_ids, extra_ids]).astype(np.int64))[:total_controls]
                if len(ctrl_ids_exp4) < total_controls:
                    remaining = np.array([e for e in control_enrolids if int(e) not in set(int(x) for x in ctrl_ids_exp4)])
                    extra = sample_random_controls(remaining, total_controls - len(ctrl_ids_exp4), seed + 7777)
                    ctrl_ids_exp4 = np.concatenate([ctrl_ids_exp4, extra])[:total_controls]
                train_df = pd.concat([cases, controls[controls["ENROLID"].isin(ctrl_ids_exp4)]], ignore_index=True)
                m = train_and_evaluate_oct(
                    train_df, val_pd, test_pd, feature_cols, target_col,
                    CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                    results_dir, f"exp4_{variant}_s{seed}", random_seed=TRAIN_TEST_SEED,
                    save_predictions=True,
                )
                all_rows.append({"experiment": "exp4", "variant": variant, "seed": seed, "n_cases": N, "n_controls": total_controls, **m})
                print(f"    PR-AUC={m.get('pr_auc', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_rows.append({"experiment": "exp4", "variant": variant, "seed": seed, "error": str(e)})

    if not all_rows:
        print("No rows recovered.")
        sys.exit(0)

    # Load existing summary and merge
    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    existing = []
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        recovered_keys = {(r["experiment"], r["variant"], r["seed"]) for r in all_rows}
        for _, row in existing_df.iterrows():
            key = (row["experiment"], row["variant"], row["seed"])
            if key not in recovered_keys:
                existing.append(row.to_dict())
        print(f"Kept {len(existing)} existing rows from summary")

    df_new = pd.DataFrame(all_rows)
    df_existing = pd.DataFrame(existing) if existing else pd.DataFrame()
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out = df_out.sort_values(["experiment", "variant", "seed"]).reset_index(drop=True)
    df_out.to_csv(summary_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {summary_path} (recovered {len(all_rows)} new)")

    agg_cols = ["pr_auc", "auc", "best_mcc", "balanced_recall_gmean", "balanced_specificity_gmean", "optimal_f1"]
    agg_cols = [c for c in agg_cols if c in df_out.columns]
    if agg_cols:
        agg = df_out.groupby(["experiment", "variant"]).agg(
            {c: ["mean", "std"] for c in agg_cols}
        ).round(4)
        agg.to_csv(os.path.join(results_dir, "experiment_summary_aggregated.csv"))
        print("\nAggregated (mean ± std):")
        print(agg)

    print("\nDone.")


if __name__ == "__main__":
    main()
