#!/usr/bin/env python
"""
run_experiments_0_to_4_gower.py
===============================
Re-run experiments 0, 1, 2, 3, 4 from experiments_compare_random_vs_curation using
Gower P-N and Gower D-N-N from precomputed_distances_msk_za_tfidf_svd_cosine_qcost.

Reproducibility:
  - Train set: defined by precomputed enrolids (D-N-N controls + P-N minority) so we never
    recompute distances. Val/test from remainder, split 50/50 with stratified seed.
  - Fixed: TRAIN_TEST_SEED (123), OCT training random_seed, val/test split.
  - Fixed: OCT parameters (cp=0.0001, depth=7, minbuckets=150)
  - Varied: sampling seed (--seeds) drives random sampling and k-center init.

Output: /Users/cat2510/scratch/exp_0_to_4_gower/ with metrics, predictions, config.

Usage
-----
  cd msk_analysis
  python scripts/run_experiments_0_to_4_gower.py [--seeds 0,1,2,3,4]
  python scripts/run_experiments_0_to_4_gower.py --seeds 100-200 --experiments 3,4 --resume
  python scripts/run_experiments_0_to_4_gower.py --seeds 100-200 --experiments 3,4 --no_save_artifacts
"""

from __future__ import annotations

import sys
import os
import argparse
import json
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
    sample_stageA_dispersed_controls,
    sample_stageB_matched_controls,
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
DISTANCES_DIR = "/Users/cat2510/scratch/precomputed_distances_exp6_ablation/"
SCRATCH_OUTDIR = "/Users/cat2510/scratch/exp_0_to_4_gower"


def _parse_seeds(seeds_str: str) -> List[int]:
    """Parse seeds: comma-separated list or 'start-end' range (inclusive)."""
    s = seeds_str.strip()
    if "-" in s and "," not in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                start, end = int(parts[0].strip()), int(parts[1].strip())
                return list(range(start, end + 1))
            except ValueError:
                pass
    return [int(x.strip()) for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser(description="Exp 0-4 with Gower distances, precomputed split")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds or 'start-end' range (e.g. 100-200)")
    p.add_argument("--experiments", type=str, default="0,1,2,3,4", help="Comma-separated experiment indices to run (e.g. 3,4 for exp3 and exp4 only)")
    p.add_argument("--outdir", type=str, default=SCRATCH_OUTDIR)
    p.add_argument("--distances_dir", type=str, default=DISTANCES_DIR)
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_no_meds.parquet")
    p.add_argument("--stageA_seed_method", choices=["centroid", "density", "random", "smart"], default="random")
    p.add_argument("--M_pool", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip runs already in summary; load predictions/train CSV")
    p.add_argument("--no_save_artifacts", action="store_true", help="Do not save train CSVs or prediction CSVs (cannot use --resume)")
    p.add_argument("--use_kmeanspp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    experiments_to_run = set(int(x.strip()) for x in args.experiments.split(","))
    results_dir = os.path.join(args.outdir, "results")
    preds_dir = os.path.join(results_dir, "predictions")
    cache_dir = os.path.join(results_dir, "cache_ids")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    target_col = "top_2_pct_cost_2018"
    cost_col = "annual_cost_2018_deflated"

    # Gower paths
    pn_h5 = os.path.join(args.distances_dir, "distances_majority_minority_gower.h5")
    dnn_dir = os.path.join(args.distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_gower")
    dnn_matrix = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_path = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")

    for p in [pn_h5, dnn_matrix, dnn_enrolids_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # Load parquet and define train from precomputed enrolids (no recompute)
    df = pd.read_parquet(args.parquet_path)
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
    n_controls = len(controls)
    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)

    M_pool = args.M_pool if args.M_pool is not None else n_controls // 2
    K = 2 * N

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    dnn_ids = np.load(dnn_enrolids_path)
    id_to_pos = {int(e): i for i, e in enumerate(dnn_ids)}

    X_majority_leaf = None
    if args.stageA_seed_method == "centroid":
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

    config = {
        "distances_dir": args.distances_dir,
        "pn_h5": pn_h5,
        "dnn_matrix": dnn_matrix,
        "dnn_enrolids": dnn_enrolids_path,
        "train_test_seed": TRAIN_TEST_SEED,
        "oct_random_seed": TRAIN_TEST_SEED,
        "sampling_seeds_varied": seeds,
        "stageA_seed_method": args.stageA_seed_method,
        "use_kmeanspp": args.use_kmeanspp,
        "parquet_path": args.parquet_path,
        "M_pool": M_pool,
        "n_train": len(train_pd),
        "n_val": len(val_pd),
        "n_test": len(test_pd),
        "N_cases": N,
        "n_controls": n_controls,
        "K": K,
    }
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    print("=" * 80)
    print("EXPERIMENTS 0-4 with Gower P-N + Gower D-N-N (precomputed split)")
    print("=" * 80)
    print(f"  Seeds: {len(seeds)} seeds" + (f" ({seeds[0]}-{seeds[-1]})" if len(seeds) > 5 else f" {seeds}"))
    print(f"  Experiments: {sorted(experiments_to_run)}")
    print(f"  Train (from enrolids): {len(train_pd):,}  Val: {len(val_pd):,}  Test: {len(test_pd):,}")
    print(f"  N={N:,}  n_controls={n_controls:,}  K={K}")
    print()

    def _pred_path(exp_name: str, variant: str, seed: int) -> str:
        return os.path.join(preds_dir, f"oct_predictions_{exp_name}_{variant}_s{seed}.csv")

    def _load_or_create_train(exp_name: str, variant: str, seed: int, create_fn):
        path = os.path.join(results_dir, f"{exp_name}_{variant}_s{seed}_train.csv")
        if not args.no_save_artifacts and args.resume and os.path.exists(path):
            return pd.read_csv(path)
        train_df = create_fn()
        if not args.no_save_artifacts:
            train_df.to_csv(path, index=False)
        return train_df

    def _cache_path(name: str, seed: int) -> str:
        return os.path.join(cache_dir, f"{name}_s{seed}.npy")

    def load_or_compute_ids(name: str, seed: int, compute_fn) -> np.ndarray:
        path = _cache_path(name, seed)
        if os.path.exists(path):
            ids = np.load(path)
            print(f"    [cache] Loaded {len(ids)} IDs from {os.path.basename(path)}")
            return ids
        ids = compute_fn()
        np.save(path, ids)
        print(f"    [cache] Saved {len(ids)} IDs to {os.path.basename(path)}")
        return ids

    def get_matched_1N(seed: int) -> np.ndarray:
        name = f"matched_1N_seedmethod_{args.stageA_seed_method}_Mpool_{M_pool}"
        return load_or_compute_ids(name, seed, lambda: sample_stageB_matched_controls(
            control_enrolids, case_enrolids, dnn_matrix, dnn_enrolids_path, pn_h5,
            target_count=N, matching_ratio=1, M_pool=M_pool,
            seed_method=args.stageA_seed_method, seed=seed,
            X_majority_leaf=X_majority_leaf, verbose=True, use_kmeanspp=args.use_kmeanspp,
        ))

    def get_random_perm_excluding(seed: int, exclude_ids: np.ndarray) -> np.ndarray:
        exclude = set(int(x) for x in exclude_ids)
        remaining = np.array([e for e in control_enrolids if int(e) not in exclude], dtype=np.int64)
        rng = np.random.RandomState(seed + 202603)
        return rng.permutation(remaining)

    def _exp_key(exp_name: str, variant: str, s: int) -> tuple:
        return (exp_name, variant, s)

    def _keys_this_run() -> set:
        keys = set()
        if 0 in experiments_to_run:
            keys.update(_exp_key("exp0_rnd", "random", s) for s in seeds)
        if 1 in experiments_to_run:
            keys.update(_exp_key("exp1", "stageA", s) for s in seeds)
        if 2 in experiments_to_run:
            keys.update(_exp_key("exp2", "stageB", s) for s in seeds)
        if 3 in experiments_to_run:
            keys.update(_exp_key("exp3", "mix", s) for s in seeds)
        if 4 in experiments_to_run:
            keys.update(_exp_key("exp4", "matched_1N_plus_1N_random", s) for s in seeds)
        return keys

    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    keys_this_run = _keys_this_run()
    all_rows: List[dict] = []
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        for _, row in existing_df.iterrows():
            key = (str(row["experiment"]), str(row["variant"]), int(row["seed"]))
            if key not in keys_this_run:
                all_rows.append(row.to_dict())
        if all_rows:
            print(f"Loaded {len(all_rows)} existing rows from summary (preserving)")

    for seed in seeds:
        # --- Exp 0: Random 2N ---
        if 0 in experiments_to_run:
            print("\n" + "#" * 80)
            print("EXPERIMENT 0: Random 2N")
            print("#" * 80)
            print(f"  seed={seed}")
            exp_name = "exp0_rnd"
            pred_rnd = _pred_path(exp_name, "random", seed)
            try:
                if args.resume and not args.no_save_artifacts and os.path.exists(pred_rnd):
                    m = load_metrics_from_predictions(pred_rnd, test_pd[target_col])
                    all_rows.append({"experiment": exp_name, "variant": "random", "seed": seed, "n_cases": N, "n_controls": K, **m})
                    print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
                else:
                    def _create_rnd():
                        rnd_ids = sample_random_controls(control_enrolids, K, seed)
                        return pd.concat([cases, controls[controls["ENROLID"].isin(rnd_ids)]], ignore_index=True)
                    rnd_train = _load_or_create_train(exp_name, "random", seed, _create_rnd)
                    m = train_and_evaluate_oct(
                        rnd_train, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"{exp_name}_random_s{seed}", random_seed=TRAIN_TEST_SEED,
                        save_predictions=not args.no_save_artifacts,
                    )
                    all_rows.append({"experiment": exp_name, "variant": "random", "seed": seed, "n_cases": N, "n_controls": K, **m})
                    print(f"    PR-AUC={m.get('pr_auc', 0):.4f} AUC={m.get('auc', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

        # --- Exp 1: Stage A only 2N ---
        if 1 in experiments_to_run:
            print("\n" + "#" * 80)
            print("EXPERIMENT 1: Stage A only (2N dispersed)")
            print("#" * 80)
            print(f"  seed={seed}")
            exp_name = "exp1"
            pred_a = _pred_path(exp_name, "stageA", seed)
            try:
                if args.resume and not args.no_save_artifacts and os.path.exists(pred_a):
                    m = load_metrics_from_predictions(pred_a, test_pd[target_col])
                    all_rows.append({"experiment": exp_name, "variant": "stageA", "seed": seed, "n_cases": N, "n_controls": K, **m})
                    print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
                else:
                    stageA_ids, _ = sample_stageA_dispersed_controls(
                        control_enrolids, dnn_matrix, dnn_enrolids_path, pn_h5,
                        case_enrolids, K, args.stageA_seed_method, seed, M_pool,
                        X_majority_leaf=X_majority_leaf, verbose=True, use_kmeanspp=args.use_kmeanspp,
                    )
                    def _create_stageA():
                        return pd.concat([cases, controls[controls["ENROLID"].isin(stageA_ids)]], ignore_index=True)
                    stageA_train = _load_or_create_train(exp_name, "stageA", seed, _create_stageA)
                    m = train_and_evaluate_oct(
                        stageA_train, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"{exp_name}_stageA_s{seed}", random_seed=TRAIN_TEST_SEED,
                        save_predictions=not args.no_save_artifacts,
                    )
                    all_rows.append({"experiment": exp_name, "variant": "stageA", "seed": seed, "n_cases": N, "n_controls": len(stageA_ids), **m})
                    print(f"    PR-AUC={m.get('pr_auc', 0):.4f} AUC={m.get('auc', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

        # --- Exp 2: Stage B 1:2 matching 2N ---
        if 2 in experiments_to_run:
            print("\n" + "#" * 80)
            print("EXPERIMENT 2: Stage B 1:2 matching (2N)")
            print("#" * 80)
            print(f"  seed={seed}")
            exp_name = "exp2"
            pred_b = _pred_path(exp_name, "stageB", seed)
            try:
                if args.resume and not args.no_save_artifacts and os.path.exists(pred_b):
                    m = load_metrics_from_predictions(pred_b, test_pd[target_col])
                    all_rows.append({"experiment": exp_name, "variant": "stageB", "seed": seed, "n_cases": N, "n_controls": K, **m})
                    print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
                else:
                    stageB_ids = sample_stageB_matched_controls(
                        control_enrolids, case_enrolids, dnn_matrix, dnn_enrolids_path, pn_h5,
                        K, matching_ratio=2, M_pool=M_pool, seed_method=args.stageA_seed_method,
                        seed=seed, X_majority_leaf=X_majority_leaf, verbose=True, use_kmeanspp=args.use_kmeanspp,
                    )
                    def _create_stageB():
                        return pd.concat([cases, controls[controls["ENROLID"].isin(stageB_ids)]], ignore_index=True)
                    stageB_train = _load_or_create_train(exp_name, "stageB", seed, _create_stageB)
                    m = train_and_evaluate_oct(
                        stageB_train, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"{exp_name}_stageB_s{seed}", random_seed=TRAIN_TEST_SEED,
                        save_predictions=not args.no_save_artifacts,
                    )
                    all_rows.append({"experiment": exp_name, "variant": "stageB", "seed": seed, "n_cases": N, "n_controls": len(stageB_ids), **m})
                    print(f"    PR-AUC={m.get('pr_auc', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

        # --- Exp 3: Stage B 1:1 + extra dispersed (2N) ---
        if 3 in experiments_to_run:
            print("\n" + "#" * 80)
            print("EXPERIMENT 3: Stage B 1:1 + extra dispersed (2N)")
            print("#" * 80)
            print(f"  seed={seed}")
            exp_name = "exp3"
            pred_mix = _pred_path(exp_name, "mix", seed)
            try:
                if args.resume and not args.no_save_artifacts and os.path.exists(pred_mix):
                    m = load_metrics_from_predictions(pred_mix, test_pd[target_col])
                    all_rows.append({"experiment": exp_name, "variant": "mix", "seed": seed, "n_cases": N, "n_controls": K, **m})
                    print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
                else:
                    match_ids = get_matched_1N(seed)
                    disp_ids = sample_stageA_on_restricted_pool(
                        control_enrolids, match_ids, dnn_matrix, dnn_enrolids_path, pn_h5,
                        case_enrolids, N, args.stageA_seed_method, seed,
                        X_majority_leaf, id_to_pos, verbose=True, use_kmeanspp=args.use_kmeanspp,
                    )
                    mix_ids = np.unique(np.concatenate([match_ids, disp_ids]))[:K]
                    if len(mix_ids) < K:
                        remaining = np.array([e for e in control_enrolids if int(e) not in set(int(x) for x in mix_ids)])
                        extra = sample_random_controls(remaining, K - len(mix_ids), seed + 9999)
                        mix_ids = np.concatenate([mix_ids, extra])[:K]
                    def _create_mix():
                        return pd.concat([cases, controls[controls["ENROLID"].isin(mix_ids)]], ignore_index=True)
                    mix_train = _load_or_create_train(exp_name, "mix", seed, _create_mix)
                    m = train_and_evaluate_oct(
                        mix_train, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"{exp_name}_mix_s{seed}", random_seed=TRAIN_TEST_SEED,
                        save_predictions=not args.no_save_artifacts,
                    )
                    all_rows.append({"experiment": exp_name, "variant": "mix", "seed": seed, "n_cases": N, "n_controls": len(mix_ids), **m})
                    print(f"    PR-AUC={m.get('pr_auc', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

        # --- Exp 4: 1N matched + 1N random = 2N total ---
        if 4 in experiments_to_run:
            print("\n" + "#" * 80)
            print("EXPERIMENT 4: 1N matched + 1N random (2N)")
            print("#" * 80)
            print(f"  seed={seed}")
            exp_name = "exp4"
            variant = "matched_1N_plus_1N_random"
            total_controls = 2 * N
            pred_path = _pred_path(exp_name, variant, seed)
            try:
                if args.resume and not args.no_save_artifacts and os.path.exists(pred_path):
                    m = load_metrics_from_predictions(pred_path, test_pd[target_col])
                    all_rows.append({"experiment": exp_name, "variant": variant, "seed": seed, "n_cases": N, "n_controls": total_controls, **m})
                    print(f"    SKIP (loaded) PR-AUC={m.get('pr_auc', 0):.4f}")
                else:
                    match_ids = get_matched_1N(seed)
                    perm = get_random_perm_excluding(seed, match_ids)
                    extra_ids = perm[:N].astype(np.int64)
                    ctrl_ids_exp4 = np.concatenate([match_ids, extra_ids]).astype(np.int64)
                    ctrl_ids_exp4 = np.unique(ctrl_ids_exp4)[:total_controls]
                    if len(ctrl_ids_exp4) < total_controls:
                        remaining = np.array([e for e in control_enrolids if int(e) not in set(int(x) for x in ctrl_ids_exp4)])
                        extra = sample_random_controls(remaining, total_controls - len(ctrl_ids_exp4), seed + 7777)
                        ctrl_ids_exp4 = np.concatenate([ctrl_ids_exp4, extra])[:total_controls]
                    def _create_exp4():
                        return pd.concat([cases, controls[controls["ENROLID"].isin(ctrl_ids_exp4)]], ignore_index=True)
                    train_df = _load_or_create_train(exp_name, variant, seed, _create_exp4)
                    m = train_and_evaluate_oct(
                        train_df, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"{exp_name}_{variant}_s{seed}", random_seed=TRAIN_TEST_SEED,
                        save_predictions=not args.no_save_artifacts,
                    )
                    all_rows.append({"experiment": exp_name, "variant": variant, "seed": seed, "n_cases": N, "n_controls": total_controls, **m})
                    print(f"    PR-AUC={m.get('pr_auc', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_rows.append({"experiment": exp_name, "variant": variant, "seed": seed, "error": str(e)})

        # Save summary after each seed (interrupt-safe)
        df_out = pd.DataFrame(all_rows)
        df_out = df_out.sort_values(["experiment", "variant", "seed"]).reset_index(drop=True)
        df_out.to_csv(summary_path, index=False)
        print(f"    [saved] experiment_summary.csv ({len(df_out)} rows)")

    # Final save and aggregates
    df_out = pd.DataFrame(all_rows)
    print(f"\nSaved {len(df_out)} rows to {summary_path}")

    agg_cols = ["pr_auc", "auc", "best_mcc", "balanced_recall_gmean", "balanced_specificity_gmean", "optimal_f1"]
    agg_cols = [c for c in agg_cols if c in df_out.columns]
    if agg_cols:
        agg = df_out.groupby(["experiment", "variant"]).agg(
            {c: ["mean", "std"] for c in agg_cols}
        ).round(4)
        print("\nAggregated (mean ± std):")
        print(agg)
        agg.to_csv(os.path.join(results_dir, "experiment_summary_aggregated.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
