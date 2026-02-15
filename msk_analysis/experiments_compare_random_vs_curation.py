#!/usr/bin/env python
"""
experiments_compare_random_vs_curation.py
=========================================
Three experiments to diagnose when/why optimal curation underperforms
random undersampling.

Experiments:
  1. Stage A only: k-center dispersed 2N controls vs random 2N
  2. Stage B 1:2 matching: full two-stage with matching_ratio=2 vs random
  3. Stage B 1:1 + extra dispersed: N matched + N dispersed vs random

All use same data split, OCT pipeline, and training set size (N cases + 2N controls).

Usage
-----
  cd msk_analysis
  python experiments_compare_random_vs_curation.py [--seeds 0,1,2,3,4] [--outdir ./exp_random_vs_curation]

  Alignment diagnostics only (no training):
  python experiments_compare_random_vs_curation.py --debug_alignment --seeds 0 --distances_dir ./precomputed_distances_msk_medical_only

  Flags:
    --seeds           Comma-separated seeds (default: 0,1,2,3,4)
    --outdir          Output directory (default: ./exp_random_vs_curation)
    --stageA_seed_method  {centroid,density,random,smart} (default: centroid)
    --M_pool          Candidate pool size for Stage A/B (default: min(50000, n_controls))
    --quota_enabled   Enable quota constraints (default: False)
    --parquet_path    Path to parquet dataset (default: msk_2017_18_full.parquet)
    --distances_dir   Precomputed distances directory
    --debug_alignment Run ID-alignment diagnostics and hard assertions, then exit
"""

from __future__ import annotations

import sys
import os
import argparse
import time
import traceback
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Path setup
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import importlib
import h5py

try:
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import (
        two_stage_kcenter_then_match,
        load_pn_hdf5,
        build_id_to_index,
        farthest_first_kcenter_indices,
        choose_seed_random,
        choose_seed_centroid,
        choose_seed_max_density,
        choose_seed_closest_to_positives_from_pn,
    )
except ImportError:
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import (
        two_stage_kcenter_then_match,
        load_pn_hdf5,
        build_id_to_index,
        farthest_first_kcenter_indices,
        choose_seed_random,
        choose_seed_centroid,
        choose_seed_max_density,
        choose_seed_closest_to_positives_from_pn,
    )

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    finetune_oct,
    evaluate_binary_oct,
    best_mcc_threshold,
    best_balanced_threshold,
)
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
from pyspark.sql import SparkSession

TRAIN_TEST_SEED = 123
OCT_DEPTHS = [7]
OCT_MINBUCKETS = [150]
OCT_CPS = [0.0001, 0.001, 0.01]


# =============================================================================
# Sampling functions
# =============================================================================

def sample_random_controls(
    control_enrolids: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Select k controls uniformly at random without replacement."""
    n = len(control_enrolids)
    if k > n:
        raise ValueError(f"sample_random_controls: k={k} > n_controls={n}")
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    return control_enrolids[idx]


def sample_stageA_dispersed_controls(
    leaf_controls_enrolids: np.ndarray,
    leaf_nn_matrix_npy: str,
    leaf_nn_enrolids_npy: str,
    pn_h5_path: str,
    leaf_cases_enrolids: np.ndarray,
    k: int,
    seed_method: str,
    seed: int,
    M_pool: int,
    X_majority_leaf: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Stage A only: farthest-first k-center to select k dispersed controls.
    Returns (selected_control_enrolids, mean_min_dist_to_case or None).
    """
    d_nn = np.load(leaf_nn_matrix_npy, mmap_mode="r")
    dnn_ids = np.load(leaf_nn_enrolids_npy)

    if d_nn.shape[0] != len(dnn_ids):
        raise ValueError("d_nn and enrolids length mismatch")

    # Align controls to d_nn order
    id2pos = {int(e): i for i, e in enumerate(leaf_controls_enrolids)}
    if not all(int(e) in id2pos for e in dnn_ids):
        raise ValueError("dnn_ids must be subset of leaf_controls_enrolids")
    # d_nn rows/cols correspond to dnn_ids; leaf_controls may differ
    n_avail = d_nn.shape[0]
    M_eff = min(k, M_pool, n_avail)
    if M_eff < k and verbose:
        print(f"  [Stage A] WARNING: M_pool/k cap: requested k={k}, using M_eff={M_eff}")

    k_eff = min(k, M_eff, n_avail)
    if k_eff != k and verbose:
        print(f"  [Stage A] WARNING: requested k={k}, available {n_avail}; using k_eff={k_eff}")

    # Load d_pn for seed selection (smart) and optional mean-distance stats
    f_pn, d_pn, pn_maj_ids, pn_min_ids = load_pn_hdf5(pn_h5_path)
    pn_maj_id2idx = build_id_to_index(pn_maj_ids)
    pn_min_id2idx = build_id_to_index(pn_min_ids)
    try:
        pn_rows = np.array([pn_maj_id2idx[int(e)] for e in dnn_ids], dtype=np.int64)
        pn_cols = np.array([pn_min_id2idx[int(e)] for e in leaf_cases_enrolids], dtype=np.int64)
        rows_sort = np.argsort(pn_rows)
        cols_sort = np.argsort(pn_cols)
        d_pn_sorted = np.array(
            d_pn[pn_rows[rows_sort], :][:, pn_cols[cols_sort]], dtype=np.float32
        )
        rows_unsort = np.argsort(rows_sort)
        cols_unsort = np.argsort(cols_sort)
        d_pn_leaf = d_pn_sorted[rows_unsort, :][:, cols_unsort]
    finally:
        f_pn.close()

    # Choose seed
    if seed_method == "random":
        seed_idx = choose_seed_random(n_avail, random_state=seed)
    elif seed_method == "centroid":
        if X_majority_leaf is None:
            raise ValueError("seed_method='centroid' requires X_majority_leaf")
        if X_majority_leaf.shape[0] != n_avail:
            raise ValueError("X_majority_leaf shape mismatch")
        seed_idx = choose_seed_centroid(X_majority_leaf)
    elif seed_method == "density":
        seed_idx = choose_seed_max_density(d_nn, percentile=10.0)
    elif seed_method == "smart":
        seed_idx = int(np.argmin(d_pn_leaf.mean(axis=1)))
    else:
        raise ValueError(f"Unknown seed_method: {seed_method}")

    # Run k-center
    cand_idx = farthest_first_kcenter_indices(d_nn, k_eff, seed_idx)
    selected_enrolids = dnn_ids[cand_idx]

    # Mean min distance from selected controls to their nearest case (for logging)
    mean_min = float(d_pn_leaf[cand_idx, :].min(axis=1).mean()) if len(cand_idx) else None

    return selected_enrolids, mean_min


def sample_stageB_matched_controls(
    leaf_controls_enrolids: np.ndarray,
    leaf_cases_enrolids: np.ndarray,
    leaf_nn_matrix_npy: str,
    leaf_nn_enrolids_npy: str,
    pn_h5_path: str,
    target_count: int,
    matching_ratio: int,
    M_pool: int,
    seed_method: str,
    seed: int,
    quota_cfg: Optional[dict],
    X_majority_leaf: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Full two-stage: k-center + min-cost matching. Returns exactly target_count
    distinct control enrolids. If matching produces duplicates, deduplicate and
    top-up from candidate pool (or remaining controls) deterministically.
    """
    n_controls = len(leaf_controls_enrolids)
    M_eff = min(M_pool, n_controls)
    if M_eff < target_count and verbose:
        print(f"  [Stage B] WARNING: M_pool={M_pool} < target_count={target_count}; M_eff={M_eff}")

    result = two_stage_kcenter_then_match(
        leaf_controls_enrolids=leaf_controls_enrolids,
        leaf_cases_enrolids=leaf_cases_enrolids,
        leaf_nn_matrix_npy=leaf_nn_matrix_npy,
        leaf_nn_enrolids_npy=leaf_nn_enrolids_npy,
        pn_h5_path=pn_h5_path,
        M=M_eff,
        use_adaptive_pool=False,
        force_nearest_per_case=False,
        force_topm=1,
        assignment_topk_start=None,
        seed_method=seed_method,
        random_state=seed,
        matching_ratio=matching_ratio,
        case_weighting=None,
        quota_cfg=quota_cfg,
        X_majority_leaf=X_majority_leaf,
    )

    raw_ids = result["selected_control_enrolids"]
    unique_ids = list(set(int(x) for x in raw_ids))

    if len(unique_ids) >= target_count:
        # Deterministic: take first target_count by sorted enrolid
        unique_ids = sorted(unique_ids)[:target_count]
        return np.array(unique_ids, dtype=np.int64)

    # Top-up from candidate pool
    cand_ids = set(result.get("candidate_majority_enrolids", raw_ids))
    if isinstance(cand_ids, np.ndarray):
        cand_ids = set(int(x) for x in cand_ids.ravel())
    remaining = sorted(cand_ids - set(unique_ids))
    need = target_count - len(unique_ids)
    for eid in remaining:
        if need <= 0:
            break
        unique_ids.append(int(eid))
        need -= 1

    # If still short, top-up from controls_train not yet selected
    if need > 0:
        all_ctrl = set(int(e) for e in leaf_controls_enrolids)
        already = set(unique_ids)
        extra = sorted(all_ctrl - already)
        for eid in extra[:need]:
            unique_ids.append(int(eid))

    return np.array(unique_ids[:target_count], dtype=np.int64)


def sample_stageA_on_restricted_pool(
    leaf_controls_enrolids: np.ndarray,
    exclude_enrolids: np.ndarray,
    leaf_nn_matrix_npy: str,
    leaf_nn_enrolids_npy: str,
    pn_h5_path: str,
    leaf_cases_enrolids: np.ndarray,
    k: int,
    seed_method: str,
    seed: int,
    X_majority_leaf: Optional[np.ndarray],
    id_to_position_in_full: dict,
    verbose: bool = True,
    debug_alignment: bool = False,
) -> np.ndarray:
    """
    Run Stage A on the subset of controls excluding exclude_enrolids.
    Returns k selected control enrolids from the restricted pool.
    """
    exclude_set = set(int(e) for e in exclude_enrolids)
    remaining_mask = np.array(
        [int(e) not in exclude_set for e in leaf_controls_enrolids], dtype=bool
    )
    remaining_ids = leaf_controls_enrolids[remaining_mask]
    if len(remaining_ids) < k:
        if verbose:
            print(f"  [Stage A restricted] WARNING: only {len(remaining_ids)} remaining, requested k={k}")
        k = len(remaining_ids)

    d_nn = np.load(leaf_nn_matrix_npy, mmap_mode="r")
    dnn_ids = np.load(leaf_nn_enrolids_npy)
    rem_positions = [id_to_position_in_full[int(e)] for e in remaining_ids]

    # Hard assertion: verify id_to_position_in_full round-trips correctly
    if debug_alignment:
        bad = []
        for eid in remaining_ids:
            eid_int = int(eid)
            pos = id_to_position_in_full[eid_int]
            actual = int(dnn_ids[pos])
            if actual != eid_int:
                bad.append((eid_int, pos, actual))
        if bad:
            sample = bad[:10]
            raise AssertionError(
                f"[debug_alignment] sample_stageA_on_restricted_pool: "
                f"{len(bad)} remaining IDs fail round-trip dnn_ids[id_to_pos[eid]] == eid. "
                f"First 10: {sample}"
            )
        else:
            print(f"  [debug_alignment] sample_stageA_on_restricted_pool: "
                  f"all {len(remaining_ids)} remaining IDs pass round-trip check")

    d_nn_sub = np.array(d_nn[np.ix_(rem_positions, rem_positions)], dtype=np.float32)

    f_pn, d_pn, pn_maj_ids, pn_min_ids = load_pn_hdf5(pn_h5_path)
    pn_maj_id2idx = build_id_to_index(pn_maj_ids)
    pn_min_id2idx = build_id_to_index(pn_min_ids)
    try:
        pn_rows = np.array([pn_maj_id2idx[int(e)] for e in remaining_ids], dtype=np.int64)
        pn_cols = np.array([pn_min_id2idx[int(e)] for e in leaf_cases_enrolids], dtype=np.int64)
        rows_sort = np.argsort(pn_rows)
        cols_sort = np.argsort(pn_cols)
        d_pn_sorted = np.array(
            d_pn[pn_rows[rows_sort], :][:, pn_cols[cols_sort]], dtype=np.float32
        )
        rows_unsort = np.argsort(rows_sort)
        cols_unsort = np.argsort(cols_sort)
        d_pn_leaf = d_pn_sorted[rows_unsort, :][:, cols_unsort]
    finally:
        f_pn.close()

    if seed_method == "random":
        seed_idx = choose_seed_random(len(remaining_ids), random_state=seed)
    elif seed_method == "centroid" and X_majority_leaf is not None:
        X_sub = X_majority_leaf[rem_positions]
        seed_idx = choose_seed_centroid(X_sub)
    elif seed_method == "smart":
        seed_idx = int(np.argmin(d_pn_leaf.mean(axis=1)))
    else:
        seed_idx = choose_seed_random(len(remaining_ids), random_state=seed)

    cand_idx = farthest_first_kcenter_indices(d_nn_sub, k, seed_idx)
    return remaining_ids[cand_idx]


# =============================================================================
# OCT training and evaluation
# =============================================================================

def load_metrics_from_predictions(
    pred_path: str,
    y_test: pd.Series,
) -> dict:
    """
    Load a predictions CSV and compute metrics (AUC, PR-AUC, MCC, etc.).
    Use when retraining is skipped because predictions already exist.
    """
    pred_df = pd.read_csv(pred_path)
    if "predicted_proba" not in pred_df.columns:
        raise ValueError(f"predictions file missing 'predicted_proba': {pred_path}")
    y_proba = pred_df["predicted_proba"].values
    y_test_arr = np.asarray(y_test).astype(int)
    if len(y_proba) != len(y_test_arr):
        raise ValueError(f"length mismatch: pred={len(y_proba)}, test={len(y_test_arr)}")

    auc = roc_auc_score(y_test_arr, y_proba)
    pr_auc = average_precision_score(y_test_arr, y_proba)
    mcc_res = best_mcc_threshold(y_test_arr, y_proba)
    best_mcc = mcc_res["mcc"]
    y_pred_mcc = mcc_res["y_pred"]

    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred_mcc).ravel()
    recall_mcc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_mcc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    balanced = best_balanced_threshold(y_test_arr, y_proba)
    gmean_recall = balanced["gmean_opt"]["recall"]
    gmean_specificity = balanced["gmean_opt"]["specificity"]

    num_leaves = int(pred_df["leaf_assignment"].nunique()) if "leaf_assignment" in pred_df.columns else np.nan

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": best_mcc,
        "recall_mcc": recall_mcc,
        "precision_mcc": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "balanced_recall_gmean": gmean_recall,
        "balanced_specificity_gmean": gmean_specificity,
        "num_leaves": num_leaves,
        "best_depth": np.nan,
        "best_minbucket": np.nan,
        "best_cp": np.nan,
    }


def train_and_evaluate_oct(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    results_dir: str,
    save_suffix: str,
    random_seed: int = TRAIN_TEST_SEED,
) -> dict:
    """Train OCT, evaluate on test, return metrics dict."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model, params, _, preprocessor, feat_names = finetune_oct(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        binary_cols=bin_cols,
        depths=OCT_DEPTHS,
        minbuckets=OCT_MINBUCKETS,
        cps=OCT_CPS,
        tree_kind="oct",
        verbose=False,
        random_seed=random_seed,
    )

    metrics = evaluate_binary_oct(
        model, X_test, y_test, preprocessor, feat_names,
        X_val_df=X_val, y_val=y_val,
        results_dir=results_dir, save_suffix=save_suffix,
    )

    # IAI model.apply() expects a DataFrame, not a Matrix
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()
    X_test_proc = pd.DataFrame(X_test_proc, columns=feat_names)
    num_leaves = int(len(pd.unique(model.apply(X_test_proc))))
    if isinstance(params, dict):
        bp = (params.get("depth"), params.get("minbucket"), params.get("cp"))
    else:
        bp = (params[0], params[1], params[2])
    return {
        "best_depth": bp[0],
        "best_minbucket": bp[1],
        "best_cp": bp[2],
        "num_leaves": num_leaves,
        **{k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float))},
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compare random vs curation experiments")
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated seeds (default: 0)")
    p.add_argument("--outdir", type=str, default="./exp_random_vs_curation_medical_only_features",
                   help="Output directory")
    p.add_argument("--stageA_seed_method", choices=["centroid", "density", "random", "smart"],
                   default="random", help="Stage A seed selection")
    p.add_argument("--M_pool", type=int, default=None,
                   help="Candidate pool size (default: n_controls//2))")
    p.add_argument("--quota_enabled", action="store_true", help="Enable quota constraints") #- Sets to True when present (default: False)
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--distances_dir", type=str,
                   default="./precomputed_distances_msk_medical_only") 
    p.add_argument("--feature_set", type=str, default="medical_only", choices=["medical_only", "all_cost", "less_cost"])
    p.add_argument("--resume", action="store_true", #- Sets to True when present (default: False)
                   help="Skip runs already in experiment_summary.csv; load training CSV when it exists")
    p.add_argument("--debug_alignment", action="store_true",
                   help="Run ID-alignment diagnostics and hard assertions (no training)")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    outdir = args.outdir
    results_dir = os.path.join(outdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    quota_cfg = {"enabled": True, "T": 5, "mode": "pool_mass", "K_per_bin": 25} if args.quota_enabled else None

    print("=" * 80)
    print("EXPERIMENTS: Random vs Curation")
    print("=" * 80)
    print(f"  Seeds: {seeds}")
    print(f"  Stage A seed method: {args.stageA_seed_method}")
    print(f"  Quota enabled: {args.quota_enabled}")
    print()

    # Load data (same as two_stage_iterative)
    spark = SparkSession.builder.appName("ExpRandomVsCuration").getOrCreate()
    df = spark.read.format("parquet").load(args.parquet_path).toPandas()

    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    COST_COLUMNS = [
        col for col in df.columns 
        if ("cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower() or 
            "decreasing" in col.lower() or "skewness" in col.lower() or "kurtosis" in col.lower() or
            "cv" in col.lower() or "range" in col.lower())
        and "2018" not in col  # Exclude 2018 columns to prevent leakage
    ]
    AUXILIARY_COST_COLUMNS =[col for col in df.columns if col.startswith("comorbidity_only") or col.startswith("msk_procedure")]
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    if args.feature_set == "medical_only":
        exclude_cols += COST_COLUMNS
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    elif args.feature_set == "all_cost":
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    elif args.feature_set == "less_cost":
        exclude_cols += AUXILIARY_COST_COLUMNS
        feature_cols = [c for c in df.columns if c not in exclude_cols] 
    #print(f"Feature columns: {feature_cols}")
    print(f"Exclude columns: {exclude_cols}")
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

    M_pool = args.M_pool if args.M_pool is not None else n_controls//2

    print(f"Train: {len(train_pd):,}  Val: {len(val_pd):,}  Test: {len(test_pd):,}")
    print(f"  N (cases): {N:,}  |  Controls: {n_controls:,}  |  M_pool: {M_pool:,}")
    print()

    # Distance paths
    pn_h5 = os.path.join(args.distances_dir, "distances_majority_minority.h5")
    dnn_dir = os.path.join(args.distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}")
    dnn_matrix = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")
    for p in [pn_h5, dnn_matrix, dnn_enrolids]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    dnn_ids = np.load(dnn_enrolids)
    id_to_pos = {int(e): i for i, e in enumerate(dnn_ids)}

    # ------------------------------------------------------------------
    # Debug alignment: run diagnostics and exit
    # ------------------------------------------------------------------
    if args.debug_alignment:
        from public.debug_id_alignment import run_full_alignment_check
        run_full_alignment_check(
            dnn_ids=dnn_ids,
            control_enrolids=control_enrolids,
            case_enrolids=case_enrolids,
            d_nn_path=dnn_matrix,
            pn_h5_path=pn_h5,
            train_test_seed=TRAIN_TEST_SEED,
            distances_dir=args.distances_dir,
            strict=True,
        )
        print("--debug_alignment complete. Exiting (no experiments run).")
        return

    X_majority_leaf = None
    if args.stageA_seed_method == "centroid":
        numeric_feature_cols = [
            c for c in feature_cols
            if c in df.select_dtypes(include="number").columns
        ]
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

    all_rows = []
    K = 2 * N  # target majority count
    preds_dir = os.path.join(results_dir, "predictions")

    def _pred_path(exp_name, variant, seed):
        return os.path.join(preds_dir, f"oct_predictions_{exp_name}_{variant}_s{seed}.csv")

    def _load_or_create_train(exp_name, variant, seed, create_fn):
        """Load training CSV if it exists, else create and save. Returns train_df."""
        path = os.path.join(results_dir, f"{exp_name}_{variant}_s{seed}_train.csv")
        if args.resume and os.path.exists(path):
            return pd.read_csv(path)
        train_df = create_fn()
        train_df.to_csv(path, index=False)
        return train_df

    
    print("\n" + "#" * 80)
    print("EXPERIMENT 0: Random undersample")
    print("#" * 80)
    for seed in seeds:
        print(f"\n  --- seed={seed} ---")
        # S_random
        exp_name = "random_undersample"
        pred_rnd = _pred_path(exp_name, "random", seed)
        if args.resume and os.path.exists(pred_rnd):
            m_rnd = load_metrics_from_predictions(pred_rnd, test_pd[target_col])
            all_rows.append({
                "experiment": exp_name, "variant": "random", "seed": seed,
                "n_cases": N, "n_controls": K, **m_rnd,
            })
            print(f"    Random: SKIP (loaded from predictions) PR-AUC={m_rnd.get('pr_auc', 0):.4f}")
        else:
            print(f"    Random: SAMPLING AND TRAINING...")
            def _create_rnd():
                rnd_ids = sample_random_controls(control_enrolids, K, seed)
                return pd.concat([cases, controls[controls["ENROLID"].isin(rnd_ids)]], ignore_index=True)
            rnd_train = _load_or_create_train(exp_name, "random", seed, _create_rnd)
            m_rnd = train_and_evaluate_oct(
                rnd_train, val_pd, test_pd, feature_cols, target_col,
                CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                results_dir, f"{exp_name}_random_s{seed}", TRAIN_TEST_SEED,
            )
            all_rows.append({
                "experiment": exp_name, "variant": "random", "seed": seed,
                "n_cases": N, "n_controls": K, **m_rnd,
            })
            print(f"    Random: PR-AUC={m_rnd.get('pr_auc', 0):.4f} AUC={m_rnd.get('auc', 0):.4f}")
    print("\n" + "#" * 80)
    print("EXPERIMENT 1: Stage A only (k-center dispersed)")
    print("#" * 80)
    for seed in seeds:  
        exp_name = "exp1"
        # ----- Experiment 1: Stage A only -----
        print(f"seed={seed} Stage A: SAMPLING AND TRAINING...")

        try:
            pred_a = _pred_path(exp_name, "stageA", seed)
            if args.resume and os.path.exists(pred_a):
                m_A = load_metrics_from_predictions(pred_a, test_pd[target_col])
                all_rows.append({
                    "experiment": exp_name, "variant": "stageA", "seed": seed,
                    "n_cases": N, "n_controls": K, **m_A,
                })
                print(f"    Stage A: SKIP (loaded from predictions) PR-AUC={m_A.get('pr_auc', 0):.4f}")
            else:
                stageA_ids, mean_min = sample_stageA_dispersed_controls(
                    control_enrolids, dnn_matrix, dnn_enrolids, pn_h5,
                    case_enrolids, K, args.stageA_seed_method, seed, M_pool,
                    X_majority_leaf=X_majority_leaf, verbose=True,
                )

                def _create_stageA():
                    return pd.concat([
                        cases,
                        controls[controls["ENROLID"].isin(stageA_ids)],
                    ], ignore_index=True)
                stageA_train = _load_or_create_train(exp_name, "stageA", seed, _create_stageA)
                m_A = train_and_evaluate_oct(
                    stageA_train, val_pd, test_pd, feature_cols, target_col,
                    CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                    results_dir, f"{exp_name}_stageA_s{seed}", TRAIN_TEST_SEED,
                )
                all_rows.append({
                    "experiment": exp_name, "variant": "stageA", "seed": seed,
                    "n_cases": N, "n_controls": len(stageA_ids), **m_A,
                })
                print(f"    Stage A: PR-AUC={m_A.get('pr_auc', 0):.4f} AUC={m_A.get('auc', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

    # ----- Experiment 2: Stage B 1:2 matching -----
    print("\n" + "#" * 80)
    print("EXPERIMENT 2: Stage B 1:2 matching")
    print("#" * 80)

    for seed in seeds:
        print("\n" + "#" * 80)
        print("EXPERIMENT 2: Stage B 1:2 matching")
        print("#" * 80)
        print(f"\n  --- seed={seed} ---")
        exp_name = "exp2"
        try:
            pred_b = _pred_path(exp_name, "stageB", seed)
            if args.resume and os.path.exists(pred_b):
                m_B = load_metrics_from_predictions(pred_b, test_pd[target_col])
                all_rows.append({
                    "experiment": exp_name, "variant": "stageB", "seed": seed,
                    "n_cases": N, "n_controls": K, **m_B,
                })
                print(f"    StageB: SKIP (loaded from predictions) PR-AUC={m_B.get('pr_auc', 0):.4f}")
            else:
                stageB_ids = sample_stageB_matched_controls(
                    control_enrolids, case_enrolids, dnn_matrix, dnn_enrolids, pn_h5,
                    K, matching_ratio=2, M_pool=M_pool, seed_method=args.stageA_seed_method,
                    seed=seed, quota_cfg=quota_cfg, X_majority_leaf=X_majority_leaf, verbose=True,
                )

                def _create_stageB():
                    return pd.concat([
                        cases,
                        controls[controls["ENROLID"].isin(stageB_ids)],
                    ], ignore_index=True)
                stageB_train = _load_or_create_train(exp_name, "stageB", seed, _create_stageB)
                m_B = train_and_evaluate_oct(
                    stageB_train, val_pd, test_pd, feature_cols, target_col,
                    CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                    results_dir, f"{exp_name}_stageB_s{seed}", TRAIN_TEST_SEED,
                )
                all_rows.append({
                    "experiment": exp_name, "variant": "stageB", "seed": seed,
                    "n_cases": N, "n_controls": len(stageB_ids), **m_B,
                })
                print(f"    StageB: PR-AUC={m_B.get('pr_auc', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

    # ----- Experiment 3: Stage B 1:1 + extra dispersed -----
    print("\n" + "#" * 80)
    print("EXPERIMENT 3: Stage B 1:1 + extra dispersed")
    print("#" * 80)

    for seed in seeds:
        print(f"\n  --- seed={seed} ---")
        exp_name = "exp3"
        try:
            pred_mix = _pred_path(exp_name, "mix", seed)
            if args.resume and os.path.exists(pred_mix):
                m_mix = load_metrics_from_predictions(pred_mix, test_pd[target_col])
                all_rows.append({
                    "experiment": exp_name, "variant": "mix", "seed": seed,
                    "n_cases": N, "n_controls": K, **m_mix,
                })
                print(f"    Mix: SKIP (loaded from predictions) PR-AUC={m_mix.get('pr_auc', 0):.4f}")
            else:
                # N matched via 1:1
                match_ids = sample_stageB_matched_controls(
                    control_enrolids, case_enrolids, dnn_matrix, dnn_enrolids, pn_h5,
                    N, matching_ratio=1, M_pool=M_pool, seed_method=args.stageA_seed_method,
                    seed=seed, quota_cfg=quota_cfg, X_majority_leaf=X_majority_leaf, verbose=True,
                )
                # N dispersed from remaining
                disp_ids = sample_stageA_on_restricted_pool(
                    control_enrolids, match_ids, dnn_matrix, dnn_enrolids, pn_h5,
                    case_enrolids, N, args.stageA_seed_method, seed,
                    X_majority_leaf, id_to_pos, verbose=True,
                )
                mix_ids = np.unique(np.concatenate([match_ids, disp_ids]))[:K]
                if len(mix_ids) < K:
                    mix_set = set(int(x) for x in mix_ids)
                    remaining = np.array([e for e in control_enrolids if int(e) not in mix_set])
                    extra = sample_random_controls(remaining, K - len(mix_ids), seed + 9999)
                    mix_ids = np.concatenate([mix_ids, extra])[:K]

                def _create_mix():
                    return pd.concat([
                        cases,
                        controls[controls["ENROLID"].isin(mix_ids)],
                    ], ignore_index=True)
                mix_train = _load_or_create_train(exp_name, "mix", seed, _create_mix)
                m_mix = train_and_evaluate_oct(
                    mix_train, val_pd, test_pd, feature_cols, target_col,
                    CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                    results_dir, f"{exp_name}_mix_s{seed}", TRAIN_TEST_SEED,
                )
                all_rows.append({
                    "experiment": exp_name, "variant": "mix", "seed": seed,
                    "n_cases": N, "n_controls": len(mix_ids), **m_mix,
                })
                print(f"    Mix: PR-AUC={m_mix.get('pr_auc', 0):.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_rows.append({"experiment": exp_name, "variant": "error", "seed": seed, "error": str(e)})

    # ----- Save summary -----
    df_out = pd.DataFrame(all_rows)
    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    df_out.to_csv(summary_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {summary_path}")

    agg = df_out.groupby(["experiment", "variant"]).agg({
        "pr_auc": ["mean", "std"],
        "auc": ["mean", "std"],
        "best_mcc": ["mean", "std"],
        "balanced_recall_gmean": ["mean", "std"],
        "balanced_specificity_gmean": ["mean", "std"],
        "optimal_f1": ["mean", "std"],
    }).round(4)
    print("\nAggregated (mean ± std):")
    print(agg)
    agg.to_csv(os.path.join(results_dir, "experiment_summary_aggregated.csv"))
    print("\nDone.")


if __name__ == "__main__":
    main()
