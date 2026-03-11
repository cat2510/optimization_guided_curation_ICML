#!/usr/bin/env python
"""
precompute_msk_ZA_variants.py
=============================
Precompute Z_A variant distance artifacts (Stage A k-center).
Supports ZA_v0_flags_only, ZA_v1_flags_plus_counts, ZA_v2_flags_plus_intensity_norm.
Metrics: euclidean | gower (gower default for v1/v2 with continuous components).

Usage:
  cd msk_analysis
  python scripts/precompute_msk_ZA_variants.py --za_preset ZA_v1_flags_plus_counts --metric gower
  python scripts/precompute_msk_ZA_variants.py --za_preset ZA_v0_flags_only --run_stageA_overlap_test 1
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
msk_analysis_dir = os.path.dirname(script_dir)
parent_dir = os.path.dirname(msk_analysis_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, msk_analysis_dir)
sys.path.insert(0, script_dir)

from public.model_IAI import (
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    train_test_split_enrol,
)
from msk_feature_groups import (
    get_ZA_columns,
    get_cost_columns_2017,
    get_utilization_columns,
    validate_no_2018_leakage,
    ZA_PRESETS,
)
from public.precompute_distances import (
    compute_distances_batched,
    precompute_leaf_dnn_memmap,
    save_distances_hdf5,
)
from tie_diagnostics import run_tie_diagnostics
# Use pd.read_parquet (not Spark) for deterministic row order; Spark may shuffle and change train/test split.

TRAIN_TEST_SEED = 123

# Gower distance helpers
def _compute_gower_block(X_A: np.ndarray, X_B: np.ndarray, binary_set: set, ranges: np.ndarray, p: int) -> np.ndarray:
    n_A, n_B = X_A.shape[0], X_B.shape[0]
    block = np.zeros((n_A, n_B), dtype=np.float32)
    for j in range(p):
        a_vals, b_vals = X_A[:, j], X_B[:, j]
        if j in binary_set:
            block += (a_vals[:, None] != b_vals[None, :]).astype(np.float32)
        else:
            rj = ranges[j]
            block += np.abs(a_vals[:, None].astype(np.float64) - b_vals[None, :].astype(np.float64)).astype(np.float32) / rj
    return block / p


def compute_gower_pn(X_maj: np.ndarray, X_min: np.ndarray, bin_idx: List[int], ranges: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    from tqdm import tqdm
    n_maj, p = X_maj.shape[0], X_maj.shape[1]
    binary_set = set(bin_idx)
    distances = np.zeros((n_maj, X_min.shape[0]), dtype=np.float32)
    for b in tqdm(range((n_maj + batch_size - 1) // batch_size), desc="PN Gower"):
        s, e = b * batch_size, min((b + 1) * batch_size, n_maj)
        distances[s:e] = _compute_gower_block(X_maj[s:e], X_min, binary_set, ranges, p)
    return distances


def precompute_gower_dnn_memmap(X_maj: np.ndarray, bin_idx: List[int], ranges: np.ndarray, out_dir: str, batch_size: int = 750) -> str:
    from tqdm import tqdm
    os.makedirs(out_dir, exist_ok=True)
    n, p = X_maj.shape[0], X_maj.shape[1]
    binary_set = set(bin_idx)
    path = os.path.join(out_dir, "leaf_global_dnn_matrix.npy")
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, n))
    for b in tqdm(range((n + batch_size - 1) // batch_size), desc="DNN Gower"):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        mm[s:e] = _compute_gower_block(X_maj[s:e], X_maj, binary_set, ranges, p)
    del mm
    return path


def prepare_za_matrix_gower(
    df: pd.DataFrame,
    cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Build [binary|OHE(cat)|numeric] matrix, bin_indices, ranges for Gower."""
    bin_in = [c for c in bin_cols if c in cols]
    cat_in = [c for c in cat_cols if c in cols]
    num_in = [c for c in num_cols if c in cols]
    parts, n_bin = [], 0
    if bin_in:
        parts.append(df[bin_in].fillna(0).values.astype(np.float64))
        n_bin += len(bin_in)
    if cat_in:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        mat = ohe.fit_transform(df[cat_in].fillna("__MISSING__"))
        parts.append(mat.astype(np.float64))
        n_bin += mat.shape[1]
    if num_in:
        parts.append(df[num_in].fillna(0).values.astype(np.float64))
    X = np.hstack(parts) if parts else np.zeros((len(df), 0))
    bin_idx = list(range(n_bin))
    ranges = np.ones(X.shape[1])
    for j in range(n_bin, X.shape[1]):
        r = float(np.nanmax(X[:, j]) - np.nanmin(X[:, j]))
        ranges[j] = r if r > 0 else 1.0
    return X, bin_idx, ranges


def prepare_za_matrix_euclidean(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess for Euclidean (OHE cat, scale num, passthrough bin)."""
    from public.precompute_distances import get_preprocessor
    train = pd.concat([cases[cols], controls[cols]], ignore_index=True)
    prep = get_preprocessor(
        X=train,
        cat_cols=[c for c in cat_cols if c in cols],
        num_cols=[c for c in num_cols if c in cols],
        binary_cols=[c for c in bin_cols if c in cols],
        verbose=False,
    )
    X_maj = prep.fit_transform(controls[cols]).astype(np.float64)
    X_min = prep.transform(cases[cols]).astype(np.float64)
    return X_maj, X_min


def run_stageA_overlap_test(
    dnn_matrix_path: str,
    dnn_enrolids_path: str,
    control_enrolids: np.ndarray,
    case_enrolids: np.ndarray,
    pn_h5_path: str,
    N: int,
    M_pool: int,
) -> None:
    """Run Stage A twice with different seeds, report overlap@K for K in {N, 2N, 5N}."""
    from experiments_compare_random_vs_curation import sample_stageA_dispersed_controls
    id_to_pos = {int(e): i for i, e in enumerate(np.load(dnn_enrolids_path))}
    K_vals = [N, 2 * N, 5 * N]
    seeds = [42, 999]
    sets = []
    for seed in seeds:
        sel, _ = sample_stageA_dispersed_controls(
            control_enrolids,
            dnn_matrix_path,
            dnn_enrolids_path,
            pn_h5_path,
            case_enrolids,
            k=min(5 * N, M_pool, len(control_enrolids)),
            seed_method="random",
            seed=seed,
            M_pool=M_pool,
            use_kmeanspp=False,
            verbose=False,
        )
        sets.append(set(int(x) for x in sel))
    print("  [Stage A overlap test] seeds 42 vs 999:")
    for K in K_vals:
        a = set(list(sets[0])[:K])
        b = set(list(sets[1])[:K])
        overlap = len(a & b) / K if K else 0
        print(f"    overlap@{K}: {overlap:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--seed", type=int, default=TRAIN_TEST_SEED)
    p.add_argument("--za_preset", type=str, choices=ZA_PRESETS, default="ZA_v1_flags_plus_counts")
    p.add_argument("--metric", type=str, choices=["euclidean", "gower"], default="gower")
    p.add_argument("--compute_dnn", type=int, default=1)
    p.add_argument("--compute_pn", type=int, default=0)
    p.add_argument("--run_diagnostics", type=int, default=1)
    p.add_argument("--run_stageA_overlap_test", type=int, default=1)
    p.add_argument("--unique_threshold", type=int, default=200)
    p.add_argument("--top5_threshold", type=float, default=0.05)
    args = p.parse_args()

    # Use Gower by default for v1/v2 (they have continuous)
    if args.metric == "gower" or (args.za_preset != "ZA_v0_flags_only" and args.metric != "euclidean"):
        metric = "gower"
    else:
        metric = "euclidean"

    df = pd.read_parquet(args.parquet)
    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)
    train_ids, _, train_pd, _ = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.seed
    )
    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]

    BIN = get_bin_flag_columns(controls)
    CAT = get_cat_columns(controls)
    NUM = get_true_num_columns(controls, CAT, BIN)
    COST = get_cost_columns_2017(controls)
    UTIL = get_utilization_columns(controls)

    controls_df, za_cols = get_ZA_columns(controls, args.za_preset, BIN, CAT, COST, UTIL)
    cases_df, _ = get_ZA_columns(cases, args.za_preset, BIN, CAT, COST, UTIL)
    ok, bad = validate_no_2018_leakage(za_cols)
    if not ok:
        raise ValueError(f"Z_A 2018 leakage: {bad}")

    out_dir = os.path.join(os.getcwd(), f"precomputed_distances_msk_{args.za_preset}")
    os.makedirs(out_dir, exist_ok=True)
    dnn_dir = os.path.join(out_dir, f"global_dnn_seed_{args.seed}")
    os.makedirs(dnn_dir, exist_ok=True)
    maj_ids = controls["ENROLID"].values.astype(np.int64)
    min_ids = cases["ENROLID"].values.astype(np.int64)

    print(f"Z_A preset: {args.za_preset} | metric: {metric} | cols: {len(za_cols)}")

    if metric == "gower":
        X_maj, bin_idx, ranges = prepare_za_matrix_gower(controls_df, za_cols, CAT, NUM, BIN)
        X_min, _, _ = prepare_za_matrix_gower(cases_df, za_cols, CAT, NUM, BIN)
        binary_set = set(bin_idx)
        if args.compute_pn:
            pn_h5 = os.path.join(out_dir, "distances_majority_minority.h5")
            dist_pn = compute_gower_pn(X_maj, X_min, bin_idx, ranges, batch_size=1000)
            save_distances_hdf5(dist_pn, maj_ids, min_ids, pn_h5)
        if args.compute_dnn:
            precompute_gower_dnn_memmap(X_maj, bin_idx, ranges, dnn_dir, batch_size=750)
            np.save(os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy"), maj_ids)
    else:
        X_maj, X_min = prepare_za_matrix_euclidean(cases_df, controls_df, za_cols, CAT, NUM, BIN)
        if args.compute_pn:
            pn_h5 = os.path.join(out_dir, "distances_majority_minority.h5")
            dist_pn = compute_distances_batched(X_maj, X_min, batch_size=1000, metric="euclidean")
            save_distances_hdf5(dist_pn, maj_ids, min_ids, pn_h5)
        if args.compute_dnn:
            precompute_leaf_dnn_memmap(
                X_majority_leaf=X_maj,
                majority_enrolids_leaf=maj_ids,
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric="euclidean",
            )
            np.save(os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy"), maj_ids)

    if args.run_diagnostics:
        pn_h5 = os.path.join(out_dir, "distances_majority_minority.h5")
        dnn_mat = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
        run_tie_diagnostics(
            pn_h5_path=pn_h5 if os.path.exists(pn_h5) else None,
            dnn_matrix_path=dnn_mat if os.path.exists(dnn_mat) else None,
            unique_threshold=args.unique_threshold,
            top5_threshold=args.top5_threshold,
            show_top10=True,
        )

    if args.run_stageA_overlap_test:
        pn_h5 = os.path.join(out_dir, "distances_majority_minority.h5")
        dnn_mat = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
        dnn_ids = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")
        if not os.path.exists(pn_h5):
            print("  [WARN] PN H5 not found; skipping Stage A overlap test (needs PN for seed selection)")
        elif os.path.exists(dnn_mat):
            N = len(cases)
            M_pool = min(50000, len(controls) // 2)
            run_stageA_overlap_test(
                dnn_mat, dnn_ids,
                controls["ENROLID"].values.astype(np.int64),
                cases["ENROLID"].values.astype(np.int64),
                pn_h5, N, M_pool,
            )

    print("Done.")


if __name__ == "__main__":
    main()
    """
    python scripts/precompute_msk_ZA_variants.py --za_preset ZA_v1_flags_plus_counts --metric gower --compute_dnn 1 --run_diagnostics 1
    python scripts/precompute_msk_ZA_variants.py --za_preset ZA_v2_flags_plus_intensity_norm --metric gower --compute_dnn 1 --run_stageA_overlap_test 1
    """