#!/usr/bin/env python
"""
precompute_msk_distances_za_zb.py
=================================
Precompute Z_A (Stage A dispersion) and Z_B (Stage B matching) distance artifacts
for MSK two-stage sampling. Does not modify two_stage_kcenter_match.

Output folders (same schema as existing precomputed_distances_msk_*):
  precomputed_distances_msk_za_coarse_phenotype/
    distances_majority_minority.h5
    global_dnn_seed_123/leaf_global_dnn_matrix.npy
    global_dnn_seed_123/leaf_global_dnn_enrolids.npy

  precomputed_distances_msk_zb_intensity_context/
    distances_majority_minority.h5
    global_dnn_seed_123/leaf_global_dnn_matrix.npy
    global_dnn_seed_123/leaf_global_dnn_enrolids.npy

Usage:
  cd msk_analysis
  python scripts/precompute_msk_distances_za_zb.py [--parquet msk_2017_18_full.parquet] [--seed 123]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Path setup
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from public.model_IAI import (
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    train_test_split_enrol,
)
from msk_analysis.scripts.msk_feature_groups import (
    get_cost_columns_2017,
    get_utilization_columns,
    get_za_coarse_phenotype_columns,
    get_zb_intensity_context_columns,
    validate_no_2018_leakage,
)
from public.precompute_distances import (
    compute_distances_batched,
    precompute_leaf_dnn_memmap,
    save_distances_hdf5,
)
from pyspark.sql import SparkSession

TRAIN_TEST_SEED = 123

# Gower distance helpers (mixed binary/categorical/continuous)
def _compute_gower_distances(
    X_A: np.ndarray,
    X_B: np.ndarray,
    binary_col_indices: set,
    ranges: np.ndarray,
    p: int,
    batch_size: int = 1000,
) -> np.ndarray:
    """Compute Gower distances. binary_col_indices: set of column indices."""
    n_A, n_B = X_A.shape[0], X_B.shape[0]
    distances = np.zeros((n_A, n_B), dtype=np.float32)
    n_batches = (n_A + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n_A)
        block = np.zeros((e - s, n_B), dtype=np.float32)
        for j in range(p):
            a_vals = X_A[s:e, j]
            b_vals = X_B[:, j]
            if j in binary_col_indices:
                # δ = 1[a≠b]
                diff = (a_vals[:, None] != b_vals[None, :]).astype(np.float32)
            else:
                # continuous: δ = |a-b|/range_j
                rj = ranges[j]
                diff = np.abs(
                    a_vals[:, None].astype(np.float64) - b_vals[None, :].astype(np.float64)
                ) / rj
                diff = diff.astype(np.float32)
            block += diff
        distances[s:e, :] = block / p
    return distances


def compute_gower_distances_batched(
    X_majority: np.ndarray,
    X_minority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    batch_size: int = 1000,
) -> np.ndarray:
    """P-N distances with Gower."""
    p = X_majority.shape[1]
    binary_set = set(bin_col_indices)
    return _compute_gower_distances(
        X_majority, X_minority, binary_set, ranges, p, batch_size
    )


def precompute_gower_dnn_memmap(
    X_majority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    out_dir: str,
    batch_size: int = 750,
) -> Tuple[str, str]:
    """D-N-N Gower distances. Returns (dnn_matrix_path, dnn_enrolids_path)."""
    os.makedirs(out_dir, exist_ok=True)
    n, p = X_majority.shape[0], X_majority.shape[1]
    binary_set = set(bin_col_indices)
    dnn_matrix_path = os.path.join(out_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_path = os.path.join(out_dir, "leaf_global_dnn_enrolids.npy")
    dnn_mm = np.lib.format.open_memmap(
        dnn_matrix_path, mode="w+", dtype=np.float32, shape=(n, n)
    )
    from tqdm import tqdm
    n_batches = (n + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="D-N-N Gower"):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        block = _compute_gower_distances(
            X_majority[s:e], X_majority, binary_set, ranges, p, batch_size=n
        )
        dnn_mm[s:e, :] = block
    del dnn_mm
    return dnn_matrix_path, dnn_enrolids_path


def run_tie_degeneracy_diagnostics(pn_h5_path: str, sample_size: int = 50000) -> None:
    """
    Sample distances from PN H5 and print:
      - number of unique values
      - fraction of top-5 most frequent distances
    Detects tie-degeneracy (e.g., sqrt(k) for binary Euclidean).
    """
    with h5py.File(pn_h5_path, "r") as f:
        d = f["distances"]
        n_maj, n_min = d.shape[0], d.shape[1]
        rng = np.random.default_rng(42)
        n_sample = min(sample_size, n_maj * n_min)
        idx_maj = rng.integers(0, n_maj, size=n_sample)
        idx_min = rng.integers(0, n_min, size=n_sample)
        # h5py requires fancy indices in increasing order; read row-by-row
        sample = np.empty(n_sample, dtype=np.float32)
        for row in np.unique(idx_maj):
            mask = idx_maj == row
            cols = idx_min[mask]
            row_data = d[row, :]  # full row slice (h5py allows this)
            sample[mask] = row_data[cols]
    uniq, counts = np.unique(sample.ravel(), return_counts=True)
    n_unique = len(uniq)
    total = counts.sum()
    top5_frac = counts[np.argsort(-counts)[:5]].sum() / total
    print(f"  [Tie diagnostics] unique values: {n_unique:,} | top-5 freq frac: {top5_frac:.4f}")
    if n_unique < 20:
        print(f"  [WARN] Very few unique distances ({n_unique}) - possible tie degeneracy")
    if top5_frac > 0.8:
        print(f"  [WARN] Top-5 distances account for {top5_frac:.1%} - possible tie degeneracy")


def run_tie_degeneracy_diagnostics_dnn(dnn_matrix_path: str, sample_size: int = 50000) -> None:
    """
    Sample control-control distances from D-N-N matrix and compute:
      - unique values, top-5 freq frac
    If top-5 freq is small and uniques are reasonably large, Stage A is fine.
    """
    dnn = np.load(dnn_matrix_path, mmap_mode="r")
    n = dnn.shape[0]
    rng = np.random.default_rng(42)
    n_sample = min(sample_size, n * n)
    idx_i = rng.integers(0, n, size=n_sample)
    idx_j = rng.integers(0, n, size=n_sample)
    sample = dnn[idx_i, idx_j]
    del dnn
    uniq, counts = np.unique(sample.ravel(), return_counts=True)
    n_unique = len(uniq)
    total = counts.sum()
    top5_frac = counts[np.argsort(-counts)[:5]].sum() / total
    print(f"  [D-N-N tie diagnostics] unique values: {n_unique:,} | top-5 freq frac: {top5_frac:.4f}")
    if n_unique < 20:
        print(f"  [WARN] Very few unique D-N-N distances ({n_unique}) - possible tie degeneracy")
    if top5_frac > 0.8:
        print(f"  [WARN] Top-5 D-N-N distances account for {top5_frac:.1%} - possible tie degeneracy")


def prepare_za_features(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Z_A: coarse phenotype. Use preprocessor (OHE cat, scale num, passthrough bin).
    Euclidean is fine in this space. Returns (X_majority, X_minority, feature_order).
    """
    from public.precompute_distances import get_preprocessor
    train = pd.concat([cases[feature_cols], controls[feature_cols]], ignore_index=True)
    preprocessor = get_preprocessor(
        X=train,
        cat_cols=[c for c in cat_cols if c in feature_cols],
        num_cols=[c for c in num_cols if c in feature_cols],
        binary_cols=[c for c in bin_cols if c in feature_cols],
        verbose=False,
    )
    X_min = preprocessor.fit_transform(cases[feature_cols])
    X_maj = preprocessor.transform(controls[feature_cols])
    # feature_order after transform (simplified - we don't need names for distances)
    return X_maj.astype(np.float64), X_min.astype(np.float64), feature_cols


def prepare_zb_features_gower(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
    """
    Z_B: intensity + context. Build matrix: [binary | OHE(cat) | numeric].
    binary indices and ranges for Gower. Returns (X_maj, X_min, bin_col_indices, ranges).
    """
    bin_in = [c for c in bin_cols if c in feature_cols]
    cat_in = [c for c in cat_cols if c in feature_cols]
    num_in = [c for c in num_cols if c in feature_cols]
    parts_min, parts_maj = [], []
    n_bin = 0
    if bin_in:
        parts_min.append(cases[bin_in].fillna(0).values.astype(np.float64))
        parts_maj.append(controls[bin_in].fillna(0).values.astype(np.float64))
        n_bin += len(bin_in)
    if cat_in:
        all_cat = pd.concat([cases[cat_in], controls[cat_in]], ignore_index=True)
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_mat = ohe.fit_transform(all_cat.fillna("__MISSING__"))
        parts_min.append(ohe_mat[: len(cases)].astype(np.float64))
        parts_maj.append(ohe_mat[len(cases) :].astype(np.float64))
        n_bin += ohe_mat.shape[1]
    if num_in:
        parts_min.append(cases[num_in].fillna(0).values.astype(np.float64))
        parts_maj.append(controls[num_in].fillna(0).values.astype(np.float64))
    X_min = np.hstack(parts_min) if parts_min else np.zeros((len(cases), 0))
    X_maj = np.hstack(parts_maj) if parts_maj else np.zeros((len(controls), 0))
    bin_col_indices = list(range(n_bin))
    # Ranges for continuous columns
    ranges = np.ones(X_min.shape[1])
    for j in range(n_bin, X_min.shape[1]):
        col = np.concatenate([X_min[:, j], X_maj[:, j]])
        r = float(np.nanmax(col) - np.nanmin(col))
        ranges[j] = r if r > 0 else 1.0
    return X_maj, X_min, bin_col_indices, ranges


def run_za_coarse_phenotype(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    out_dir: str,
    seed: int,
) -> None:
    """Z_A: coarse phenotype. Euclidean. Stage A dispersion."""
    os.makedirs(out_dir, exist_ok=True)
    BIN = get_bin_flag_columns(cases)
    CAT = get_cat_columns(cases)
    NUM = get_true_num_columns(cases, CAT, BIN)
    COST = get_cost_columns_2017(cases)
    UTIL = get_utilization_columns(cases)
    za_cols = get_za_coarse_phenotype_columns(cases, BIN, CAT, COST, UTIL)
    ok, bad = validate_no_2018_leakage(za_cols)
    if not ok:
        raise ValueError(f"Z_A has 2018 leakage: {bad}")
    if not za_cols:
        raise ValueError("Z_A feature list is empty")
    print(f"  Z_A columns: {len(za_cols)}")
    X_maj, X_min, _ = prepare_za_features(cases, controls, za_cols, CAT, NUM, BIN)
    pn_h5 = os.path.join(out_dir, "distances_majority_minority.h5")
    dnn_dir = os.path.join(out_dir, f"global_dnn_seed_{seed}")
    maj_ids = controls["ENROLID"].values.astype(np.int64)
    min_ids = cases["ENROLID"].values.astype(np.int64)
    if not os.path.exists(pn_h5):
        print("  Computing PN (Euclidean)...")
        dist_pn = compute_distances_batched(
            X_maj, X_min, batch_size=1000, dtype=np.float32, metric="euclidean"
        )
        save_distances_hdf5(dist_pn, maj_ids, min_ids, pn_h5)
    else:
        print(f"  Found PN: {pn_h5}")
    run_tie_degeneracy_diagnostics(pn_h5)
    dnn_mat = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
    if not os.path.exists(dnn_mat):
        print("  Computing D-N-N (Euclidean)...")
        precompute_leaf_dnn_memmap(
            X_majority_leaf=X_maj,
            majority_enrolids_leaf=maj_ids,
            out_dir=dnn_dir,
            leaf_id="global",
            batch_size=750,
            metric="euclidean",
        )
        np.save(os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy"), maj_ids)
    else:
        print(f"  Found D-N-N: {dnn_dir}")
    run_tie_degeneracy_diagnostics_dnn(dnn_mat)


def run_zb_intensity_context(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    out_dir: str,
    seed: int,
) -> None:
    """Z_B: intensity + context. Gower. Stage B matching."""
    os.makedirs(out_dir, exist_ok=True)
    BIN = get_bin_flag_columns(cases)
    CAT = get_cat_columns(cases)
    NUM = get_true_num_columns(cases, CAT, BIN)
    COST = get_cost_columns_2017(cases)
    UTIL = get_utilization_columns(cases)
    zb_cols = get_zb_intensity_context_columns(cases, CAT, COST, UTIL, BIN)
    ok, bad = validate_no_2018_leakage(zb_cols)
    if not ok:
        raise ValueError(f"Z_B has 2018 leakage: {bad}")
    if not zb_cols:
        raise ValueError("Z_B feature list is empty")
    print(f"  Z_B columns: {len(zb_cols)}")
    X_maj, X_min, bin_idx, ranges = prepare_zb_features_gower(
        cases, controls, zb_cols, CAT, NUM, BIN
    )
    pn_h5 = os.path.join(out_dir, "distances_majority_minority.h5")
    dnn_dir = os.path.join(out_dir, f"global_dnn_seed_{seed}")
    maj_ids = controls["ENROLID"].values.astype(np.int64)
    min_ids = cases["ENROLID"].values.astype(np.int64)
    if not os.path.exists(pn_h5):
        print("  Computing PN (Gower)...")
        dist_pn = compute_gower_distances_batched(
            X_maj, X_min, bin_idx, ranges, batch_size=1000
        )
        save_distances_hdf5(dist_pn, maj_ids, min_ids, pn_h5)
    else:
        print(f"  Found PN: {pn_h5}")
    run_tie_degeneracy_diagnostics(pn_h5)
    dnn_mat = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
    if not os.path.exists(dnn_mat):
        print("  Computing D-N-N (Gower)...")
        precompute_gower_dnn_memmap(X_maj, bin_idx, ranges, dnn_dir, batch_size=750)
        np.save(os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy"), maj_ids)
    else:
        print(f"  Found D-N-N: {dnn_dir}")


def main():
    p = argparse.ArgumentParser(description="Precompute Z_A and Z_B distances for MSK")
    p.add_argument("--parquet", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--seed", type=int, default=TRAIN_TEST_SEED)
    p.add_argument("--za_only", action="store_true", help="Only Z_A")
    p.add_argument("--zb_only", action="store_true", help="Only Z_B")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if files exist")
    args = p.parse_args()
    if args.overwrite:
        # Will be implemented by deleting/skipping exists checks in run_* - for now we skip overwrite
        pass
    spark = SparkSession.builder.appName("PrecomputeZAZB").getOrCreate()
    df = spark.read.format("parquet").load(args.parquet).toPandas()
    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)
    train_ids, _, train_pd, _ = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.seed
    )
    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]
    print(f"Train cases: {len(cases):,}  controls: {len(controls):,}")
    cwd = os.getcwd()
    if not args.zb_only:
        out_za = os.path.join(cwd, "precomputed_distances_msk_za_coarse_phenotype")
        print(f"\n--- Z_A (coarse phenotype) -> {out_za} ---")
        run_za_coarse_phenotype(cases, controls, out_za, args.seed)
    if not args.za_only:
        out_zb = os.path.join(cwd, "precomputed_distances_msk_zb_intensity_context")
        print(f"\n--- Z_B (intensity + context) -> {out_zb} ---")
        run_zb_intensity_context(cases, controls, out_zb, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
