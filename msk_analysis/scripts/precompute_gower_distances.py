#!/usr/bin/env python
"""
precompute_gower_distances.py
=============================
Standalone precomputation of pairwise Gower distances (P-N and D-N-N) using a
fast v2 kernel with numexpr, float32, binary validation, and tolerance-based
binary comparison.

Outputs (compatible with exp6 / two_stage / experiments_compare_random_vs_curation):
  - P-N: distances_majority_minority_gower.h5 (controls x cases)
  - D-N-N: leaf_global_dnn_matrix.npy (or .h5) + leaf_global_dnn_enrolids.npy

Usage:
  cd msk_analysis
  python scripts/precompute_gower_distances.py --parquet msk_2017_18_full.parquet --outdir ./precomputed_distances_gower
  python scripts/precompute_gower_distances.py --feature_set all_cost --use_hdf5_dnn --dnn_batch_size 750
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

# Path setup (match exp6)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

try:
    import numexpr as ne
except ImportError:
    ne = None  # type: ignore[assignment]

# Optional: use public package for save_distances_hdf5 and model_IAI
try:
    from public.model_IAI import (
        get_bin_flag_columns,
        get_cat_columns,
        get_true_num_columns,
        train_test_split_enrol,
    )
    from public.precompute_distances import save_distances_hdf5
except ImportError:
    # Fallback: define minimal save and use local feature helpers if needed
    save_distances_hdf5 = None
    get_bin_flag_columns = get_cat_columns = get_true_num_columns = train_test_split_enrol = None

TRAIN_TEST_SEED = 123
_GOWER_BINARY_ATOL = 1e-6
_GOWER_BINARY_RTOL = 1e-5


# -------------------------------------------------------------------------------
# Binary column validation
# -------------------------------------------------------------------------------
def _validate_gower_binary_columns(
    X_A: np.ndarray,
    X_B: np.ndarray,
    binary_col_indices: np.ndarray,
    atol: float = _GOWER_BINARY_ATOL,
    rtol: float = _GOWER_BINARY_RTOL,
) -> None:
    """Raise ValueError if any column marked as binary contains values not in {0, 1} (within tolerance)."""
    for c in binary_col_indices:
        for name, X in [("A", X_A), ("B", X_B)]:
            col = np.asarray(X[:, c], dtype=np.float64).ravel()
            if np.any(np.isnan(col)):
                uniq = np.unique(col[~np.isnan(col)])
                raise ValueError(
                    f"Gower binary column #{c} in X_{name} contains NaN. "
                    f"Unique non-NaN values (sample): {uniq[:20].tolist()}"
                )
            close_to_0 = np.abs(col) <= atol + rtol * np.maximum(np.abs(col), 1e-12)
            close_to_1 = np.abs(col - 1.0) <= atol + rtol * np.maximum(np.abs(col), 1e-12)
            if not np.all(close_to_0 | close_to_1):
                uniq = np.unique(col)
                if len(uniq) > 10:
                    uniq_repr = f"{uniq[:5].tolist()} ... {uniq[-3:].tolist()} (n_unique={len(uniq)})"
                else:
                    uniq_repr = uniq.tolist()
                raise ValueError(
                    f"Gower binary column #{c} in X_{name} is not binary (values must be 0 or 1). "
                    f"Unique values: {uniq_repr}"
                )


# -------------------------------------------------------------------------------
# _compute_gower_distances_v2 — fast kernel (pre-normalized continuous + numexpr)
# -------------------------------------------------------------------------------
def _compute_gower_distances_v2(
    X_A: np.ndarray,
    X_B: np.ndarray,
    binary_col_indices: np.ndarray,
    continuous_col_indices: np.ndarray,
    ranges: np.ndarray,
    binary_atol: float = _GOWER_BINARY_ATOL,
    binary_rtol: float = _GOWER_BINARY_RTOL,
    use_tolerance_binary: bool = True,
) -> np.ndarray:
    """
    Compute Gower distances between X_A (n_A x p) and X_B (n_B x p).
    Returns (n_A, n_B) float32.
    Uses pre-normalized continuous columns and numexpr; ~2x faster than batched per-row loop.
    """
    if ne is None:
        raise ImportError("Gower v2 requires 'numexpr'. Install with: pip install numexpr")

    _validate_gower_binary_columns(X_A, X_B, binary_col_indices, atol=binary_atol, rtol=binary_rtol)

    X_A = np.asarray(X_A, dtype=np.float32)
    X_B = np.asarray(X_B, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.float32)

    p = X_A.shape[1]
    n_A, n_B = X_A.shape[0], X_B.shape[0]
    distances = np.zeros((n_A, n_B), dtype=np.float32)

    # --- continuous columns: pre-normalize by range, then add |a - b| per column ---
    if len(continuous_col_indices) > 0:
        r = ranges[continuous_col_indices].copy()
        r[r == 0] = 1.0
        A_cont = X_A[:, continuous_col_indices] / r  # (n_A, n_cont)
        B_cont = X_B[:, continuous_col_indices] / r  # (n_B, n_cont)
        for k in range(A_cont.shape[1]):
            a = A_cont[:, k][:, None]
            b = B_cont[:, k][None, :]
            distances += ne.evaluate("abs(a - b)").astype(np.float32)

    # --- binary columns: δ = 1[a≠b], with optional tolerance ---
    if len(binary_col_indices) > 0:
        A_bin = X_A[:, binary_col_indices]
        B_bin = X_B[:, binary_col_indices]
        for k in range(A_bin.shape[1]):
            a = A_bin[:, k][:, None]
            b = B_bin[:, k][None, :]
            if use_tolerance_binary:
                diff = ne.evaluate(
                    "where(abs(a - b) <= atol + rtol * abs(b), 0.0, 1.0)",
                    local_dict={"a": a, "b": b, "atol": binary_atol, "rtol": binary_rtol},
                ).astype(np.float32)
            else:
                diff = ne.evaluate("a != b").astype(np.float32)
            distances += diff

    return distances / p


def _save_distances_hdf5_inline(
    distances: np.ndarray,
    majority_enrolids: np.ndarray,
    minority_enrolids: np.ndarray,
    path: str,
) -> None:
    """Write P-N distances to HDF5 (compatible with load_pn_hdf5)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("distances", data=distances.astype(np.float32), dtype=np.float32)
        f.create_dataset("majority_enrolids", data=np.asarray(majority_enrolids, dtype=np.int64))
        f.create_dataset("minority_enrolids", data=np.asarray(minority_enrolids, dtype=np.int64))


# -------------------------------------------------------------------------------
# Feature matrix construction (same logic as exp6 Gower block)
# -------------------------------------------------------------------------------
def build_gower_feature_matrices(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Build X_minority (cases), X_majority (controls), ranges, bin_col_indices.
    All in float32. Order: binary | OHE(cat) | numeric.
    """
    bin_in_feature = [c for c in bin_cols if c in feature_cols]
    num_in_feature = [c for c in num_cols if c in feature_cols]
    cat_in_feature = [c for c in cat_cols if c in feature_cols]
    parts_min, parts_maj = [], []
    n_bin = 0
    if bin_in_feature:
        parts_min.append(cases[bin_in_feature].values.astype(np.float32))
        parts_maj.append(controls[bin_in_feature].values.astype(np.float32))
        n_bin += len(bin_in_feature)
    if cat_in_feature:
        all_cat = pd.concat([cases[cat_in_feature], controls[cat_in_feature]], ignore_index=True)
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_mat = ohe.fit_transform(all_cat)
        parts_min.append(ohe_mat[: len(cases)].astype(np.float32))
        parts_maj.append(ohe_mat[len(cases) :].astype(np.float32))
        n_bin += ohe_mat.shape[1]
    if num_in_feature:
        parts_min.append(cases[num_in_feature].values.astype(np.float32))
        parts_maj.append(controls[num_in_feature].values.astype(np.float32))
    X_minority = np.hstack(parts_min) if parts_min else np.zeros((len(cases), 0), dtype=np.float32)
    X_majority = np.hstack(parts_maj) if parts_maj else np.zeros((len(controls), 0), dtype=np.float32)
    ranges = np.ones(X_minority.shape[1], dtype=np.float32)
    for j in range(n_bin, X_minority.shape[1]):
        col = np.concatenate([X_minority[:, j], X_majority[:, j]])
        r = float(np.nanmax(col) - np.nanmin(col))
        ranges[j] = np.float32(r if r > 0 else 1.0)
    bin_col_indices = list(range(n_bin))
    return X_majority, X_minority, ranges, bin_col_indices


# -------------------------------------------------------------------------------
# P-N and D-N-N precomputation (batched for D-N-N to limit memory)
# -------------------------------------------------------------------------------
def compute_gower_pn_v2(
    X_majority: np.ndarray,
    X_minority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
) -> np.ndarray:
    """P-N Gower distances (controls x cases) with v2 kernel. Full matrix in memory."""
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_arr = np.array(bin_col_indices)
    return _compute_gower_distances_v2(
        X_majority, X_minority, bin_arr, cont_col_indices, ranges
    )


def precompute_gower_dnn_v2(
    X_majority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    out_dir: str,
    batch_size: int = 750,
    use_hdf5: bool = False,
    compression: str = "gzip",
    compression_opts: int = 9,
) -> Tuple[str, str]:
    """
    D-N-N (control-control) Gower with v2 kernel, batched to avoid huge allocation.
    Returns (dnn_matrix_path, dnn_enrolids_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    n = X_majority.shape[0]
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_arr = np.array(bin_col_indices)

    dnn_matrix_path = os.path.join(out_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_path = os.path.join(out_dir, "leaf_global_dnn_enrolids.npy")

    if use_hdf5:
        dnn_matrix_path = os.path.join(out_dir, "leaf_global_dnn_matrix.h5")
        chunk_rows = min(batch_size, n)
        with h5py.File(dnn_matrix_path, "w") as f:
            dset = f.create_dataset(
                "distances",
                shape=(n, n),
                dtype=np.float32,
                chunks=(chunk_rows, n),
                compression=compression,
                compression_opts=compression_opts,
            )
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                block = _compute_gower_distances_v2(
                    X_majority[s:e], X_majority, bin_arr, cont_col_indices, ranges
                )
                dset[s:e, :] = block
        return dnn_matrix_path, dnn_enrolids_path

    dnn_mm = np.lib.format.open_memmap(
        dnn_matrix_path, mode="w+", dtype=np.float32, shape=(n, n)
    )
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        block = _compute_gower_distances_v2(
            X_majority[s:e], X_majority, bin_arr, cont_col_indices, ranges
        )
        dnn_mm[s:e, :] = block
    del dnn_mm
    return dnn_matrix_path, dnn_enrolids_path


# -------------------------------------------------------------------------------
# Feature set (match exp6)
# -------------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame, feature_set: str, target_col: str) -> List[str]:
    """Return feature column list for medical_only | all_cost | less_cost."""
    BIN = get_bin_flag_columns(df)
    CAT = get_cat_columns(df)
    TRUE_NUM = get_true_num_columns(df, CAT, BIN)
    COST_COLUMNS = [
        c for c in df.columns
        if ("cost" in c.lower() or "quarterly" in c.lower() or "increasing" in c.lower()
            or "decreasing" in c.lower() or "skewness" in c.lower() or "kurtosis" in c.lower()
            or "cv" in c.lower() or "range" in c.lower())
        and "2018" not in c
    ]
    AUXILIARY_COST_COLUMNS = [c for c in df.columns if c.startswith("comorbidity_only") or c.startswith("msk_procedure")]
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    if feature_set == "medical_only":
        exclude_cols = exclude_cols + COST_COLUMNS
    elif feature_set == "less_cost":
        exclude_cols = exclude_cols + AUXILIARY_COST_COLUMNS
    return [c for c in df.columns if c not in exclude_cols]


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute Gower P-N and D-N-N distances (v2 kernel)")
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--outdir", type=str, default="./precomputed_distances_gower")
    p.add_argument("--feature_set", type=str, default="all_cost", choices=["medical_only", "all_cost", "less_cost"])
    p.add_argument("--seed", type=int, default=TRAIN_TEST_SEED)
    p.add_argument("--dnn_batch_size", type=int, default=750)
    p.add_argument("--use_hdf5_dnn", action="store_true", help="Save D-N-N as HDF5 (compressed)")
    p.add_argument("--skip_pn", action="store_true", help="Skip P-N computation")
    p.add_argument("--skip_dnn", action="store_true", help="Skip D-N-N computation")
    p.add_argument("--resume", action="store_true", help="Skip if output files already exist")
    args = p.parse_args()

    if get_bin_flag_columns is None or train_test_split_enrol is None:
        raise ImportError("Run from msk_analysis with public on PYTHONPATH (public.model_IAI, public.precompute_distances)")

    outdir = args.outdir
    dnn_dir = os.path.join(outdir, f"global_dnn_seed_{args.seed}_gower")
    pn_h5 = os.path.join(outdir, "distances_majority_minority_gower.h5")
    dnn_matrix_path = os.path.join(dnn_dir, "leaf_global_dnn_matrix.h5" if args.use_hdf5_dnn else "leaf_global_dnn_matrix.npy")
    dnn_enrolids_path = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(dnn_dir, exist_ok=True)

    target_col = "top_2_pct_cost_2018"
    df = pd.read_parquet(args.parquet_path)
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    train_ids, _, train_pd, _ = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.seed
    )
    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]
    BIN = get_bin_flag_columns(df)
    CAT = get_cat_columns(df)
    NUM = get_true_num_columns(df, CAT, BIN)
    feature_cols = get_feature_columns(df, args.feature_set, target_col)

    X_majority, X_minority, ranges, bin_col_indices = build_gower_feature_matrices(
        cases, controls, feature_cols, CAT, NUM, BIN
    )
    n_controls, n_cases = X_majority.shape[0], X_minority.shape[0]
    print(f"Features: {X_majority.shape[1]} (binary: {len(bin_col_indices)})")
    print(f"Controls: {n_controls:,}  Cases: {n_cases:,}")

    majority_enrolids = controls["ENROLID"].values.astype(np.int64)
    minority_enrolids = cases["ENROLID"].values.astype(np.int64)

    # P-N
    if not args.skip_pn:
        if args.resume and os.path.exists(pn_h5):
            print(f"  P-N already exists: {pn_h5}")
        else:
            print("  Computing P-N (Gower v2)...")
            dist_pn = compute_gower_pn_v2(X_majority, X_minority, bin_col_indices, ranges)
            if save_distances_hdf5 is not None:
                save_distances_hdf5(dist_pn, majority_enrolids, minority_enrolids, pn_h5)
            else:
                _save_distances_hdf5_inline(dist_pn, majority_enrolids, minority_enrolids, pn_h5)
            print(f"  Saved P-N: {pn_h5}")

    # D-N-N
    if not args.skip_dnn:
        if args.resume and os.path.exists(dnn_matrix_path) and os.path.exists(dnn_enrolids_path):
            print(f"  D-N-N already exists: {dnn_matrix_path}")
        else:
            print("  Computing D-N-N (Gower v2, batched)...")
            precompute_gower_dnn_v2(
                X_majority, bin_col_indices, ranges, dnn_dir,
                batch_size=args.dnn_batch_size,
                use_hdf5=args.use_hdf5_dnn,
            )
            np.save(dnn_enrolids_path, majority_enrolids)
            print(f"  Saved D-N-N: {dnn_matrix_path}")

    print("Done.")


if __name__ == "__main__":
    main()
