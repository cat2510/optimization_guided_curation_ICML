#!/usr/bin/env python
"""
precompute_gower_distances.py
=============================
Precomputation of pairwise Gower distances (P-N and D-N-N matrices) using a
fast kernel with numexpr, float16/float32 (no float8 in NumPy), binary validation, and tolerance-based
binary comparison. (see Omid's test.py sent 2026-03-16, v2 kernel)

Outputs (compatible with exp6 / two_stage / experiments_compare_random_vs_curation):
  - P-N: distances_majority_minority_gower.h5 (controls x cases)
  - D-N-N: leaf_global_dnn_matrix.npy or .npz + leaf_global_dnn_enrolids.npy

Library API: ``precompute_gower_pn_and_dnn`` — MSK (or other) drivers live outside
``public/`` (e.g. ``msk_analysis/scripts/precompute_msk_gower_distances.py``).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Collection, List, Optional, Tuple

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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    from public.model_IAI import (
        GOWER_BINARY_ATOL,
        GOWER_BINARY_RTOL,
        get_bin_flag_columns,
        get_bin_flag_columns_with_provenance,
        get_cat_columns,
        get_true_num_columns,
        is_binary_01_series,
        train_test_split_enrol,
    )
    from public.precompute_distances import save_distances_hdf5
except ImportError:
    save_distances_hdf5 = None
    get_bin_flag_columns = get_bin_flag_columns_with_provenance = None
    get_cat_columns = get_true_num_columns = train_test_split_enrol = None
    is_binary_01_series = None  # type: ignore[assignment]
    GOWER_BINARY_ATOL = 1e-6
    GOWER_BINARY_RTOL = 1e-5

TRAIN_TEST_SEED = 123

from public.dnn_matrix_storage import (
    dnn_enrolids_npy_path,
    dnn_matrix_npy_path,
    dnn_matrix_path,
    dnn_matrix_storage_exists,
    ensure_dnn_matrix_npy,
)

# NumPy has no float8. Supported storage/compute dtypes for distances & features:
GOWER_DISTANCE_DTYPES = (np.float32, np.float16)


def _as_gower_dtype(name_or_dtype) -> np.dtype:
    """Resolve float32 | float16 (default float16). float8 is not available in NumPy."""
    if isinstance(name_or_dtype, np.dtype):
        dt = name_or_dtype
    else:
        s = str(name_or_dtype).lower().replace("float", "")
        if s in ("32", ""):
            dt = np.dtype(np.float32)
        elif s == "16":
            dt = np.dtype(np.float16)
        else:
            dt = np.dtype(name_or_dtype)
    if dt.type not in GOWER_DISTANCE_DTYPES:
        raise ValueError(f"Gower distance dtype must be float32 or float16, got {dt}")
    return dt


# -------------------------------------------------------------------------------
# Binary column validation
# -------------------------------------------------------------------------------
def _validate_gower_binary_columns(
    X_A: np.ndarray,
    X_B: np.ndarray,
    binary_col_indices: np.ndarray,
    atol: float = GOWER_BINARY_ATOL,
    rtol: float = GOWER_BINARY_RTOL,
    col_names: Optional[List[str]] = None,
) -> None:
    """Raise ValueError if any column marked as binary contains values not in {0, 1} (within tolerance).
    Important difference from binary_01 check, this checks for after OHE of categorical columns."""
    for c in binary_col_indices:
        for mat_name, X in [("A", X_A), ("B", X_B)]:
            col = np.asarray(X[:, c], dtype=np.float64).ravel()
            feat_name = col_names[c] if col_names is not None and c < len(col_names) else f"col_{c}"
            if np.any(np.isnan(col)):
                uniq = np.unique(col[~np.isnan(col)])
                raise ValueError(
                    f"Gower binary column #{c} ({feat_name!r}) in X_{mat_name} contains NaN. "
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
                    f"Gower binary column #{c} ({feat_name!r}) in X_{mat_name} is not binary (values must be 0 or 1). "
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
    binary_atol: float = GOWER_BINARY_ATOL,
    binary_rtol: float = GOWER_BINARY_RTOL,
    use_tolerance_binary: bool = True,
    col_names: Optional[List[str]] = None,
    out_dtype: np.dtype | type = np.float16,
) -> np.ndarray:
    """
    Compute Gower distances between X_A (n_A x p) and X_B (n_B x p).
    Returns (n_A, n_B) with dtype out_dtype (float16 or float32).
    Uses pre-normalized continuous columns and numexpr (intermediate often float32).
    """
    if ne is None:
        raise ImportError("Gower v2 requires 'numexpr'. Install with: pip install numexpr")

    dt = _as_gower_dtype(out_dtype)

    _validate_gower_binary_columns(
        X_A, X_B, binary_col_indices, atol=binary_atol, rtol=binary_rtol, col_names=col_names
    )

    X_A = np.asarray(X_A, dtype=dt)
    X_B = np.asarray(X_B, dtype=dt)
    ranges = np.asarray(ranges, dtype=dt)
    p = float(X_A.shape[1])

    n_A, n_B = X_A.shape[0], X_B.shape[0]
    distances = np.zeros((n_A, n_B), dtype=dt)

    # --- continuous columns: pre-normalize by range, then add |a - b| per column ---
    if len(continuous_col_indices) > 0:
        r = ranges[continuous_col_indices].copy()
        r[r == 0] = np.asarray(1.0, dtype=dt)
        A_cont = X_A[:, continuous_col_indices] / r  # (n_A, n_cont)
        B_cont = X_B[:, continuous_col_indices] / r  # (n_B, n_cont)
        for k in range(A_cont.shape[1]):
            a = A_cont[:, k][:, None]
            b = B_cont[:, k][None, :]
            # numexpr promotes float16 inputs to float32; cast back for accumulation
            distances += ne.evaluate("abs(a - b)").astype(dt)

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
                ).astype(dt)
            else:
                diff = ne.evaluate("a != b").astype(dt)
            distances += diff

    return (distances / np.asarray(p, dtype=dt)).astype(dt)


def _save_distances_hdf5_inline(
    distances: np.ndarray,
    majority_enrolids: np.ndarray,
    minority_enrolids: np.ndarray,
    path: str,
    distances_dtype: np.dtype | type = np.float16,
) -> None:
    """Write P-N distances to HDF5 (compatible with load_pn_hdf5)."""
    dt = _as_gower_dtype(distances_dtype)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        d = np.asarray(distances, dtype=dt)
        f.create_dataset("distances", data=d, dtype=dt)
        f.create_dataset("majority_enrolids", data=np.asarray(majority_enrolids, dtype=np.int64))
        f.create_dataset("minority_enrolids", data=np.asarray(minority_enrolids, dtype=np.int64))


def build_gower_feature_matrices(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    feature_dtype: np.dtype | type = np.float16,
    bin_cols_verified_by_values: Optional[Collection[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Build X_minority (cases), X_majority (controls), ranges, bin_col_indices, col_names.
    All in feature_dtype (default float16). Order: binary | OHE(cat) | numeric.
    Numeric columns are min–max scaled in float32 (avoids float16 overflow on costs, etc.),
    then stored; their Gower range is 1 (kernel sees normalized [0,1]-ish values).
    Columns in bin_cols that are not 0/1 (Gower-tolerant) on cases∪controls are demoted to continuous.

    Pass bin_cols_verified_by_values from get_bin_flag_columns_with_provenance(train_features_df)[1]
    to skip re-scanning columns already confirmed strict-binary on that frame (same pipeline as OCT).
    """
    bin_in_feature = [c for c in bin_cols if c in feature_cols]
    num_in_feature = [c for c in num_cols if c in feature_cols]
    cat_in_feature = [c for c in cat_cols if c in feature_cols]
    verified = bin_cols_verified_by_values
    real_binary: List[str] = []
    demoted_to_num: List[str] = []
    for c in bin_in_feature:
        if verified is not None and c in verified:
            real_binary.append(c)
            continue
        comb = pd.Series(
            np.concatenate([cases[c].values.ravel(), controls[c].values.ravel()])
        )
        gower_ok = bool(
            is_binary_01_series(
                comb, atol=GOWER_BINARY_ATOL, rtol=GOWER_BINARY_RTOL
            )
        )
        if gower_ok:
            real_binary.append(c)
        else:
            demoted_to_num.append(c)
    num_in_feature = demoted_to_num + num_in_feature
    fd = _as_gower_dtype(feature_dtype)
    col_names: List[str] = []
    parts_min, parts_maj = [], []
    n_bin = 0
    if real_binary:
        col_names.extend(real_binary)
        parts_min.append(cases[real_binary].values.astype(fd))
        parts_maj.append(controls[real_binary].values.astype(fd))
        n_bin += len(real_binary)
    if cat_in_feature:
        all_cat = pd.concat([cases[cat_in_feature], controls[cat_in_feature]], ignore_index=True)
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_mat = ohe.fit_transform(all_cat)
        try:
            ohe_names = ohe.get_feature_names_out(cat_in_feature)
        except Exception:
            ohe_names = [f"ohe_{i}" for i in range(ohe_mat.shape[1])]
        col_names.extend(ohe_names.tolist() if hasattr(ohe_names, "tolist") else list(ohe_names))
        parts_min.append(ohe_mat[: len(cases)].astype(fd))
        parts_maj.append(ohe_mat[len(cases) :].astype(fd))
        n_bin += ohe_mat.shape[1]
    if num_in_feature:
        col_names.extend(num_in_feature)
        num_parts_min, num_parts_maj = [], []
        for c in num_in_feature:
            v_case = np.asarray(cases[c].values, dtype=np.float32).ravel()
            v_ctrl = np.asarray(controls[c].values, dtype=np.float32).ravel()
            col = np.concatenate([v_case, v_ctrl])
            cmin = np.nanmin(col)
            cmax = np.nanmax(col)
            if not (np.isfinite(cmin) and np.isfinite(cmax)):
                nc = np.zeros_like(v_case)
                nm = np.zeros_like(v_ctrl)
            else:
                r = float(cmax - cmin)
                if r <= 0.0 or not np.isfinite(r):
                    r = 1.0
                nc = (v_case - cmin) / r
                nm = (v_ctrl - cmin) / r
                nc = np.where(np.isfinite(nc), nc, 0.0)
                nm = np.where(np.isfinite(nm), nm, 0.0)
            num_parts_min.append(nc.astype(fd, copy=False))
            num_parts_maj.append(nm.astype(fd, copy=False))
        parts_min.append(np.column_stack(num_parts_min))
        parts_maj.append(np.column_stack(num_parts_maj))
    X_minority = np.hstack(parts_min) if parts_min else np.zeros((len(cases), 0), dtype=fd)
    X_majority = np.hstack(parts_maj) if parts_maj else np.zeros((len(controls), 0), dtype=fd)
    ranges = np.ones(X_minority.shape[1], dtype=fd)
    # Continuous block is already range-normalized; Gower uses |Δ|/r with r=1.
    for j in range(n_bin, X_minority.shape[1]):
        ranges[j] = np.asarray(1.0, dtype=fd)
    bin_col_indices = list(range(n_bin))
    return X_majority, X_minority, ranges, bin_col_indices, col_names


# -------------------------------------------------------------------------------
# P-N and D-N-N precomputation (batched for D-N-N to limit memory)
# -------------------------------------------------------------------------------
def compute_gower_pn_v2(
    X_majority: np.ndarray,
    X_minority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    col_names: Optional[List[str]] = None,
    out_dtype: np.dtype | type = np.float16,
) -> np.ndarray:
    """P-N Gower distances (controls x cases) with v2 kernel. Full matrix in memory."""
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_arr = np.array(bin_col_indices)
    return _compute_gower_distances_v2(
        X_majority, X_minority, bin_arr, cont_col_indices, ranges,
        col_names=col_names, out_dtype=out_dtype,
    )


def precompute_gower_dnn_v2(
    X_majority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    out_dir: str,
    batch_size: int = 750,
    col_names: Optional[List[str]] = None,
    out_dtype: np.dtype | type = np.float16,
    verbose: bool = True,
    dnn_full_matrix: bool = False,
    dnn_save_format: str = "npy",
) -> Tuple[str, str]:
    """
    D-N-N (control-control) Gower with v2 kernel.

    When dnn_full_matrix=False: batched memmap to limit memory. Always writes .npy.
    When dnn_full_matrix=True: compute full (n,n) in memory, then save via np.save or
    np.savez. Use dnn_save_format="npy" or "npz". Requires O(n²) RAM.

    Returns (dnn_matrix_path, dnn_enrolids_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    dt = _as_gower_dtype(out_dtype)
    n = X_majority.shape[0]
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_arr = np.array(bin_col_indices)
    dnn_enrolids_path = dnn_enrolids_npy_path(out_dir)

    if dnn_full_matrix:
        # Full matrix in memory, then save
        fmt = "npz" if dnn_save_format == "npz" else "npy"
        out_path = dnn_matrix_path(out_dir, fmt=fmt)
        t0 = time.perf_counter()
        distances = _compute_gower_distances_v2(
            X_majority,
            X_majority,
            bin_arr,
            cont_col_indices,
            ranges,
            col_names=col_names,
            out_dtype=dt,
        )
        t_compute = time.perf_counter() - t0
        t1 = time.perf_counter()
        if fmt == "npz":
            np.savez(out_path, distances=distances)
        else:
            np.save(out_path, distances)
        t_save = time.perf_counter() - t1
        if verbose:
            print(f"D-N-N Gower: compute {t_compute:.1f}s, save {t_save:.1f}s (full matrix, {fmt})")
        return out_path, dnn_enrolids_path

    # Batched memmap path
    dnn_matrix_out = dnn_matrix_npy_path(out_dir)
    dnn_mm = np.lib.format.open_memmap(
        dnn_matrix_out, mode="w+", dtype=dt, shape=(n, n)
    )
    batch_starts = range(0, n, batch_size)
    n_batches = (n + batch_size - 1) // batch_size
    iterator = (
        tqdm(batch_starts, total=n_batches, desc="D-N-N Gower", unit="batch")
        if (verbose and tqdm is not None)
        else batch_starts
    )
    t_compute_total = 0.0
    t_write_total = 0.0
    for s in iterator:
        e = min(s + batch_size, n)
        t0 = time.perf_counter()
        block = _compute_gower_distances_v2(
            X_majority[s:e],
            X_majority,
            bin_arr,
            cont_col_indices,
            ranges,
            col_names=col_names,
            out_dtype=dt,
        )
        t_compute_total += time.perf_counter() - t0
        t1 = time.perf_counter()
        dnn_mm[s:e, :] = block
        t_write_total += time.perf_counter() - t1
    del dnn_mm
    if verbose:
        print(f"D-N-N Gower: compute {t_compute_total:.1f}s, save {t_write_total:.1f}s (batched)")
    return dnn_matrix_out, dnn_enrolids_path


# -------------------------------------------------------------------------------
# Feature set 
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


def precompute_gower_pn_and_dnn(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    *,
    bin_cols_verified_by_values: Optional[Collection[str]] = None,
    outdir: str,
    dnn_subdir_seed: int,
    gower_dtype=np.float16,
    dnn_batch_size: int = 750,
    skip_pn: bool = False,
    skip_dnn: bool = False,
    resume: bool = False,
    verbose: bool = True,
    dnn_full_matrix: bool = False,
    dnn_save_format: str = "npy",
) -> Tuple[str, str, str]:
    """
    Write Gower P-N HDF5 and D-N-N memmap under ``outdir`` (cohort-specific layout).

    Returns ``(pn_h5_path, dnn_matrix_npy_path, dnn_enrolids_npy_path)``.
    """
    if get_bin_flag_columns_with_provenance is None:
        raise ImportError("public.model_IAI required on PYTHONPATH")

    gdt = _as_gower_dtype(gower_dtype)
    os.makedirs(outdir, exist_ok=True)
    dnn_dir = os.path.join(outdir, f"global_dnn_seed_{dnn_subdir_seed}_gower")
    pn_h5 = os.path.join(outdir, "distances_majority_minority_gower.h5")
    dnn_enrolids_path = dnn_enrolids_npy_path(dnn_dir)
    os.makedirs(dnn_dir, exist_ok=True)

    verified = bin_cols_verified_by_values
    if verified is None:
        train_feat = pd.concat(
            [cases[feature_cols], controls[feature_cols]], ignore_index=True
        )
        _, verified = get_bin_flag_columns_with_provenance(train_feat)

    if verbose:
        print("Building Gower feature matrices...")
    X_majority, X_minority, ranges, bin_col_indices, col_names = build_gower_feature_matrices(
        cases,
        controls,
        feature_cols,
        cat_cols,
        num_cols,
        bin_cols,
        feature_dtype=gdt,
        bin_cols_verified_by_values=verified,
    )
    maj_ids = controls["ENROLID"].values.astype(np.int64)
    min_ids = cases["ENROLID"].values.astype(np.int64)

    if not skip_pn:
        if resume and os.path.isfile(pn_h5):
            if verbose:
                print("P-N: skipping (resume, file exists).")
        else:
            if verbose:
                print("Computing P-N Gower distances (controls x cases)...")
            dist_pn = compute_gower_pn_v2(
                X_majority,
                X_minority,
                bin_col_indices,
                ranges,
                col_names=col_names,
                out_dtype=gdt,
            )
            if gdt == np.dtype(np.float32) and save_distances_hdf5 is not None:
                save_distances_hdf5(dist_pn, maj_ids, min_ids, pn_h5)
            else:
                _save_distances_hdf5_inline(
                    dist_pn, maj_ids, min_ids, pn_h5, distances_dtype=gdt
                )
            if verbose:
                print("P-N done.")

    if not skip_dnn:
        if resume and dnn_matrix_storage_exists(dnn_dir) and os.path.isfile(dnn_enrolids_path):
            if verbose:
                print("D-N-N: skipping (resume, files exist).")
        else:
            if verbose:
                mode = "full matrix" if dnn_full_matrix else "batched"
                print(f"Computing D-N-N Gower ({mode})...")
            precompute_gower_dnn_v2(
                X_majority,
                bin_col_indices,
                ranges,
                dnn_dir,
                batch_size=dnn_batch_size,
                col_names=col_names,
                out_dtype=gdt,
                verbose=verbose,
                dnn_full_matrix=dnn_full_matrix,
                dnn_save_format=dnn_save_format,
            )
            np.save(dnn_enrolids_path, maj_ids)
            if verbose:
                print("D-N-N done.")

    return pn_h5, ensure_dnn_matrix_npy(dnn_dir), dnn_enrolids_path


if __name__ == "__main__":
    print(
        "Use your cohort driver to precompute; this module is library-only."
    )
