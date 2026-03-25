#!/usr/bin/env python
"""
precompute_gower_distances_warped.py
====================================
Gower P-N + D-N-N precompute with optional log-clip warping on continuous columns only.

Uses the same v2 kernel and output layout as ``precompute_gower_pn_and_dnn``:
  - P-N: distances_majority_minority_gower.h5
  - D-N-N: global_dnn_seed_{seed}_gower/

PN and DNN may use different continuous column lists; when lists are identical,
a single ``build_gower_feature_matrices`` call is reused for both.
"""
from __future__ import annotations

import os
from typing import Collection, List, Optional, Tuple

import numpy as np
import pandas as pd

from public.precompute_gower_distances import (
    _as_gower_dtype,
    _save_distances_hdf5_inline,
    build_gower_feature_matrices,
    compute_gower_pn_v2,
    dnn_enrolids_npy_path,
    dnn_matrix_storage_exists,
    ensure_dnn_matrix_npy,
    precompute_gower_dnn_v2,
    save_distances_hdf5,
)

try:
    from public.model_IAI import get_bin_flag_columns_with_provenance
except ImportError:
    get_bin_flag_columns_with_provenance = None  # type: ignore[misc,assignment]


def fit_warp_stats(X_maj: np.ndarray, col_mask_binary: np.ndarray) -> dict:
    """Fit log-clip + robust-scale statistics on majority (controls) rows."""
    stats: dict = {}
    log_x = np.log10(1.0 + np.maximum(X_maj, 0.0))
    stats["log_clip_lo"] = np.percentile(log_x, 2, axis=0)
    stats["log_clip_hi"] = np.percentile(log_x, 97.5, axis=0)
    stats["log_median"] = np.median(log_x, axis=0)
    log_q25 = np.percentile(log_x, 25, axis=0)
    log_q75 = np.percentile(log_x, 75, axis=0)
    stats["log_iqr"] = np.maximum(log_q75 - log_q25, 1e-10)
    return stats


def apply_warp_log_clip(
    X: np.ndarray, col_mask_binary: np.ndarray, fit_stats: dict
) -> np.ndarray:
    """log10(1+x) -> clip -> robust scale; binary columns clip to [0,1] only."""
    out = X.copy().astype(np.float64)
    for j in range(X.shape[1]):
        if col_mask_binary[j]:
            out[:, j] = np.clip(X[:, j], 0, 1)
        else:
            raw = np.maximum(X[:, j], 0)
            log_x = np.log10(1.0 + raw)
            lo, hi = fit_stats["log_clip_lo"][j], fit_stats["log_clip_hi"][j]
            med, iqr = fit_stats["log_median"][j], fit_stats["log_iqr"][j]
            clipped = np.clip(log_x, lo, hi)
            if iqr > 1e-10:
                out[:, j] = (clipped - med) / iqr
            else:
                out[:, j] = 0.0
    return out.astype(np.float32)


def _subset_log_clip_stats(fit_stats: dict, indices: np.ndarray) -> dict:
    return {
        "log_clip_lo": fit_stats["log_clip_lo"][indices],
        "log_clip_hi": fit_stats["log_clip_hi"][indices],
        "log_median": fit_stats["log_median"][indices],
        "log_iqr": fit_stats["log_iqr"][indices],
    }


def _apply_warp_to_frame(
    df: pd.DataFrame,
    cols: List[str],
    warp_ordered: List[str],
    fit_stats_full: dict,
    warp_fit_order: List[str],
) -> pd.DataFrame:
    """Copy df[cols]; replace warped columns (subset of cols, order warp_fit_order)."""
    out = df[cols].copy()
    warp_names = [c for c in warp_fit_order if c in warp_ordered and c in cols]
    if not warp_names:
        return out
    idx = np.array([warp_fit_order.index(c) for c in warp_names], dtype=np.intp)
    fs = _subset_log_clip_stats(fit_stats_full, idx)
    col_mask = np.zeros(len(warp_names), dtype=bool)
    X = out[warp_names].fillna(0).values.astype(np.float64)
    Xw = apply_warp_log_clip(X, col_mask, fs)
    for i, c in enumerate(warp_names):
        out[c] = Xw[:, i]
    return out


def _build_warped_gower_pair(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    warp_fit_order: List[str],
    warp_apply: List[str],
    fit_stats_full: dict,
    gdt: np.dtype,
    verified: Collection[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    cases_w = _apply_warp_to_frame(cases, feature_cols, warp_apply, fit_stats_full, warp_fit_order)
    ctr_w = _apply_warp_to_frame(controls, feature_cols, warp_apply, fit_stats_full, warp_fit_order)
    return build_gower_feature_matrices(
        cases_w,
        ctr_w,
        feature_cols,
        [],
        feature_cols,
        [],
        feature_dtype=gdt,
        bin_cols_verified_by_values=verified,
    )


def precompute_gower_pn_and_dnn_warped(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    pn_continuous_cols: List[str],
    dnn_continuous_cols: List[str],
    *,
    warp_continuous_cols: Optional[List[str]] = None,
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
    Like ``precompute_gower_pn_and_dnn`` but:
      - Only continuous numeric columns (pass explicit lists; no cat/bin in matrix).
      - Optionally applies log-clip warp (fit on controls) before Gower encoding.

    ``warp_continuous_cols``: columns to warp; default None = warp all columns in
    the union of pn_continuous_cols and dnn_continuous_cols.

    Returns ``(pn_h5_path, dnn_matrix_npy_path, dnn_enrolids_npy_path)``.
    """
    if get_bin_flag_columns_with_provenance is None:
        raise ImportError("public.model_IAI required on PYTHONPATH")

    if not pn_continuous_cols or not dnn_continuous_cols:
        raise ValueError("pn_continuous_cols and dnn_continuous_cols must be non-empty")

    gdt = _as_gower_dtype(gower_dtype)
    os.makedirs(outdir, exist_ok=True)
    dnn_dir = os.path.join(outdir, f"global_dnn_seed_{dnn_subdir_seed}_gower")
    pn_h5 = os.path.join(outdir, "distances_majority_minority_gower.h5")
    dnn_enrolids_path = dnn_enrolids_npy_path(dnn_dir)
    os.makedirs(dnn_dir, exist_ok=True)

    warp_fit_order = sorted(set(pn_continuous_cols) | set(dnn_continuous_cols))
    if warp_continuous_cols is None:
        warp_apply = list(warp_fit_order)
    else:
        extra = set(warp_continuous_cols) - set(warp_fit_order)
        if extra:
            raise ValueError(f"warp_continuous_cols not in PN/DNN union: {sorted(extra)}")
        warp_apply = [c for c in warp_fit_order if c in warp_continuous_cols]

    X_maj_fit = controls[warp_fit_order].fillna(0).values.astype(np.float64)
    col_mask_fit = np.zeros(len(warp_fit_order), dtype=bool)
    fit_stats_full = fit_warp_stats(X_maj_fit, col_mask_fit)

    verified = bin_cols_verified_by_values
    if verified is None:
        ucols = sorted(set(pn_continuous_cols) | set(dnn_continuous_cols))
        train_feat = pd.concat(
            [cases[ucols], controls[ucols]], ignore_index=True
        )
        _, verified = get_bin_flag_columns_with_provenance(train_feat)

    maj_ids = controls["ENROLID"].values.astype(np.int64)
    min_ids = cases["ENROLID"].values.astype(np.int64)

    same_lists = len(pn_continuous_cols) == len(dnn_continuous_cols) and all(
        a == b for a, b in zip(pn_continuous_cols, dnn_continuous_cols)
    )

    if verbose:
        print(
            f"Building warped Gower feature matrices (PN cols={len(pn_continuous_cols)}, "
            f"DNN cols={len(dnn_continuous_cols)}, same_build={same_lists})..."
        )

    if same_lists:
        X_majority, X_minority, ranges, bin_col_indices, col_names = _build_warped_gower_pair(
            cases,
            controls,
            pn_continuous_cols,
            warp_fit_order,
            warp_apply,
            fit_stats_full,
            gdt,
            verified,
        )
        X_majority_dnn = X_majority
        ranges_dnn = ranges
        bin_dnn = bin_col_indices
        col_names_dnn = col_names
    else:
        X_majority, X_minority, ranges, bin_col_indices, col_names = _build_warped_gower_pair(
            cases,
            controls,
            pn_continuous_cols,
            warp_fit_order,
            warp_apply,
            fit_stats_full,
            gdt,
            verified,
        )
        X_majority_dnn, _, ranges_dnn, bin_dnn, col_names_dnn = _build_warped_gower_pair(
            cases,
            controls,
            dnn_continuous_cols,
            warp_fit_order,
            warp_apply,
            fit_stats_full,
            gdt,
            verified,
        )

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
                X_majority_dnn,
                bin_dnn,
                ranges_dnn,
                dnn_dir,
                batch_size=dnn_batch_size,
                col_names=col_names_dnn,
                out_dtype=gdt,
                verbose=verbose,
                dnn_full_matrix=dnn_full_matrix,
                dnn_save_format=dnn_save_format,
            )
            np.save(dnn_enrolids_path, maj_ids)
            if verbose:
                print("D-N-N done.")

    return pn_h5, ensure_dnn_matrix_npy(dnn_dir), dnn_enrolids_path
