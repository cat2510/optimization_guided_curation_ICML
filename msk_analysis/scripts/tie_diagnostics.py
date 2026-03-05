"""
Tie degeneracy diagnostics for PN (minority-majority) and DNN (majority-majority) distance matrices.
Configurable thresholds: WARN if unique < unique_threshold OR top5_frac > top5_threshold.
"""
from __future__ import annotations

from typing import Optional, Tuple

import h5py
import numpy as np


def tie_diagnostics_from_h5(
    pn_h5_path: str,
    sample_size: int = 50000,
    dataset_key: str = "distances",
    unique_threshold: int = 200,
    top5_threshold: float = 0.05,
    show_top10: bool = False,
) -> dict:
    """
    Sample distances from PN H5, return diagnostics dict.
    WARN if unique < unique_threshold OR top5_frac > top5_threshold.
    """
    with h5py.File(pn_h5_path, "r") as f:
        d = f[dataset_key]
        n_maj, n_min = d.shape[0], d.shape[1]
        rng = np.random.default_rng(42)
        n_sample = min(sample_size, n_maj * n_min)
        idx_maj = rng.integers(0, n_maj, size=n_sample)
        idx_min = rng.integers(0, n_min, size=n_sample)
        sample = np.empty(n_sample, dtype=np.float32)
        for row in np.unique(idx_maj):
            mask = idx_maj == row
            cols = idx_min[mask]
            row_data = d[row, :]
            sample[mask] = row_data[cols]
    return _compute_diagnostics(
        sample,
        "PN",
        unique_threshold,
        top5_threshold,
        show_top10,
    )


def tie_diagnostics_from_dnn(
    dnn_matrix_path: str,
    sample_size: int = 50000,
    unique_threshold: int = 200,
    top5_threshold: float = 0.05,
    show_top10: bool = False,
) -> dict:
    """Sample control-control distances from DNN .npy matrix."""
    dnn = np.load(dnn_matrix_path, mmap_mode="r")
    n = dnn.shape[0]
    rng = np.random.default_rng(42)
    n_sample = min(sample_size, n * n)
    idx_i = rng.integers(0, n, size=n_sample)
    idx_j = rng.integers(0, n, size=n_sample)
    sample = dnn[idx_i, idx_j].copy()
    del dnn
    return _compute_diagnostics(
        sample,
        "DNN",
        unique_threshold,
        top5_threshold,
        show_top10,
    )


def _compute_diagnostics(
    sample: np.ndarray,
    label: str,
    unique_threshold: int,
    top5_threshold: float,
    show_top10: bool,
) -> dict:
    uniq, counts = np.unique(sample.ravel(), return_counts=True)
    n_unique = len(uniq)
    total = counts.sum()
    top5_frac = counts[np.argsort(-counts)[:5]].sum() / total if total > 0 else 0.0
    out = {
        "n_unique": n_unique,
        "top5_frac": top5_frac,
        "degenerate": n_unique < unique_threshold or top5_frac > top5_threshold,
    }
    if show_top10:
        order = np.argsort(-counts)[:10]
        out["top10_values"] = uniq[order].tolist()
        out["top10_counts"] = counts[order].tolist()
    return out


def run_tie_diagnostics(
    pn_h5_path: Optional[str] = None,
    dnn_matrix_path: Optional[str] = None,
    sample_size: int = 50000,
    unique_threshold: int = 200,
    top5_threshold: float = 0.05,
    show_top10: bool = False,
) -> None:
    """Run and print diagnostics for PN and/or DNN. WARN if degenerate."""
    if pn_h5_path:
        d = tie_diagnostics_from_h5(
            pn_h5_path, sample_size,
            unique_threshold=unique_threshold,
            top5_threshold=top5_threshold,
            show_top10=show_top10,
        )
        print(f"  [PN tie diagnostics] unique={d['n_unique']:,} | top5_frac={d['top5_frac']:.4f}")
        if d.get("top10_values") is not None:
            print(f"    top-10 values: {d['top10_values'][:10]}")
        if d["degenerate"]:
            print(f"  [WARN] PN degenerate (unique<{unique_threshold} or top5_frac>{top5_threshold})")
    if dnn_matrix_path:
        d = tie_diagnostics_from_dnn(
            dnn_matrix_path, sample_size,
            unique_threshold=unique_threshold,
            top5_threshold=top5_threshold,
            show_top10=show_top10,
        )
        print(f"  [DNN tie diagnostics] unique={d['n_unique']:,} | top5_frac={d['top5_frac']:.4f}")
        if d.get("top10_values") is not None:
            print(f"    top-10 values: {d['top10_values'][:10]}")
        if d["degenerate"]:
            print(f"  [WARN] DNN degenerate (unique<{unique_threshold} or top5_frac>{top5_threshold})")
