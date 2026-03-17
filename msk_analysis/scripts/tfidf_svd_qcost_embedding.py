"""
TF-IDF + TruncatedSVD + QuantileTransformer embedding for Stage A DNN.
Produces L2-normalized combined embedding for control-control cosine distances.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import QuantileTransformer


SVD_DIM = 100
TRAIN_TEST_SEED = 123
SUBSAMPLE = 200_000


def get_cost_columns_2017(df: pd.DataFrame) -> List[str]:
    """Numeric 2017 cost columns only (exclude 2018)."""
    cost_cols = [
        col
        for col in df.columns
        if (
            "cost" in col.lower()
            or "quarterly" in col.lower()
            or "total_increasing" in col.lower()
            or "total_decreasing" in col.lower()
            or "skewness" in col.lower()
            or "kurtosis" in col.lower()
            or "cv" in col.lower()
            or "range" in col.lower()
        )
        and "2018" not in col
    ]
    return [c for c in cost_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def build_tfidf_svd_qcost_embedding(
    controls: pd.DataFrame,
    code_columns: List[str],
    cost_columns: List[str],
    out_dir: str,
    svd_dim: int = SVD_DIM,
    subsample: int | None = SUBSAMPLE,
    random_state: int = TRAIN_TEST_SEED,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build combined embedding for TRAIN controls.
    Returns (Z, control_enrolids, meta) where Z is (n_controls, d) L2-normalized.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_controls = len(controls)
    enrolids = controls["ENROLID"].values.astype(np.int64)

    # 1. Sparse clinical code matrix
    X_codes = controls[code_columns].fillna(0).values.astype(np.float64)
    X_codes_sp = csr_matrix(X_codes)
    n_codes = len(code_columns)

    code_order_path = os.path.join(out_dir, "code_column_order.json")
    with open(code_order_path, "w") as f:
        json.dump(code_columns, f, indent=2)
    print(f"  Saved code column order: {code_order_path} ({n_codes} columns)")

    # 2. TF-IDF
    df_j = np.array((X_codes_sp > 0).sum(axis=0)).flatten()
    idf = np.log((n_controls + 1) / (df_j + 1)) + 1
    X_tfidf = X_codes_sp.multiply(idf)

    # 3. TruncatedSVD (n_components cannot exceed n_features)
    svd_dim_eff = min(svd_dim, n_codes)
    svd = TruncatedSVD(n_components=svd_dim_eff, random_state=random_state, n_iter=7)
    Z_codes = svd.fit_transform(X_tfidf).astype(np.float64)
    Z_codes_norm = Z_codes / (np.linalg.norm(Z_codes, axis=1, keepdims=True) + 1e-10)

    # 4. Quantile transform 2017 cost (sklearn expects 2D: n_samples x n_features)
    X_cost = controls[cost_columns].fillna(0).values.astype(np.float64)  # (n_controls, d_cost)
    n_quantiles = min(1000, max(1, X_cost.shape[0]))
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=n_quantiles,
        subsample=subsample,
        random_state=random_state,
    )
    Z_cost = qt.fit_transform(X_cost).astype(np.float64)  # (n_controls, d_cost)
    Z_cost_norm = Z_cost / (np.linalg.norm(Z_cost, axis=1, keepdims=True) + 1e-10)

    d_cost = X_cost.shape[1]
    # 5. Combine with alpha = sqrt(svd_dim_eff / d_cost) for scale matching
    alpha = np.sqrt(float(svd_dim_eff) / d_cost) if d_cost > 0 else 1.0
    Z = np.hstack([Z_codes_norm, alpha * Z_cost_norm])
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-10)

    meta = {
        "n_codes": n_codes,
        "svd_dim": svd_dim_eff,
        "d_cost": d_cost,
        "alpha": float(alpha),
        "code_column_order": code_columns,
    }
    return Z, enrolids, meta


def run_cosine_distance_diagnostics(
    Z: np.ndarray, n_sample: int = 200_000, random_state: int = 123
) -> np.ndarray:
    """Sample random pairs, compute cosine distance, report stats."""
    n = Z.shape[0]
    n_sample = min(n_sample, n * (n - 1) // 2)
    rng = np.random.RandomState(random_state)
    i = rng.randint(0, n, size=n_sample)
    j = rng.randint(0, n, size=n_sample)
    bad = i == j
    j[bad] = (j[bad] + 1) % n
    dots = (Z[i] * Z[j]).sum(axis=1)
    dists = 1.0 - dots
    dists = np.clip(dists, 0.0, 2.0)
    uniq, counts = np.unique(np.round(dists, 6), return_counts=True)
    frac = counts / n_sample
    top5_idx = np.argsort(frac)[-5:][::-1]
    print("  Cosine distance diagnostics (sample ~200k pairs):")
    print(f"    unique values: {len(uniq)}")
    print("    top-5 freq frac:")
    for idx in top5_idx:
        print(f"      {uniq[idx]:.6f}: {frac[idx]:.6f}")
    if frac[top5_idx[0]] > 0.05:
        print("  WARNING: top-5 freq frac > 0.05")
    return dists
