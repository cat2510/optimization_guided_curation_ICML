#!/usr/bin/env python
"""
exp6_distance_metric_ablation.py
================================
Distance metric ablation for Exp 5 vs Exp 6 sampling.

Exp 5: 1N matched + 2N random (random undersampling baseline)
Exp 6: 1N matched + 2N max-dispersed (curated), per distance metric

Two-stage: k-center then match, random init, no k-means++, no adaptive pool.
Metrics: Euclidean, Manhattan, Hamming, Jaccard, Cosine, Chebyshev.

Then: plot cost distribution, train OCT on Exp 5 and on best Exp 6 (lowest cost).

Usage
-----
  cd msk_analysis
  python exp6_distance_metric_ablation.py [--seeds 0] [--outdir ./exp6_distance_ablation]
  python exp6_distance_metric_ablation.py --metrics euclidean,manhattan --seeds 0,1
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
import traceback
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path setup
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import importlib
import public.precompute_distances
importlib.reload(public.precompute_distances)
from public.precompute_distances import (
    get_preprocessor,
    compute_distances_batched,
    save_distances_hdf5,
    precompute_leaf_dnn_memmap,
    precompute_leaf_dnn_hdf5,
)

try:
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import (
        load_pn_hdf5,
        build_id_to_index,
        farthest_first_kcenter_indices,
        choose_seed_random,
    )
except ImportError:
    from public.two_stage_kcenter_match import (
        load_pn_hdf5,
        build_id_to_index,
        farthest_first_kcenter_indices,
        choose_seed_random,
    )

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    finetune_oct,
    evaluate_binary_oct,
)
# Use pd.read_parquet (not Spark) for deterministic row order; Spark may shuffle and change train/test split.
from experiments_compare_random_vs_curation import (
    sample_random_controls,
    sample_stageA_dispersed_controls,
    sample_stageB_matched_controls,
    sample_stageA_on_restricted_pool,
    train_and_evaluate_oct,
)

TRAIN_TEST_SEED = 123
OCT_DEPTHS = [7, 9]
OCT_MINBUCKETS = [100, 150]
OCT_CPS = [0.00001, 0.0001, 0.001, 0.01]

# Exp 6: 1N matched + 2N max_dispersed
EXP6_TOTAL_CONTROLS_MULT = 3  # 1 + 2

DEFAULT_METRICS = ["euclidean", "manhattan", "hamming", "jaccard", "cosine", "chebyshev", "gower"]


def parse_args():
    p = argparse.ArgumentParser(description="Exp 6 distance metric ablation")
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds (default: 0)")
    p.add_argument("--outdir", type=str, default="./exp6_distance_ablation", help="Output directory")
    p.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated distance metrics (default: {','.join(DEFAULT_METRICS)})",
    )
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--distances_dir", type=str, default="./precomputed_distances_exp6_ablation")
    p.add_argument("--feature_set", type=str, default="all_cost", choices=["medical_only", "all_cost", "less_cost"])
    p.add_argument("--M_pool", type=int, default=None, help="Candidate pool size (default: n_controls//2)")
    p.add_argument("--skip_plot", action="store_true", help="Skip cost distribution plot")
    p.add_argument("--skip_oct", action="store_true", help="Skip OCT training")
    p.add_argument("--resume", action="store_true", help="Reuse existing CSVs/predictions")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gower mixed-type distance: binary uses δ=1[a≠b], continuous uses δ=|a-b|/range_j
# range_j = max-min from train; if range_j=0 set to 1 (constant feature contributes 0)
# ---------------------------------------------------------------------------
def _compute_gower_distances(
    X_A: np.ndarray,
    X_B: np.ndarray,
    binary_col_indices: np.ndarray,
    continuous_col_indices: np.ndarray,
    ranges: np.ndarray,
    batch_size: int = 1000,
) -> np.ndarray:
    """Compute Gower distances between X_A (n_A x p) and X_B (n_B x p). Returns (n_A, n_B)."""
    n_A, n_B = X_A.shape[0], X_B.shape[0]
    p = X_A.shape[1]
    distances = np.zeros((n_A, n_B), dtype=np.float32)

    n_batches = (n_A + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n_A)
        block = np.zeros((e - s, n_B), dtype=np.float32)
        for j in range(p):
            a_vals = X_A[s:e, j]
            b_vals = X_B[:, j]
            if j in set(binary_col_indices):
                # δ = 1[a≠b]
                diff = (a_vals[:, None] != b_vals[None, :]).astype(np.float32)
            else:
                # continuous: δ = |a-b|/range_j
                rj = ranges[j]
                diff = np.abs(a_vals[:, None].astype(np.float64) - b_vals[None, :].astype(np.float64)) / rj
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
    """P-N distances (controls x cases) with Gower."""
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_col_indices_arr = np.array(bin_col_indices)
    return _compute_gower_distances(
        X_majority, X_minority, bin_col_indices_arr, cont_col_indices, ranges, batch_size
    )


def precompute_gower_dnn_memmap(
    X_majority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    out_dir: str,
    batch_size: int = 750,
) -> Tuple[str, str]:
    """D-N-N (control-control) Gower distances. Returns (dnn_matrix_path, dnn_enrolids_path)."""
    os.makedirs(out_dir, exist_ok=True)
    n = X_majority.shape[0]
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_col_indices_arr = np.array(bin_col_indices)

    dnn_matrix_path = os.path.join(out_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_path = os.path.join(out_dir, "leaf_global_dnn_enrolids.npy")

    dnn_mm = np.lib.format.open_memmap(dnn_matrix_path, mode="w+", dtype=np.float32, shape=(n, n))
    for b in range((n + batch_size - 1) // batch_size):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        block = _compute_gower_distances(
            X_majority[s:e], X_majority, bin_col_indices_arr, cont_col_indices, ranges, batch_size=n
        )
        dnn_mm[s:e, :] = block
    del dnn_mm
    return dnn_matrix_path, dnn_enrolids_path


def precompute_gower_dnn_hdf5(
    X_majority: np.ndarray,
    bin_col_indices: List[int],
    ranges: np.ndarray,
    out_dir: str,
    batch_size: int = 750,
    compression: str = "gzip",
    compression_opts: int = 9,
) -> Tuple[str, str]:
    """D-N-N Gower distances to HDF5 (chunked + compressed). Returns (h5_path, enrolids_path)."""
    os.makedirs(out_dir, exist_ok=True)
    n = X_majority.shape[0]
    n_p = X_majority.shape[1]
    cont_col_indices = np.array([j for j in range(n_p) if j not in bin_col_indices])
    bin_col_indices_arr = np.array(bin_col_indices)

    h5_path = os.path.join(out_dir, "leaf_global_dnn_matrix.h5")
    dnn_enrolids_path = os.path.join(out_dir, "leaf_global_dnn_enrolids.npy")

    with h5py.File(h5_path, "w") as f:
        chunk_rows = min(1000, n)
        dset = f.create_dataset(
            "distances",
            shape=(n, n),
            dtype=np.float32,
            chunks=(chunk_rows, n),
            compression=compression,
            compression_opts=compression_opts,
        )
        for b in range((n + batch_size - 1) // batch_size):
            s, e = b * batch_size, min((b + 1) * batch_size, n)
            block = _compute_gower_distances(
                X_majority[s:e], X_majority, bin_col_indices_arr, cont_col_indices, ranges, batch_size=n
            )
            dset[s:e, :] = block

    return h5_path, dnn_enrolids_path


def ensure_distances_for_metric(
    distance_metric: str,
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    distances_dir: str,
    use_hdf5_dnn: bool = False,
) -> Tuple[str, str, str]:
    """
    Ensure P-N and D-N-N distance matrices exist for the given metric.
    Returns (pn_h5_path, dnn_matrix_path, dnn_enrolids_path).

    use_hdf5_dnn: if True, save D-N-N as HDF5 compressed (~3-5x smaller than .npy).
    load_nn() in two_stage supports both .npy and .h5.
    """
    os.makedirs(distances_dir, exist_ok=True)
    pn_h5 = os.path.join(distances_dir, f"distances_majority_minority_{distance_metric}.h5")
    dnn_dir = os.path.join(distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_{distance_metric}")
    dnn_suffix = ".h5" if use_hdf5_dnn else ".npy"
    dnn_matrix_path = os.path.join(dnn_dir, f"leaf_global_dnn_matrix{dnn_suffix}")
    dnn_enrolids_path = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")

    # tfidf_svd_cosine_qcost: Stage A DNN only; use Gower P-N for seed selection
    if distance_metric == "tfidf_svd_cosine_qcost":
        gower_pn_h5 = os.path.join(distances_dir, "distances_majority_minority_gower.h5")
        if not os.path.exists(gower_pn_h5):
            ensure_distances_for_metric(
                "gower", cases, controls, feature_cols, cat_cols, num_cols, bin_cols,
                distances_dir, use_hdf5_dnn=use_hdf5_dnn,
            )
        pn_h5 = gower_pn_h5
        if os.path.exists(dnn_matrix_path) and os.path.exists(dnn_enrolids_path):
            return pn_h5, dnn_matrix_path, dnn_enrolids_path
        # Build TF-IDF+SVD+QuantileTransformer embedding and precompute D-N-N (cosine)
        import tfidf_svd_qcost_embedding
        importlib.reload(tfidf_svd_qcost_embedding)
        from tfidf_svd_qcost_embedding import (
            build_tfidf_svd_qcost_embedding,
            get_cost_columns_2017,
            run_cosine_distance_diagnostics,
        )
        code_cols = [c for c in bin_cols if c in feature_cols]
        cost_cols_2017 = get_cost_columns_2017(controls)
        Z, enrolids, meta = build_tfidf_svd_qcost_embedding(
            controls, code_cols, cost_cols_2017, dnn_dir,
        )
        print(f"  Sanity: n_codes={meta['n_codes']}, svd_dim={meta['svd_dim']}, d_cost={meta['d_cost']}, alpha={meta['alpha']}")
        run_cosine_distance_diagnostics(Z)
        dnn_matrix_path, dnn_enrolids_path = precompute_leaf_dnn_hdf5(
            X_majority_leaf=Z,
            majority_enrolids_leaf=enrolids,
            out_dir=dnn_dir,
            leaf_id="global",
            batch_size=750,
            metric="cosine",
        )
        return pn_h5, dnn_matrix_path, dnn_enrolids_path

    if os.path.exists(dnn_matrix_path) and os.path.exists(dnn_enrolids_path):
        return pn_h5, dnn_matrix_path, dnn_enrolids_path
    if distance_metric == "gower" and (not os.path.exists(pn_h5) or not os.path.exists(dnn_matrix_path) or not os.path.exists(dnn_enrolids_path)):
        # Gower: binary δ=1[a≠b], continuous δ=|a-b|/range_j (range from train, 1 if 0)
        from sklearn.preprocessing import OneHotEncoder
        bin_in_feature = [c for c in bin_cols if c in feature_cols]
        num_in_feature = [c for c in num_cols if c in feature_cols]
        cat_in_feature = [c for c in cat_cols if c in feature_cols]
        parts_min, parts_maj = [], []
        n_bin = 0
        if bin_in_feature:
            parts_min.append(cases[bin_in_feature].values.astype(np.float64))
            parts_maj.append(controls[bin_in_feature].values.astype(np.float64))
            n_bin += len(bin_in_feature)
        if cat_in_feature:
            all_cat = pd.concat([cases[cat_in_feature], controls[cat_in_feature]], ignore_index=True)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            ohe_mat = ohe.fit_transform(all_cat)
            parts_min.append(ohe_mat[: len(cases)].astype(np.float64))
            parts_maj.append(ohe_mat[len(cases) :].astype(np.float64))
            n_bin += ohe_mat.shape[1]
        if num_in_feature:
            parts_min.append(cases[num_in_feature].values.astype(np.float64))
            parts_maj.append(controls[num_in_feature].values.astype(np.float64))
        X_minority = np.hstack(parts_min) if parts_min else np.zeros((len(cases), 0))
        X_majority = np.hstack(parts_maj) if parts_maj else np.zeros((len(controls), 0))
        ranges = np.ones(X_minority.shape[1])
        for j in range(n_bin, X_minority.shape[1]):
            col = np.concatenate([X_minority[:, j], X_majority[:, j]])
            r = float(np.nanmax(col) - np.nanmin(col))
            ranges[j] = r if r > 0 else 1.0
        bin_col_indices = list(range(n_bin))
        print(f"  Precomputing P-N distances (metric=gower)...")
        dist_pn = compute_gower_distances_batched(X_majority, X_minority, bin_col_indices, ranges)
        save_distances_hdf5(
            dist_pn,
            controls["ENROLID"].values.astype(np.int64),
            cases["ENROLID"].values.astype(np.int64),
            pn_h5,
        )
        if not os.path.exists(dnn_matrix_path) or not os.path.exists(dnn_enrolids_path):
            print(f"  Precomputing D-N-N distances (metric=gower)...")
            if use_hdf5_dnn:
                dnn_matrix_path, dnn_enrolids_path = precompute_gower_dnn_hdf5(
                    X_majority, bin_col_indices, ranges, dnn_dir
                )
            else:
                dnn_matrix_path, dnn_enrolids_path = precompute_gower_dnn_memmap(
                    X_majority, bin_col_indices, ranges, dnn_dir
                )
            np.save(dnn_enrolids_path, controls["ENROLID"].values.astype(np.int64))
        return pn_h5, dnn_matrix_path, dnn_enrolids_path

    elif distance_metric in ("jaccard", "hamming"):
        # Use only binary features (0/1) - appropriate for Jaccard and Hamming
        bin_in_feature = [c for c in bin_cols if c in feature_cols]
        if not bin_in_feature:
            raise ValueError(
                f"metric={distance_metric} requires binary features, but none of {bin_cols} are in feature_cols"
            )
        X_minority = cases[bin_in_feature].values.astype(np.float64)
        X_majority = controls[bin_in_feature].values.astype(np.float64)
    else:
        preprocessor = get_preprocessor(
            X=pd.concat([cases[feature_cols], controls[feature_cols]], ignore_index=True),
            cat_cols=cat_cols,
            num_cols=num_cols,
            binary_cols=bin_cols,
            verbose=False,
        )
        X_minority = preprocessor.fit_transform(cases[feature_cols])
        X_majority = preprocessor.transform(controls[feature_cols])

    if not os.path.exists(pn_h5):
        print(f"  Precomputing P-N distances (metric={distance_metric})...")
        distances_pn = compute_distances_batched(
            X_majority, X_minority, batch_size=1000, dtype=np.float32, metric=distance_metric
        )
        save_distances_hdf5(
            distances_pn,
            controls["ENROLID"].values.astype(np.int64),
            cases["ENROLID"].values.astype(np.int64),
            pn_h5,
        )
    if not os.path.exists(dnn_matrix_path) or not os.path.exists(dnn_enrolids_path):
        print(f"  Precomputing D-N-N distances (metric={distance_metric})...")
        if use_hdf5_dnn:
            dnn_matrix_path, dnn_enrolids_path = precompute_leaf_dnn_hdf5(
                X_majority_leaf=X_majority,
                majority_enrolids_leaf=controls["ENROLID"].values.astype(np.int64),
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric=distance_metric,
            )
        else:
            precompute_leaf_dnn_memmap(
                X_majority_leaf=X_majority,
                majority_enrolids_leaf=controls["ENROLID"].values.astype(np.int64),
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric=distance_metric,
            )
    return pn_h5, dnn_matrix_path, dnn_enrolids_path

def run_rnd_1to1_sampling(
    control_enrolids: np.ndarray,
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    N: int,
    seed: int,
) -> pd.DataFrame:
    """RND 1-1: N cases + N random controls. Returns train_df."""
    rnd_ids = sample_random_controls(control_enrolids, N, seed)
    return pd.concat([cases, controls[controls["ENROLID"].isin(rnd_ids)]], ignore_index=True)


def run_ours_1to1_sampling(
    control_enrolids: np.ndarray,
    case_enrolids: np.ndarray,
    controls: pd.DataFrame,
    cases: pd.DataFrame,
    dnn_matrix_path: str,
    dnn_enrolids_path: str,
    pn_h5_path: str,
    N: int,
    M_pool: int,
    seed: int,
) -> pd.DataFrame:
    """Ours 1-1: 1:1 k-center then match (matching_ratio=1). Returns train_df."""
    match_ids = sample_stageB_matched_controls(
        control_enrolids, case_enrolids, dnn_matrix_path, dnn_enrolids_path, pn_h5_path,
        target_count=N, matching_ratio=1, M_pool=M_pool, seed_method="random", seed=seed,
        X_majority_leaf=None, verbose=False, use_kmeanspp=False,
    )
    return pd.concat([cases, controls[controls["ENROLID"].isin(match_ids)]], ignore_index=True)


def run_exp5_sampling(
    control_enrolids: np.ndarray,
    case_enrolids: np.ndarray,
    controls: pd.DataFrame,
    cases: pd.DataFrame,
    dnn_matrix_path: str,
    dnn_enrolids_path: str,
    pn_h5_path: str,
    N: int,
    M_pool: int,
    seed: int,
) -> pd.DataFrame:
    """Run Exp 5: 1N matched + 2N random extras. Returns train_df (cases + selected controls)."""
    total_controls = EXP6_TOTAL_CONTROLS_MULT * N  # 3N

    # 1) 1N matched (same two-stage matching as Exp 6, but extras are random)
    match_ids = sample_stageB_matched_controls(
        control_enrolids,
        case_enrolids,
        dnn_matrix_path,
        dnn_enrolids_path,
        pn_h5_path,
        target_count=N,
        matching_ratio=1,
        M_pool=M_pool,
        seed_method="random",
        seed=seed,
        X_majority_leaf=None,
        verbose=False,
        use_kmeanspp=False,
    )

    # 2) 2N random from remaining (excluding matched)
    exclude = set(int(x) for x in match_ids)
    remaining = np.array([e for e in control_enrolids if int(e) not in exclude], dtype=np.int64)
    rng = np.random.RandomState(seed + 202603)
    perm = rng.permutation(remaining)
    extra_ids = perm[: 2 * N].astype(np.int64)

    ctrl_ids = np.concatenate([match_ids, extra_ids])
    train_df = pd.concat([cases, controls[controls["ENROLID"].isin(ctrl_ids)]], ignore_index=True)
    return train_df


def run_exp6_sampling(
    control_enrolids: np.ndarray,
    case_enrolids: np.ndarray,
    controls: pd.DataFrame,
    cases: pd.DataFrame,
    dnn_matrix_path: str,
    dnn_enrolids_path: str,
    pn_h5_path: str,
    N: int,
    M_pool: int,
    seed: int,
) -> pd.DataFrame:
    """Run Exp 6: 1N matched + 2N max_dispersed. Returns train_df (cases + selected controls)."""
    total_controls = EXP6_TOTAL_CONTROLS_MULT * N  # 3N
    STAGEA_KMAX = min(M_pool, len(control_enrolids), total_controls + 512)

    # 1) 1N matched
    match_ids = sample_stageB_matched_controls(
        control_enrolids,
        case_enrolids,
        dnn_matrix_path,
        dnn_enrolids_path,
        pn_h5_path,
        target_count=N,
        matching_ratio=1,
        M_pool=M_pool,
        seed_method="random",
        seed=seed,
        X_majority_leaf=None,
        verbose=False,
        use_kmeanspp=False,
    )

    # 2) 2N max-dispersed from remaining
    id_to_pos = {int(e): i for i, e in enumerate(np.load(dnn_enrolids_path))}
    extra_k = 2 * N
    disp_ids = sample_stageA_on_restricted_pool(
        control_enrolids,
        match_ids,
        dnn_matrix_path,
        dnn_enrolids_path,
        pn_h5_path,
        case_enrolids,
        extra_k,
        "random",
        seed,
        None,
        id_to_pos,
        verbose=False,
        use_kmeanspp=False,
    )

    ctrl_ids = np.unique(np.concatenate([match_ids, disp_ids]))
    if len(ctrl_ids) < total_controls:
        exclude = set(int(x) for x in ctrl_ids)
        remaining = np.array([e for e in control_enrolids if int(e) not in exclude], dtype=np.int64)
        extra = sample_random_controls(remaining, total_controls - len(ctrl_ids), seed + 9999)
        ctrl_ids = np.concatenate([ctrl_ids, extra])[:total_controls]

    train_df = pd.concat([cases, controls[controls["ENROLID"].isin(ctrl_ids)]], ignore_index=True)
    return train_df


def plot_majority_cost_by_metric(
    results_dir: str,
    csv_pattern: str = "exp*_*_s*.csv",
    target_col: str = "annual_cost_2018_deflated",
    filter_col: str = "top_2_pct_cost_2018",
) -> str:
    """
    Plot cost distribution (majority class) across methods.
    Exp 5 = 1N matched + 2N random; Exp 6 = 1N matched + 2N max-dispersed per metric.
    Returns path to saved plot.
    """
    base_path = Path(results_dir)
    all_files = list(base_path.glob("rnd_1to1_s*.csv")) + list(base_path.glob("ours_1to1_s*.csv")) + list(base_path.glob("exp5_s*.csv")) + list(base_path.glob("exp6_*_s*.csv"))
    if not all_files:
        print(f"No sampling CSVs in {results_dir}")
        return ""

    data_list = []
    for f_path in all_files:
        fname = f_path.name.lower()
        if "rnd_1to1_" in fname:
            method = "RND 1-1"
        elif "ours_1to1_" in fname:
            method = "Ours 1-1"
        elif "exp5_" in fname:
            method = "Exp 5 (1N matched + 2N random)"
        elif "exp6_" in fname:
            # Extract metric: exp6_euclidean_s0 -> Euclidean
            parts = fname.replace(".csv", "").split("_")
            if len(parts) >= 3:
                metric = parts[1]
                method = f"Exp 6 ({metric.capitalize()})"
            else:
                method = "Exp 6 (Unknown)"
        else:
            continue

        try:
            df = pd.read_csv(f_path, usecols=[target_col, filter_col])
            majority_df = df[df[filter_col] == 0].copy()
            majority_df["Method"] = method
            data_list.append(majority_df)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if not data_list:
        print("No data for plotting.")
        return ""

    full_df = pd.concat(data_list, ignore_index=True)
    # Order: Exp 5 first (baseline), then Exp 6 metrics alphabetically
    base_order = ["RND 1-1", "Ours 1-1", "Exp 5 (1N matched + 2N random)"]
    rest = sorted([m for m in full_df["Method"].unique() if m not in base_order])
    method_order = [m for m in base_order if m in full_df["Method"].unique()] + rest
    full_df["Method"] = pd.Categorical(full_df["Method"], categories=method_order, ordered=True)
    full_df = full_df.sort_values("Method")

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", n_colors=len(method_order))
    ax = sns.boxplot(data=full_df, x="Method", y=target_col, hue="Method", palette=dict(zip(method_order, palette)))
    plt.yscale("log")
    plt.title("Majority Class Cost Distribution: Exp 5 vs Exp 6 by Distance Metric\n(top_2_pct_cost_2018 == 0)", fontsize=14)
    plt.ylabel("Annual Cost 2018 (Log Scale $)")
    plt.xlabel("Sampling Method")
    plt.xticks(rotation=45, ha="right")
    plt.legend().set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "exp6_cost_distribution_by_metric.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Cost plot saved: {out_path}")
    return out_path


def get_median_majority_cost(csv_path: str, target_col: str = "annual_cost_2018_deflated", filter_col: str = "top_2_pct_cost_2018") -> float:
    """Return median annual cost for majority class (filter_col==0)."""
    df = pd.read_csv(csv_path, usecols=[target_col, filter_col])
    maj = df[df[filter_col] == 0][target_col]
    return float(maj.median())


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    metrics = [m.strip().lower() for m in args.metrics.split(",")]
    outdir = args.outdir
    results_dir = os.path.join(outdir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.distances_dir, exist_ok=True)

    print("=" * 80)
    print("EXP 6: Distance Metric Ablation")
    print("=" * 80)
    print(f"  Seeds: {seeds}")
    print(f"  Metrics: {metrics}")
    print(f"  Two-stage: k-center then match | random init | no k-means++ | no adaptive pool")
    print()

    df = pd.read_parquet(args.parquet_path)

    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    COST_COLUMNS = [
        col for col in df.columns
        if ("cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower()
            or "decreasing" in col.lower() or "skewness" in col.lower() or "kurtosis" in col.lower()
            or "cv" in col.lower() or "range" in col.lower())
        and "2018" not in col
    ]
    AUXILIARY_COST_COLUMNS = [col for col in df.columns if col.startswith("comorbidity_only") or col.startswith("msk_procedure")]
    exclude_cols = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    if args.feature_set == "medical_only":
        exclude_cols += COST_COLUMNS
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    elif args.feature_set == "all_cost":
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    else:
        exclude_cols += AUXILIARY_COST_COLUMNS
        feature_cols = [c for c in df.columns if c not in exclude_cols]

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
    M_pool = args.M_pool if args.M_pool is not None else n_controls // 2

    print(f"Train: {len(train_pd):,}  Val: {len(val_pd):,}  Test: {len(test_pd):,}")
    print(f"  N (cases): {N:,}  Controls: {n_controls:,}  M_pool: {M_pool:,}")
    print()

    metric_median_costs = {}
    K = EXP6_TOTAL_CONTROLS_MULT * N

    # --- RND 1-1: N cases + N random controls ---
    print("\n--- RND 1-1: 1:1 random undersampling ---")
    for seed in seeds:
        rnd_path = os.path.join(results_dir, f"rnd_1to1_s{seed}.csv")
        if args.resume and os.path.exists(rnd_path):
            print(f"  RND 1-1 s{seed}: using cached CSV")
        else:
            rnd_train = run_rnd_1to1_sampling(control_enrolids, cases, controls, N, seed)
            rnd_train.to_csv(rnd_path, index=False)
            print(f"  RND 1-1 s{seed}: saved {rnd_path}")
        metric_median_costs[f"rnd1to1_s{seed}"] = get_median_majority_cost(rnd_path)

    # --- Ours 1-1 and Exp 5/6 use Gower ---
    pn_h5_gower, dnn_mat_gower, dnn_ids_gower = ensure_distances_for_metric(
        "gower", cases, controls, feature_cols,
        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
        args.distances_dir,
    )

    # --- Ours 1-1: 1:1 k-center then match (Gower) ---
    print("\n--- Ours 1-1: 1:1 k-center then match (Gower) ---")
    for seed in seeds:
        ours_path = os.path.join(results_dir, f"ours_1to1_s{seed}.csv")
        if args.resume and os.path.exists(ours_path):
            print(f"  Ours 1-1 s{seed}: using cached CSV")
        else:
            ours_train = run_ours_1to1_sampling(
                control_enrolids, case_enrolids, controls, cases,
                dnn_mat_gower, dnn_ids_gower, pn_h5_gower, N, M_pool, seed,
            )
            ours_train.to_csv(ours_path, index=False)
            print(f"  Ours 1-1 s{seed}: saved {ours_path}")
        metric_median_costs[f"ours1to1_s{seed}"] = get_median_majority_cost(ours_path)

    # --- Exp 5 (1N matched + 2N random) - Gower ---
    print("\n--- Exp 5: 1N matched + 2N random (Gower) ---")
    for seed in seeds:
        exp5_path = os.path.join(results_dir, f"exp5_s{seed}.csv")
        if args.resume and os.path.exists(exp5_path):
            print(f"  Exp 5 s{seed}: using cached CSV")
        else:
            exp5_train = run_exp5_sampling(
                control_enrolids, case_enrolids, controls, cases,
                dnn_mat_gower, dnn_ids_gower, pn_h5_gower, N, M_pool, seed,
            )
            exp5_train.to_csv(exp5_path, index=False)
            print(f"  Exp 5 s{seed}: saved {exp5_path}")
        med = get_median_majority_cost(exp5_path)
        metric_median_costs[f"exp5_s{seed}"] = med

    # --- Per-metric Exp 6 sampling (1N matched + 2N max-dispersed) ---
    for metric in metrics:
        print(f"\n--- Metric: {metric} ---")
        try:
            pn_h5, dnn_mat, dnn_ids = ensure_distances_for_metric(
                metric, cases, controls, feature_cols,
                CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                args.distances_dir,
            )
        except Exception as e:
            print(f"  ERROR precomputing distances for {metric}: {e}")
            traceback.print_exc()
            continue

        for seed in seeds:
            csv_path = os.path.join(results_dir, f"exp6_{metric}_s{seed}.csv")
            if args.resume and os.path.exists(csv_path):
                print(f"  {metric} s{seed}: using cached CSV")
            else:
                try:
                    train_df = run_exp6_sampling(
                        control_enrolids, case_enrolids, controls, cases,
                        dnn_mat, dnn_ids, pn_h5, N, M_pool, seed,
                    )
                    train_df.to_csv(csv_path, index=False)
                    print(f"  {metric} s{seed}: saved {csv_path}")
                except Exception as e:
                    print(f"  ERROR {metric} s{seed}: {e}")
                    traceback.print_exc()
                    continue

            med = get_median_majority_cost(csv_path)
            metric_median_costs[f"{metric}_s{seed}"] = med

    # Aggregate median per method (mean over seeds)
    metric_medians = {}
    for key in ["rnd1to1", "ours1to1", "exp5"]:
        vals = [metric_median_costs.get(f"{key}_s{s}") for s in seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            metric_medians[key] = np.mean(vals)
    for metric in metrics:
        vals = [metric_median_costs.get(f"{metric}_s{s}") for s in seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            metric_medians[metric] = np.mean(vals)
    if metric_medians:
        print(f"\n--- Cost medians: {metric_medians} ---")

    # --- Plot cost distribution ---
    if not args.skip_plot:
        plot_majority_cost_by_metric(results_dir)

    # --- Train OCT on all methods (RND 1-1, Ours 1-1, Exp 5, Exp 6 per metric) ---
    if not args.skip_oct:
        seed = seeds[0]
        train_configs = [
            ("rnd_1to1", "RND 1-1"),
            ("ours_1to1", "Ours 1-1"),
            ("exp5", "Exp 5"),
        ]
        for csv_base, label in train_configs:
            csv_path = os.path.join(results_dir, f"{csv_base}_s{seed}.csv")
            if os.path.exists(csv_path):
                train_df = pd.read_csv(csv_path)
                print(f"\nTraining OCT on {label}...")
                try:
                    m = train_and_evaluate_oct(
                        train_df, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"{csv_base}_s{seed}", TRAIN_TEST_SEED,
                    )
                    print(f"  PR-AUC: {m.get('pr_auc', 0):.4f}  AUC: {m.get('auc', 0):.4f}")
                except Exception as e:
                    print(f"  OCT ERROR ({label}): {e}")
                    traceback.print_exc()
            else:
                print(f"  {label} train CSV not found: {csv_path}")
        for metric in metrics:
            exp6_path = os.path.join(results_dir, f"exp6_{metric}_s{seed}.csv")
            if os.path.exists(exp6_path):
                train_df = pd.read_csv(exp6_path)
                print(f"\nTraining OCT on Exp 6 ({metric})...")
                try:
                    m = train_and_evaluate_oct(
                        train_df, val_pd, test_pd, feature_cols, target_col,
                        CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                        results_dir, f"exp6_{metric}_s{seed}", TRAIN_TEST_SEED,
                    )
                    print(f"  PR-AUC: {m.get('pr_auc', 0):.4f}  AUC: {m.get('auc', 0):.4f}")
                except Exception as e:
                    print(f"  OCT ERROR (Exp 6 {metric}): {e}")
                    traceback.print_exc()
            else:
                print(f"  Exp 6 {metric} train CSV not found: {exp6_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

