"""
Ensure P-N + D-N-N distance files exist for ablation metrics (MSK train split).

  * **gower**: ``public.precompute_gower_distances`` v2 → same paths as precompute_msk_gower_distances.
  * **tfidf_svd_cosine_qcost**: Gower P-N + TF-IDF embedding DNN (cosine).
  * **jaccard / hamming**: binary features only.
  * **other** (euclidean, manhattan, cosine, …): sklearn preprocessor + batched distances.

Replaces legacy ``exp6_distance_metric_ablation.ensure_distances_for_metric``.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

_scripts = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_scripts, ".."))
sys.path.insert(0, _parent)

from public.model_IAI import get_bin_flag_columns_with_provenance
from public.precompute_distances import (
    compute_distances_batched,
    get_preprocessor,
    precompute_leaf_dnn_hdf5,
    precompute_leaf_dnn_memmap,
    save_distances_hdf5,
)
import public.precompute_gower_distances as gower_mod
from public.dnn_matrix_storage import (
    dnn_enrolids_npy_path,
    dnn_matrix_npy_path,
    dnn_matrix_storage_exists,
    ensure_dnn_matrix_npy,
)

TRAIN_TEST_SEED = 123


def _ensure_gower(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    distances_dir: str,
    dnn_batch_size: int,
    gower_dtype: str,
) -> Tuple[str, str, str]:
    dnn_dir = os.path.join(distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_gower")
    pn_h5 = os.path.join(distances_dir, "distances_majority_minority_gower.h5")
    enrol = dnn_enrolids_npy_path(dnn_dir)
    if (
        os.path.isfile(pn_h5)
        and dnn_matrix_storage_exists(dnn_dir)
        and os.path.isfile(enrol)
    ):
        return pn_h5, ensure_dnn_matrix_npy(dnn_dir), enrol

    train_feat = pd.concat(
        [cases[feature_cols], controls[feature_cols]], ignore_index=True
    )
    _, verified = get_bin_flag_columns_with_provenance(train_feat)
    gower_mod.precompute_gower_pn_and_dnn(
        cases,
        controls,
        feature_cols,
        cat_cols,
        num_cols,
        bin_cols,
        bin_cols_verified_by_values=verified,
        outdir=distances_dir,
        dnn_subdir_seed=TRAIN_TEST_SEED,
        gower_dtype=gower_dtype,
        dnn_batch_size=dnn_batch_size,
        resume=False,
    )
    return pn_h5, ensure_dnn_matrix_npy(dnn_dir), enrol


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
    dnn_batch_size: int = 750,
    gower_dtype: str = "float16",
) -> Tuple[str, str, str]:
    """
    Returns ``(pn_h5, dnn_matrix_path, dnn_enrolids_path)``.
    """
    os.makedirs(distances_dir, exist_ok=True)
    metric = distance_metric.strip().lower()
    pn_h5 = os.path.join(distances_dir, f"distances_majority_minority_{metric}.h5")
    dnn_dir = os.path.join(distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_{metric}")
    dnn_npy = dnn_matrix_npy_path(dnn_dir) if metric == "gower" else os.path.join(
        dnn_dir, "leaf_global_dnn_matrix.npy"
    )
    dnn_h5 = os.path.join(dnn_dir, "leaf_global_dnn_matrix.h5")
    dnn_enrolids = dnn_enrolids_npy_path(dnn_dir)

    if metric == "gower":
        return _ensure_gower(
            cases,
            controls,
            feature_cols,
            cat_cols,
            num_cols,
            bin_cols,
            distances_dir,
            dnn_batch_size,
            gower_dtype,
        )

    dnn_matrix_path = dnn_h5 if use_hdf5_dnn else dnn_npy
    if (
        os.path.isfile(pn_h5)
        and os.path.isfile(dnn_matrix_path)
        and os.path.isfile(dnn_enrolids)
    ):
        return pn_h5, dnn_matrix_path, dnn_enrolids

    if metric == "tfidf_svd_cosine_qcost":
        gower_pn = os.path.join(distances_dir, "distances_majority_minority_gower.h5")
        if not os.path.isfile(gower_pn):
            _ensure_gower(
                cases,
                controls,
                feature_cols,
                cat_cols,
                num_cols,
                bin_cols,
                distances_dir,
                dnn_batch_size,
                gower_dtype,
            )
        pn_h5 = gower_pn
        if (
            os.path.isfile(pn_h5)
            and os.path.isfile(dnn_matrix_path)
            and os.path.isfile(dnn_enrolids)
        ):
            return pn_h5, dnn_matrix_path, dnn_enrolids
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
        print(
            f"  tfidf_svd: n_codes={meta['n_codes']}, svd_dim={meta['svd_dim']}, "
            f"d_cost={meta['d_cost']}, alpha={meta['alpha']}"
        )
        run_cosine_distance_diagnostics(Z)
        if use_hdf5_dnn:
            dnn_matrix_path, dnn_enrolids = precompute_leaf_dnn_hdf5(
                X_majority_leaf=Z,
                majority_enrolids_leaf=enrolids,
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric="cosine",
            )
        else:
            dnn_matrix_path, dnn_enrolids = precompute_leaf_dnn_memmap(
                X_majority_leaf=Z,
                majority_enrolids_leaf=enrolids,
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric="cosine",
            )
        return pn_h5, dnn_matrix_path, dnn_enrolids

    if metric in ("jaccard", "hamming"):
        bin_in = [c for c in bin_cols if c in feature_cols]
        if not bin_in:
            raise ValueError(f"{metric} needs binary columns in feature_cols")
        X_min = cases[bin_in].values.astype(np.float64)
        X_maj = controls[bin_in].values.astype(np.float64)
    else:
        pre = get_preprocessor(
            X=pd.concat([cases[feature_cols], controls[feature_cols]], ignore_index=True),
            cat_cols=cat_cols,
            num_cols=num_cols,
            binary_cols=bin_cols,
            verbose=False,
        )
        X_min = pre.fit_transform(cases[feature_cols])
        X_maj = pre.transform(controls[feature_cols])

    if not os.path.isfile(pn_h5):
        d_pn = compute_distances_batched(
            X_maj, X_min, batch_size=1000, dtype=np.float32, metric=metric
        )
        save_distances_hdf5(
            d_pn,
            controls["ENROLID"].values.astype(np.int64),
            cases["ENROLID"].values.astype(np.int64),
            pn_h5,
        )
    if not os.path.isfile(dnn_matrix_path) or not os.path.isfile(dnn_enrolids):
        if use_hdf5_dnn:
            dnn_matrix_path, dnn_enrolids = precompute_leaf_dnn_hdf5(
                X_majority_leaf=X_maj,
                majority_enrolids_leaf=controls["ENROLID"].values.astype(np.int64),
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric=metric,
            )
        else:
            dnn_matrix_path, dnn_enrolids = precompute_leaf_dnn_memmap(
                X_majority_leaf=X_maj,
                majority_enrolids_leaf=controls["ENROLID"].values.astype(np.int64),
                out_dir=dnn_dir,
                leaf_id="global",
                batch_size=750,
                metric=metric,
            )
    return pn_h5, dnn_matrix_path, dnn_enrolids
