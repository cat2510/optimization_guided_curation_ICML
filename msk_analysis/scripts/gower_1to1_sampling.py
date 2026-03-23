"""
Shared RND 1:1 vs Gower two-stage 1:1 control sampling (MSK).

Used by run_1to1_sampling_gower.py, ensure_distances-free hybrid eval scripts, etc.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_scripts = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_scripts, ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from experiments_compare_random_vs_curation import (
    sample_random_controls,
    sample_stageB_matched_controls,
)


def run_rnd_1to1_sampling(
    control_enrolids: np.ndarray,
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    n_cases: int,
    seed: int,
) -> pd.DataFrame:
    rnd_ids = sample_random_controls(control_enrolids, n_cases, seed)
    return pd.concat([cases, controls[controls["ENROLID"].isin(rnd_ids)]], ignore_index=True)


def run_ours_1to1_sampling(
    control_enrolids: np.ndarray,
    case_enrolids: np.ndarray,
    controls: pd.DataFrame,
    cases: pd.DataFrame,
    dnn_matrix_path: str,
    dnn_enrolids_path: str,
    pn_h5_path: str,
    n_cases: int,
    m_pool: int,
    seed: int,
    seed_method: str = "random",
) -> pd.DataFrame:
    match_ids = sample_stageB_matched_controls(
        control_enrolids,
        case_enrolids,
        dnn_matrix_path,
        dnn_enrolids_path,
        pn_h5_path,
        target_count=n_cases,
        matching_ratio=1,
        M_pool=m_pool,
        seed_method=seed_method,
        seed=seed,
        use_kmeanspp=False,
        verbose=False,
    )
    return pd.concat(
        [cases, controls[controls["ENROLID"].isin(match_ids)]],
        ignore_index=True,
    )
