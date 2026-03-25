#!/usr/bin/env python
"""
Precompute MSK Gower P-N + D-N-N (v2 kernel). MSK-specific: parquet, flexible top-% target.

  cd msk_analysis
  python scripts/precompute_msk_gower_distances.py --parquet_path msk_2017_18_full.parquet --outdir ../../../scratch/msk_analysis/precomputed_distances_gower --target_top_pct 2

Or pass an explicit target column name (created from cost if missing):

  python scripts/precompute_msk_gower_distances.py --target_col top_1_pct_cost_2018 ...

Warped continuous-only Gower (log-clip, notebook-style FEAT_CONT):

  python scripts/precompute_msk_gower_distances.py --parquet_path ... --outdir ... --warp
"""
from __future__ import annotations

import argparse
import os
import sys

_scripts = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(_scripts, "../.."))
sys.path.insert(0, _scripts)
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_bin_flag_columns_with_provenance,
    get_cat_columns,
    get_true_num_columns,
)
from public.precompute_gower_distances import (
    get_feature_columns,
    precompute_gower_pn_and_dnn,
)
from msk_gower_target import ensure_msk_top_pct_cost_target, target_column_name_for_top_pct


def msk_notebook_style_feat_cont(
    train_pd: pd.DataFrame,
    df: pd.DataFrame,
    target_col: str,
    feature_set: str,
) -> list:
    """
    Continuous columns aligned with two_stage_log_clip_kcenter: FEAT_COLS minus binary
    (nunique<=2 on train), intersected with get_feature_columns(feature_set).
    """
    BIN = get_bin_flag_columns(df)
    CAT = get_cat_columns(df)
    base_feat = get_feature_columns(df, feature_set, target_col)
    feat_cols = [
        c
        for c in train_pd.columns
        if c not in BIN
        and c not in CAT
        and c not in [target_col, "ENROLID"]
        and "2018" not in c
        and c in base_feat
    ]
    return [c for c in feat_cols if train_pd[c].nunique(dropna=False) > 2]


def parse_args():
    p = argparse.ArgumentParser(description="MSK Gower P-N + D-N-N precompute (v2)")
    p.add_argument("--parquet_path", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument(
        "--target_top_pct",
        type=float,
        default=None,
        help="Top cost %% (e.g. 2, 1, 0.5). Ignored if --target_col is set.",
    )
    p.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Override target column, e.g. top_1_pct_cost_2018. Derived from cost if absent.",
    )
    p.add_argument(
        "--feature_set",
        type=str,
        default="all_cost",
        choices=["medical_only", "all_cost", "less_cost"],
    )
    p.add_argument("--seed", type=int, default=123, help="Train split + global_dnn_seed_{seed}_gower")
    p.add_argument("--dnn_batch_size", type=int, default=750)
    p.add_argument("--dnn_full_matrix", action="store_true", help="Compute full D-N-N in memory (O(n²) RAM), then save")
    p.add_argument("--dnn_save_format", type=str, default="npy", choices=["npy", "npz"], help="Format when --dnn_full_matrix")
    p.add_argument("--skip_pn", action="store_true")
    p.add_argument("--skip_dnn", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gower_dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument(
        "--warp",
        action="store_true",
        help="Log-clip warp continuous columns only (notebook-style FEAT_CONT), then Gower P-N + D-N-N",
    )
    p.add_argument(
        "--pn_continuous_cols",
        type=str,
        default=None,
        help="Comma-separated PN columns (default: all FEAT_CONT); must be subset of train columns",
    )
    p.add_argument(
        "--dnn_continuous_cols",
        type=str,
        default=None,
        help="Comma-separated DNN columns (default: same as PN)",
    )
    p.add_argument(
        "--warp_continuous_cols",
        type=str,
        default=None,
        help="Comma subset of PN∪DNN to warp (default: warp all of that union)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.target_col:
        target_col = args.target_col
    elif args.target_top_pct is not None:
        target_col = target_column_name_for_top_pct(args.target_top_pct)
    else:
        target_col = target_column_name_for_top_pct(2.0)

    df = pd.read_parquet(args.parquet_path)
    ensure_msk_top_pct_cost_target(df, target_col)

    train_ids, _, train_pd, _ = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.seed
    )
    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]

    BIN = get_bin_flag_columns(df)
    CAT = get_cat_columns(df)
    feature_cols = get_feature_columns(df, args.feature_set, target_col)

    if args.warp:
        from public.precompute_gower_distances_warped import precompute_gower_pn_and_dnn_warped

        feat_cont = msk_notebook_style_feat_cont(train_pd, df, target_col, args.feature_set)
        if not feat_cont:
            raise ValueError(
                "Warped precompute: no continuous columns (nunique>2) after "
                "notebook-style FEAT_COLS ∩ feature_set."
            )
        if args.pn_continuous_cols:
            pn_cols = [c.strip() for c in args.pn_continuous_cols.split(",") if c.strip()]
        else:
            pn_cols = list(feat_cont)

        if args.dnn_continuous_cols:
            dnn_cols = [c.strip() for c in args.dnn_continuous_cols.split(",") if c.strip()]
        elif args.pn_continuous_cols:
            dnn_cols = list(pn_cols)
        else:
            dnn_cols = list(feat_cont)
        for name, cols in [("PN", pn_cols), ("DNN", dnn_cols)]:
            missing = [c for c in cols if c not in train_pd.columns]
            if missing:
                raise ValueError(f"{name} columns not in training frame: {missing}")

        warp_subset = None
        if args.warp_continuous_cols:
            warp_subset = [c.strip() for c in args.warp_continuous_cols.split(",") if c.strip()]

        ucols = sorted(set(pn_cols) | set(dnn_cols))
        train_feat = pd.concat(
            [cases[ucols], controls[ucols]], ignore_index=True
        )
        _, bin_verified = get_bin_flag_columns_with_provenance(train_feat)

        pn_h5, dnn_npy, enrol = precompute_gower_pn_and_dnn_warped(
            cases,
            controls,
            pn_cols,
            dnn_cols,
            warp_continuous_cols=warp_subset,
            bin_cols_verified_by_values=bin_verified,
            outdir=args.outdir,
            dnn_subdir_seed=args.seed,
            gower_dtype=args.gower_dtype,
            dnn_batch_size=args.dnn_batch_size,
            skip_pn=args.skip_pn,
            skip_dnn=args.skip_dnn,
            resume=args.resume,
            dnn_full_matrix=args.dnn_full_matrix,
            dnn_save_format=args.dnn_save_format,
        )
    else:
        NUM = get_true_num_columns(df, CAT, BIN)
        train_feat = pd.concat(
            [cases[feature_cols], controls[feature_cols]], ignore_index=True
        )
        _, bin_verified = get_bin_flag_columns_with_provenance(train_feat)

        pn_h5, dnn_npy, enrol = precompute_gower_pn_and_dnn(
            cases,
            controls,
            feature_cols,
            CAT,
            NUM,
            BIN,
            bin_cols_verified_by_values=bin_verified,
            outdir=args.outdir,
            dnn_subdir_seed=args.seed,
            gower_dtype=args.gower_dtype,
            dnn_batch_size=args.dnn_batch_size,
            skip_pn=args.skip_pn,
            skip_dnn=args.skip_dnn,
            resume=args.resume,
            dnn_full_matrix=args.dnn_full_matrix,
            dnn_save_format=args.dnn_save_format,
        )
    print("Done.")
    print(f"  P-N: {pn_h5}")
    print(f"  D-N-N: {dnn_npy}")
    print(f"  Enrolids: {enrol}")


if __name__ == "__main__":
    main()
