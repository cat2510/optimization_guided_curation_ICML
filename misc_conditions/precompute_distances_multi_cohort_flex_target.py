#!/usr/bin/env python3
"""
precompute_distances_multi_cohort_flex_target.py
================================================

Variant of precompute_distances_multi_cohort.py with flexible target selection,
aligned with vanilla_oct_multi_cohort_flex_target.py.

Supports --target_pct (5, 2, 1, 0.5) and --target_col override.
Merges annual_cost_*_deflated from _with_meds when using 100feat parquets.
distances_dir_template includes {target_suffix} so different targets use separate dirs.

Sampling for giant cohorts is aligned with vanilla_oct_multi_cohort_flex_target:
  when len(df) > 500_000, keep exactly 250_000 rows (stratified, same seed).

  python precompute_distances_multi_cohort_flex_target.py --metric gower --codes F32 --target_pct 1

"""
import os
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

# Gower kernel: public.precompute_gower_distances (my_projects/public); parent_dir must be on path.

from sklearn.metrics import pairwise_distances
import importlib
import public.precompute_distances
importlib.reload(public.precompute_distances)
from public.precompute_distances import (
    get_preprocessor,
    compute_distances_batched,
    save_distances_hdf5,
    precompute_leaf_dnn_memmap,
)

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)

from vanilla_oct_multi_cohort_flex_target import (
    pick_target_and_features,
    _merge_annual_cost_from_with_meds,
    VALID_TARGET_PCTS,
    DEFAULT_TARGET_PCT,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--codes", type=str,
                   default="E78, E66",
                   help="Comma-separated ICD10 root codes")

    p.add_argument("--features_dir", type=str,
                   default="/Users/cat2510/my_projects/misc_conditions/misc_conditions_features_with_meds",
                   help="Directory containing <CODE>_features_<baseline>_<outcome>([_with_meds]).parquet")

    p.add_argument("--baseline_year", type=int, default=2017)
    p.add_argument("--outcome_year", type=int, default=2018)

    p.add_argument("--target_pct", type=float, default=DEFAULT_TARGET_PCT,
                   choices=VALID_TARGET_PCTS,
                   help=f"Top-cost percentile: one of {VALID_TARGET_PCTS}. Default: {DEFAULT_TARGET_PCT}%%")
    p.add_argument("--target_col", type=str, default=None,
                   help="Override: explicit target column (e.g. top_1_pct_cost_2018). If set, ignores --target_pct.")

    # Where to write distances; template uses {code} and {target_suffix} (e.g. top2pct, top0_5pct)
    p.add_argument("--distances_dir_template", type=str,
                   default="/Users/cat2510/scratch/precomputed_distances_{code}_top{target_suffix}pct",
                   help="Where to write distances per cohort. Use {code} and {target_suffix} (auto from --target_pct).")

    p.add_argument("--train_test_seed", type=int, default=123)

    p.add_argument("--pn_batch_size", type=int, default=1000)
    p.add_argument("--dnn_batch_size", type=int, default=750)
    p.add_argument("--metric", type=str, default="gower",
                   help="Distance metric. Gower uses public.precompute_gower_distances v2 kernel.")

    p.add_argument("--use_hdf5_dnn", action="store_true",
                   help="save as HDF5 (compressed) instead of .npy memmap.")

    p.add_argument("--feature_regex", type=str, default="",
                   help="Optional regex to keep only matching feature columns (applied after leakage exclusion). "
                        "Example: '^(med_|has_)'")

    p.add_argument("--overwrite", action="store_true",
                   help="Force recomputation even if files exist and ENROLIDs match")

    p.add_argument("--gower_dtype", type=str, default="float16", choices=["float32", "float16"],
                   help="Gower feature + distance dtype (default float16). Use float32 for max precision.")

    return p.parse_args()


def _target_suffix_from_args(args) -> str:
    """Derive target_suffix for distances_dir_template: '2', '1', '5', '0_5'."""
    if args.target_col:
        m = re.match(r"top_(\d+(?:_\d+)?)_pct_cost_", args.target_col)
        if m:
            return m.group(1)  # e.g. "0_5", "1", "2"
        return "custom"
    return "0_5" if args.target_pct == 0.5 else str(int(args.target_pct))


def resolve_feature_path(features_dir: Path, code: str, baseline_year: int, outcome_year: int) -> Path:
    """Same candidate order as vanilla_oct_multi_cohort_flex_target."""
    candidates = [
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_100feat.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No features parquet found for {code}. Tried: {candidates}")


def maybe_compute_pn_gower(pn_h5_path, X_maj, X_min, maj_ids, min_ids, bin_col_indices, ranges, args, col_names=None):
    if pn_h5_path.exists() and not args.overwrite:
        return
    import public.precompute_gower_distances as gower_module
    print(f"  Computing P-N (gower v2): {X_maj.shape[0]:,} x {X_min.shape[0]:,}")
    gdt = gower_module._as_gower_dtype(getattr(args, "gower_dtype", "float16"))
    dist_pn = gower_module.compute_gower_pn_v2(
        X_maj, X_min, bin_col_indices, ranges, col_names=col_names, out_dtype=gdt,
    )
    if gdt == np.dtype(np.float32):
        save_distances_hdf5(dist_pn, maj_ids, min_ids, str(pn_h5_path))
    else:
        gower_module._save_distances_hdf5_inline(
            dist_pn, maj_ids, min_ids, str(pn_h5_path), distances_dtype=gdt,
        )


def maybe_compute_dnn_gower(dnn_out_dir, X_maj, maj_ids, bin_col_indices, ranges, args, col_names=None):
    suffix = ".h5" if getattr(args, "use_hdf5_dnn", False) else ".npy"
    dnn_matrix_path = dnn_out_dir / f"leaf_global_dnn_matrix{suffix}"
    dnn_enrolids_path = dnn_out_dir / "leaf_global_dnn_enrolids.npy"
    if dnn_matrix_path.exists() and not args.overwrite:
        if not dnn_enrolids_path.exists():
            np.save(dnn_enrolids_path, maj_ids)
            print(f"  Saved missing enrolids: {dnn_enrolids_path}")
        return
    import public.precompute_gower_distances as gower_module
    print(f"  Computing D-N-N (gower v2, batched)...")
    gdt = gower_module._as_gower_dtype(getattr(args, "gower_dtype", "float16"))
    gower_module.precompute_gower_dnn_v2(
        X_maj, bin_col_indices, ranges,
        out_dir=str(dnn_out_dir),
        batch_size=args.dnn_batch_size,
        use_hdf5=getattr(args, "use_hdf5_dnn", False),
        col_names=col_names,
        out_dtype=gdt,
    )
    np.save(dnn_enrolids_path, maj_ids)


def run_one(code: str, args):
    print(f"\n{'='*88}\nCOHORT {code}\n{'='*88}")

    feat_path = resolve_feature_path(Path(args.features_dir), code, args.baseline_year, args.outcome_year)
    print(f"  Features: {feat_path}")
    df = pd.read_parquet(str(feat_path))

    outcome_years = [2018, 2019] if code in ("I25", "I50") else args.outcome_year

    # 100feat parquets lack annual_cost_*_deflated; merge from _with_meds if needed
    df = _merge_annual_cost_from_with_meds(
        df, feat_path, Path(args.features_dir), code, args.baseline_year, args.outcome_year
    )

    target_col, feat_cols = pick_target_and_features(
        df,
        args.baseline_year,
        outcome_years,
        target_col=args.target_col,
        target_pct=args.target_pct,
        feature_regex=args.feature_regex,
    )

    # Align with vanilla_oct_multi_cohort_flex_target: same 250k subset for giant cohorts
    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df, train_size=250_000, stratify=df[target_col], random_state=args.train_test_seed
        )
        print(f"  Sampled to {len(df):,} rows (stratified on {target_col})")
    else:
        print(f"  Using {len(df):,} rows")

    print(f"  Target: {target_col} | Features: {len(feat_cols)}")

    _, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.train_test_seed
    )

    cases_mask = train_pd[target_col] == 1
    controls_mask = train_pd[target_col] == 0
    maj_ids = train_pd.loc[controls_mask, "ENROLID"].values.astype(np.int64)
    min_ids = train_pd.loc[cases_mask, "ENROLID"].values.astype(np.int64)

    target_suffix = _target_suffix_from_args(args)
    dist_dir = Path(args.distances_dir_template.format(code=code, target_suffix=target_suffix))
    dist_dir.mkdir(parents=True, exist_ok=True)
    pn_h5_path = dist_dir / "distances_majority_minority.h5"
    dnn_out_dir = dist_dir / f"global_dnn_seed_{args.train_test_seed}"

    if args.metric == "gower":
        import public.precompute_gower_distances as gower_module
        cases_pd = train_pd.loc[cases_mask]
        controls_pd = train_pd.loc[controls_mask]
        X_train_raw = train_pd[feat_cols]
        BIN_FLAG_COLUMNS = get_bin_flag_columns(X_train_raw)
        CAT_COLUMNS = get_cat_columns(X_train_raw)
        TRUE_NUM_COLUMNS = get_true_num_columns(X_train_raw, CAT_COLUMNS, BIN_FLAG_COLUMNS)
        gdt = gower_module._as_gower_dtype(getattr(args, "gower_dtype", "float16"))
        X_majority, X_minority, ranges, bin_col_indices, col_names = gower_module.build_gower_feature_matrices(
            cases_pd, controls_pd, feat_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
            feature_dtype=gdt,
        )
        maybe_compute_pn_gower(pn_h5_path, X_majority, X_minority, maj_ids, min_ids, bin_col_indices, ranges, args, col_names=col_names)
        maybe_compute_dnn_gower(dnn_out_dir, X_majority, maj_ids, bin_col_indices, ranges, args, col_names=col_names)

    print(f"  ✓ Cohort {code} Complete.")


def main():
    """
  Gower with flexible target (aligned with vanilla_oct_multi_cohort_flex_target):
  python precompute_distances_multi_cohort_flex_target.py \\
    --metric gower \\
    --codes E78,E66 \\
    --target_pct 2 \\
    --features_dir /path/to/features \\
    --distances_dir_template /scratch/precomputed_distances_{code}_top{target_suffix}pct \\
    --train_test_seed 123

  Use --target_pct 1 for top 1% cost, --target_pct 0.5 for top 0.5%.
  Requires my_projects on path (../ from misc_conditions) for public.* and numexpr for Gower.
  """
    args = parse_args()
    codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]

    target_suffix = _target_suffix_from_args(args)
    print(f"Processing Codes: {codes}")
    print(f"Target: {args.target_col or f'top {args.target_pct}%'}")
    print(f"Distances dir template: {args.distances_dir_template} (target_suffix={target_suffix})")

    for code in codes:
        try:
            run_one(code, args)
        except Exception as e:
            print(f"\nERROR in cohort {code}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
