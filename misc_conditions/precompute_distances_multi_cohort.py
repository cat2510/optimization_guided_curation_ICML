#!/usr/bin/env python3
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

from pyspark.sql import SparkSession


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--codes", type=str,
                   default="I25,I50", #E66,F32,  ## E78 is GIANT!!
                   # E10, E11, C50 are in scratch folder! C61 in current folder
                   help="Comma-separated ICD10 root codes")

    p.add_argument("--features_dir", type=str,
                   default="/Users/cat2510/my_projects/misc_conditions/misc_conditions_features_with_meds",
                   help="Directory containing <CODE>_features_<baseline>_<outcome>([_with_meds]).parquet")

    p.add_argument("--baseline_year", type=int, default=2017)
    p.add_argument("--outcome_year", type=int, default=2018)

    # Where to write distances; uses template to match later training script
    p.add_argument("--distances_dir_template", type=str,
                   default="./precomputed_distances_{code}_with_cost_features",
                   help="Where to write distances per cohort")

    p.add_argument("--train_test_seed", type=int, default=123)

    p.add_argument("--pn_batch_size", type=int, default=1000)
    p.add_argument("--dnn_batch_size", type=int, default=750)
    p.add_argument("--metric", type=str, default="euclidean")

    # Feature selection knobs
    p.add_argument("--feature_regex", type=str, default="",
                   help="Optional regex to keep only matching feature columns (applied after leakage exclusion). "
                        "Example: '^(med_|has_)'")

    p.add_argument("--overwrite", action="store_true",
                   help="Force recomputation even if files exist and ENROLIDs match")

    return p.parse_args()


def resolve_feature_path(features_dir: Path, code: str, baseline_year: int, outcome_year: int) -> Path:
    candidates = [
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_100feat.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No features parquet found for {code}. Tried: {candidates}")


def ensure_target(df: pd.DataFrame, outcome_year: int) -> str:
    target = f"top_2_pct_cost_{outcome_year}"
    annual = f"annual_cost_{outcome_year}_deflated"

    if target in df.columns:
        return target

    if annual in df.columns:
        thr = float(df[annual].quantile(0.98))
        df[target] = (df[annual] >= thr).astype(int)
        print(f"  Created {target} from {annual} @98th pct = {thr:,.2f}")
        return target

    raise ValueError(f"Need either '{target}' or '{annual}' in features parquet.")


def pick_feature_cols(df: pd.DataFrame, target_col: str, outcome_year: int, feature_regex: str) -> list[str]:
    # leakage guard: remove outcome-year columns + id + target
    exclude = ["ENROLID", target_col] + [c for c in df.columns if str(outcome_year) in c]
    cols = [c for c in df.columns if c not in exclude]

    if feature_regex:
        pat = re.compile(feature_regex)
        cols = [c for c in cols if pat.search(c)]

    if not cols:
        raise ValueError("No feature columns left after exclusions/regex.")
    return cols




# Keep only the logic-specific helpers that aren't in your utility file
def _same_id_set(ids_a: np.ndarray, ids_b: np.ndarray) -> bool:
    if ids_a.shape != ids_b.shape: return False
    return np.array_equal(ids_a, ids_b)

def maybe_compute_pn(pn_h5_path, X_maj, X_min, maj_ids, min_ids, args):
    if pn_h5_path.exists() and not args.overwrite:
        # ID validation logic...
        return

    print(f"  Computing PN: {X_maj.shape[0]:,} x {X_min.shape[0]:,}")
    # Use the utility function
    dist_pn = compute_distances_batched(
        X_maj, X_min, batch_size=args.pn_batch_size, metric=args.metric
    )
    save_distances_hdf5(dist_pn, maj_ids, min_ids, str(pn_h5_path))

def maybe_compute_dnn(dnn_out_dir, X_maj, maj_ids, args):
    dnn_matrix = dnn_out_dir / "leaf_global_dnn_matrix.npy"
    if dnn_matrix.exists() and not args.overwrite:
        return

    # Use the utility memmap function to avoid OOM
    precompute_leaf_dnn_memmap(
        X_majority_leaf=X_maj,
        majority_enrolids_leaf=maj_ids,
        out_dir=str(dnn_out_dir),
        leaf_id="global",
        batch_size=args.dnn_batch_size,
        metric=args.metric
    )
def run_one(code: str, args):
    print(f"\n{'='*88}\nCOHORT {code}\n{'='*88}")

    feat_path = resolve_feature_path(Path(args.features_dir), code, args.baseline_year, args.outcome_year)
    print(f"  Features: {feat_path}")

    # FIX: Bypass Spark to avoid Java Heap OOM
    # Use PyArrow to load only the necessary columns directly into Pandas
    print(f"  Loading features directly via PyArrow...")
    df = pd.read_parquet(str(feat_path))

    target_col = ensure_target(df, args.outcome_year)
    feat_cols = pick_feature_cols(df, target_col, args.outcome_year, args.feature_regex)

    print(f"  Target: {target_col} | Features: {len(feat_cols)}")

    # Split (Using your existing helper)
    _, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.train_test_seed
    )
    
    # Identify Cases and Controls
    cases_mask = train_pd[target_col] == 1
    controls_mask = train_pd[target_col] == 0
    
    # Memory Management: Don't .copy() the whole dataframe if not needed
    # Extract just the features we need for the preprocessor
    X_train_raw = train_pd[feat_cols] 

    # Define preprocessing column groups
    BIN_FLAG_COLUMNS = get_bin_flag_columns(X_train_raw)
    CAT_COLUMNS = get_cat_columns(X_train_raw)
    TRUE_NUM_COLUMNS = get_true_num_columns(X_train_raw, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    preprocessor = get_preprocessor(
        X=X_train_raw,
        cat_cols=CAT_COLUMNS,
        num_cols=TRUE_NUM_COLUMNS,
        binary_cols=BIN_FLAG_COLUMNS,
        verbose=True,
    )
    
    print("  Fitting preprocessor...")
    preprocessor.fit(X_train_raw)

    # Transform into Minority (Cases) and Majority (Controls)
    # We slice the already-preprocessed data to save memory
    X_transformed = preprocessor.transform(X_train_raw)
    X_minority = X_transformed[cases_mask.values]
    X_majority = X_transformed[controls_mask.values]
    
    maj_ids = train_pd.loc[controls_mask, "ENROLID"].values.astype(np.int64)
    min_ids = train_pd.loc[cases_mask, "ENROLID"].values.astype(np.int64)

    # Output setup
    dist_dir = Path(args.distances_dir_template.format(code=code))
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. PN Distances (Majority vs Minority)
    pn_h5_path = dist_dir / "distances_majority_minority.h5"
    maybe_compute_pn(pn_h5_path, X_majority, X_minority, maj_ids, min_ids, args)

    # 2. DNN Distances (Majority vs Majority - Memmap)
    dnn_out_dir = dist_dir / f"global_dnn_seed_{args.train_test_seed}"
    maybe_compute_dnn(dnn_out_dir, X_majority, maj_ids, args)

    print(f"  ✓ Cohort {code} Complete.")

def main():
    """ python precompute_distances_multi_cohort.py \
  --features_dir /Users/charles/DATA/misc_conditions_features_augmented \
  --distances_dir_template ./precomputed_distances_{code}_with_cost_features \
  --train_test_seed 123

  or If you want “medical-only” distances like your MSK experiment:
  python precompute_distances_multi_cohort.py \
  --features_dir /Users/charles/DATA/misc_conditions_features_augmented \
  --distances_dir_template ./precomputed_distances_{code}_medical_only \
  --feature_regex '^(med_|has_)' \
  --train_test_seed 123 
  """
    args = parse_args()
    codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]

    print(f"Processing Codes: {codes}")
    
    # WE REMOVED THE SPARK SESSION INITIALIZATION
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
