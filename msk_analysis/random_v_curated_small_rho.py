#!/usr/bin/env python
"""
random_v_curated_under1to1.py
=============================
"Below 1:1" ratio sweep: compare deterministic curated selection
(two-stage k-center + matching) vs. random undersampling baseline
when the number of majority controls used for training is LESS than
the number of positives.

Approach – *truncate cases* (deterministic, no changes to solver):
  For each ratio r:
      nC = ceil(r * nP)
      CURATED:  pick nC "hardest" (or sorted-enrolid) cases P',
                run standard 1:1 matching on P' only  → nC controls.
                Training set = ALL positives  +  matched controls.
      RANDOM:   for each seed, sample nC controls uniformly from
                training controls.
                Training set = ALL positives  +  sampled controls.


Usage
-----
    cd msk_analysis
    python random_v_curated_under1to1.py \\
        [--ratios 0.25 0.35 0.5 0.65 0.8 1.0] \\
        [--random_seeds 0 1 2 3 4 5 6 7 8 9] \\
        [--M_pool 50000] \\
        [--case_subset_mode hardest] \\
        [--output_dir ./random_vs_curated_small_ratios]
"""

import sys, glob
import os
import argparse
import time
import math
import traceback

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup (mirrors two_stage_iterative.py)
# ---------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import importlib
import h5py

import public.precompute_distances
importlib.reload(public.precompute_distances)
from public.precompute_distances import (
    get_preprocessor,
    compute_distances_batched,
    save_distances_hdf5,
    precompute_leaf_dnn_memmap,
)

try:
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import (
        two_stage_kcenter_then_match,
        load_pn_hdf5,
        build_id_to_index,
    )
except ImportError:
    parent_projects_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if parent_projects_dir not in sys.path:
        sys.path.insert(0, parent_projects_dir)
    import public.two_stage_kcenter_match
    importlib.reload(public.two_stage_kcenter_match)
    from public.two_stage_kcenter_match import (
        two_stage_kcenter_then_match,
        load_pn_hdf5,
        build_id_to_index,
    )

from public.model_IAI import *
from pyspark.sql import SparkSession


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Below-1:1 ratio sweep: curated (k-center + matching) "
                    "vs random undersampling",
    )
    p.add_argument(
        "--ratios", nargs="+", type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Control-to-positive ratios to sweep "
             "(default: 0.25 0.5 0.75 1.0)",
    )
    p.add_argument(
        "--random_seeds", nargs="+", type=int,
        default=list(range(10)),
        help="Seeds for random baseline (default: 0 1 2 ... 9)",
    )
    p.add_argument(
        "--M_pool", type=int, default=80000,
        help="Candidate pool size M for k-center (default: 50000)",
    )
    p.add_argument(
        "--case_subset_mode", choices=["hardest", "sorted_enrolid"],
        default="hardest",
        help="How to select deterministic case subset (default: hardest)",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="./random_vs_curated_small_ratios_kmeanspp",
        help="Output directory (default: ./random_vs_curated_small_ratios_kmeanspp)",
    )
    p.add_argument(
        "--train_test_seed", type=int, default=123,
        help="Random seed for train/test split (default: 123)",
    )
    p.add_argument(
        "--depths", nargs="+", type=int, default=[5, 7],
        help="OCT tree depths to try (default: 5 7)",
    )
    p.add_argument(
        "--minbuckets", nargs="+", type=int, default=[150],
        help="OCT minbucket values (default: 150)",
    )
    p.add_argument(
        "--cps", nargs="+", type=float, default=[0.0001, 0.001, 0.01],
        help="OCT complexity parameters (default: 0.0001 0.001 0.01)",
    )
    p.add_argument(
        "--seed_method", type=str, default="smart",
        help="k-center seed selection method (default: smart)",
    )
    p.add_argument(
        "--distances_dir", type=str,
        default="./precomputed_distances_msk_medical_only",
        help="Directory with precomputed distances",
    )
    p.add_argument(
        "--parquet_path", type=str,
        default="msk_2017_18_full.parquet",
        help="Path to parquet dataset",
    )

    p.add_argument("--resume", action="store_true",
                   help="Resume from existing results")
    p.add_argument("--use_kmeanspp", action="store_true",
                   help="Use k-means++ for seed selection")
    return p.parse_args()



# ============================================================================
# HELPERS
# ============================================================================

def load_dpn_leaf(pn_h5_path, control_enrolids, case_enrolids):
    """
    Load the full control-to-case distance sub-matrix d_pn_leaf from HDF5.

    Returns
    -------
    d_pn_leaf : ndarray, shape (n_controls, n_cases), float32
    """
    f_pn, d_pn, pn_maj_ids, pn_min_ids = load_pn_hdf5(pn_h5_path)
    pn_maj_id2idx = build_id_to_index(pn_maj_ids)
    pn_min_id2idx = build_id_to_index(pn_min_ids)

    try:
        pn_rows = np.array(
            [pn_maj_id2idx[int(e)] for e in control_enrolids], dtype=np.int64
        )
        pn_cols = np.array(
            [pn_min_id2idx[int(e)] for e in case_enrolids], dtype=np.int64
        )

        # HDF5 fancy indexing requires sorted indices
        rows_sort_idx = np.argsort(pn_rows)
        cols_sort_idx = np.argsort(pn_cols)

        pn_rows_sorted = pn_rows[rows_sort_idx]
        pn_cols_sorted = pn_cols[cols_sort_idx]

        d_pn_sorted = np.array(
            d_pn[pn_rows_sorted, :][:, pn_cols_sorted], dtype=np.float32
        )

        # Restore original order
        rows_unsort_idx = np.argsort(rows_sort_idx)
        cols_unsort_idx = np.argsort(cols_sort_idx)
        d_pn_leaf = d_pn_sorted[rows_unsort_idx, :][:, cols_unsort_idx]
    finally:
        f_pn.close()

    return d_pn_leaf


def select_case_subset(mode, nC, d_pn_leaf, case_enrolids):
    """
    Select *nC* case indices (into ``case_enrolids``) deterministically.

    Parameters
    ----------
    mode : str
        ``"hardest"`` – pick cases with smallest min-distance to any control
        (hardest boundary cases).
        ``"sorted_enrolid"`` – pick the *nC* smallest ENROLIDs.
    nC : int
        Number of cases to select.
    d_pn_leaf : ndarray, shape (n_controls, n_cases)
        Full control-to-case distance matrix.
    case_enrolids : ndarray
        ENROLID array for cases.

    Returns
    -------
    subset_indices : ndarray of int, length nC
        Sorted indices into ``case_enrolids``.
    """
    nP = len(case_enrolids)
    if nC >= nP:
        return np.arange(nP)

    if mode == "hardest":
        # h_i = min distance from any control to case i
        h = d_pn_leaf.min(axis=0)  # (n_cases,)
        # smallest h_i  ⟹  closest to majority  ⟹  hardest boundary
        subset_indices = np.argsort(h)[:nC]
    elif mode == "sorted_enrolid":
        subset_indices = np.argsort(case_enrolids)[:nC]
    else:
        raise ValueError(f"Unknown case_subset_mode: {mode}")

    return np.sort(subset_indices)


def compute_num_leaves(model, X_df, preprocessor, feature_names):
    """Return the number of distinct leaves the model assigns on *X_df*."""
    X_proc = preprocessor.transform(X_df)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
    leaves = model.apply(X_proc_df)
    return int(len(pd.unique(leaves)))


def load_metrics_from_predictions(
    pred_path: str,
    y_test: pd.Series,
) -> dict:
    """
    Load a predictions CSV and compute metrics (AUC, PR-AUC, MCC, etc.).
    Use when retraining is skipped because predictions already exist.
    """
    pred_df = pd.read_csv(pred_path)
    if "predicted_proba" not in pred_df.columns:
        raise ValueError(f"predictions file missing 'predicted_proba': {pred_path}")
    y_proba = pred_df["predicted_proba"].values
    y_test_arr = np.asarray(y_test).astype(int)
    if len(y_proba) != len(y_test_arr):
        raise ValueError(f"length mismatch: pred={len(y_proba)}, test={len(y_test_arr)}")

    auc = roc_auc_score(y_test_arr, y_proba)
    pr_auc = average_precision_score(y_test_arr, y_proba)
    mcc_res = best_mcc_threshold(y_test_arr, y_proba)
    best_mcc = mcc_res["mcc"]
    y_pred_mcc = mcc_res["y_pred"]

    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred_mcc).ravel()
    recall_mcc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_mcc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    balanced = best_balanced_threshold(y_test_arr, y_proba)
    gmean_recall = balanced["gmean_opt"]["recall"]
    gmean_specificity = balanced["gmean_opt"]["specificity"]

    from sklearn.metrics import precision_recall_curve, f1_score
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_test_arr, y_proba)
    f1_scores = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-10)
    optimal_f1 = float(f1_scores.max())

    num_leaves = int(pred_df["leaf_assignment"].nunique()) if "leaf_assignment" in pred_df.columns else np.nan
    row = {     "num_leaves": num_leaves,
                "test_auc": auc,
                "test_pr_auc": pr_auc,
                "test_best_mcc": best_mcc,
                "gmean_recall": gmean_recall,
                "gmean_specificity": gmean_specificity,
    }
    return row


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    print(f"{'='*80}")
    print("BELOW-1:1 RATIO SWEEP: CURATED vs RANDOM UNDERSAMPLING")
    print(f"{'='*80}")
    print(f"  Ratios:            {args.ratios}")
    print(f"  Random seeds:      {args.random_seeds}")
    print(f"  M_pool:            {args.M_pool:,}")
    print(f"  Case subset mode:  {args.case_subset_mode}")
    print(f"  Seed method:       {args.seed_method}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  OCT grid:          depths={args.depths}, "
          f"minbuckets={args.minbuckets}, cps={args.cps}")
    print()

    # ==================================================================
    # 1. DATA LOADING
    # ==================================================================
    print(f"{'='*80}")
    print("PHASE 0: Loading data")
    print(f"{'='*80}\n")

    spark = SparkSession.builder.appName("BelowOneToOne").getOrCreate()
    df_msk_spark = spark.read.format("parquet").load(args.parquet_path)
    df_og = df_msk_spark.toPandas()

    TRAIN_TEST_SEED = args.train_test_seed

    # Feature column definitions (same as two_stage_iterative.py)
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df_og)
    CAT_COLUMNS = get_cat_columns(df_og)
    TRUE_NUM_COLUMNS = get_true_num_columns(df_og, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    # Target column
    if "top_2_pct_cost_2018" in df_og.columns:
        target_col = "top_2_pct_cost_2018"
        print(f"Using {target_col} as target column")
    elif "annual_cost_2018_deflated" in df_og.columns:
        threshold = df_og["annual_cost_2018_deflated"].quantile(0.98)
        df_og["top_2_pct_cost_2018"] = (
            df_og["annual_cost_2018_deflated"] >= threshold
        ).astype(int)
        target_col = "top_2_pct_cost_2018"
        print(f"Created {target_col} using threshold ${threshold:,.2f}")
    else:
        raise ValueError(
            "No 2018 target column found.  Need either "
            "'top_2_pct_cost_2018' or 'annual_cost_2018_deflated'"
        )

    # Feature columns (exclude 2018 cols to prevent leakage)
    exclude_cols = (
        ["ENROLID", target_col]
        + [c for c in df_og.columns if "2018" in c]
    )
    feature_cols = [c for c in df_og.columns if c not in exclude_cols]

    # ==================================================================
    # 2. TRAIN / VAL / TEST SPLIT
    # ==================================================================
    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df_og,
        target_col=target_col,
        test_size=0.3,
        verbose=False,
        random_state=TRAIN_TEST_SEED,
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd,
        target_col=target_col,
        test_size=0.5,
        verbose=False,
        random_state=TRAIN_TEST_SEED,
    )

    X_test = test_pd[feature_cols]
    y_test = test_pd[target_col]
    X_val = val_pd[feature_cols]
    y_val = val_pd[target_col]

    print(f"Train: {train_pd.shape},  Val: {val_pd.shape},  Test: {test_pd.shape}")

    cases = train_pd[train_pd[target_col] == 1].copy()
    controls = train_pd[train_pd[target_col] == 0].copy()
    nP = len(cases)
    n_controls = len(controls)
    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)

    print(f"  Cases  (minority / positives): {nP:,}")
    print(f"  Controls (majority):           {n_controls:,}")
    print(f"  Original ratio:                {n_controls / nP:.2f}:1\n")

    # ==================================================================
    # 3. DISTANCE FILE PATHS
    # ==================================================================
    PN_H5_PATH = os.path.join(
        args.distances_dir, "distances_majority_minority.h5"
    )
    DNN_OUT_DIR = os.path.join(
        args.distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}"
    )
    dnn_matrix_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(DNN_OUT_DIR, "leaf_global_dnn_enrolids.npy")

    for fpath in [PN_H5_PATH, dnn_matrix_npy, dnn_enrolids_npy]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    # ==================================================================
    # 4. PRELOAD d_pn_leaf FOR CASE-HARDNESS COMPUTATION
    # ==================================================================
    print("Loading full d_pn_leaf for case-hardness computation ...")
    t0 = time.perf_counter()
    d_pn_leaf_full = load_dpn_leaf(
        PN_H5_PATH, control_enrolids, case_enrolids
    )
    print(
        f"  d_pn_leaf shape: {d_pn_leaf_full.shape}  "
        f"({time.perf_counter() - t0:.1f}s)\n"
    )

    # ==================================================================
    # 5. OUTPUT DIRECTORY SETUP
    # ==================================================================
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    # ==================================================================
    # 6. SWEEP OVER RATIOS
    # ==================================================================
    total_curated = len(args.ratios)
    total_random = len(args.ratios) * len(args.random_seeds)
    total_runs = total_curated + total_random
    run_idx = 0
    
    for ratio_idx, ratio in enumerate(args.ratios, 1):
        run_dir = os.path.join(OUTPUT_DIR, f"curated_r{ratio:.2f}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
        nC = math.ceil(ratio * nP)
        nC = min(nC, n_controls)  # safety cap

        if args.resume and os.path.exists(run_dir):
            pattern = os.path.join(
                run_dir,
                "predictions",
                f"oct_predictions_{ratio:.2f}_*.csv"
            )
            files = glob.glob(pattern)  
            if len(files) > 0:
                path = files[0]
                print(f"Loading metrics from {path}")
                metrics = load_metrics_from_predictions(path, y_test)
                row = {
                    "method": "curated",
                    "ratio": ratio,
                    "nP": nP,
                    "nC": nC,
                    "seed": np.nan,
                    "best_depth": np.nan,
                    "best_minbucket": np.nan,
                    "best_cp": np.nan,
                    **metrics,
                    "matching_time_s": np.nan,
                    "matching_mean_cost": np.nan,
                }
                all_results.append(row)
                continue
       
            
        print(f"\n{'#'*80}")
        print(
            f"RATIO {ratio_idx}/{len(args.ratios)}: r={ratio:.2f}  |  "
            f"nP={nP:,}  nC={nC:,}  "
            f"(effective ratio {nC / nP:.3f})"
        )
        print(f"{'#'*80}\n")

        # ==============================================================
        # 6a. CURATED (deterministic)
        # ==============================================================
        run_idx += 1
        print(f"  {'='*72}")
        print(
            f"  RUN {run_idx}/{total_runs}:  method=CURATED  "
            f"ratio={ratio:.2f}"
        )
        print(f"  {'='*72}")

        
        try:
            curated_start = time.perf_counter()

            # --- Select case subset ---
            subset_idx = select_case_subset(
                args.case_subset_mode, nC, d_pn_leaf_full, case_enrolids,
            )
            subset_case_enrolids = case_enrolids[subset_idx]
            print(
                f"    Case subset ({args.case_subset_mode}): "
                f"{len(subset_idx)} of {nP} cases selected"
            )

            # --- K-center + matching on the SUBSET of cases ---
            matching_start = time.perf_counter()
            matching_result = two_stage_kcenter_then_match(
                leaf_controls_enrolids=control_enrolids.copy(),
                leaf_cases_enrolids=subset_case_enrolids,
                leaf_nn_matrix_npy=dnn_matrix_npy,
                leaf_nn_enrolids_npy=dnn_enrolids_npy,
                pn_h5_path=PN_H5_PATH,
                M=args.M_pool,
                use_kmeanspp=args.use_kmeanspp,
                use_adaptive_pool=False,
                force_nearest_per_case=False,
                force_topm=1,
                assignment_topk_start=None,  # exact matching
                seed_method=args.seed_method,
                matching_ratio=1,
                case_weighting=None,
            )
            matching_time = time.perf_counter() - matching_start

            selected_ctrl_enrolids = matching_result["selected_control_enrolids"]
            mean_match_cost = float(matching_result["match_costs"].mean())

            print(
                f"    Matched {len(selected_ctrl_enrolids)} controls  "
                f"(mean cost={mean_match_cost:.4f},  "
                f"time={matching_time:.1f}s)"
            )

            # --- Build training set: ALL positives + matched controls ---
            all_minority = train_pd[train_pd[target_col] == 1].copy()
            unique_ctrl = list(set(selected_ctrl_enrolids))
            selected_majority = train_pd[
                (train_pd[target_col] == 0)
                & (train_pd["ENROLID"].isin(unique_ctrl))
            ].copy()
            undersampled = pd.concat(
                [all_minority, selected_majority], axis=0, ignore_index=True
            )

            assert len(selected_majority) == nC, (
                f"Expected {nC} controls, got {len(selected_majority)}"
            )
            print(
                f"    Training set: {len(undersampled):,}  "
                f"(nP={len(all_minority)}, nC={len(selected_majority)})"
            )

            # --- Save undersampled CSV ---
            us_path = os.path.join(
                run_dir, f"undersampled_curated_r{ratio:.2f}.csv"
            )
            undersampled.to_csv(us_path, index=False)
            print(f"    Saved undersampled dataset: {us_path}")

            # --- Train OCT ---
            print(f"\n    TRAINING OCT ...")
            model, params, grid_df, preprocessor, feat_names = finetune_oct(
                X_train=undersampled[feature_cols],
                y_train=undersampled[target_col],
                X_val=X_val,
                y_val=y_val,
                categorical_cols=CAT_COLUMNS,
                numeric_cols=TRUE_NUM_COLUMNS,
                binary_cols=BIN_FLAG_COLUMNS,
                depths=args.depths,
                minbuckets=args.minbuckets,
                cps=args.cps,
                verbose=False,
                random_seed=TRAIN_TEST_SEED,
            )
            save_sfx = (
                f"curated_r{ratio:.2f}_"
                f"{params['depth']}_{params['minbucket']}_{params['cp']}"
            )
            metrics = evaluate_binary_oct(
                model,
                X_test,
                y_test,
                preprocessor,
                feat_names,
                X_val_df=X_val,
                y_val=y_val,
                results_dir=run_dir,
                save_suffix=save_sfx,
            )
            total_time = time.perf_counter() - curated_start

            num_leaves = compute_num_leaves(
                model, X_test, preprocessor, feat_names
            )

            print(
                f"\n    CURATED r={ratio:.2f}  DONE  "
                f"PR-AUC={metrics.get('pr_auc', float('nan')):.4f}  "
                f"AUC={metrics.get('auc', float('nan')):.4f}  "
                f"MCC={metrics.get('best_mcc', float('nan')):.4f}  "
                f"leaves={num_leaves}  "
                f"({total_time:.1f}s)"
            )

            # --- Collect result row ---
            row = {
                "method": "curated",
                "ratio": ratio,
                "nP": nP,
                "nC": nC,
                "seed": np.nan,
                "best_depth": params['depth'],
                "best_minbucket": params['minbucket'],
                "best_cp": params['cp'],
                "num_leaves": num_leaves,
                "test_auc": metrics.get("auc"),
                "test_pr_auc": metrics.get("pr_auc"),
                "test_best_mcc": metrics.get("best_mcc"),
                "gmean_recall": metrics.get("balanced_recall_gmean"),
                "gmean_specificity": metrics.get("balanced_specificity_gmean"),
                "matching_time_s": matching_time,
                "matching_mean_cost": mean_match_cost,
            }
            all_results.append(row)

        except Exception as e:
            print(f"\n    ERROR (curated r={ratio:.2f}): {e}")
            traceback.print_exc()
            all_results.append({
                "method": "curated",
                "ratio": ratio,
                "nP": nP,
                "nC": nC,
                "seed": np.nan,
                "error": str(e),
            })

        # ==============================================================
        # 6b. RANDOM BASELINE (per seed)
        # ==============================================================
        for seed_idx, seed in enumerate(args.random_seeds, 1):

            run_idx += 1
            print(f"\n  {'='*72}")
            print(
                f"  RUN {run_idx}/{total_runs}:  method=RANDOM  "
                f"ratio={ratio:.2f}  seed={seed}"
            )
            print(f"  {'='*72}")

            rnd_dir = os.path.join(
                OUTPUT_DIR, f"random_r{ratio:.2f}_s{seed}"
            )
            os.makedirs(rnd_dir, exist_ok=True)
            os.makedirs(os.path.join(rnd_dir, "predictions"), exist_ok=True)

            if args.resume:
                pattern = os.path.join(
                    rnd_dir,
                    "predictions",
                    f"oct_predictions_random_r{ratio:.2f}_s{seed}_*.csv"
                )
                files = glob.glob(pattern)  
                if len(files) > 0:
                    path = files[0]
                    print(f"Loading metrics from {path}")
                    metrics = load_metrics_from_predictions(path, y_test)
                    row = {
                        "method": "random",
                        "ratio": ratio,
                        "nP": nP,
                        "nC": nC,
                        "seed": seed,
                        "best_depth": np.nan,
                        "best_minbucket": np.nan,
                        "best_cp": np.nan,
                        **metrics,
                        "matching_time_s": np.nan,
                        "matching_mean_cost": np.nan,
                    }
                    all_results.append(row)

            else:
                try:
                    rnd_start = time.perf_counter()

                    # --- Sample nC controls uniformly ---
                    rng = np.random.RandomState(seed)
                    sampled_idx = rng.choice(n_controls, size=nC, replace=False)
                    sampled_ctrl_enrolids = control_enrolids[sampled_idx]

                    # --- Build training set ---
                    all_minority = train_pd[train_pd[target_col] == 1].copy()
                    sampled_majority = train_pd[
                        (train_pd[target_col] == 0)
                        & (train_pd["ENROLID"].isin(set(sampled_ctrl_enrolids)))
                    ].copy()
                    undersampled_rnd = pd.concat(
                        [all_minority, sampled_majority],
                        axis=0,
                        ignore_index=True,
                    )
                    
                    print(
                        f"    Training set: {len(undersampled_rnd):,}  "
                        f"(nP={len(all_minority)}, nC={len(sampled_majority)})"
                    )

                    # --- Save undersampled CSV ---
                    us_path = os.path.join(
                        rnd_dir,
                        f"undersampled_random_r{ratio:.2f}_s{seed}.csv",
                    )
                    undersampled_rnd.to_csv(us_path, index=False)
                    print(f"    Saved undersampled dataset: {us_path}")

                    # --- Train OCT ---
                    print(f"\n    TRAINING OCT ...")
                    model, params, grid_df, preprocessor, feat_names = finetune_oct(
                        X_train=undersampled_rnd[feature_cols],
                        y_train=undersampled_rnd[target_col],
                        X_val=X_val,
                        y_val=y_val,
                        categorical_cols=CAT_COLUMNS,
                        numeric_cols=TRUE_NUM_COLUMNS,
                        binary_cols=BIN_FLAG_COLUMNS,
                        depths=args.depths,
                        minbuckets=args.minbuckets,
                        cps=args.cps,
                        verbose=False,
                        random_seed=TRAIN_TEST_SEED,
                    )
                    
                    # --- Evaluate on test ---
                    save_sfx = (
                        f"random_r{ratio:.2f}_s{seed}_"
                        f"{params['depth']}_{params['minbucket']}_{params['cp']}"
                    )
                    metrics = evaluate_binary_oct(
                        model,
                        X_test,
                        y_test,
                        preprocessor,
                        feat_names,
                        X_val_df=X_val,
                        y_val=y_val,
                        results_dir=rnd_dir,
                        save_suffix=save_sfx,
                    )
                    total_time = time.perf_counter() - rnd_start

                    num_leaves = compute_num_leaves(
                        model, X_test, preprocessor, feat_names
                    )

                    print(
                        f"\n    RANDOM r={ratio:.2f} s={seed}  DONE  "
                        f"PR-AUC={metrics.get('pr_auc', float('nan')):.4f}  "
                        f"AUC={metrics.get('auc', float('nan')):.4f}  "
                        f"MCC={metrics.get('best_mcc', float('nan')):.4f}  "
                        f"leaves={num_leaves}  "
                        f"({total_time:.1f}s)"
                    )

                    # --- Collect result row ---
                    row = {
                        "method": "random",
                        "ratio": ratio,
                        "nP": nP,
                        "nC": nC,
                        "seed": seed,
                        "best_depth": params['depth'],
                        "best_minbucket": params['minbucket'],
                        "best_cp": params['cp'],
                        "num_leaves": num_leaves,
                        "test_auc": metrics.get("auc"),
                        "test_pr_auc": metrics.get("pr_auc"),
                        "test_best_mcc": metrics.get("best_mcc"),
                        "gmean_recall": metrics.get("balanced_recall_gmean"),
                        "gmean_specificity": metrics.get("balanced_specificity_gmean"),
                        "matching_time_s": np.nan,
                        "matching_mean_cost": np.nan,
                    }
                    all_results.append(row)

                except Exception as e:
                    print(f"\n    ERROR (random r={ratio:.2f} s={seed}): {e}")
                    traceback.print_exc()
                    all_results.append({
                        "method": "random",
                        "ratio": ratio,
                        "nP": nP,
                        "nC": nC,
                        "seed": seed,
                        "error": str(e),
                    })

    # ==================================================================
    # 7. SAVE SUMMARY CSV & PRINT COMPARISON
    # ==================================================================
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    if not all_results:
        print("No results collected.  Check for errors above.")
        return

    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "random_vs_curated_comparison.csv")
    results_df.to_csv(summary_path, mode='a', index=False, header=not os.path.exists(summary_path))
    print(f"Saved {len(results_df)} rows to: {summary_path}\n")

    # --- Pretty-print comparison table ---
    display_cols = ["method", "ratio", "nP", "nC", "seed"]
    metric_cols = [
        "val_pr_auc",
        "test_pr_auc",
        "test_auc",
        "test_best_mcc",
        "gmean_recall",
        "gmean_specificity",
        "num_leaves",
        "matching_mean_cost",
    ]
    for c in metric_cols:
        if c in results_df.columns:
            display_cols.append(c)

    print(results_df[display_cols].to_string(index=False))

    # --- Aggregated comparison per ratio ---
    print(f"\n{'='*80}")
    print("AGGREGATED COMPARISON  (mean +/- std for random)")
    print(f"{'='*80}\n")

    for ratio in args.ratios:
        sub = results_df[results_df["ratio"] == ratio]
        cur = sub[sub["method"] == "curated"]
        rnd = sub[sub["method"] == "random"]

        if len(cur) and "test_pr_auc" in cur.columns:
            c_prauc = cur["test_pr_auc"].iloc[0]
            c_auc = cur["test_auc"].iloc[0]
            c_mcc = cur["test_best_mcc"].iloc[0]
        else:
            c_prauc = c_auc = c_mcc = float("nan")

        if len(rnd) and "test_pr_auc" in rnd.columns:
            r_prauc_mean = rnd["test_pr_auc"].mean()
            r_prauc_std = rnd["test_pr_auc"].std()
            r_auc_mean = rnd["test_auc"].mean()
            r_mcc_mean = rnd["test_best_mcc"].mean()
        else:
            r_prauc_mean = r_prauc_std = r_auc_mean = r_mcc_mean = float("nan")

        delta = c_prauc - r_prauc_mean
        print(
            f"  ratio={ratio:.2f}  |  "
            f"curated PR-AUC={c_prauc:.4f}  AUC={c_auc:.4f}  MCC={c_mcc:.4f}  |  "
            f"random PR-AUC={r_prauc_mean:.4f} +/- {r_prauc_std:.4f}  "
            f"AUC={r_auc_mean:.4f}  MCC={r_mcc_mean:.4f}  |  "
            f"delta(PR-AUC)={delta:+.4f}"
        )

    print(f"\n{'='*80}")
    print("Done.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
