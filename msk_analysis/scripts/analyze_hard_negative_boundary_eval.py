#!/usr/bin/env python
"""
analyze_hard_negative_boundary_eval.py
======================================
Hard-negative boundary evaluation: curated vs random OCT on hardest majority test subsets.

Scientific question: Does curated training excel at separating minority from "hard"
majority test patients (closest to minority manifold), even if random wins on full iid test?

Hardness is defined via Gower-distance neighborhood in the same prediction-feature space
used by the exp6 pipeline. Uses TRAINING-REFERENCE distances only (no test-test geometry).

Usage
-----
  cd msk_analysis
  python scripts/analyze_hard_negative_boundary_eval.py --outdir /Users/cat2510/scratch/hard_negative_boundary_eval

Outputs
-------
  {outdir}/predictions_ours_s0.csv, predictions_rnd_s0.csv
  {outdir}/hardness_scores_test_majority.csv
  {outdir}/metrics_summary.csv
  {outdir}/fig_*.png
  {outdir}/INTERPRETATION.md
"""

from __future__ import annotations

import sys
import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path setup
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

warnings.filterwarnings("ignore", category=UserWarning)

# Exp6 / model imports
from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    finetune_oct,
)
from msk_analysis.experiments_compare_random_vs_curation import (
    sample_random_controls,
    sample_stageB_matched_controls,
    train_and_evaluate_oct,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
from precompute_msk_distances_za_zb import _compute_gower_distances
from ensure_distances_for_metric import TRAIN_TEST_SEED, ensure_distances_for_metric
from gower_1to1_sampling import run_ours_1to1_sampling, run_rnd_1to1_sampling

# OCT config matching exp6
OCT_DEPTHS = [7]
OCT_MINBUCKETS = [100, 150]
OCT_CPS = [0.00001, 0.0001, 0.001]

TARGET_COL = "top_2_pct_cost_2018"
EPS = 1e-9
DEFAULT_K = 10
K_SENSITIVITY = [5, 10, 20]
HARD_PCT = [100, 20, 10, 5, 2]


def parse_args():
    p = argparse.ArgumentParser(
        description="Hard-negative boundary evaluation: curated vs random on hardest majority"
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/Users/cat2510/scratch/hard_negative_boundary_eval2",
        help="Output directory",
    )
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_no_meds.parquet")
    p.add_argument(
        "--distances_dir",
        type=str,
        default="/Users/cat2510/scratch/precomputed_distances_exp6_ablation",
    )
    p.add_argument(
        "--exp6_results",
        type=str,
        default="./exp6_distance_ablation/results",
        help="Path to exp6 results (ours_1to1_s0.csv, rnd_1to1_s0.csv)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_retrain", action="store_true", help="Skip retraining if predictions exist")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="k for hardness neighbors")
    return p.parse_args()


def build_gower_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    bin_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    ohe: Optional[OneHotEncoder] = None,
    ohe_fit_df: Optional[pd.DataFrame] = None,
    ranges: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, OneHotEncoder, np.ndarray]:
    """
    Build Gower-compatible feature matrix (same as exp6 ensure_distances_for_metric).
    Returns (X, ohe, ranges). If ohe/ranges provided, use for transform only.
    ohe_fit_df: when fitting OHE, use this dataframe (e.g. concat train cases+controls).
    """
    bin_in_feature = [c for c in bin_cols if c in feature_cols]
    num_in_feature = [c for c in num_cols if c in feature_cols]
    cat_in_feature = [c for c in cat_cols if c in feature_cols]
    parts = []
    n_bin = 0
    if bin_in_feature:
        parts.append(df[bin_in_feature].values.astype(np.float64))
        n_bin += len(bin_in_feature)
    if cat_in_feature:
        if ohe is None:
            fit_df = ohe_fit_df if ohe_fit_df is not None else df
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            ohe.fit(fit_df[cat_in_feature])
        ohe_mat = ohe.transform(df[cat_in_feature])
        parts.append(ohe_mat.astype(np.float64))
        n_bin += ohe_mat.shape[1]
    if num_in_feature:
        parts.append(df[num_in_feature].values.astype(np.float64))
    X = np.hstack(parts) if parts else np.zeros((len(df), 0))

    if ranges is None:
        ranges = np.ones(X.shape[1])
        for j in range(n_bin, X.shape[1]):
            c = X[:, j]
            r = float(np.nanmax(c) - np.nanmin(c))
            ranges[j] = r if r > 0 else 1.0

    return X, ohe, ranges


def compute_hardness_scores(
    X_test_maj: np.ndarray,
    X_train_min: np.ndarray,
    X_train_maj: np.ndarray,
    bin_col_indices: np.ndarray,
    k: int = DEFAULT_K,
    batch_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each majority test patient x, compute:
    - d_minority_k: avg distance to k nearest minority TRAIN
    - d_majority_k: avg distance to k nearest majority TRAIN
    - ratio_hardness: d_minority_k / (d_majority_k + eps). Smaller = closer to minority.
    - margin_hardness: d_majority_k - d_minority_k. Smaller = closer to minority.
    - minority_neighbor_fraction: among k nearest TRAIN (all), fraction that are minority.

    Convention: harder = larger. We use:
    - hardness_ratio = 1 / (ratio_raw + eps)  so larger = harder
    - hardness_margin = -margin  so larger = harder
    - minority_neighbor_fraction: already larger = harder
    """
    n_test = X_test_maj.shape[0]
    n_min = X_train_min.shape[0]
    n_maj = X_train_maj.shape[0]
    n_train = n_min + n_maj
    X_train_all = np.vstack([X_train_min, X_train_maj])
    train_is_minority = np.array([True] * n_min + [False] * n_maj)

    n_p = X_test_maj.shape[1]
    cont_indices = np.array([j for j in range(n_p) if j not in set(bin_col_indices)])
    ranges = np.ones(n_p)
    for j in range(len(bin_col_indices), n_p):
        col = np.concatenate([X_train_all[:, j], X_test_maj[:, j]])
        r = float(np.nanmax(col) - np.nanmin(col))
        ranges[j] = r if r > 0 else 1.0

    d_minority = np.full((n_test,), np.nan, dtype=np.float32)
    d_majority = np.full((n_test,), np.nan, dtype=np.float32)
    minority_frac = np.full((n_test,), np.nan, dtype=np.float32)

    k_min = min(k, n_min)
    k_maj = min(k, n_maj)
    k_all = min(k, n_train)

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        block = X_test_maj[start:end]
        # Distances to minority train
        D_min = _compute_gower_distances(
            block, X_train_min, bin_col_indices, cont_indices, ranges, batch_size=n_train
        )
        # Distances to majority train
        D_maj = _compute_gower_distances(
            block, X_train_maj, bin_col_indices, cont_indices, ranges, batch_size=n_train
        )
        # Distances to all train (for k-NN over combined)
        D_all = _compute_gower_distances(
            block, X_train_all, bin_col_indices, cont_indices, ranges, batch_size=n_train
        )

        for i in range(end - start):
            idx = start + i
            # k nearest minority
            d_min_i = np.partition(D_min[i], k_min - 1)[:k_min]
            d_minority[idx] = float(np.mean(d_min_i))
            # k nearest majority
            d_maj_i = np.partition(D_maj[i], k_maj - 1)[:k_maj]
            d_majority[idx] = float(np.mean(d_maj_i))
            # k nearest in all train
            nn_idx = np.argpartition(D_all[i], k_all - 1)[:k_all]
            n_min_in_nn = np.sum(train_is_minority[nn_idx])
            minority_frac[idx] = n_min_in_nn / k_all

    ratio_raw = d_minority / (d_majority + EPS)
    hardness_ratio = 1.0 / (ratio_raw + EPS)
    hardness_margin = -(d_majority - d_minority)

    return hardness_ratio, hardness_margin, minority_frac, d_minority


def train_and_save_predictions_with_enrolid(
    train_df: pd.DataFrame,
    val_pd: pd.DataFrame,
    test_pd: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    out_path: str,
    model_tag: str,
) -> pd.DataFrame:
    """
    Train OCT, predict on val and test, save with ENROLID, y_true, split, score, model_tag.
    """
    from public.model_IAI import finetune_oct

    # Restrict to columns present in all splits (ours/rnd CSVs may have fewer cols than parquet)
    cols_avail = set(train_df.columns) & set(val_pd.columns) & set(test_pd.columns)
    feature_cols_use = [c for c in feature_cols if c in cols_avail]
    if len(feature_cols_use) < len(feature_cols):
        print(f"  [WARN] Using {len(feature_cols_use)}/{len(feature_cols)} features (missing in train CSV)")
    feature_cols = feature_cols_use
    cat_cols = [c for c in cat_cols if c in feature_cols]
    num_cols = [c for c in num_cols if c in feature_cols]
    bin_cols = [c for c in bin_cols if c in feature_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_pd[feature_cols]
    y_val = val_pd[target_col]
    X_test = test_pd[feature_cols]
    y_test = test_pd[target_col]

    model, params, _, preprocessor, feat_names = finetune_oct(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        binary_cols=bin_cols,
        depths=OCT_DEPTHS,
        minbuckets=OCT_MINBUCKETS,
        cps=OCT_CPS,
        tree_kind="oct",
        verbose=False,
        random_seed=TRAIN_TEST_SEED,
    )

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        X = preprocessor.transform(df[feature_cols])
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_df = pd.DataFrame(X, columns=feat_names)
        p = model.predict_proba(X_df)
        if isinstance(p, pd.DataFrame):
            return p.iloc[:, 1].values
        return np.asarray(p)[:, 1]

    preds_list = []
    for split_name, df in [("val", val_pd), ("test", test_pd)]:
        proba = predict_proba(df)
        preds_list.append(
            pd.DataFrame(
                {
                    "ENROLID": df["ENROLID"].values,
                    "split": split_name,
                    "y_true": df[target_col].values,
                    "score": proba,
                    "model_tag": model_tag,
                }
            )
        )

    out_df = pd.concat(preds_list, ignore_index=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 80)
    print("HARD-NEGATIVE BOUNDARY EVALUATION")
    print("=" * 80)

    # Load data
    df = pd.read_parquet(args.parquet_path)
    if TARGET_COL not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[TARGET_COL] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    COST_COLUMNS = [
        c
        for c in df.columns
        if (
            "cost" in c.lower()
            or "quarterly" in c.lower()
            or "increasing" in c.lower()
            or "decreasing" in c.lower()
            or "skewness" in c.lower()
            or "kurtosis" in c.lower()
            or "cv" in c.lower()
            or "range" in c.lower()
        )
        and "2018" not in c
    ]
    AUXILIARY_COST_COLUMNS = [
        c for c in df.columns if "comorbidity_only" in c.lower()
    ]
    exclude_cols = ["ENROLID", TARGET_COL] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Reconstruct split (identical to exp6)
    train_ids, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=TARGET_COL, test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=TARGET_COL, test_size=0.5, verbose=False, random_state=TRAIN_TEST_SEED
    )

    cases = train_pd[train_pd[TARGET_COL] == 1]
    controls = train_pd[train_pd[TARGET_COL] == 0]
    N = len(cases)
    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)
    M_pool = len(controls) // 2

    print(f"Train: {len(train_pd):,}  Val: {len(val_pd):,}  Test: {len(test_pd):,}")
    print(f"  N (cases): {N:,}  Controls: {len(controls):,}")

    # Load or create sampled training sets
    exp6_res = Path(args.exp6_results)
    #ours_path = exp6_res / f"ours_1to1_s{args.seed}_gower.csv"
    ours_path = exp6_res / f"ours_1to1_A_za_coarse_phenotype__B_zb_intensity_context.csv"
    rnd_path = exp6_res / f"rnd_1to1_s{args.seed}.csv"
    if not ours_path.exists() or not rnd_path.exists():
        # Create distances and sample if missing
        os.makedirs(args.distances_dir, exist_ok=True)
        pn_h5, dnn_mat, dnn_ids = ensure_distances_for_metric(
            "gower",
            cases,
            controls,
            feature_cols,
            CAT_COLUMNS,
            TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS,
            args.distances_dir,
        )
        if not ours_path.exists():
            ours_train = run_ours_1to1_sampling(
                control_enrolids,
                case_enrolids,
                controls,
                cases,
                dnn_mat,
                dnn_ids,
                pn_h5,
                N,
                M_pool,
                args.seed,
            )
            ours_train.to_csv(ours_path, index=False)
        if not rnd_path.exists():
            rnd_train = run_rnd_1to1_sampling(
                control_enrolids, cases, controls, N, args.seed
            )
            rnd_train.to_csv(rnd_path, index=False)

    ours_train = pd.read_csv(ours_path)
    rnd_train = pd.read_csv(rnd_path)

    # Retrain and save predictions with ENROLID (or load if skip_retrain)
    pred_ours_path = os.path.join(args.outdir, f"predictions_ours_s{args.seed}.csv")
    pred_rnd_path = os.path.join(args.outdir, f"predictions_rnd_s{args.seed}.csv")

    if args.skip_retrain and os.path.exists(pred_ours_path) and os.path.exists(pred_rnd_path):
        print("\n[SKIP] Loading existing predictions (--skip_retrain)")
        pred_ours = pd.read_csv(pred_ours_path)
        pred_rnd = pd.read_csv(pred_rnd_path)
    else:
        print("\n--- Retraining OCT (ours) ---")
        pred_ours = train_and_save_predictions_with_enrolid(
            ours_train,
            val_pd,
            test_pd,
            feature_cols,
            TARGET_COL,
            CAT_COLUMNS,
            TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS,
            pred_ours_path,
            "ours",
        )
        print("\n--- Retraining OCT (rnd) ---")
        pred_rnd = train_and_save_predictions_with_enrolid(
            rnd_train,
            val_pd,
            test_pd,
            feature_cols,
            TARGET_COL,
            CAT_COLUMNS,
            TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS,
            pred_rnd_path,
            "rnd",
        )

    # Option 1: Hardness from FULL original train (model-independent)
    print("\n--- Computing hardness (Option 1: full original train) ---")
    train_combined = pd.concat([cases, controls], ignore_index=True)
    X_train_min, ohe, ranges = build_gower_features(
        cases, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe_fit_df=train_combined
    )
    X_train_maj, _, _ = build_gower_features(
        controls, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe=ohe, ohe_fit_df=train_combined
    )
    # Ranges from full train
    X_train_all_temp = np.vstack([X_train_min, X_train_maj])
    ranges = np.ones(X_train_all_temp.shape[1])
    n_bin = sum(1 for c in BIN_FLAG_COLUMNS if c in feature_cols)
    if CAT_COLUMNS and any(c in feature_cols for c in CAT_COLUMNS):
        n_bin += ohe.transform(train_combined[[c for c in CAT_COLUMNS if c in feature_cols]]).shape[1]
    for j in range(n_bin, X_train_all_temp.shape[1]):
        r = float(np.nanmax(X_train_all_temp[:, j]) - np.nanmin(X_train_all_temp[:, j]))
        ranges[j] = r if r > 0 else 1.0

    test_maj = test_pd[test_pd[TARGET_COL] == 0]
    test_min = test_pd[test_pd[TARGET_COL] == 1]
    X_test_maj, _, _ = build_gower_features(
        test_maj, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe=ohe, ohe_fit_df=train_combined
    )
    n_bin = sum(1 for c in BIN_FLAG_COLUMNS if c in feature_cols)
    if CAT_COLUMNS:
        n_bin += ohe.transform(test_maj[[c for c in CAT_COLUMNS if c in feature_cols]]).shape[1]
    bin_col_indices = np.array(list(range(n_bin)))

    h_ratio, h_margin, h_frac, d_min = compute_hardness_scores(
        X_test_maj, X_train_min, X_train_maj, bin_col_indices, k=args.k
    )

    hardness_df = pd.DataFrame(
        {
            "ENROLID": test_maj["ENROLID"].values,
            "y_true": 0,
            "split": "test",
            "hardness_ratio": h_ratio,
            "hardness_margin": h_margin,
            "minority_neighbor_frac": h_frac,
        }
    )
    hardness_df["hardness_composite"] = (
        0.33 * (h_ratio / (np.nanmax(h_ratio) + EPS))
        + 0.33 * (h_margin / (np.nanmax(h_margin) + EPS))
        + 0.34 * h_frac
    )

    for pct in HARD_PCT:
        if pct == 100:
            hardness_df[f"hard_{pct}"] = True
        else:
            n_top = max(1, int(len(hardness_df) * pct / 100))
            thresh = np.nanpercentile(hardness_df["hardness_composite"], 100 - pct)
            hardness_df[f"hard_{pct}"] = hardness_df["hardness_composite"] >= thresh

    hardness_path = os.path.join(args.outdir, "hardness_scores_test_majority.csv")
    hardness_df.to_csv(hardness_path, index=False)
    print(f"Saved {hardness_path}")

    # Merge predictions into test set and hardness df
    pred_ours_test = pred_ours[pred_ours["split"] == "test"][["ENROLID", "score"]].rename(
        columns={"score": "score_ours"}
    )
    pred_rnd_test = pred_rnd[pred_rnd["split"] == "test"][["ENROLID", "score"]].rename(
        columns={"score": "score_rnd"}
    )
    test_full = (
        test_pd[["ENROLID", TARGET_COL]]
        .merge(pred_ours_test, on="ENROLID", how="left")
        .merge(pred_rnd_test, on="ENROLID", how="left")
    )
    test_maj_ids = set(test_maj["ENROLID"].astype(int))
    hardness_df = hardness_df.merge(
        pred_ours_test, on="ENROLID", how="left"
    ).merge(pred_rnd_test, on="ENROLID", how="left")
    hardness_df["y_true"] = 0

    # Per-test-patient export: majority with hardness + minority (hardness=NaN)
    cols_exp = [
        "ENROLID", "y_true", "split", "hardness_ratio", "hardness_margin",
        "minority_neighbor_frac", "hardness_composite",
        "hard_100", "hard_20", "hard_10", "hard_5", "hard_2", "score_ours", "score_rnd",
    ]
    maj_part = hardness_df.copy()
    maj_part["split"] = "test"
    min_part = test_full[test_full[TARGET_COL] == 1][["ENROLID", "score_ours", "score_rnd"]].copy()
    min_part["y_true"] = 1
    min_part["split"] = "test"
    for c in cols_exp:
        if c not in min_part.columns:
            min_part[c] = np.nan
    for c in cols_exp:
        if c not in maj_part.columns:
            maj_part[c] = np.nan
    per_patient_full = pd.concat(
        [min_part[cols_exp], maj_part[cols_exp]], ignore_index=True
    )
    per_patient_path = os.path.join(args.outdir, "per_test_patient_hardness_scores.csv")
    per_patient_full.to_csv(per_patient_path, index=False)
    print(f"Saved {per_patient_path}")

    # Build evaluation subsets H_q = minority_test + hard majority
    test_min_ids = set(test_min["ENROLID"].astype(int))
    def eval_subset(enrolids: List[int], tag: str) -> Dict[str, float]:
        sub = test_full[test_full["ENROLID"].isin(enrolids)].dropna(
            subset=["score_ours", "score_rnd"]
        )
        if len(sub) < 10 or sub[TARGET_COL].nunique() < 2:
            return {"auc_ours": np.nan, "auc_rnd": np.nan, "prauc_ours": np.nan, "prauc_rnd": np.nan}
        y = sub[TARGET_COL].astype(int).values
        return {
            "auc_ours": roc_auc_score(y, sub["score_ours"].values),
            "auc_rnd": roc_auc_score(y, sub["score_rnd"].values),
            "prauc_ours": average_precision_score(y, sub["score_ours"].values),
            "prauc_rnd": average_precision_score(y, sub["score_rnd"].values),
        }

    metrics_rows = []
    for pct in HARD_PCT:
        if pct == 100:
            enrolids = test_pd["ENROLID"].astype(int).tolist()
        else:
            hard_ids = hardness_df[hardness_df[f"hard_{pct}"]]["ENROLID"].astype(int).tolist()
            enrolids = list(test_min_ids) + hard_ids
        m = eval_subset(enrolids, f"hard_{pct}")
        m["subset"] = f"hard_{pct}"
        m["n"] = len(enrolids)
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.outdir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # Overall iid test
    overall = eval_subset(test_pd["ENROLID"].astype(int).tolist(), "full")
    overall["subset"] = "full_iid"
    overall["n"] = len(test_pd)
    metrics_df = pd.concat([pd.DataFrame([overall]), metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nOverall iid test: AUC ours={overall['auc_ours']:.4f} rnd={overall['auc_rnd']:.4f}")

    # Plots
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = metrics_df[metrics_df["subset"] != "full_iid"]
    x = range(len(sub))
    ax.plot(x, sub["auc_ours"], "o-", label="Ours (curated)", color="C0")
    ax.plot(x, sub["auc_rnd"], "s-", label="RND", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["subset"])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC vs Hardness Subset (minority + hard majority)")
    ax.legend()
    ax.axhline(overall["auc_ours"], color="C0", ls="--", alpha=0.5)
    ax.axhline(overall["auc_rnd"], color="C1", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_auc_vs_hardness.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, sub["prauc_ours"], "o-", label="Ours (curated)", color="C0")
    ax.plot(x, sub["prauc_rnd"], "s-", label="RND", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["subset"])
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC vs Hardness Subset")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_prauc_vs_hardness.png"), dpi=150)
    plt.close()

    # Score distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (tag, col) in zip(axes, [("Ours", "score_ours"), ("RND", "score_rnd")]):
        t = test_full.dropna(subset=[col])
        ax.hist(
            t[t[TARGET_COL] == 1][col].dropna(),
            bins=30,
            alpha=0.6,
            label="Minority",
            color="C1",
            density=True,
        )
        ax.hist(
            t[t[TARGET_COL] == 0][col].dropna(),
            bins=30,
            alpha=0.6,
            label="Majority (all)",
            color="C0",
            density=True,
        )
        hard_ids = hardness_df[hardness_df["hard_20"]]["ENROLID"].astype(int)
        ax.hist(
            t[t["ENROLID"].isin(hard_ids)][col].dropna(),
            bins=30,
            alpha=0.5,
            label="Hard 20% majority",
            color="C2",
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax.set_xlabel(f"Score ({tag})")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(tag)
    plt.suptitle("Score distributions: minority / majority / hard majority")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_score_distributions.png"), dpi=150)
    plt.close()

    # Interpretation memo
    memo = f"""# Hard-Negative Boundary Evaluation — Interpretation Memo

## Hardness Definition
- **Primary (Option 1)**: Full original training split (before undersampling).
- Gower distance on all prediction features (same as exp6).
- For each majority test patient x:
  - `d_minority_k`: avg distance to k={args.k} nearest minority TRAIN
  - `d_majority_k`: avg distance to k nearest majority TRAIN
  - `hardness_ratio` = 1/(d_minority_k/(d_majority_k+eps)) — larger = harder
  - `hardness_margin` = -(d_majority_k - d_minority_k) — larger = harder
  - `minority_neighbor_frac`: fraction of k nearest TRAIN that are minority — larger = harder
- Composite: equal-weight combination of normalized scores.

## Retraining
- **Performed**: Yes (predictions did not contain ENROLID).
- Saved predictions with ENROLID, y_true, split, score, model_tag for both ours and rnd.

## Split Reconstruction
- Same as exp6: train_test_split_enrol test_size=0.3, then 0.5 of remainder for val/test.
- TRAIN_TEST_SEED=123.
- Train controls from ours_1to1_s{args.seed}.csv and rnd_1to1_s{args.seed}.csv.

## Results Summary
- Overall iid test: AUC ours={overall['auc_ours']:.4f}, rnd={overall['auc_rnd']:.4f}
- Overall iid test: PR-AUC ours={overall['prauc_ours']:.4f}, rnd={overall['prauc_rnd']:.4f}

## Desirable Pattern?
- Desired: random better overall iid, but curated catches up or wins on hardest subsets.
- Check metrics_summary.csv and figures for subset-wise comparison.

## Caveats
- Hardness uses Gower on all prediction features (including cost); possible leakage if cost
  is part of the decision boundary.
- Subgroup sample sizes for hard_5 and hard_2 may be small; metrics can be unstable.
"""
    with open(os.path.join(args.outdir, "INTERPRETATION.md"), "w") as f:
        f.write(memo)
    print(f"\nSaved INTERPRETATION.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
