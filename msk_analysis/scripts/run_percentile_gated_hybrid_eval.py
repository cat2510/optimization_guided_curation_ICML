#!/usr/bin/env python
"""
run_percentile_gated_hybrid_eval.py
===================================
Percentile-gated hybrid inference: route hard patients to curated OCT, easy to random OCT.

Uses msk_2017_18_full.parquet, tfidf_svd_cosine_qcost distances (from distances_dir),
runs random 1:1 and ours two-stage 1:1 sampling in-script, trains both OCTs,
calibrates, chooses gate q on validation, evaluates on test and hard subgroups.

Usage:
  cd msk_analysis
  python scripts/run_percentile_gated_hybrid_eval.py
  python scripts/run_percentile_gated_hybrid_eval.py --parquet_path msk_2017_18_full.parquet --distances_dir /path/to/precomputed
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

# Path setup (same as analyze_hard_negative_boundary_eval.py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)

warnings.filterwarnings("ignore", category=UserWarning)

from public.model_IAI import (
    train_test_split_enrol,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    finetune_oct,
)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
# Reuse helpers from analyze script (uses exp6 for Gower)
from analyze_hard_negative_boundary_eval import (
    build_gower_features,
    compute_hardness_scores,
    train_and_save_predictions_with_enrolid,
    TARGET_COL,
    OCT_DEPTHS,
    OCT_MINBUCKETS,
    OCT_CPS,
    EPS,
    DEFAULT_K,
)
import exp6_distance_metric_ablation as exp6
from exp6_distance_metric_ablation import (
    ensure_distances_for_metric,
    run_ours_1to1_sampling,
    run_rnd_1to1_sampling,
)

# Gate percentile candidates (validation selects best)
GATE_Q_CANDIDATES = [50, 40, 30, 20, 15, 10, 5, 2]
HARD_PCT = [100, 50, 30, 20, 10, 5, 2]  # hard subsets for subgroup eval
TRAIN_TEST_SEED = 123

def parse_args():
    p = argparse.ArgumentParser(
        description="Percentile-gated hybrid: curated for hard, random for easy"
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/Users/cat2510/my_projects/msk_analysis/hybrid_gated_with_meds_gower_s0",
        help="Output directory",
    )
    p.add_argument("--parquet_path", type=str, default="msk_2017_18_full.parquet")
    p.add_argument(
        "--distances_dir",
        type=str,
        default="/Users/cat2510/scratch/precomputed_distances_msk_za_tfidf_svd_cosine_qcost",
        help="Distances dir for tfidf_svd_cosine_qcost (Gower P-N + DNN); TRAIN_TEST_SEED=123",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_retrain", action="store_true")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    return p.parse_args()


def compute_hardness_for_all_samples(
    X_query: np.ndarray,
    X_train_min: np.ndarray,
    X_train_maj: np.ndarray,
    bin_col_indices: np.ndarray,
    k: int = DEFAULT_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hardness for any query points (val or test, any label).
    Same formula as compute_hardness_scores. Returns (hardness_ratio, hardness_margin, minority_frac).
    hardness_composite = 0.33*norm(h_ratio) + 0.33*norm(h_margin) + 0.34*h_frac
    """
    h_ratio, h_margin, h_frac, _ = compute_hardness_scores(
        X_query, X_train_min, X_train_maj, bin_col_indices, k=k
    )
    h_ratio_norm = h_ratio / (np.nanmax(h_ratio) + EPS)
    h_margin_norm = h_margin / (np.nanmax(h_margin) + EPS)
    hardness_composite = 0.33 * h_ratio_norm + 0.33 * h_margin_norm + 0.34 * h_frac
    return hardness_composite, h_ratio, h_margin


def calibrate_scores(
    score_raw: np.ndarray, y_true: np.ndarray, isotonic: IsotonicRegression
) -> np.ndarray:
    """Apply fitted isotonic regression to raw scores."""
    s = np.clip(score_raw, 0.0, 1.0).reshape(-1, 1)
    return np.clip(isotonic.predict(s.flatten()), 1e-6, 1 - 1e-6)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PERCENTILE-GATED HYBRID EVALUATION")
    print("=" * 80)

    # Load parquet (resolve path if needed)
    parquet_path = args.parquet_path
    if not os.path.exists(parquet_path):
        for alt in [
            os.path.join(os.getcwd(), args.parquet_path),
            os.path.join(parent_dir, "msk_analysis", args.parquet_path),
            os.path.join(_scripts_dir, "..", args.parquet_path),
        ]:
            if os.path.exists(alt):
                parquet_path = alt
                break
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {args.parquet_path} (tried cwd, parent/msk_analysis)")
    print(f"Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if TARGET_COL not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[TARGET_COL] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    exclude_cols = ["ENROLID", TARGET_COL] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_ids, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=TARGET_COL, test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=TARGET_COL, test_size=0.5, verbose=False, random_state=TRAIN_TEST_SEED
    )
    train_enrolids = set(train_pd["ENROLID"].astype(np.int64))
    val_enrolids = set(val_pd["ENROLID"].astype(np.int64))
    test_enrolids = set(test_pd["ENROLID"].astype(np.int64))
    cases = train_pd[train_pd[TARGET_COL] == 1]
    controls = train_pd[train_pd[TARGET_COL] == 0]
    N = len(cases)
    case_enrolids = cases["ENROLID"].values.astype(np.int64)
    control_enrolids = controls["ENROLID"].values.astype(np.int64)
    M_pool = len(controls) // 2

    print(f"Train: {len(train_pd):,}  Val: {len(val_pd):,}  Test: {len(test_pd):,}")

    # Ensure distances (tfidf_svd_cosine_qcost: DNN + Gower P-N) and run sampling
    os.makedirs(args.distances_dir, exist_ok=True)
    print(f"\nDistances dir: {args.distances_dir}")
    print("--- Ensuring distances (tfidf_svd_cosine_qcost) ---")
    pn_h5, dnn_mat, dnn_ids = ensure_distances_for_metric(
        "gower",
        cases,
        controls,
        feature_cols,
        CAT_COLUMNS,
        TRUE_NUM_COLUMNS,
        BIN_FLAG_COLUMNS,
        args.distances_dir,
        use_hdf5_dnn=False,
    )
    sampled_dir = outdir / "sampled"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    ours_path = sampled_dir / f"ours_1to1_s{args.seed}.csv"
    rnd_path = sampled_dir / f"rnd_1to1_s{args.seed}.csv"
    if not ours_path.exists() or not rnd_path.exists():
        print("\n--- Running sampling (1:1 random and ours two-stage) ---")
        ours_train = exp6.run_ours_1to1_sampling(
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
        rnd_train = exp6.run_rnd_1to1_sampling(
            control_enrolids, cases, controls, N, args.seed
        )
        ours_train.to_csv(ours_path, index=False)
        rnd_train.to_csv(rnd_path, index=False)
        print(f"  Saved {ours_path} ({len(ours_train):,} rows), {rnd_path} ({len(rnd_train):,} rows)")
    else:
        ours_train = pd.read_csv(ours_path)
        rnd_train = pd.read_csv(rnd_path)
        print(f"  Loaded sampled: ours {len(ours_train):,}, rnd {len(rnd_train):,}")

    # Predictions: reuse from outdir if sufficient (same split/data)
    pred_ours_path = outdir / f"predictions_curated_s{args.seed}.csv"
    pred_rnd_path = outdir / f"predictions_random_s{args.seed}.csv"
    pred_sufficient = (
        pred_ours_path.exists()
        and pred_rnd_path.exists()
        and all(
            col in pd.read_csv(pred_ours_path, nrows=0).columns
            for col in ["ENROLID", "split", "y_true", "score"]
        )
    )

    def _filter_predictions_to_split(pred: pd.DataFrame) -> pd.DataFrame:
        mask = (
            ((pred["split"] == "val") & (pred["ENROLID"].astype(np.int64).isin(val_enrolids)))
            | ((pred["split"] == "test") & (pred["ENROLID"].astype(np.int64).isin(test_enrolids)))
        )
        return pred[mask].copy()

    use_loaded_preds = False
    if args.skip_retrain and pred_sufficient:
        print("\n[REUSE] Loading predictions from outdir")
        pred_ours = _filter_predictions_to_split(pd.read_csv(pred_ours_path))
        pred_rnd = _filter_predictions_to_split(pd.read_csv(pred_rnd_path))
        pred_val_ids = set(pred_ours[pred_ours["split"] == "val"]["ENROLID"].astype(np.int64))
        pred_test_ids = set(pred_ours[pred_ours["split"] == "test"]["ENROLID"].astype(np.int64))
        missing_val = val_enrolids - pred_val_ids
        missing_test = test_enrolids - pred_test_ids
        if missing_val or missing_test:
            print(
                f"  [WARN] Loaded predictions missing {len(missing_val)} val, {len(missing_test)} test ENROLIDs; "
                "retraining for full alignment"
            )
        else:
            use_loaded_preds = True

    if not use_loaded_preds:
        print("\n--- Retraining OCT (curated) ---")
        pred_ours_path_out = outdir / f"predictions_curated_s{args.seed}.csv"
        pred_ours = train_and_save_predictions_with_enrolid(
            ours_train,
            val_pd,
            test_pd,
            feature_cols,
            TARGET_COL,
            CAT_COLUMNS,
            TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS,
            str(pred_ours_path_out),
            "curated",
        )
        pred_ours["model_tag"] = "curated"
        pred_ours.to_csv(pred_ours_path_out, index=False)
        print("\n--- Retraining OCT (random) ---")
        pred_rnd_path_out = outdir / f"predictions_random_s{args.seed}.csv"
        pred_rnd = train_and_save_predictions_with_enrolid(
            rnd_train,
            val_pd,
            test_pd,
            feature_cols,
            TARGET_COL,
            CAT_COLUMNS,
            TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS,
            str(pred_rnd_path_out),
            "random",
        )
        pred_rnd["model_tag"] = "random"
        pred_rnd.to_csv(pred_rnd_path_out, index=False)

    pred_ours_val = pred_ours[pred_ours["split"] == "val"][["ENROLID", "score"]].rename(
        columns={"score": "score_curated"}
    )
    pred_ours_test = pred_ours[pred_ours["split"] == "test"][["ENROLID", "score"]].rename(
        columns={"score": "score_curated"}
    )
    pred_rnd_val = pred_rnd[pred_rnd["split"] == "val"][["ENROLID", "score"]].rename(
        columns={"score": "score_random"}
    )
    pred_rnd_test = pred_rnd[pred_rnd["split"] == "test"][["ENROLID", "score"]].rename(
        columns={"score": "score_random"}
    )

    # Build Gower features and hardness (full original train reference)
    train_combined = pd.concat([cases, controls], ignore_index=True)
    X_train_min, ohe, _ = build_gower_features(
        cases, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe_fit_df=train_combined
    )
    X_train_maj, _, _ = build_gower_features(
        controls, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe=ohe, ohe_fit_df=train_combined
    )
    X_train_all_temp = np.vstack([X_train_min, X_train_maj])
    ranges = np.ones(X_train_all_temp.shape[1])
    n_bin = sum(1 for c in BIN_FLAG_COLUMNS if c in feature_cols)
    if CAT_COLUMNS and any(c in feature_cols for c in CAT_COLUMNS):
        n_bin += ohe.transform(
            train_combined[[c for c in CAT_COLUMNS if c in feature_cols]]
        ).shape[1]
    for j in range(n_bin, X_train_all_temp.shape[1]):
        r = float(np.nanmax(X_train_all_temp[:, j]) - np.nanmin(X_train_all_temp[:, j]))
        ranges[j] = r if r > 0 else 1.0
    bin_col_indices = np.array(list(range(n_bin)))

    # Hardness for val and test (all samples)
    X_val, _, _ = build_gower_features(
        val_pd, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe=ohe, ohe_fit_df=train_combined
    )
    X_test, _, _ = build_gower_features(
        test_pd, feature_cols, BIN_FLAG_COLUMNS, CAT_COLUMNS, TRUE_NUM_COLUMNS,
        ohe=ohe, ohe_fit_df=train_combined
    )

    print("\n--- Computing hardness for val and test ---")
    h_val, _, _ = compute_hardness_for_all_samples(
        X_val, X_train_min, X_train_maj, bin_col_indices, k=args.k
    )
    h_test, _, _ = compute_hardness_for_all_samples(
        X_test, X_train_min, X_train_maj, bin_col_indices, k=args.k
    )

    # Validation data
    val_df = (
        val_pd[["ENROLID", TARGET_COL]]
        .merge(pred_ours_val, on="ENROLID", how="left")
        .merge(pred_rnd_val, on="ENROLID", how="left")
    )
    val_df["hardness"] = h_val
    val_df["split"] = "val"

    # Calibration on validation
    mask = val_df[["score_curated", "score_random"]].notna().all(axis=1)
    X_cal = val_df.loc[mask]
    y_cal = X_cal[TARGET_COL].astype(int).values
    iso_curated = IsotonicRegression(out_of_bounds="clip")
    iso_curated.fit(X_cal["score_curated"].values, y_cal)
    iso_random = IsotonicRegression(out_of_bounds="clip")
    iso_random.fit(X_cal["score_random"].values, y_cal)

    val_df["score_curated_cal"] = calibrate_scores(
        val_df["score_curated"].values, val_df[TARGET_COL].values, iso_curated
    )
    val_df["score_random_cal"] = calibrate_scores(
        val_df["score_random"].values, val_df[TARGET_COL].values, iso_random
    )
    val_df["score_avg"] = 0.5 * val_df["score_curated_cal"] + 0.5 * val_df["score_random_cal"]

    # Test data
    test_df = (
        test_pd[["ENROLID", TARGET_COL]]
        .merge(pred_ours_test, on="ENROLID", how="left")
        .merge(pred_rnd_test, on="ENROLID", how="left")
    )
    test_df["hardness"] = h_test
    test_df["split"] = "test"
    test_df["score_curated_cal"] = calibrate_scores(
        test_df["score_curated"].values, test_df[TARGET_COL].values, iso_curated
    )
    test_df["score_random_cal"] = calibrate_scores(
        test_df["score_random"].values, test_df[TARGET_COL].values, iso_random
    )
    test_df["score_avg"] = 0.5 * test_df["score_curated_cal"] + 0.5 * test_df["score_random_cal"]

    # Gate selection on validation
    hardness_val = val_df["hardness"].dropna().values
    best_q = None
    best_val_auc = -1
    val_metrics_by_q = []

    for q in GATE_Q_CANDIDATES:
        t_q = np.percentile(hardness_val, 100 - q)
        routed_to_curated = val_df["hardness"].values >= t_q
        score_hybrid = np.where(
            routed_to_curated,
            val_df["score_curated_cal"].values,
            val_df["score_random_cal"].values,
        )
        mask = ~np.isnan(score_hybrid) & val_df[TARGET_COL].notna()
        if mask.sum() < 10 or val_df.loc[mask, TARGET_COL].nunique() < 2:
            auc = np.nan
            prauc = np.nan
        else:
            auc = roc_auc_score(val_df.loc[mask, TARGET_COL].astype(int), score_hybrid[mask])
            prauc = average_precision_score(
                val_df.loc[mask, TARGET_COL].astype(int), score_hybrid[mask]
            )
        val_metrics_by_q.append({"q": q, "t_q": t_q, "auc": auc, "prauc": prauc})
        if not np.isnan(auc) and auc > best_val_auc:
            best_val_auc = auc
            best_q = q

    val_metrics_df = pd.DataFrame(val_metrics_by_q)
    if best_q is None:
        best_q = 20
        print(f"[WARN] No valid q; defaulting to {best_q}")
    t_q_best = float(np.percentile(hardness_val, 100 - best_q))
    print(f"\n--- Best gate on validation: q={best_q}, t_q={t_q_best:.6f} ---")

    # Apply frozen gate to test
    routed = test_df["hardness"].values >= t_q_best
    test_df["gate_percentile_q"] = best_q
    test_df["gate_threshold"] = t_q_best
    test_df["routed_to_curated"] = routed.astype(int)
    test_df["score_hybrid"] = np.where(
        routed,
        test_df["score_curated_cal"].values,
        test_df["score_random_cal"].values,
    )

    # Same for val (for completeness)
    val_df["gate_percentile_q"] = best_q
    val_df["gate_threshold"] = t_q_best
    val_df["routed_to_curated"] = (val_df["hardness"].values >= t_q_best).astype(int)
    val_df["score_hybrid"] = np.where(
        val_df["routed_to_curated"].astype(bool),
        val_df["score_curated_cal"].values,
        val_df["score_random_cal"].values,
    )

    # Metrics
    def eval_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
        mask = df[score_col].notna() & df[TARGET_COL].notna()
        if mask.sum() < 10 or df.loc[mask, TARGET_COL].nunique() < 2:
            return {"auc": np.nan, "prauc": np.nan}
        y = df.loc[mask, TARGET_COL].astype(int).values
        s = df.loc[mask, score_col].values
        return {
            "auc": roc_auc_score(y, s),
            "prauc": average_precision_score(y, s),
        }

    methods = [
        ("curated", "score_curated_cal"),
        ("random", "score_random_cal"),
        ("avg_ensemble", "score_avg"),
        (f"hybrid_q{best_q}", "score_hybrid"),
    ]

    metrics_rows = []
    for split_name, dff in [("val", val_df), ("test", test_df)]:
        for method, col in methods:
            m = eval_metrics(dff, col)
            m["method"] = method
            m["split"] = split_name
            metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(outdir / "metrics_summary.csv", index=False)
    print(f"Saved {outdir / 'metrics_summary.csv'}")

    # Subgroup evaluation (test only)
    test_maj = test_pd[test_pd[TARGET_COL] == 0]
    test_min_ids = set(test_pd[test_pd[TARGET_COL] == 1]["ENROLID"].astype(int))
    hardness_maj = pd.DataFrame({
        "ENROLID": test_maj["ENROLID"].values,
        "hardness": h_test[test_pd[TARGET_COL].values == 0],
    })
    subgroup_rows = []
    for pct in HARD_PCT:
        if pct == 100:
            hard_ids = set(test_maj["ENROLID"].astype(int))
        else:
            thresh = np.percentile(hardness_maj["hardness"].values, 100 - pct)
            hard_ids = set(
                hardness_maj[hardness_maj["hardness"] >= thresh]["ENROLID"].astype(int)
            )
        enrolids = list(test_min_ids | hard_ids)
        sub = test_df[test_df["ENROLID"].isin(enrolids)].dropna(
            subset=["score_curated_cal", "score_random_cal", "score_avg", "score_hybrid"]
        )
        if len(sub) < 10 or sub[TARGET_COL].nunique() < 2:
            continue
        y = sub[TARGET_COL].astype(int).values
        subgroup_rows.append({
            "subset": f"hard_{pct}",
            "n": len(sub),
            "auc_curated": roc_auc_score(y, sub["score_curated_cal"].values),
            "auc_random": roc_auc_score(y, sub["score_random_cal"].values),
            "auc_avg_ensemble": roc_auc_score(y, sub["score_avg"].values),
            f"auc_hybrid_q{best_q}": roc_auc_score(y, sub["score_hybrid"].values),
            "prauc_curated": average_precision_score(y, sub["score_curated_cal"].values),
            "prauc_random": average_precision_score(y, sub["score_random_cal"].values),
            "prauc_avg_ensemble": average_precision_score(y, sub["score_avg"].values),
            f"prauc_hybrid_q{best_q}": average_precision_score(y, sub["score_hybrid"].values),
        })

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(outdir / "subgroup_metrics.csv", index=False)
    print(f"Saved {outdir / 'subgroup_metrics.csv'}")

    # Save predictions
    out_cols = [
        "ENROLID", "split", "y_true",
        "score_curated", "score_random", "score_curated_cal", "score_random_cal",
        "score_avg", "hardness", "gate_percentile_q", "gate_threshold",
        "routed_to_curated", "score_hybrid",
    ]
    val_out = val_df.copy()
    val_out["y_true"] = val_out[TARGET_COL]
    val_out = val_out[[c for c in out_cols if c in val_out.columns]]
    val_out.to_csv(outdir / "predictions_validation.csv", index=False)
    test_out = test_df.copy()
    test_out["y_true"] = test_out[TARGET_COL]
    test_out = test_out[[c for c in out_cols if c in test_out.columns]]
    test_out.to_csv(outdir / "predictions_test.csv", index=False)
    print(f"Saved {outdir / 'predictions_validation.csv'}, {outdir / 'predictions_test.csv'}")

    # Plots
    sns.set_style("whitegrid")

    # Validation metric vs q
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_metrics_df["q"], val_metrics_df["auc"], "o-", label="ROC-AUC")
    ax.axvline(best_q, color="gray", ls="--", label=f"chosen q={best_q}")
    ax.set_xlabel("Gate percentile q (top q% → curated)")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Validation AUC vs gate percentile q")
    ax.legend()
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(outdir / "fig_validation_auc_vs_q.png", dpi=150)
    plt.close()

    # Test comparison
    test_m = metrics_df[metrics_df["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(test_m))
    axes[0].bar([i - 0.2 for i in x], test_m["auc"], 0.4, label="ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(test_m["method"])
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Test ROC-AUC by method")
    axes[1].bar([i - 0.2 for i in x], test_m["prauc"], 0.4, label="PR-AUC", color="C1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(test_m["method"])
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("Test PR-AUC by method")
    plt.tight_layout()
    plt.savefig(outdir / "fig_test_comparison.png", dpi=150)
    plt.close()

    # Hardness histogram with threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(test_df["hardness"].dropna(), bins=50, alpha=0.7, label="Test hardness")
    ax.axvline(t_q_best, color="red", ls="--", linewidth=2, label=f"Gate t_q (q={best_q})")
    ax.set_xlabel("Hardness")
    ax.set_ylabel("Count")
    ax.set_title("Test hardness distribution with chosen gate threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_hardness_histogram.png", dpi=150)
    plt.close()

    # Subgroup AUC curves
    if len(subgroup_df) > 0 and "auc_curated" in subgroup_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(subgroup_df))
        ax.plot(x, subgroup_df["auc_curated"], "o-", label="Curated", color="C0")
        ax.plot(x, subgroup_df["auc_random"], "s-", label="Random", color="C1")
        ax.plot(x, subgroup_df["auc_avg_ensemble"], "^-", label="Avg ensemble", color="C2")
        ax.plot(x, subgroup_df[f"auc_hybrid_q{best_q}"], "d-", label=f"Hybrid q={best_q}", color="C3")
        ax.set_xticks(x)
        ax.set_xticklabels(subgroup_df["subset"])
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Hard subset ROC-AUC (minority + top X% hardest majority)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(outdir / "fig_subgroup_auc.png", dpi=150)
        plt.close()

    # Interpretation memo
    test_curated = metrics_df[(metrics_df["split"] == "test") & (metrics_df["method"] == "curated")]
    test_random = metrics_df[(metrics_df["split"] == "test") & (metrics_df["method"] == "random")]
    test_hybrid = metrics_df[(metrics_df["split"] == "test") & (metrics_df["method"] == f"hybrid_q{best_q}")]
    curated_auc = test_curated["auc"].values[0] if len(test_curated) else np.nan
    random_auc = test_random["auc"].values[0] if len(test_random) else np.nan
    hybrid_auc = test_hybrid["auc"].values[0] if len(test_hybrid) else np.nan

    memo = f"""# Percentile-Gated Hybrid Evaluation — Summary

## Setup
- **Retraining**: {"No (reused aligned predictions)" if use_loaded_preds else "Yes"}
- **Data**: msk_2017_18_full.parquet; Split: train_test_split_enrol, TRAIN_TEST_SEED=123
- **Sampling**: tfidf_svd_cosine_qcost (DNN + Gower P-N) from distances_dir; ours 1:1 + rnd 1:1
- **Hardness**: Full original train, Gower on all prediction features, k={args.k}
- **Calibration**: Isotonic on validation; applied to both models before ensemble/hybrid

## Gate Selection
- **Percentile reference**: Validation hardness distribution
- **Candidates q**: {GATE_Q_CANDIDATES}
- **Chosen q**: {best_q} (validation ROC-AUC)
- **Gate threshold t_q**: {t_q_best:.6f}

## Results

### Validation
| Method | AUC | PR-AUC |
|--------|-----|--------|
"""
    for _, row in metrics_df[metrics_df["split"] == "val"].iterrows():
        memo += f"| {row['method']} | {row['auc']:.4f} | {row['prauc']:.4f} |\n"

    memo += f"""
### Test
| Method | AUC | PR-AUC |
|--------|-----|--------|
"""
    for _, row in metrics_df[metrics_df["split"] == "test"].iterrows():
        memo += f"| {row['method']} | {row['auc']:.4f} | {row['prauc']:.4f} |\n"

    memo += f"""
## Answers
- **Hybrid vs random baseline on test**: {"Yes" if not np.isnan(hybrid_auc) and hybrid_auc > random_auc else "No"} (hybrid AUC {hybrid_auc:.4f} vs random {random_auc:.4f})
- **Hard-subset advantage preserved**: See subgroup_metrics.csv and fig_subgroup_auc.png

## Caveats
- Percentile threshold depends on validation hardness distribution
- Possible calibration mismatch between models
- If hybrid does not improve overall or is unstable, report plainly
"""

    with open(outdir / "summary.md", "w") as f:
        f.write(memo)
    print(f"Saved {outdir / 'summary.md'}")

    # Experiment setup doc
    setup_md = """# Percentile-Gated Hybrid — Experiment Setup

## Data & Split
- Parquet: msk_2017_18_full.parquet
- Split: train_test_split_enrol, TRAIN_TEST_SEED=123
- Distances: tfidf_svd_cosine_qcost (from distances_dir)
- Sampling: ours 1:1 (two-stage) + rnd 1:1

## Hardness & Gate
- Hardness: full original train, Gower on prediction features, k={k}
- Gate: t_q from validation hardness; q chosen by validation ROC-AUC
""".format(k=args.k)
    with open(outdir / "experiment_setup.md", "w") as f:
        f.write(setup_md)
    print(f"Saved {outdir / 'experiment_setup.md'}")

    # Final print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Chosen gate: q={best_q}, t_q={t_q_best:.6f}")
    print("\nValidation performance:")
    for _, row in metrics_df[metrics_df["split"] == "val"].iterrows():
        print(f"  {row['method']}: AUC={row['auc']:.4f} PR-AUC={row['prauc']:.4f}")
    print("\nTest performance:")
    for _, row in metrics_df[metrics_df["split"] == "test"].iterrows():
        print(f"  {row['method']}: AUC={row['auc']:.4f} PR-AUC={row['prauc']:.4f}")
    print(f"\nHybrid improves over random: {hybrid_auc > random_auc}")
    print(f"Output dir: {outdir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
