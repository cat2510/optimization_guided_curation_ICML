#!/usr/bin/env python3
"""
CLI for the UCI credit-fraud competing-methods benchmark (adaptive OCT grid).

Equivalent intent to ``competing_methods_fraud_dataset_parameter_oct.ipynb``:
Vanilla (optional cached preds), Ours (k-center), resampler baselines, leaf deltas, summary CSV.

Run from the directory that contains ``creditcard.csv``, or pass ``--workdir``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


# -----------------------------------------------------------------------------
# Defaults (match notebook)
# -----------------------------------------------------------------------------
TRAIN_TEST_SEED = 123
TARGET_COL = "target"
RESULTS_DIR = "./uci_competing_methods_results"

REF_N_TRAIN_OCT = 199_364
REF_OCT_DEPTHS = [5, 7]
REF_OCT_MINBUCKETS = [50, 100, 150]
REF_OCT_CPS = [1e-5, 1e-3]

VANILLA_PREDS_CANDIDATE = os.environ.get(
    "VANILLA_PREDS_CANDIDATE",
    "/Users/cat2510/scratch/credit_fraud_main/uci_experiments_results/predictions/fraud_vanilla_oct_predictions.csv",
)
USE_CACHED_VANILLA_FOR_LEAF_REF = os.environ.get("USE_CACHED_VANILLA_FOR_LEAF_REF", "1") not in (
    "0",
    "false",
    "False",
)
SKIP_VANILLA_TRAIN_IF_CACHE_VALID = os.environ.get("SKIP_VANILLA_TRAIN_IF_CACHE_VALID", "1") not in (
    "0",
    "false",
    "False",
)

OURS_MATCHING_RATIO = 1
OURS_CASE_WEIGHTING = None
OURS_USE_ADAPTIVE_POOL = True
OURS_SEED_METHOD = "smart"


def oct_hyperparameter_grid(n_train: int, _n_positive: int):
    """OCT grids with an anchor + strong scaling for SMOTE-like blow-ups.

    Small train (n_train < 4k, ~0.2%-of-majority scale): minbuckets [25,50,100,150].
    For REF_N_TRAIN_OCT and below (but >= 4k): [50,100,150].
    For n_train around 2x REF (~398k), we scale minbuckets more aggressively
    (linear in n/REF) and increase cp regularization.
    """
    n_train = max(int(n_train), 2)

    if n_train < 4000:
        depths = [7]
    elif n_train < 15_000:
        depths = [7, 9]
    else:
        depths = list(REF_OCT_DEPTHS)

    # ~0.2%-of-majority-scale train (n_train < 4k): allow smaller leaves than [50,100,150].
    if n_train < 4000:
        minbuckets = [25] + list(REF_OCT_MINBUCKETS)
    elif n_train <= REF_N_TRAIN_OCT:
        minbuckets = list(REF_OCT_MINBUCKETS)
    elif n_train < 1.5 * REF_N_TRAIN_OCT:
        scale = (n_train / REF_N_TRAIN_OCT) ** 0.5
        minbuckets = [int(round(b * scale)) for b in REF_OCT_MINBUCKETS]
    else:
        scale = n_train / REF_N_TRAIN_OCT
        minbuckets = [int(round(b * scale)) for b in REF_OCT_MINBUCKETS]

    cap = n_train - 1
    minbuckets = sorted({min(mb, cap) for mb in minbuckets})
    minbuckets = [mb for mb in minbuckets if mb >= 25]
    if not minbuckets:
        minbuckets = [max(10, min(cap, n_train // 3 or 1))]

    if n_train >= 2 * REF_N_TRAIN_OCT:
        cps = [1e-4, 1e-3, 1e-2]
    elif n_train > 150_000:
        cps = [1e-2]
    else:
        cps = [1e-5]

    return depths, minbuckets, cps


def print_oct_grid_for_majority_scenarios(
    n_majority: int = 199_020,
) -> None:
    """Print grids for the four illustrative n_train sizes (second arg to grid is unused)."""
    scenarios = [
        ("0.2% * n_majority", 0.002 * n_majority),
        ("99.6% * n_majority", 0.996 * n_majority),
        ("99.8% * n_majority", 0.998 * n_majority),
        ("2 * n_majority", 2 * n_majority),
    ]
    print(f"n_majority = {n_majority:,}")
    print(f"REF_N_TRAIN_OCT = {REF_N_TRAIN_OCT:,}  (2 * REF = {2 * REF_N_TRAIN_OCT:,})")
    print()
    for label, n_raw in scenarios:
        n_int = int(round(n_raw))
        d, m, c = oct_hyperparameter_grid(n_int, 1)
        print(f"{label}")
        print(f"  n_train = {n_int:,}  (raw = {n_raw:.6g})")
        print(f"  depths    = {d}")
        print(f"  minbuckets = {m}")
        print(f"  cps       = {c}")
        print()


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name))


def _verify_cached_vanilla_preds(csv_path: str, X_test_df: pd.DataFrame, feature_cols: list) -> bool:
    if not os.path.isfile(csv_path):
        print(f"[vanilla cache] File missing: {csv_path}")
        return False
    van = pd.read_csv(csv_path)
    required = {"predicted_proba", "leaf_assignment", *feature_cols}
    missing = required - set(van.columns)
    if missing:
        print(f"[vanilla cache] Missing columns: {missing}")
        return False
    if len(van) != len(X_test_df):
        print(f"[vanilla cache] Row count: csv={len(van):,} vs test={len(X_test_df):,}")
        return False
    A = X_test_df[feature_cols].to_numpy(dtype=np.float64)
    B = van[feature_cols].to_numpy(dtype=np.float64)
    if not np.allclose(A, B, rtol=1e-5, atol=1e-6, equal_nan=True):
        print("[vanilla cache] Feature matrix mismatch vs current test split (order or values).")
        return False
    print("[vanilla cache] OK - features and row count match this notebook's test set.")
    return True


def _metrics_from_cached_vanilla_csv(y_test_series: pd.Series, pred_tbl: pd.DataFrame, best_mcc_fn) -> dict:
    y = np.asarray(y_test_series).astype(int)
    p = np.asarray(pred_tbl["predicted_proba"]).astype(float)
    auc = float(roc_auc_score(y, p))
    pr_auc = float(average_precision_score(y, p))
    mcc_pack = best_mcc_fn(y, p)
    y_hat = mcc_pack["y_pred"]
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    recall_mcc = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": float(mcc_pack["mcc"]),
        "recall_mcc": recall_mcc,
    }


def apply_sampler_return_train_features(
    name: str,
    sampler,
    train_pd: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list,
    target_col: str,
    n_majority_train: int,
):
    X_np = np.asarray(X_train[feature_cols], dtype=np.float64)
    y_np = np.asarray(y_train).astype(int)

    print(f"\n--- Resampling: {name} ---")
    print(f"    Before: n={len(y_np):,}  maj={int((y_np == 0).sum()):,}  min={int((y_np == 1).sum()):,}")

    X_res, y_res = sampler.fit_resample(X_np, y_np)

    if isinstance(sampler, SMOTE):
        n_syn = int(len(y_res) - len(y_np))
        print(f"    SMOTE added {n_syn:,} synthetic minority rows (not in train_pd index).")
        X_out = pd.DataFrame(X_res, columns=feature_cols)
        y_out = pd.Series(y_res, name=target_col)
    else:
        idx = getattr(sampler, "sample_indices_", None)
        if idx is None:
            raise RuntimeError(f"{name}: expected sample_indices_ from undersampler")
        idx = np.asarray(idx, dtype=int)
        if X_res.shape[0] != idx.size:
            print(f"    WARN: len(X_res)={X_res.shape[0]} vs len(sample_indices_)={idx.size}")
        tr_sub = train_pd.iloc[idx].reset_index(drop=True)
        if not np.array_equal(tr_sub[target_col].to_numpy(), y_res):
            raise RuntimeError(f"{name}: train_pd.iloc[sample_indices_] labels != y_res")
        X_out = tr_sub[feature_cols]
        y_out = tr_sub[target_col]
        print(f"    sample_indices_: {idx.size:,} rows, {len(np.unique(idx)):,} unique")

    maj_after = int((y_out == 0).sum())
    min_after = int((y_out == 1).sum())
    keep_pct = 100.0 * maj_after / n_majority_train
    print(f"    After:  n={len(y_out):,}  maj={maj_after:,}  min={min_after:,}  (% maj retained: {keep_pct:.2f}%)")
    return X_out, y_out, keep_pct


def run_benchmark(args: argparse.Namespace) -> None:
    workdir = os.path.abspath(args.workdir)
    os.chdir(workdir)
    repo_root = os.path.abspath(os.path.join(workdir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import (
        NearMiss,
        TomekLinks,
        EditedNearestNeighbours,
        NeighbourhoodCleaningRule,
    )

    import public.model_IAI
    import utils_fraud
    import kcenter_hyperparameter_search_global
    import symmetric_excess_AUC

    if args.reload:
        importlib.reload(public.model_IAI)
        importlib.reload(utils_fraud)
        importlib.reload(kcenter_hyperparameter_search_global)
        importlib.reload(symmetric_excess_AUC)

    from public.model_IAI import finetune_oct, evaluate_binary_oct, best_mcc_threshold
    from utils_fraud import (
        prepare_dataset_for_kcenter,
        setup_feature_columns,
        create_train_test_split,
        precompute_case_control_distances,
    )
    from kcenter_hyperparameter_search_global import run_global_kcenter_matching, build_undersampled_dataset
    from symmetric_excess_AUC import symmetric_leaf_evaluation_oct

    seed = args.seed
    target_col = args.target_col
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    data_path = args.data_csv
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data not found: {data_path} (cwd={os.getcwd()})")

    # --- closures over train_eval (val/test filled after load_data) ---
    X_val = y_val = X_test = y_test = None
    CAT_COLUMNS: list = []
    TRUE_NUM_COLUMNS: list = []
    BIN_COLUMNS: list = []

    def train_eval_finetuned_oct(method_name: str, X_tr, y_tr, majority_retained_pct: float) -> dict:
        save_slug = _safe_slug(method_name)
        method_dir = os.path.join(results_dir, save_slug)
        os.makedirs(method_dir, exist_ok=True)

        n_tr = len(X_tr)
        n_pos = int(np.asarray(y_tr).astype(int).sum())
        depths, minbuckets, cps = oct_hyperparameter_grid(n_tr, n_pos)
        print(
            f"\n[OCT grid] {method_name}: n_train={n_tr:,}, n_pos={n_pos:,} -> "
            f"depths={depths}, minbuckets={minbuckets}, cps={cps}"
        )

        model, best_params, _, preprocessor, feature_names = finetune_oct(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            categorical_cols=CAT_COLUMNS,
            numeric_cols=TRUE_NUM_COLUMNS,
            binary_cols=BIN_COLUMNS,
            depths=depths,
            minbuckets=minbuckets,
            cps=cps,
            random_seed=seed,
            verbose=True,
        )
        metrics = evaluate_binary_oct(
            iai_model=model,
            X_test_df=X_test,
            y_test=y_test,
            preprocessor=preprocessor,
            feature_names=feature_names,
            results_dir=method_dir,
            save_suffix=save_slug,
            X_val_df=X_val,
            y_val=y_val,
        )
        pred_path = os.path.join(method_dir, "predictions", f"oct_predictions_{save_slug}.csv")
        return {
            "method": method_name,
            "best_params": best_params,
            "majority_retained_pct": float(majority_retained_pct),
            "auc": float(metrics["auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "mcc": float(metrics["best_mcc"]),
            "recall": float(metrics["recall_mcc"]),
            "pred_path": pred_path,
        }

    df_raw = pd.read_csv(data_path)
    X_raw = df_raw.drop(columns=["Class"])
    y_raw = df_raw["Class"]

    df = prepare_dataset_for_kcenter(X_raw, y_raw, dataset_name="creditcard_fraud")
    df, feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_COLUMNS = setup_feature_columns(
        df, target_col=target_col, drop_time=False, log1p_amount=False
    )
    train_pd, val_pd, test_pd = create_train_test_split(df, target_col=target_col, random_state=seed)

    X_train = train_pd[feature_cols].copy()
    y_train = train_pd[target_col].copy()
    X_val = val_pd[feature_cols].copy()
    y_val = val_pd[target_col].copy()
    X_test = test_pd[feature_cols].copy()
    y_test = test_pd[target_col].copy()
    n_majority_train = int((y_train == 0).sum())

    rows = []
    vanilla_mv_pred_path = None

    use_cache = USE_CACHED_VANILLA_FOR_LEAF_REF and not args.no_vanilla_cache
    skip_train = SKIP_VANILLA_TRAIN_IF_CACHE_VALID and not args.force_vanilla_train
    cache_ok = use_cache and _verify_cached_vanilla_preds(VANILLA_PREDS_CANDIDATE, X_test, feature_cols)

    if cache_ok and skip_train:
        print(
            "\n[INFO] Using cached Vanilla predictions - skipping Vanilla OCT training.\n"
            "       Vanilla MCC/Recall: MCC threshold chosen on **test** (others use val from evaluate_binary_oct).\n"
            "       Set --force-vanilla-train to retrain Vanilla for val-aligned metrics."
        )
        van_tbl = pd.read_csv(VANILLA_PREDS_CANDIDATE)
        mcached = _metrics_from_cached_vanilla_csv(y_test, van_tbl, best_mcc_threshold)
        vanilla_mv_pred_path = os.path.abspath(VANILLA_PREDS_CANDIDATE)
        rows.append(
            {
                "method": "Vanilla",
                "majority_retained_pct": 100.0,
                "auc": mcached["auc"],
                "pr_auc": mcached["pr_auc"],
                "mcc": mcached["best_mcc"],
                "recall": mcached["recall_mcc"],
                "pred_path": vanilla_mv_pred_path,
                "best_params": None,
            }
        )
    else:
        if cache_ok:
            print("[INFO] Cache valid but training Vanilla (--force-vanilla-train or cache disabled).")
        rows.append(train_eval_finetuned_oct("Vanilla", X_train, y_train, 100.0))
        vanilla_mv_pred_path = rows[-1]["pred_path"]

    if not args.skip_ours:
        print("\n=== Ours (M_c): two-stage k-center ===")
        pn_h5_path, _, _ = precompute_case_control_distances(
            train_df=train_pd,
            target_col=target_col,
            feature_cols=feature_cols,
            cat_columns=CAT_COLUMNS,
            true_num_columns=TRUE_NUM_COLUMNS,
            dataset_name="creditcard_fraud",
            seed=seed,
        )
        matching_result = run_global_kcenter_matching(
            train_pd=train_pd,
            target_col=target_col,
            feature_cols=feature_cols,
            pn_h5_path=pn_h5_path,
            matching_ratio=OURS_MATCHING_RATIO,
            case_weighting=OURS_CASE_WEIGHTING,
            use_adaptive_pool=OURS_USE_ADAPTIVE_POOL,
            seed_method=OURS_SEED_METHOD,
            CAT_COLUMNS=CAT_COLUMNS,
            TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
            COST_COLUMNS=[],
            dnn_out_dir=f"./precomputed_distances/global_dnn_uci_seed_{seed}",
        )
        ours_train = build_undersampled_dataset(
            train_pd=train_pd,
            matching_result=matching_result,
            target_col=target_col,
            matching_ratio=OURS_MATCHING_RATIO,
        )
        ours_keep = 100.0 * (ours_train[target_col].eq(0).sum() / n_majority_train)
        rows.append(
            train_eval_finetuned_oct("Ours (M_c)", ours_train[feature_cols], ours_train[target_col], ours_keep)
        )

    resamplers = {
        "NearMiss-1": NearMiss(version=1, n_neighbors=3),
        "TomekLinks": TomekLinks(sampling_strategy="majority"),
        "ENN": EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, kind_sel="all"),
        "NCL": NeighbourhoodCleaningRule(sampling_strategy="majority", n_neighbors=3, kind_sel="all"),
        "SMOTE": SMOTE(random_state=seed),
    }

    for name, sampler in resamplers.items():
        X_rs, y_rs, keep_pct = apply_sampler_return_train_features(
            name, sampler, train_pd, X_train, y_train, feature_cols, target_col, n_majority_train
        )
        rows.append(train_eval_finetuned_oct(name, X_rs, y_rs, keep_pct))

    ref_path = vanilla_mv_pred_path or next(r["pred_path"] for r in rows if r["method"] == "Vanilla")
    for r in rows:
        if r["method"] == "Vanilla":
            r["delta_s_given_o"] = np.nan
            r["delta_o_given_s"] = np.nan
            continue
        out = symmetric_leaf_evaluation_oct(
            mv_pred_path=ref_path,
            ms_pred_path=r["pred_path"],
            y_test=y_test.reset_index(drop=True),
            enrolid_col=None,
            B=1000,
            rng=seed,
        )
        r["delta_s_given_o"] = float(out["scores"]["excess_ROC_s|v"])
        r["delta_o_given_s"] = float(out["scores"]["excess_ROC_v|s"])

    df_out = pd.DataFrame(rows)
    order = ["Ours (M_c)", "NearMiss-1", "NCL", "ENN", "TomekLinks", "SMOTE", "Vanilla"]
    df_out["_order"] = df_out["method"].map(lambda x: order.index(x) if x in order else 999)
    df_out = df_out.sort_values(["_order", "pr_auc"], ascending=[True, False]).drop(columns=["_order"])

    final_table = df_out[
        ["method", "majority_retained_pct", "pr_auc", "auc", "mcc", "recall", "delta_s_given_o", "delta_o_given_s"]
    ].rename(
        columns={
            "method": "Method",
            "majority_retained_pct": "% Majority Retained",
            "pr_auc": "PR-AUC",
            "auc": "AUC",
            "mcc": "MCC",
            "recall": "Recall",
            "delta_s_given_o": "Delta_s|o",
            "delta_o_given_s": "Delta_o|s",
        }
    )
    final_table["% Majority Retained"] = final_table["% Majority Retained"].map(lambda v: f"{v:.1f}%")
    for c in ["PR-AUC", "AUC", "MCC", "Recall", "Delta_s|o", "Delta_o|s"]:
        final_table[c] = final_table[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")

    print(final_table.to_string(index=False))
    out_csv = os.path.join(results_dir, "uci_competing_methods_summary.csv")
    final_table.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UCI fraud competing methods (adaptive OCT grid)")
    p.add_argument("--workdir", default=".", help="Directory containing creditcard.csv")
    p.add_argument("--data-csv", default="creditcard.csv", help="Filename of data CSV under workdir")
    p.add_argument("--results-dir", default=RESULTS_DIR, help="Where to write method folders + summary CSV")
    p.add_argument("--seed", type=int, default=TRAIN_TEST_SEED)
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--reload", action="store_true", help="importlib.reload project modules")
    p.add_argument("--print-oct-grid-only", action="store_true", help="Print oct_hyperparameter_grid for demo sizes and exit")
    p.add_argument("--n-majority", type=int, default=199_020, help="For --print-oct-grid-only: majority count anchor")
    p.add_argument("--skip-ours", action="store_true", help="Skip k-center Ours (M_c) block")
    p.add_argument("--no-vanilla-cache", action="store_true", help="Do not use cached vanilla CSV even if present")
    p.add_argument("--force-vanilla-train", action="store_true", help="Train Vanilla OCT even if cache is valid")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.print_oct_grid_only:
        print_oct_grid_for_majority_scenarios(n_majority=args.n_majority)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
