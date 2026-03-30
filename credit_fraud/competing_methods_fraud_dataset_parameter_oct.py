#!/usr/bin/env python3
"""
CLI for the UCI credit-fraud competing-methods benchmark (fixed OCT grid).

Equivalent intent to ``competing_methods_fraud_dataset_parameter_oct.ipynb``:
Vanilla (optional cached preds), resampler baselines, leaf deltas, summary CSV.

Run from the directory that contains ``creditcard.csv``, or pass ``--workdir``.

cd /Users/cat2510/my_projects/credit_fraud
screen -S fraud_oct
# activate your env if needed; use a persistent session (tmux/screen, remote shell, VNC desktop, etc.) for long runs
python3 competing_methods_fraud_dataset_parameter_oct.py > uci_competing_methods_run_new.log &
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

# Fixed OCT search for every method (depth × minbucket × cp; see ``finetune_oct``).
OCT_DEPTHS = [5, 7, 9]
OCT_MINBUCKETS = [25, 50, 100]
OCT_CPS = [1e-5, 1e-4, 1e-3, 1e-2]

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


def print_fixed_oct_grid() -> None:
    """Print the fixed OCT search used for all methods (same as ``finetune_oct`` inputs)."""
    print("Fixed OCT hyperparameter search (all methods):")
    print(f"  depths     = {OCT_DEPTHS}")
    print(f"  minbuckets = {OCT_MINBUCKETS}")
    print(f"  cps        = {OCT_CPS}")


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

    # Avoid module-level SMOTE import (imblearn only loaded inside run_benchmark).
    if type(sampler).__name__ == "SMOTE":
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
    import symmetric_excess_AUC

    if args.reload:
        importlib.reload(public.model_IAI)
        importlib.reload(utils_fraud)
        importlib.reload(symmetric_excess_AUC)

    from public.model_IAI import finetune_oct, evaluate_binary_oct, best_mcc_threshold
    from utils_fraud import (
        prepare_dataset_for_kcenter,
        setup_feature_columns,
        create_train_test_split,
    )
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
        print(
            f"\n[OCT grid] {method_name}: n_train={n_tr:,}, n_pos={n_pos:,} -> "
            f"depths={OCT_DEPTHS}, minbuckets={OCT_MINBUCKETS}, cps={OCT_CPS}"
        )

        model, best_params, _, preprocessor, feature_names = finetune_oct(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            categorical_cols=CAT_COLUMNS,
            numeric_cols=TRUE_NUM_COLUMNS,
            binary_cols=BIN_COLUMNS,
            depths=OCT_DEPTHS,
            minbuckets=OCT_MINBUCKETS,
            cps=OCT_CPS,
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
    order = ["NearMiss-1", "NCL", "ENN", "TomekLinks", "SMOTE", "Vanilla"]
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
    p = argparse.ArgumentParser(description="UCI fraud competing methods (fixed OCT grid)")
    p.add_argument("--workdir", default=".", help="Directory containing creditcard.csv")
    p.add_argument("--data-csv", default="creditcard.csv", help="Filename of data CSV under workdir")
    p.add_argument("--results-dir", default=RESULTS_DIR, help="Where to write method folders + summary CSV")
    p.add_argument("--seed", type=int, default=TRAIN_TEST_SEED)
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--reload", action="store_true", help="importlib.reload project modules")
    p.add_argument(
        "--print-oct-grid-only",
        action="store_true",
        help="Print fixed OCT depths/minbuckets/cps and exit",
    )
    p.add_argument("--no-vanilla-cache", action="store_true", help="Do not use cached vanilla CSV even if present")
    p.add_argument("--force-vanilla-train", action="store_true", help="Train Vanilla OCT even if cache is valid")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.print_oct_grid_only:
        print_fixed_oct_grid()
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
