#!/usr/bin/env python3
"""
K-center curated OCT for UCI credit-card fraud (logic from uci_experiments.ipynb),
with data prep / OCT grid / logging aligned to competing_methods_fraud_dataset_parameter_oct.py.

K-center **distances** (case–control + global DNN) use features without ``Time``, and
``Amount_log = log1p(Amount)`` instead of ``Amount``. OCT still trains on the same
features as the competing script (``Time`` + raw ``Amount``).

After training and symmetric leaf evaluation vs Vanilla OCT, appends one summary row
``Ours (M_c)`` to uci_competing_methods_summary.csv (or creates the file with headers).

Run from credit_fraud (directory containing creditcard.csv):

  cd /path/to/credit_fraud
  python3 run_uci_experiments_kcenter_oct.py

Requires: creditcard.csv, parent repo on PYTHONPATH (for public/, kcenter_hyperparameter_search_global.py).
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults — match competing_methods_fraud_dataset_parameter_oct.py
# -----------------------------------------------------------------------------
TRAIN_TEST_SEED = 123
TARGET_COL = "target"
RESULTS_DIR_DEFAULT = "./uci_competing_methods_results"

OCT_DEPTHS = [5, 7, 9]
OCT_MINBUCKETS = [25,50,20]
OCT_CPS = [1e-5, 1e-4, 1e-3, 1e-2]

# K-center settings from uci_experiments.ipynb (fixed comparison run)
KCENTER_SEED_METHOD = "smart"
KCENTER_CASE_WEIGHTING = "boundary"
KCENTER_USE_ADAPTIVE_POOL = False
KCENTER_MATCHING_RATIO = 1
# Distinct from plain creditcard_fraud so H5 cache is not reused when distance features change.
DATASET_NAME_DIST = "creditcard_fraud_dist_notime_logamount"


def _add_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    """Add Amount_log = log1p(clip(Amount)) for distance geometry (skewed Amount)."""
    out = df.copy()
    if "Amount" in out.columns:
        amt = pd.to_numeric(out["Amount"], errors="coerce").fillna(0.0)
        amt = np.clip(amt.astype(float), 0.0, None)
        out["Amount_log"] = np.log1p(amt)
    return out


def build_distance_feature_cols(model_feature_cols: list[str]) -> list[str]:
    """
    Features used only for k-center distances: drop Time; use Amount_log instead of Amount.
    OCT still uses model_feature_cols (Time + raw Amount) unchanged.
    """
    out: list[str] = []
    for c in model_feature_cols:
        if c == "Time":
            continue
        if c == "Amount":
            out.append("Amount_log")
        else:
            out.append(c)
    return out


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name))


def _append_ours_row_to_summary_csv(
    summary_path: str,
    *,
    majority_retained_pct: float,
    pr_auc: float,
    auc: float,
    mcc: float,
    recall: float,
    delta_s_given_o: float,
    delta_o_given_s: float,
) -> None:
    """Append or replace row Method == 'Ours (M_c)' with formatted strings matching existing CSV."""
    new_row = {
        "Method": "Ours (M_c)",
        "% Majority Retained": f"{majority_retained_pct:.1f}%",
        "PR-AUC": f"{pr_auc:.3f}",
        "AUC": f"{auc:.3f}",
        "MCC": f"{mcc:.3f}",
        "Recall": f"{recall:.3f}",
        "Delta_s|o": f"{delta_s_given_o:.3f}",
        "Delta_o|s": f"{delta_o_given_s:.3f}",
    }
    cols = list(new_row.keys())
    if os.path.isfile(summary_path):
        df = pd.read_csv(summary_path)
        df = df[df["Method"] != "Ours (M_c)"]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".", exist_ok=True)
        df = pd.DataFrame([new_row], columns=cols)
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary row 'Ours (M_c)' -> {summary_path}")


def run(args: argparse.Namespace) -> None:
    workdir = os.path.abspath(args.workdir)
    os.chdir(workdir)

    repo_root = os.path.abspath(os.path.join(workdir, ".."))
    for p in (repo_root, workdir):
        if p not in sys.path:
            sys.path.insert(0, p)

    import kcenter_hyperparameter_search_global
    import public.model_IAI
    import symmetric_excess_AUC
    import utils_fraud

    if args.reload:
        importlib.reload(kcenter_hyperparameter_search_global)
        importlib.reload(public.model_IAI)
        importlib.reload(symmetric_excess_AUC)
        importlib.reload(utils_fraud)

    from kcenter_hyperparameter_search_global import (
        run_global_kcenter_matching,
        build_undersampled_dataset,
    )
    from public.model_IAI import finetune_oct, evaluate_binary_oct
    from symmetric_excess_AUC import symmetric_leaf_evaluation_oct
    from utils_fraud import (
        prepare_dataset_for_kcenter,
        setup_feature_columns,
        create_train_test_split,
        precompute_case_control_distances,
    )

    seed = args.seed
    target_col = args.target_col
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if args.vanilla_pred_path:
        vanilla_path = os.path.abspath(args.vanilla_pred_path)
    else:
        vanilla_path = os.path.join(
            results_dir, "Vanilla", "predictions", "oct_predictions_Vanilla.csv"
        )

    data_path = args.data_csv
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data not found: {data_path} (cwd={os.getcwd()})")

    df_raw = pd.read_csv(data_path)
    X_raw = df_raw.drop(columns=["Class"])
    y_raw = df_raw["Class"]

    df = prepare_dataset_for_kcenter(X_raw, y_raw, dataset_name="creditcard_fraud")
    df, feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_COLUMNS = setup_feature_columns(
        df, target_col=target_col, drop_time=False, log1p_amount=False
    )
    # log(Amount) for k-center distances only (see build_distance_feature_cols).
    df = _add_amount_log(df)
    train_pd, val_pd, test_pd = create_train_test_split(
        df, target_col=target_col, random_state=seed
    )

    X_train = train_pd[feature_cols].copy()
    y_train = train_pd[target_col].copy()
    X_val = val_pd[feature_cols].copy()
    y_val = val_pd[target_col].copy()
    X_test = test_pd[feature_cols].copy()
    y_test = test_pd[target_col].copy()

    n_majority_train = int((y_train == 0).sum())

    distance_feature_cols = build_distance_feature_cols(feature_cols)
    if "Amount_log" in distance_feature_cols and "Amount_log" not in train_pd.columns:
        raise ValueError("Expected Amount_log on train_pd for distance features.")
    dist_cat = [c for c in CAT_COLUMNS if c in distance_feature_cols]
    dist_true_num = [c for c in distance_feature_cols if c not in dist_cat]
    print(
        "\n[K-center distances] Excluding Time; using Amount_log instead of Amount.\n"
        f"  n_distance_features={len(distance_feature_cols)}  dataset_name={DATASET_NAME_DIST}"
    )

    # --- Precompute case–control distances (utils_fraud; scaled numeric space) ---
    h5_path, _, _ = precompute_case_control_distances(
        train_pd,
        target_col,
        distance_feature_cols,
        dist_cat,
        dist_true_num,
        DATASET_NAME_DIST,
        seed=seed,
    )

    # run_global_kcenter_matching builds the control matrix from train_pd columns (see kcenter_hyperparameter_search_global).
    # Pass a frame with only ENROLID, target, and distance features so Time/raw Amount do not enter k-center geometry.
    train_kcenter = train_pd[["ENROLID", target_col] + distance_feature_cols].copy()

    dnn_dir = os.path.join(
        workdir,
        "precomputed_distances",
        f"global_dnn_{DATASET_NAME_DIST}_seed_{seed}",
    )

    matching_result = run_global_kcenter_matching(
        train_pd=train_kcenter,
        target_col=target_col,
        feature_cols=distance_feature_cols,
        pn_h5_path=h5_path,
        matching_ratio=KCENTER_MATCHING_RATIO,
        case_weighting=KCENTER_CASE_WEIGHTING,
        use_adaptive_pool=KCENTER_USE_ADAPTIVE_POOL,
        seed_method=KCENTER_SEED_METHOD,
        candidate_pool_size=None,
        CAT_COLUMNS=dist_cat,
        TRUE_NUM_COLUMNS=dist_true_num,
        COST_COLUMNS=None,
        dnn_out_dir=dnn_dir,
    )

    undersampled = build_undersampled_dataset(
        train_pd=train_pd,
        matching_result=matching_result,
        target_col=target_col,
        matching_ratio=KCENTER_MATCHING_RATIO,
    )

    n_maj_after = int((undersampled[target_col] == 0).sum())
    majority_retained_pct = 100.0 * n_maj_after / n_majority_train if n_majority_train else 0.0

    method_label = "Ours (M_c)"
    save_slug = _safe_slug(method_label)
    method_dir = os.path.join(results_dir, save_slug)
    os.makedirs(method_dir, exist_ok=True)

    print(
        f"\n[OCT grid] {method_label}: depths={OCT_DEPTHS}, "
        f"minbuckets={OCT_MINBUCKETS}, cps={OCT_CPS} (same as competing_methods_fraud_dataset_parameter_oct.py)"
    )

    model, best_params, _, preprocessor, feature_names = finetune_oct(
        X_train=undersampled[feature_cols],
        y_train=undersampled[target_col],
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
    print(f"Best OCT params: {best_params}")

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
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Expected predictions at {pred_path}")

    if not os.path.isfile(vanilla_path):
        raise FileNotFoundError(
            f"Vanilla OCT predictions not found: {vanilla_path}\n"
            "Run competing_methods_fraud_dataset_parameter_oct.py first or pass --vanilla-pred-path."
        )

    sym = symmetric_leaf_evaluation_oct(
        mv_pred_path=vanilla_path,
        ms_pred_path=pred_path,
        y_test=y_test.reset_index(drop=True),
        enrolid_col=None,
        B=args.bootstrap_B,
        rng=seed,
    )

    delta_s = float(sym["scores"]["excess_ROC_s|v"])
    delta_o = float(sym["scores"]["excess_ROC_v|s"])

    _append_ours_row_to_summary_csv(
        os.path.join(results_dir, "uci_competing_methods_summary.csv"),
        majority_retained_pct=majority_retained_pct,
        pr_auc=float(metrics["pr_auc"]),
        auc=float(metrics["auc"]),
        mcc=float(metrics["best_mcc"]),
        recall=float(metrics["recall_mcc"]),
        delta_s_given_o=delta_s,
        delta_o_given_s=delta_o,
    )

    print("\n--- Ours (M_c) test metrics ---")
    print(
        f"  majority_retained={majority_retained_pct:.2f}%  PR-AUC={metrics['pr_auc']:.4f}  "
        f"AUC={metrics['auc']:.4f}  MCC={metrics['best_mcc']:.4f}  Recall={metrics['recall_mcc']:.4f}"
    )
    print(f"  Delta_s|o={delta_s:.4f}  Delta_o|s={delta_o:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="K-center + OCT (uci_experiments) with competing-methods preprocessing; append Ours (M_c) to summary CSV."
    )
    p.add_argument("--workdir", default=".", help="Directory containing creditcard.csv")
    p.add_argument("--data-csv", default="creditcard.csv")
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--seed", type=int, default=TRAIN_TEST_SEED)
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--reload", action="store_true", help="importlib.reload project modules")
    p.add_argument(
        "--vanilla-pred-path",
        default=None,
        help="Vanilla OCT predictions CSV. Default: <results-dir>/Vanilla/predictions/oct_predictions_Vanilla.csv",
    )
    p.add_argument(
        "--bootstrap-B",
        type=int,
        default=1000,
        help="Bootstrap replicates for symmetric_leaf_evaluation_oct (match competing script default)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
