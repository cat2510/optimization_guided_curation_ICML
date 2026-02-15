#!/usr/bin/env python3
"""
Train/evaluate OCT on *existing* k-center undersampled datasets (ratio_*.csv),
WITHOUT recomputing k-center matching.

Adds a strict leakage check: if any ENROLID in the training undersample appears in
val or test splits, raise an error.

Expected existing undersampled files (default):
  /Users/cat2510/my_projects/msk_analysis/xgb_matching_ratio_results/undersampled_datasets/ratio_1.csv
  ratio_{1,5,10,15,20,25,30}.csv

Usage:
  python train_oct_on_existing_undersamples.py
  python train_oct_on_existing_undersamples.py --undersampled_dir /path/to/undersampled_datasets
  python train_oct_on_existing_undersamples.py --tree_kind oct
  python train_oct_on_existing_undersamples.py --tree_kind both --compare_random  # optional, if you add later
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# Repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from model_nonIAI_utils import train_test_split_enrol, get_bin_flag_columns, get_true_num_columns

try:
    from public.model_IAI import get_cat_columns
except ImportError:
    def get_cat_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# OCT pipeline
from public.model_IAI import finetune_oct, evaluate_binary_oct

# metrics to match your prior outputs
sys.path.insert(0, os.path.join(_REPO_ROOT, "distillation_with_OCT"))
from oct_distillation import compute_minority_metrics


MATCHING_RATIOS_DEFAULT = [1, 5, 10, 15, 20, 25, 30]
TRAIN_TEST_SEED = 123


# --------------------------- utils ---------------------------

def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("DataLoad").getOrCreate()
            return spark.read.format("parquet").load(path).toPandas()
        except Exception:
            return pd.read_parquet(path)
    return pd.read_csv(path)


def ensure_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, str]:
    if target_col in df.columns:
        return df, target_col
    if "annual_cost_2018_deflated" not in df.columns:
        raise ValueError("Need top_2_pct_cost_2018 or annual_cost_2018_deflated")
    thr = df["annual_cost_2018_deflated"].quantile(0.98)
    df = df.copy()
    df["top_2_pct_cost_2018"] = (df["annual_cost_2018_deflated"] >= thr).astype(int)
    return df, "top_2_pct_cost_2018"


def prepare_features_msk(df: pd.DataFrame, target_col: str, drop_high_corr: bool = True) -> tuple:
    exclude = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude]

    if drop_high_corr and len(feature_cols) < 500:
        numeric_cols = df[feature_cols + [target_col]].select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols and target_col in df.columns:
            corrs = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            high = [c for c in corrs[corrs > 0.95].index if c != target_col]
            feature_cols = [c for c in feature_cols if c not in high]

    BIN = [c for c in get_bin_flag_columns(df) if c in feature_cols]
    CAT = [c for c in get_cat_columns(df) if c in feature_cols]
    NUM = [c for c in get_true_num_columns(df, CAT, BIN) if c in feature_cols]
    return feature_cols, CAT, NUM, BIN


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}m{s:.1f}"


def assert_no_enrolid_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Strict leakage guard: ENROLID sets must be disjoint across train vs val/test.
    Raises ValueError with a small sample of overlapping ids.
    """
    if "ENROLID" not in train_df.columns or "ENROLID" not in val_df.columns or "ENROLID" not in test_df.columns:
        raise ValueError("Leakage check requires ENROLID column in train/val/test dataframes.")

    train_ids = set(train_df["ENROLID"].astype(np.int64).tolist())
    val_ids = set(val_df["ENROLID"].astype(np.int64).tolist())
    test_ids = set(test_df["ENROLID"].astype(np.int64).tolist())

    overlap_val = train_ids.intersection(val_ids)
    overlap_test = train_ids.intersection(test_ids)

    if overlap_val or overlap_test:
        ov_v = sorted(list(overlap_val))[:20]
        ov_t = sorted(list(overlap_test))[:20]
        raise ValueError(
            "❌ DATA LEAKAGE DETECTED: training ENROLIDs overlap with val/test.\n"
            f"  overlap(train, val): {len(overlap_val)} (sample: {ov_v})\n"
            f"  overlap(train, test): {len(overlap_test)} (sample: {ov_t})\n"
            "Fix by regenerating splits OR ensuring the saved undersampled training CSVs are drawn only from the train split."
        )


def train_eval_oct_one_ratio(
    ratio: int,
    undersampled_path: str,
    feature_cols: list,
    target_col: str,
    CAT_COLUMNS: list,
    TRUE_NUM_COLUMNS: list,
    BIN_FLAG_COLUMNS: list,
    val_pd: pd.DataFrame,
    test_pd: pd.DataFrame,
    results_dir: str,
    depths: list[int],
    minbuckets: list[int],
    cps: list[float],
    tree_kind: str,
    hyperplane_configs,
    random_seed: int,
) -> dict:
    under = pd.read_csv(undersampled_path)

    # Leakage check requires ENROLID presence in saved undersample and val/test
    assert_no_enrolid_overlap(under, val_pd, test_pd)

    X_val = val_pd[feature_cols]
    y_val = val_pd[target_col].values
    X_test = test_pd[feature_cols]
    y_test = test_pd[target_col].values

    X_train_under = under[feature_cols]
    y_train_under = under[target_col].values

    # Train + tune
    t0 = time.perf_counter()
    model, best_params, _, preprocessor, feature_names = finetune_oct(
        X_train=X_train_under,
        y_train=y_train_under,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=CAT_COLUMNS,
        numeric_cols=TRUE_NUM_COLUMNS,
        binary_cols=BIN_FLAG_COLUMNS,
        depths=depths,
        minbuckets=minbuckets,
        cps=cps,
        tree_kind=tree_kind,
        hyperplane_configs=hyperplane_configs,
        random_seed=random_seed,
        verbose=False,
    )
    t_train = time.perf_counter() - t0

    # Evaluate (and also produce plots/artifacts via your function)
    t1 = time.perf_counter()
    oct_metrics = evaluate_binary_oct(
        model,
        X_test, y_test,
        preprocessor, feature_names,
        X_val_df=X_val, y_val=y_val,
        results_dir=results_dir,
        save_suffix=f"ratio{ratio}__{str(best_params).replace(' ', '')}",
    )
    t_eval = time.perf_counter() - t1

    # Ensure we have probabilities (prefer returned dict, else compute)
    y_test_proba = None
    if isinstance(oct_metrics, dict):
        for key in ["y_test_proba", "test_proba", "proba_test", "y_score_test"]:
            if key in oct_metrics:
                y_test_proba = np.asarray(oct_metrics[key])
                break

    if y_test_proba is None:
        X_test_p = preprocessor.transform(X_test)
        X_test_df = pd.DataFrame(X_test_p, columns=feature_names)
        y_test_proba = model.predict_proba(X_test_df).iloc[:, 1].values

    metrics = compute_minority_metrics(y_test, y_test_proba, verbose=False)

    n_min = int((under[target_col] == 1).sum())
    n_maj = int((under[target_col] == 0).sum())

    return {
        "matching_ratio": ratio,
        "undersampled_path": undersampled_path,
        "n_train": int(len(under)),
        "n_train_minority": n_min,
        "n_train_majority": n_maj,

        "pr_auc": float(metrics["pr_auc"]),
        "auc": float(metrics["auc"]),

        "recall_at_mcc": float(metrics["mcc_optimal"]["recall"]),
        "specificity_at_mcc": float(metrics["mcc_optimal"]["specificity"]),
        "mcc": float(metrics["mcc_optimal"]["mcc"]),

        "recall_at_gmean": float(metrics["gmean_optimal"]["recall"]),
        "specificity_at_gmean": float(metrics["gmean_optimal"]["specificity"]),
        "gmean": float(metrics["gmean_optimal"]["gmean"]),

        "recall_at_f1": float(metrics["f1_optimal"]["recall"]),
        "specificity_at_f1": float(metrics["f1_optimal"]["specificity"]),
        "f1": float(metrics["f1_optimal"]["f1"]),

        "oct_train_time_s": float(t_train),
        "oct_eval_time_s": float(t_eval),
        "oct_best_params": best_params,
    }


# --------------------------- main ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train OCT on existing ratio_*.csv undersamples and evaluate on fixed val/test splits with leakage check"
    )
    parser.add_argument("--data", type=str, default="msk_2017_18_full.parquet",
                        help="Path to parquet (relative to cwd or repo)")
    parser.add_argument("--undersampled_dir", type=str,
                        default="/Users/cat2510/my_projects/msk_analysis/xgb_matching_ratio_results/undersampled_datasets",
                        help="Directory containing existing ratio_*.csv undersampled datasets")
    parser.add_argument("--output_dir", type=str, default="oct_matching_ratio_results_from_existing",
                        help="Output dir for CSV + plots + OCT eval artifacts")
    parser.add_argument("--ratios", type=int, nargs="+", default=MATCHING_RATIOS_DEFAULT)

    # OCT tuning grid
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 7])
    parser.add_argument("--minbuckets", type=int, nargs="+", default=[150])
    parser.add_argument("--cps", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])

    # OCT variant
    parser.add_argument("--tree_kind", type=str, default="oct", choices=["oct", "oct_h", "both"])
    parser.add_argument("--random_state", type=int, default=TRAIN_TEST_SEED)
    parser.add_argument("--no_drop_high_corr", action="store_true", help="Do not drop high-corr features")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Resolve data path (same logic as your script)
    data_path = args.data
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        alt = os.path.join(_REPO_ROOT, "msk_analysis", data_path)
        if os.path.exists(alt):
            data_path = alt

    if not os.path.isdir(args.undersampled_dir):
        raise FileNotFoundError(f"undersampled_dir does not exist: {args.undersampled_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    oct_results_dir = os.path.join(args.output_dir, "oct_eval")
    os.makedirs(oct_results_dir, exist_ok=True)

    # Load full data, create the *same* splits as before
    print("Loading data...")
    df = load_data(data_path)
    df, target_col = ensure_target(df, "top_2_pct_cost_2018")
    feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS = prepare_features_msk(
        df, target_col, drop_high_corr=not args.no_drop_high_corr
    )

    # IMPORTANT: We only need val/test here, but we reproduce splits deterministically to verify leakage.
    _, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.random_state
    )
    _, _, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=args.random_state
    )

    # Sanity: val/test are disjoint (should be by construction)
    if len(set(val_pd["ENROLID"]).intersection(set(test_pd["ENROLID"]))) > 0:
        raise ValueError("val/test overlap on ENROLID (unexpected). Check train_test_split_enrol implementation.")

    print(f"Val: {len(val_pd):,}, Test: {len(test_pd):,}")
    print(f"Using existing undersamples in: {args.undersampled_dir}")
    print(f"OCT tree_kind={args.tree_kind} | tuning depths={args.depths} minbuckets={args.minbuckets} cps={args.cps}")
    print("Leakage check: undersample ENROLIDs must be disjoint from val/test ENROLIDs.\n")

    hyperplane_configs = None
    if args.tree_kind in {"oct_h", "both"}:
        hyperplane_configs = [{"sparsity": "all"}]

    rows = []
    for ratio in args.ratios:
        p = os.path.join(args.undersampled_dir, f"ratio_{ratio}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing undersampled file: {p}")

        if args.verbose:
            print(f"{'-'*80}\nRatio 1:{ratio} | loading {p}")

        row = train_eval_oct_one_ratio(
            ratio=ratio,
            undersampled_path=p,
            feature_cols=feature_cols,
            target_col=target_col,
            CAT_COLUMNS=CAT_COLUMNS,
            TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
            BIN_FLAG_COLUMNS=BIN_FLAG_COLUMNS,
            val_pd=val_pd,
            test_pd=test_pd,
            results_dir=oct_results_dir,
            depths=args.depths,
            minbuckets=args.minbuckets,
            cps=args.cps,
            tree_kind=args.tree_kind,
            hyperplane_configs=hyperplane_configs,
            random_seed=args.random_state,
        )
        rows.append(row)

        if args.verbose:
            print(f"  PR-AUC={row['pr_auc']:.4f} | AUC={row['auc']:.4f} | MCC={row['mcc']:.4f} "
                  f"| train={format_time(row['oct_train_time_s'])}s | eval={format_time(row['oct_eval_time_s'])}s")

    out_csv = os.path.join(args.output_dir, "oct_ratio_sweep_from_existing_undersamples.csv")
    df_out = pd.DataFrame(rows).sort_values("matching_ratio").reset_index(drop=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    # Plot metrics vs ratio
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ratios_plot = df_out["matching_ratio"].values
        for metric in ["pr_auc", "auc", "recall_at_gmean", "specificity_at_gmean", "mcc"]:
            plt.figure(figsize=(6, 4))
            plt.plot(ratios_plot, df_out[metric].values, "o-")
            plt.xlabel("Matching ratio (1:k)")
            plt.ylabel(metric.upper().replace("_", "-"))
            plt.xticks(ratios_plot)
            plt.grid(True, alpha=0.3)
            plt.title(f"OCT ({args.tree_kind}) on existing undersamples: {metric}")
            p = os.path.join(args.output_dir, f"oct_{metric}_vs_ratio.png")
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close()
            print(f"Saved plot: {p}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
