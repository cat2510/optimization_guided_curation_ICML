#!/usr/bin/env python3
"""
Sensitivity over matching ratio (1:k): undersample with two_stage_kcenter_then_match
at ratios 1, 5, 10, 15, 20; train XGB or RF with class weights (no distillation);
save undersampled datasets, results CSV, and plots (PR-AUC, AUC, recall at gmean vs ratio).

Uses MSK data and precomputed distances (same as two_stage_iterative.py).
Run from msk_analysis/ or set --data and --distances_dir as needed.

Usage:
  python sensitivity_matching_ratio_xgb_rf.py --model xgb
  python sensitivity_matching_ratio_xgb_rf.py --model rf --output_dir my_run
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from model_nonIAI_utils import (
    get_preprocessor_with_impute,
    train_test_split_enrol,
    get_bin_flag_columns,
    get_true_num_columns,
)
try:
    from public.model_IAI import get_cat_columns
except ImportError:
    def get_cat_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

from public.two_stage_kcenter_match import two_stage_kcenter_then_match

# oct_distillation lives in distillation_with_OCT
sys.path.insert(0, os.path.join(_REPO_ROOT, "distillation_with_OCT"))
from oct_distillation import compute_minority_metrics

# Matching ratios to sweep: 1, 5, 10, 15, 20
MATCHING_RATIOS = [1, 5, 10, 15, 20]
TRAIN_TEST_SEED = 123


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("DataLoad").getOrCreate()
            return spark.read.format("parquet").load(path).toPandas()
        except Exception:
            return pd.read_parquet(path)
    return pd.read_csv(path)


def ensure_target(df: pd.DataFrame, target_col: str) -> tuple:
    if target_col in df.columns:
        return df, target_col
    if "annual_cost_2018_deflated" not in df.columns:
        raise ValueError("Need top_2_pct_cost_2018 or annual_cost_2018_deflated")
    threshold = df["annual_cost_2018_deflated"].quantile(0.98)
    df = df.copy()
    df["top_2_pct_cost_2018"] = (df["annual_cost_2018_deflated"] >= threshold).astype(int)
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


def run_one_ratio(
    ratio: int,
    train_pd: pd.DataFrame,
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    CAT_COLUMNS: list,
    TRUE_NUM_COLUMNS: list,
    BIN_FLAG_COLUMNS: list,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    dnn_matrix_npy: str,
    dnn_enrolids_npy: str,
    pn_h5_path: str,
    M: int,
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    early_stop_rounds: int,
    random_state: int,
    quota_cfg: dict | None,
    seed_method: str,
    undersampled_dir: str,
    verbose: bool,
) -> dict:
    """Run two_stage_kcenter_then_match for one ratio, train XGB/RF, return test metrics."""
    n_cases = len(cases)
    n_controls = len(controls)
    if n_controls < ratio * n_cases:
        return {"matching_ratio": ratio, "error": f"Not enough controls ({n_controls}) for 1:{ratio} (need >= {ratio * n_cases})"}

    # Quota only for 1:1
    use_quota = quota_cfg if ratio == 1 else None

    result = two_stage_kcenter_then_match(
        leaf_controls_enrolids=controls["ENROLID"].values.astype(np.int64),
        leaf_cases_enrolids=cases["ENROLID"].values.astype(np.int64),
        leaf_nn_matrix_npy=dnn_matrix_npy,
        leaf_nn_enrolids_npy=dnn_enrolids_npy,
        pn_h5_path=pn_h5_path,
        M=M,
        use_adaptive_pool=False,
        plateau_eps=0.01,
        force_nearest_per_case=False,
        force_topm=1,
        assignment_topk_start=None,
        seed_method=seed_method,
        matching_ratio=ratio,
        X_majority_leaf=None,
        case_weighting=None,
        quota_cfg=use_quota,
    )

    selected_control_enrolids = result["selected_control_enrolids"]
    unique_majority = list(set(selected_control_enrolids))
    all_minority = train_pd[train_pd[target_col] == 1].copy()
    selected_majority = train_pd[
        (train_pd[target_col] == 0) & (train_pd["ENROLID"].isin(unique_majority))
    ].copy()
    undersampled = pd.concat([all_minority, selected_majority], axis=0, ignore_index=True)

    os.makedirs(undersampled_dir, exist_ok=True)
    out_path = os.path.join(undersampled_dir, f"ratio_{ratio}.csv")
    undersampled.to_csv(out_path, index=False)
    if verbose:
        print(f"    Saved undersampled: {out_path} (n={len(undersampled)})")

    X_under = undersampled[feature_cols]
    y_under = undersampled[target_col].values

    preprocessor = get_preprocessor_with_impute(
        X_under, CAT_COLUMNS, TRUE_NUM_COLUMNS, binary_cols=BIN_FLAG_COLUMNS, verbose=False
    )
    preprocessor.fit(X_under)

    X_under_p = preprocessor.transform(X_under)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    if model_type == "xgb":
        import xgboost as xgb
        n_pos = int(np.sum(y_under == 1))
        n_neg = int(np.sum(y_under == 0))
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method="hist",
            early_stopping_rounds=early_stop_rounds,
        )
        model.fit(X_under_p, y_under, eval_set=[(X_val_p, y_val)], verbose=False)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_under_p, y_under)

    y_test_proba = model.predict_proba(X_test_p)[:, 1]
    metrics = compute_minority_metrics(y_test, y_test_proba, verbose=False)

    return {
        "matching_ratio": ratio,
        "n_train": len(undersampled),
        "n_train_minority": int((undersampled[target_col] == 1).sum()),
        "n_train_majority": int((undersampled[target_col] == 0).sum()),
        "pr_auc": float(metrics["pr_auc"]),
        "auc": float(metrics["auc"]),
        "mcc": float(metrics["mcc_optimal"]["mcc"]),
        "recall_at_gmean": float(metrics["gmean_optimal"]["recall"]),
        "specificity_at_gmean": float(metrics["gmean_optimal"]["specificity"]),
        "gmean": float(metrics["gmean_optimal"]["gmean"]),
    }


def train_xgb_default_no_weight(
    undersampled_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    CAT_COLUMNS: list,
    TRUE_NUM_COLUMNS: list,
    BIN_FLAG_COLUMNS: list,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stop_rounds: int = 50,
    random_state: int = 123,
) -> dict:
    """Train XGBoost with default config and no class weighting; return test metrics (auc, pr_auc, mcc)."""
    import xgboost as xgb
    X_under = undersampled_df[feature_cols]
    y_under = undersampled_df[target_col].values
    preprocessor = get_preprocessor_with_impute(
        X_under, CAT_COLUMNS, TRUE_NUM_COLUMNS, binary_cols=BIN_FLAG_COLUMNS, verbose=False
    )
    preprocessor.fit(X_under)
    X_under_p = preprocessor.transform(X_under)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=early_stop_rounds,
    )
    model.fit(X_under_p, y_under, eval_set=[(X_val_p, y_val)], verbose=False)
    y_test_proba = model.predict_proba(X_test_p)[:, 1]
    metrics = compute_minority_metrics(y_test, y_test_proba, verbose=False)
    n_min = int((undersampled_df[target_col] == 1).sum())
    n_maj = int((undersampled_df[target_col] == 0).sum())
    return {
        "n_train": len(undersampled_df),
        "n_train_minority": n_min,
        "n_train_majority": n_maj,
        "pr_auc": float(metrics["pr_auc"]),
        "auc": float(metrics["auc"]),
        "mcc": float(metrics["mcc_optimal"]["mcc"]),
    }


def random_undersample_to_size(
    train_pd: pd.DataFrame,
    target_col: str,
    n_minority: int,
    n_majority: int,
    seed: int,
) -> pd.DataFrame:
    """Create one random undersampled DataFrame with exactly n_minority and n_majority (no class weighting)."""
    rng = np.random.default_rng(seed)
    minority = train_pd[train_pd[target_col] == 1]
    majority = train_pd[train_pd[target_col] == 0]
    if len(minority) < n_minority or len(majority) < n_majority:
        raise ValueError(
            f"Not enough data: need {n_minority} minority and {n_majority} majority; "
            f"have {len(minority)} and {len(majority)}"
        )
    idx_min = rng.choice(len(minority), size=n_minority, replace=False)
    idx_maj = rng.choice(len(majority), size=n_majority, replace=False)
    out = pd.concat([
        minority.iloc[idx_min],
        majority.iloc[idx_maj],
    ], axis=0, ignore_index=True)
    return out


def run_comparison_random_vs_kcenter(
    undersampled_dir: str,
    train_pd: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    CAT_COLUMNS: list,
    TRUE_NUM_COLUMNS: list,
    BIN_FLAG_COLUMNS: list,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ratios: list,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    early_stop_rounds: int,
    random_state_base: int,
    n_random_seeds: int = 10,
    verbose: bool = True,
) -> tuple:
    """
    Load k-center undersampled CSVs (ratio_1.csv, ratio_5.csv, ...), train XGB default no weight on each.
    For each ratio, create n_random_seeds random undersamples of the same size, train XGB default no weight.
    Returns (kcenter_rows, random_rows) where each row has matching_ratio, source, pr_auc, auc, mcc, [seed for random].
    """
    import glob
    kcenter_rows = []
    random_rows = []
    for ratio in ratios:
        path = os.path.join(undersampled_dir, f"ratio_{ratio}.csv")
        if not os.path.exists(path):
            if verbose:
                print(f"  Skip ratio {ratio}: {path} not found")
            continue
        under = pd.read_csv(path)
        n_min = int((under[target_col] == 1).sum())
        n_maj = int((under[target_col] == 0).sum())
        if verbose:
            print(f"  Ratio 1:{ratio} (n_min={n_min}, n_maj={n_maj})")
        # Train XGB default no weight on k-center data
        row_k = train_xgb_default_no_weight(
            under, feature_cols, target_col,
            CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
            X_val, y_val, X_test, y_test,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, early_stop_rounds=early_stop_rounds,
            random_state=random_state_base,
        )
        row_k["matching_ratio"] = ratio
        row_k["source"] = "kcenter"
        kcenter_rows.append(row_k)
        if verbose:
            print(f"    k-center: PR-AUC={row_k['pr_auc']:.4f}, AUC={row_k['auc']:.4f}, MCC={row_k['mcc']:.4f}")
        # Random undersample, same size, n_random_seeds seeds
        for seed in range(n_random_seeds):
            try:
                under_rand = random_undersample_to_size(train_pd, target_col, n_min, n_maj, seed=seed)
                row_r = train_xgb_default_no_weight(
                    under_rand, feature_cols, target_col,
                    CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
                    X_val, y_val, X_test, y_test,
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=learning_rate, early_stop_rounds=early_stop_rounds,
                    random_state=random_state_base + seed + 1,
                )
                row_r["matching_ratio"] = ratio
                row_r["source"] = "random"
                row_r["seed"] = seed
                random_rows.append(row_r)
            except Exception as e:
                if verbose:
                    print(f"    random seed {seed}: {e}")
    return kcenter_rows, random_rows


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity over matching ratio: undersample then train XGB/RF with class weights"
    )
    parser.add_argument("--data", type=str, default="msk_2017_18_full.parquet",
                        help="Path to parquet (relative to cwd or repo)")
    parser.add_argument("--distances_dir", type=str, default="precomputed_distances_msk_with_cost_features",
                        help="Directory with distances_majority_minority.h5 and global_dnn_*")
    parser.add_argument("--output_dir", type=str, default="xgb_matching_ratio_results",
                        help="Output dir for undersampled datasets, CSV, and plot")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "rf"])
    parser.add_argument("--ratios", type=int, nargs="+", default=MATCHING_RATIOS,
                        help="Matching ratios to test (default: 1 5 10 15 20)")
    parser.add_argument("--M", type=int, default=None,
                        help="Pool size M for k-center (default: min(150000, n_controls))")
    parser.add_argument("--seed_method", type=str, default="smart")
    parser.add_argument("--quota_cfg", action="store_true",
                        help="Use bin-quota for 1:1 only (like two_stage_iterative)")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--early_stop_rounds", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=TRAIN_TEST_SEED)
    parser.add_argument("--no_drop_high_corr", action="store_true", help="Do not drop high-corr features")
    parser.add_argument("--compare_random", action="store_true",
                        help="After k-center run (or using existing undersampled_datasets), train XGB default no weight on k-center data and on 10 random same-size undersamples; save comparison CSV and plot (AUC, PR-AUC, MCC)")
    parser.add_argument("--n_random_seeds", type=int, default=10,
                        help="Number of random seeds per ratio for --compare_random (default 10)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Resolve paths
    data_path = args.data
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        alt = os.path.join(_REPO_ROOT, "msk_analysis", data_path)
        if os.path.exists(alt):
            data_path = alt
    distances_dir = args.distances_dir
    if not os.path.isabs(distances_dir) and not os.path.exists(distances_dir):
        alt = os.path.join(os.path.dirname(__file__), distances_dir)
        if os.path.exists(alt):
            distances_dir = alt

    pn_h5_path = os.path.join(distances_dir, "distances_majority_minority.h5")
    dnn_dir = os.path.join(distances_dir, f"global_dnn_seed_{args.random_state}")
    dnn_matrix_npy = os.path.join(dnn_dir, "leaf_global_dnn_matrix.npy")
    dnn_enrolids_npy = os.path.join(dnn_dir, "leaf_global_dnn_enrolids.npy")

    for p in (pn_h5_path, dnn_matrix_npy, dnn_enrolids_npy):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path not found: {p}")

    os.makedirs(args.output_dir, exist_ok=True)
    undersampled_dir = os.path.join(args.output_dir, "undersampled_datasets")
    os.makedirs(undersampled_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data(data_path)
    df, target_col = ensure_target(df, "top_2_pct_cost_2018")
    feature_cols, CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS = prepare_features_msk(
        df, target_col, drop_high_corr=not args.no_drop_high_corr
    )

    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=args.random_state
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=args.random_state
    )

    X_val = val_pd[feature_cols]
    y_val = val_pd[target_col].values
    X_test = test_pd[feature_cols]
    y_test = test_pd[target_col].values

    cases = train_pd[train_pd[target_col] == 1].copy()
    controls = train_pd[train_pd[target_col] == 0].copy()
    n_controls = len(controls)
    n_cases = len(cases)
    M = args.M if args.M is not None else min(150000, n_controls)
    M = max(M, max(args.ratios) * n_cases)

    quota_cfg = None
    if args.quota_cfg:
        quota_cfg = {
            "enabled": False,
           # "T": 5,
           # "mode": "pool_mass",
           # "binning": "population",
           # "pop_S": 50000,
           # "pop_subset": "sorted_prefix",
           # "K_per_bin": 25,
        }

    print(f"Train: {len(train_pd):,} (cases: {n_cases:,}, controls: {n_controls:,})")
    print(f"Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"Model: {args.model}, Ratios: {args.ratios}, M: {M:,}\n")

    all_results = []
    for ratio in args.ratios:
        print(f"--- Matching ratio 1:{ratio} ---")
        try:
            row = run_one_ratio(
                ratio=ratio,
                train_pd=train_pd,
                cases=cases,
                controls=controls,
                target_col=target_col,
                feature_cols=feature_cols,
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                BIN_FLAG_COLUMNS=BIN_FLAG_COLUMNS,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                dnn_matrix_npy=dnn_matrix_npy,
                dnn_enrolids_npy=dnn_enrolids_npy,
                pn_h5_path=pn_h5_path,
                M=M,
                model_type=args.model,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                early_stop_rounds=args.early_stop_rounds,
                random_state=args.random_state,
                quota_cfg=quota_cfg,
                seed_method=args.seed_method,
                undersampled_dir=undersampled_dir,
                verbose=args.verbose,
            )
            if "error" in row:
                print(f"  Skip: {row['error']}")
            else:
                print(f"  Test PR-AUC: {row['pr_auc']:.4f}, AUC: {row['auc']:.4f}, Recall(gmean): {row['recall_at_gmean']:.4f}")
            all_results.append(row)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"matching_ratio": ratio, "error": str(e)})

    # Save CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "sensitivity_matching_ratio.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Plot: PR-AUC, AUC, recall_at_gmean vs matching_ratio
    if "error" in results_df.columns:
        plot_df = results_df[results_df["error"].isna()].copy()
    else:
        plot_df = results_df.copy()
    if len(plot_df) == 0 or "matching_ratio" not in plot_df.columns:
        plot_df = results_df.copy()
    plot_df = plot_df.sort_values("matching_ratio")

    if len(plot_df) > 0 and "pr_auc" in plot_df.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            x = plot_df["matching_ratio"].values
            ax.plot(x, plot_df["pr_auc"], "o-", label="PR-AUC", color="C0")
            ax.plot(x, plot_df["auc"], "s-", label="AUC", color="C1")
            if "mcc" in plot_df.columns:
                ax.plot(x, plot_df["mcc"], "d-", label="MCC", color="C2")
            ax.plot(x, plot_df["recall_at_gmean"], "^-", label="Recall (at G-mean threshold)", color="C3")
            ax.set_xlabel("Matching ratio (1:k)")
            ax.set_ylabel("Score")
            ax.set_xticks(x)
            ax.legend()
            ax.set_title(f"Sensitivity to matching ratio ({args.model}, class-weighted)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, "sensitivity_matching_ratio.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved plot to {plot_path}")
        except Exception as e:
            print(f"Plot failed: {e}")
    else:
        print("No valid rows to plot.")

    # ----- Compare k-center (no weight) vs random same-size (no weight) -----
    if args.compare_random:
        csv_files = [f for f in os.listdir(undersampled_dir) if f.startswith("ratio_") and f.endswith(".csv")]
        if not csv_files:
            print("\n--compare_random: no ratio_*.csv in undersampled_datasets; run without --compare_random first to generate them.")
        else:
            print(f"\n{'='*80}")
            print("COMPARISON: k-center (no class weight) vs random same-size (no class weight)")
            print(f"{'='*80}\n")
            kcenter_rows, random_rows = run_comparison_random_vs_kcenter(
                undersampled_dir=undersampled_dir,
                train_pd=train_pd,
                target_col=target_col,
                feature_cols=feature_cols,
                CAT_COLUMNS=CAT_COLUMNS,
                TRUE_NUM_COLUMNS=TRUE_NUM_COLUMNS,
                BIN_FLAG_COLUMNS=BIN_FLAG_COLUMNS,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                ratios=args.ratios,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                early_stop_rounds=args.early_stop_rounds,
                random_state_base=args.random_state,
                n_random_seeds=args.n_random_seeds,
                verbose=args.verbose,
            )
            # Save comparison CSV
            comp_df = pd.DataFrame(kcenter_rows + random_rows)
            comp_path = os.path.join(args.output_dir, "comparison_random_vs_kcenter.csv")
            comp_df.to_csv(comp_path, index=False)
            print(f"\nSaved comparison to {comp_path}")

            # Plot: AUC, PR-AUC, MCC - k-center line vs random mean ± std
            if len(kcenter_rows) > 0 and len(random_rows) > 0:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    kc_df = pd.DataFrame(kcenter_rows).sort_values("matching_ratio")
                    rnd_df = pd.DataFrame(random_rows)
                    ratios_plot = sorted(kc_df["matching_ratio"].unique())
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    for ax, metric in zip(axes, ["pr_auc", "auc", "mcc"]):
                        ax.plot(kc_df["matching_ratio"], kc_df[metric], "o-", label="k-center (no weight)", color="C0", linewidth=2)
                        rnd_agg = rnd_df.groupby("matching_ratio")[metric].agg(["mean", "std"])
                        rnd_agg = rnd_agg.reindex(ratios_plot)
                        means = rnd_agg["mean"].values
                        stds = rnd_agg["std"].values
                        ax.errorbar(ratios_plot, means, yerr=stds, fmt="s--", capsize=4, label=f"random (no weight, n={args.n_random_seeds} seeds)", color="C1")
                        ax.set_xlabel("Matching ratio (1:k)")
                        ax.set_ylabel(metric.upper().replace("_", "-"))
                        ax.set_xticks(ratios_plot)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                    plt.suptitle("XGB default (no class weight): k-center vs random same-size undersample")
                    plt.tight_layout()
                    comp_plot_path = os.path.join(args.output_dir, "comparison_random_vs_kcenter.png")
                    plt.savefig(comp_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved comparison plot to {comp_plot_path}")
                except Exception as e:
                    print(f"Comparison plot failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
