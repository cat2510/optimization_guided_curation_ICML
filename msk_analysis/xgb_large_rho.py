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
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
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
    from public.model_IAI import get_cat_columns, best_mcc_threshold, best_balanced_threshold

except ImportError:
    def get_cat_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

from public.two_stage_kcenter_match import two_stage_kcenter_then_match

# oct_distillation lives in distillation_with_OCT
sys.path.insert(0, os.path.join(_REPO_ROOT, "distillation_with_OCT"))

# Matching ratios to sweep: 1, 5, 10, 15, 20
MATCHING_RATIOS =  [1, 5, 10, 15, 20]
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

def compute_scores_from_predictions(
    y_proba: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """
    Load a predictions CSV and compute metrics (AUC, PR-AUC, MCC, etc.).
    Use when retraining is skipped because predictions already exist.
    """
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

    from public.model_IAI import recall_at_specificity
    recall_at_spec_06, achieved_spec_06, threshold_spec_06 = recall_at_specificity(y_test_arr, y_proba, target_specificity=0.60)
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "best_mcc": best_mcc,
        "recall_mcc": recall_mcc,
        "specificity_mcc": specificity_mcc,
        "optimal_f1": optimal_f1,
        "balanced_recall_gmean": gmean_recall,
        "balanced_specificity_gmean": gmean_specificity,
        "recall_at_specificity_0.6": float(recall_at_spec_06),
        "achieved_specificity_0.6": float(achieved_spec_06),
        "threshold_specificity_0.6": float(threshold_spec_06),
    }


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


def kcenter_then_train_xgb_default_no_weight(
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
    seed_method: str,
    use_kmeanspp: bool,
    undersampled_dir: str,
    verbose: bool,
) -> dict:
    """Run two_stage_kcenter_then_match for one ratio, train XGB/RF, return test metrics."""
    n_cases = len(cases)
    n_controls = len(controls)
    if n_controls < ratio * n_cases:
        return {"matching_ratio": ratio, "error": f"Not enough controls ({n_controls}) for 1:{ratio} (need >= {ratio * n_cases})"}

    result = two_stage_kcenter_then_match(
        leaf_controls_enrolids=controls["ENROLID"].values.astype(np.int64),
        leaf_cases_enrolids=cases["ENROLID"].values.astype(np.int64),
        leaf_nn_matrix_npy=dnn_matrix_npy,
        leaf_nn_enrolids_npy=dnn_enrolids_npy,
        pn_h5_path=pn_h5_path,
        M=M,
        use_adaptive_pool=False,
        use_kmeanspp=use_kmeanspp,
        plateau_eps=0.01,
        force_nearest_per_case=False,
        force_topm=1,
        assignment_topk_start=None,
        seed_method=seed_method,
        matching_ratio=ratio,
        X_majority_leaf=None,
        case_weighting=None,
    )

    selected_control_enrolids = result["selected_control_enrolids"]
    unique_majority = list(set(selected_control_enrolids))
    all_minority = train_pd[train_pd[target_col] == 1].copy()
    selected_majority = train_pd[
        (train_pd[target_col] == 0) & (train_pd["ENROLID"].isin(unique_majority))
    ].copy()
    undersampled = pd.concat([all_minority, selected_majority], axis=0, ignore_index=True)

    os.makedirs(undersampled_dir, exist_ok=True)
    out_path = os.path.join(undersampled_dir, f"ratio_{ratio}_kmeanspp_{use_kmeanspp}.csv")
    undersampled.to_csv(out_path, index=False)
    if verbose:
        print(f"    Saved undersampled: {out_path} (n={len(undersampled)})")

    return undersampled

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
    metrics = compute_scores_from_predictions(y_test_proba, y_test)
    n_min = int((undersampled_df[target_col] == 1).sum())
    n_maj = int((undersampled_df[target_col] == 0).sum())
    return {
        "n_train": len(undersampled_df),
        "n_train_minority": n_min,
        "n_train_majority": n_maj,
        "pr_auc": float(metrics["pr_auc"]),
        "auc": float(metrics["auc"]),
        "best_mcc": float(metrics["best_mcc"]),
        "recall_mcc": float(metrics["recall_mcc"]),
        "specificity_mcc": float(metrics["specificity_mcc"]),
        "optimal_f1": float(metrics["optimal_f1"]),
        "balanced_recall_gmean": float(metrics["balanced_recall_gmean"]),
        "balanced_specificity_gmean": float(metrics["balanced_specificity_gmean"]),
        "recall_at_specificity_0.6": float(metrics["recall_at_specificity_0.6"]),
        "achieved_specificity_0.6": float(metrics["achieved_specificity_0.6"]),
        "threshold_specificity_0.6": float(metrics["threshold_specificity_0.6"]),
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
    use_kmeanspp: bool,
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
    kcenter_rows = []
    random_rows = []
    for ratio in ratios:
        path = os.path.join(undersampled_dir, f"ratio_{ratio}_kmeanspp_{use_kmeanspp}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Undersampled file not found: {path}")
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
            print(f"    k-center: PR-AUC={row_k['pr_auc']:.4f}, AUC={row_k['auc']:.4f}")
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
    parser.add_argument("--distances_dir", type=str, default="precomputed_distances_msk_medical_only",
                        help="Directory with distances_majority_minority.h5 and global_dnn_*")
    parser.add_argument("--output_dir", type=str, default="xgb_matching_ratio_results_kmeanspp",
                        help="Output dir for undersampled datasets, CSV, and plot")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "rf"])
    parser.add_argument("--ratios", type=int, nargs="+", default=MATCHING_RATIOS,
                        help="Matching ratios to test")
    parser.add_argument("--M", type=int, default=None,
                        help="Pool size M for k-center (default: min(150000, n_controls))")
    parser.add_argument("--seed_method", type=str, default="smart")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--early_stop_rounds", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=TRAIN_TEST_SEED, help="Random seed for train/test split (default: 123)")
    parser.add_argument("--no_drop_high_corr", action="store_true", help="Do not drop high-corr features")
    parser.add_argument("--compare_random", action="store_true",
                        help="After k-center run (or using existing undersampled_datasets), train XGB default no weight on k-center data and on 10 random same-size undersamples; save comparison CSV and plot (AUC, PR-AUC, MCC)")
    parser.add_argument("--n_random_seeds", type=int, default=3,
                        help="Number of random seeds per ratio for --compare_random (default 3)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--use_kmeanspp", action="store_true",
                        help="Use k-means++ for seed selection")
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
    M = args.M if args.M is not None else min(100000, n_controls//2)
    M = max(M, max(args.ratios) * n_cases)

    print(f"Train: {len(train_pd):,} (cases: {n_cases:,}, controls: {n_controls:,})")
    print(f"Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"Model: {args.model}, Ratios: {args.ratios}, M: {M:,}\n")

    for ratio in args.ratios:
        path = os.path.join(undersampled_dir, f"ratio_{ratio}_kmeanspp_{args.use_kmeanspp}.csv")
        if not os.path.exists(path):
            _ = kcenter_then_train_xgb_default_no_weight(
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
                seed_method=args.seed_method,
                use_kmeanspp=args.use_kmeanspp,
                undersampled_dir=undersampled_dir,
                verbose=args.verbose,
                )

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
                use_kmeanspp=args.use_kmeanspp,
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
            if os.path.exists(comp_path):
                comp_df.to_csv(comp_path, mode='a', header=False, index=False)
            else:
                comp_df.to_csv(comp_path, index=False)
            print(f"\nSaved comparison to {comp_path}")
    print("Done.")


if __name__ == "__main__":
    main()
