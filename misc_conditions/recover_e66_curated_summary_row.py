#!/usr/bin/env python3
"""
Recover the missing E66 top2pct curated summary row (Option B).

Uses the saved OCT model and preprocessor to compute validation-set thresholds,
then applies them to the existing test predictions so metrics match the main script.

1. Same data load + train/val/test split as curated_vs_random_1to1_multi_cohort_oct_flex_target.
2. Build preprocessor from undersampled_curated_1to1.csv (same as training).
3. Load OCT via interpretableai.Predictor(oct_model_*.json).
4. Get validation predictions → compute G-mean threshold on val.
5. Apply val G-mean threshold to test predictions (from CSV); recall@spec≥0.6 on test curve.
6. Append one row to summary_curated_vs_random_flex_target.csv.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.exceptions import NotFittedError

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__) or os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

from public.model_IAI import (
    train_test_split_enrol,
    best_balanced_threshold,
    recall_at_specificity,
    get_preprocessor_with_impute,
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
)
from vanilla_oct_multi_cohort_flex_target import (
    pick_target_and_features,
    _merge_annual_cost_from_with_meds,
)

# Load saved OCT via Predictor (works without Julia)
try:
    from interpretableai import Predictor as IAIPredictor
except ImportError:
    try:
        from interpretableai import iai
        IAIPredictor = iai.Predictor
    except ImportError as e:
        raise ImportError(
            "Need interpretableai (Predictor or iai.Predictor). "
            "Install with: pip install interpretableai (or use your OCT env)."
        ) from e

# Paths (match main script and your run)
CODE = "E66"
TARGET_PCT = 2
BASELINE_YEAR = 2017
OUTCOME_YEAR = 2018
TRAIN_TEST_SEED = 123
FEATURES_DIR = Path("/Users/cat2510/my_projects/misc_conditions/misc_conditions_features_with_meds")
OUTPUT_ROOT = Path("/Users/cat2510/scratch/oct_curated_vs_random_flex_target")
RUN_DIR = OUTPUT_ROOT / f"{CODE}_top{TARGET_PCT}pct/curated_M80000_random"
MODEL_JSON = RUN_DIR / "oct_model_curated_top2pct_M80000_random_7_100_0.001.json"
PREDICTIONS_PATH = RUN_DIR / "predictions/oct_predictions_curated_top2pct_M80000_random_7_100_0.001.csv"
UNDERSAMPLED_CSV = RUN_DIR / "undersampled_curated_1to1.csv"
SUMMARY_CSV = OUTPUT_ROOT / "summary_curated_vs_random_flex_target.csv"


def resolve_feature_path(features_dir: Path, code: str, baseline_year: int, outcome_year: int) -> Path:
    candidates = [
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_100feat.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.parquet",
        features_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No features parquet found for {code}. Tried: {candidates}")


def get_feature_names_from_preprocessor(preprocessor, X_train):
    """Mirror feature name extraction from model_IAI.finetune_oct (ColumnTransformer)."""
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            try:
                if hasattr(transformer, "named_steps") and "ohe" in transformer.named_steps:
                    ohe = transformer.named_steps["ohe"]
                    feature_names.extend(ohe.get_feature_names_out(columns))
                elif hasattr(transformer, "get_feature_names_out"):
                    feature_names.extend(transformer.get_feature_names_out(columns))
            except (NotFittedError, AttributeError, ValueError):
                pass
        elif name == "ohe" and columns:
            try:
                feature_names.extend(transformer.get_feature_names_out(columns))
            except (NotFittedError, AttributeError, ValueError):
                pass
        elif name == "num":
            feature_names.extend(columns)
        elif name == "binary":
            feature_names.extend(columns)
        elif name == "remainder" and getattr(preprocessor, "remainder", None) == "passthrough":
            all_cols = X_train.columns.tolist()
            used = []
            for _, _, cols in preprocessor.transformers_[:-1]:
                used.extend(cols)
            feature_names.extend(c for c in all_cols if c not in used)
    return feature_names


def main():
    feat_path = resolve_feature_path(FEATURES_DIR, CODE, BASELINE_YEAR, OUTCOME_YEAR)
    df = pd.read_parquet(str(feat_path))
    df = _merge_annual_cost_from_with_meds(
        df, feat_path, FEATURES_DIR, CODE, BASELINE_YEAR, OUTCOME_YEAR
    )
    target_col, feature_cols = pick_target_and_features(
        df, BASELINE_YEAR, OUTCOME_YEAR, target_pct=TARGET_PCT
    )
    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df, train_size=250_000, stratify=df[target_col], random_state=TRAIN_TEST_SEED
        )
    train_ids, test_ids, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED
    )
    val_ids, test_ids, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=TRAIN_TEST_SEED
    )
    y_test = np.asarray(test_pd[target_col].values)
    nP = int((train_pd[target_col] == 1).sum())

    undersampled = pd.read_csv(UNDERSAMPLED_CSV)
    for c in feature_cols:
        if c not in undersampled.columns:
            raise ValueError(f"Undersampled CSV missing feature column: {c}")
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)
    preprocessor = get_preprocessor_with_impute(
        undersampled[feature_cols], CAT_COLUMNS, TRUE_NUM_COLUMNS,
        binary_cols=BIN_FLAG_COLUMNS, verbose=False
    )
    preprocessor.fit(undersampled[feature_cols])
    feature_names = get_feature_names_from_preprocessor(preprocessor, undersampled[feature_cols])
    X_val_transformed = preprocessor.transform(val_pd[feature_cols])
    if hasattr(X_val_transformed, "toarray"):
        X_val_transformed = X_val_transformed.toarray()
    # Predictor expects exactly the model's feature names and order
    with open(MODEL_JSON) as f:
        model_json = json.load(f)
    data = model_json.get("prb_", {}).get("data", {})
    model_feature_names = data["features"]["feature_names"]
    # Build DataFrame in model order: use our columns when present, else 0 (e.g. extra OHE level)
    reordered = []
    for c in model_feature_names:
        if c in feature_names:
            idx = feature_names.index(c)
            reordered.append(X_val_transformed[:, idx])
        else:
            reordered.append(np.zeros(len(X_val_transformed)))
    X_val_processed = pd.DataFrame(
        np.column_stack(reordered), columns=model_feature_names
    )
    y_val = np.asarray(val_pd[target_col].values)

    predictor = IAIPredictor(str(MODEL_JSON))
    # Predictor expects list of dicts (or DataFrame with columns that are valid Python identifiers).
    # Column names like "2017Q4_total_claims_3month" get renamed by itertuples(), so pass records.
    X_val_records = X_val_processed.to_dict("records")
    y_proba_val = predictor.predict_proba(X_val_records)
    if hasattr(y_proba_val, "iloc"):
        y_proba_val = np.asarray(y_proba_val.iloc[:, 1])
    else:
        y_proba_val = np.asarray(y_proba_val)
        if y_proba_val.ndim >= 2:
            y_proba_val = y_proba_val[:, 1]

    balanced = best_balanced_threshold(y_val, y_proba_val)
    gmean_threshold = balanced["gmean_opt"]["threshold"]

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    if len(pred_df) != len(y_test):
        raise RuntimeError(
            f"Predictions rows ({len(pred_df)}) != test set size ({len(y_test)}). "
            "Split or data may not match the original run."
        )
    y_proba_test = pred_df["predicted_proba"].values.astype(float)

    y_pred_gmean = (y_proba_test >= gmean_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_gmean).ravel()
    balanced_recall_gmean_test = tp / (tp + fn) if (tp + fn) else 0.0
    balanced_specificity_gmean_test = tn / (tn + fp) if (tn + fp) else 0.0

    recall_at_spec_06, achieved_spec_06, _ = recall_at_specificity(
        y_test, y_proba_test, target_specificity=0.60
    )
    auc = roc_auc_score(y_test, y_proba_test)
    pr_auc = average_precision_score(y_test, y_proba_test)
    num_leaves = int(pred_df["leaf_assignment"].nunique())

    target_suffix = "2"
    row = {
        "code": CODE,
        "method": "curated",
        "target_col": target_col,
        "target_suffix": target_suffix,
        "seed": np.nan,
        "nP": nP,
        "nC": nP,
        "M_pool": 80000,
        "seed_method": "random",
        "matching_mean_cost": np.nan,
        "best_depth": 7,
        "best_minbucket": 100,
        "best_cp": 0.001,
        "val_pr_auc": np.nan,
        "num_leaves": num_leaves,
        "matching_time_s": np.nan,
        "training_time_s": np.nan,
        "total_time_s": np.nan,
        "undersample_path": str(UNDERSAMPLED_CSV),
        "run_dir": str(RUN_DIR),
        "test_recall_at_best_gmean": balanced_recall_gmean_test,
        "test_specificity_at_best_gmean": balanced_specificity_gmean_test,
        "test_recall_at_specfloor": recall_at_spec_06,
        "test_specificity_at_specfloor": achieved_spec_06,
        "test_auc": auc,
        "test_pr_auc": pr_auc,
    }
    df_row = pd.DataFrame([row])

    file_exists = os.path.isfile(SUMMARY_CSV) and SUMMARY_CSV.stat().st_size > 0
    if file_exists:
        with open(SUMMARY_CSV, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
    df_row.to_csv(SUMMARY_CSV, mode="a", index=False, header=not file_exists)
    print(f"Appended 1 row to {SUMMARY_CSV} (Option B: val-set thresholds)")
    print(row)


if __name__ == "__main__":
    main()
