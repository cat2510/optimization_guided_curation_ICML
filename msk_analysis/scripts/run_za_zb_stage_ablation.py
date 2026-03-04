#!/usr/bin/env python
"""
run_za_zb_stage_ablation.py
===========================
Runner for two-stage sampling with Stage A and Stage B pointing to different
distance folders. No changes to sampling logic - only path construction.

Usage:
  cd msk_analysis
  python scripts/run_za_zb_stage_ablation.py --stageA za_coarse_phenotype --stageB zb_intensity_context
  # not recommended: --stageA zb_intensity_context --stageB za_coarse_phenotype

Valid folder names: za_coarse_phenotype, zb_intensity_context, medical_only,
  with_cost_features, cost_only (must exist under precomputed_distances_msk_<name>/)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from public.model_IAI import (
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    train_test_split_enrol,
    finetune_oct,
    evaluate_binary_oct,
)
from public.two_stage_kcenter_match import two_stage_kcenter_then_match
from pyspark.sql import SparkSession

TRAIN_TEST_SEED = 123
DIST_DIR_TEMPLATE = "./precomputed_distances_msk_{distance_features}"


def get_distance_paths(distance_features: str, seed: int):
    dist_dir = DIST_DIR_TEMPLATE.format(distance_features=distance_features)
    pn_h5 = os.path.join(dist_dir, "distances_majority_minority.h5")
    dnn_out = os.path.join(dist_dir, f"global_dnn_seed_{seed}")
    dnn_mat = os.path.join(dnn_out, "leaf_global_dnn_matrix.npy")
    dnn_ids = os.path.join(dnn_out, "leaf_global_dnn_enrolids.npy")
    return dist_dir, pn_h5, dnn_mat, dnn_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stageA", type=str, required=True,
                   help="Stage A folder name (e.g. za_coarse_phenotype)")
    p.add_argument("--stageB", type=str, required=True,
                   help="Stage B folder name (e.g. zb_intensity_context)")
    p.add_argument("--parquet", type=str, default="msk_2017_18_full.parquet")
    p.add_argument("--M", type=int, default=50_000)
    p.add_argument("--outdir", type=str, default="./za_zb_stage_ablation_results")
    args = p.parse_args()

    _, _, dnn_mat, dnn_ids = get_distance_paths(args.stageA, TRAIN_TEST_SEED)
    _, pn_h5, _, _ = get_distance_paths(args.stageB, TRAIN_TEST_SEED)

    for path, label in [(dnn_mat, "StageA DNN"), (dnn_ids, "StageA DNN enrolids"), (pn_h5, "StageB PN")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    spark = SparkSession.builder.appName("ZAZBStageAblation").getOrCreate()
    df = spark.read.format("parquet").load(args.parquet).toPandas()
    target_col = "top_2_pct_cost_2018"
    if target_col not in df.columns and "annual_cost_2018_deflated" in df.columns:
        thresh = df["annual_cost_2018_deflated"].quantile(0.98)
        df[target_col] = (df["annual_cost_2018_deflated"] >= thresh).astype(int)

    train_ids, _, train_pd, test_pd = train_test_split_enrol(
        df, target_col=target_col, test_size=0.3, verbose=False, random_state=TRAIN_TEST_SEED
    )
    val_ids, _, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=target_col, test_size=0.5, verbose=False, random_state=TRAIN_TEST_SEED
    )

    BIN = get_bin_flag_columns(df)
    CAT = get_cat_columns(df)
    NUM = get_true_num_columns(df, CAT, BIN)
    exclude = ["ENROLID", target_col] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude]

    cases = train_pd[train_pd[target_col] == 1]
    controls = train_pd[train_pd[target_col] == 0]
    combo_tag = f"A_{args.stageA}__B_{args.stageB}"
    print(f"StageA (k-center): {args.stageA} | StageB (matching): {args.stageB}")

    result = two_stage_kcenter_then_match(
        leaf_controls_enrolids=controls["ENROLID"].values.astype(np.int64),
        leaf_cases_enrolids=cases["ENROLID"].values.astype(np.int64),
        leaf_nn_matrix_npy=dnn_mat,
        leaf_nn_enrolids_npy=dnn_ids,
        pn_h5_path=pn_h5,
        M=args.M,
        use_adaptive_pool=False,
        tau=None,
        plateau_eps=0.01,
        force_nearest_per_case=False,
        force_topm=1,
        assignment_topk_start=None,
        seed_method="smart",
        matching_ratio=1,
        X_majority_leaf=None,
        case_weighting=None,
        use_kmeanspp=False,
    )
    sel = result["selected_control_enrolids"]
    train_undersampled = pd.concat([
        cases,
        controls[controls["ENROLID"].isin(sel)],
    ], ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{combo_tag}.csv")
    train_undersampled.to_csv(csv_path, index=False)
    print(f"Saved undersampled CSV: {csv_path}")

    # OCT
    model, params, _, preprocessor, feat_names = finetune_oct(
        X_train=train_undersampled[feature_cols],
        y_train=train_undersampled[target_col],
        X_val=val_pd[feature_cols],
        y_val=val_pd[target_col],
        categorical_cols=[c for c in CAT if c in feature_cols],
        numeric_cols=[c for c in NUM if c in feature_cols],
        binary_cols=[c for c in BIN if c in feature_cols],
        depths=[5, 7],
        minbuckets=[150],
        cps=[0.0001, 0.001, 0.01],
        verbose=False,
        random_seed=TRAIN_TEST_SEED,
    )
    oct_dir = os.path.join(args.outdir, "oct", combo_tag)
    os.makedirs(oct_dir, exist_ok=True)
    bd, bm, bcp = params["depth"], params["minbucket"], params["cp"]
    metrics = evaluate_binary_oct(
        model, test_pd[feature_cols], test_pd[target_col],
        preprocessor, feat_names,
        X_val_df=val_pd[feature_cols], y_val=val_pd[target_col],
        results_dir=oct_dir,
        save_suffix=f"{bd}_{bm}_{bcp}",
    )
    print(f"PR-AUC: {metrics.get('pr_auc', 0):.4f}  AUC: {metrics.get('auc', 0):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
