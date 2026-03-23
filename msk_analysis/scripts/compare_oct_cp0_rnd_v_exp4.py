#!/usr/bin/env python3

""" Run example: 
python scripts/compare_oct_cp0_rnd_v_exp4.py \                                                                       
  --train_root compare_oct_cp0_rnd_v_exp4_train_sets \
  --outdir compare_oct_cp0_rnd_v_exp4_train_sets --seeds 5"""
import os, sys, json, argparse, traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# repo imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from msk_analysis.experiments_compare_random_vs_curation import (
    sample_random_controls,
    sample_stageA_on_restricted_pool,
    sample_stageB_matched_controls,
)

from public.model_IAI import (
    get_bin_flag_columns,
    get_cat_columns,
    get_true_num_columns,
    finetune_oct,
    evaluate_binary_oct,
    train_test_split_enrol,
)

TRAIN_TEST_SEED = 123
TARGET_COL = "top_2_pct_cost_2018"
COST_COL = "annual_cost_2018_deflated"


# ---------------------------------------------------------------------
# Leaf composition helper (train/test)
# ---------------------------------------------------------------------
def leaf_composition_from_apply(
    model, preprocessor, feature_names: List[str],
    X_df: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    Xp = preprocessor.transform(X_df)
    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()
    Xp = pd.DataFrame(Xp, columns=feature_names)

    leaf_ids = model.apply(Xp)
    y_arr = np.asarray(y).astype(int)

    tmp = pd.DataFrame({"leaf_id": leaf_ids, "y": y_arr})
    out = (tmp.groupby("leaf_id")
              .agg(n=("y", "size"), n_pos=("y", "sum"), pos_rate=("y", "mean"))
              .reset_index()
              .sort_values(["pos_rate", "n"], ascending=[False, False]))
    return out


# ---------------------------------------------------------------------
# Robust split extraction from your saved tree JSON
# ---------------------------------------------------------------------
def extract_splits_from_tree_json(tree: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - path (string of L/R)
      - feature
      - threshold
      - op
    Defensive: tries multiple formats.
    """
    splits = []

    def children(node):
        if not isinstance(node, dict):
            return None, None
        if "children" in node and isinstance(node["children"], list) and len(node["children"]) == 2:
            return node["children"][0], node["children"][1]
        if "left" in node and "right" in node:
            return node["left"], node["right"]
        if "true" in node and "false" in node:
            return node["true"], node["false"]
        return None, None

    def is_leaf(node):
        l, r = children(node)
        return l is None and r is None

    feature_keys = ["feature", "feature_name", "variable", "split_feature", "name"]
    thr_keys = ["threshold", "cut", "split_value", "value"]
    op_keys = ["op", "operator", "rule"]

    def get(node, keys):
        for k in keys:
            if isinstance(node, dict) and k in node:
                return node[k]
        return None

    def walk(node, path=""):
        if not isinstance(node, dict):
            return
        if is_leaf(node):
            return
        feat = get(node, feature_keys)
        thr = get(node, thr_keys)
        op = get(node, op_keys) or "<="
        splits.append({"path": path, "feature": feat, "op": op, "threshold": thr})
        l, r = children(node)
        if l is not None:
            walk(l, path + "L")
        if r is not None:
            walk(r, path + "R")

    root = tree.get("tree", tree.get("root", tree))
    walk(root, "")
    return pd.DataFrame(splits)


# ---------------------------------------------------------------------
# Train OCT with fixed hyperparams (no grid)
# ---------------------------------------------------------------------
def train_oct_fixed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    depth: int,
    minbucket: int,
    cp: float,
):
    model, params, _, preprocessor, feat_names = finetune_oct(
        X_train=train_df[feature_cols],
        y_train=train_df[target_col],
        X_val=val_df[feature_cols],
        y_val=val_df[target_col],
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        binary_cols=bin_cols,
        depths=[depth],
        minbuckets=[minbucket],
        cps=[cp],
        tree_kind="oct",
        verbose=False,
        random_seed=TRAIN_TEST_SEED,
    )
    return model, preprocessor, feat_names, params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_path", type=str, default="/Users/cat2510/my_projects/msk_analysis/msk_2017_18_full.parquet")
    p.add_argument("--train_root", type=str, default="/Users/cat2510/scratch/msk_analysis/exp_0_to_4_gower_v2/results")
    p.add_argument("--outdir", type=str, default="compare_oct_cp0_gower_v2")
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--distances_dir", type=str, default="/Users/cat2510/scratch/msk_analysis/precomputed_distances_gower")
    p.add_argument("--stageA_seed_method", type=str, default="density")
    p.add_argument("--M_pool", type=int, default=None)
    p.add_argument("--use_kmeanspp", action="store_true")
    p.add_argument("--generate_trains", action="store_true", help="Generate and save exp3/exp4 train sets using get_matched_1N")

    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--minbucket_unpruned", type=int, default=5)
    p.add_argument("--cp_unpruned", type=float, default=0.0)

    p.add_argument("--minbucket_pruned", type=int, default=5)
    p.add_argument("--cp_pruned", type=float, default=0.00001)

    return p.parse_args()


def generate_exp3_exp4_trains(
    df: pd.DataFrame, cases: pd.DataFrame, controls: pd.DataFrame,
    control_enrolids: np.ndarray, case_enrolids: np.ndarray, N: int, K: int,
    pn_h5: str, dnn_matrix: str, dnn_enrolids_path: str,
    dnn_ids: np.ndarray, id_to_pos: dict,
    stageA_seed_method: str, M_pool: int, use_kmeanspp: bool,
    X_majority_leaf, train_root: str, seeds: List[int],
):
    """Generate exp3 and exp4 undersampled train sets using get_matched_1N (shared) for both."""
    os.makedirs(train_root, exist_ok=True)
    cache_dir = os.path.join(train_root, "cache_ids")
    os.makedirs(cache_dir, exist_ok=True)
    cache_prefix = f"matched_1N_seedmethod_{stageA_seed_method}_Mpool_{M_pool}"

    def _cache_path(name: str, seed: int) -> str:
        return os.path.join(cache_dir, f"{name}_s{seed}.npy")

    def load_or_compute_ids(name: str, seed: int, compute_fn) -> np.ndarray:
        path = _cache_path(name, seed)
        if os.path.exists(path):
            return np.load(path)
        ids = compute_fn()
        np.save(path, ids)
        return ids

    def get_matched_1N(seed: int) -> np.ndarray:
        return load_or_compute_ids(cache_prefix, seed, lambda: sample_stageB_matched_controls(
            control_enrolids, case_enrolids, dnn_matrix, dnn_enrolids_path, pn_h5,
            target_count=N, matching_ratio=1, M_pool=M_pool,
            seed_method=stageA_seed_method, seed=seed,
            X_majority_leaf=X_majority_leaf, verbose=False, use_kmeanspp=use_kmeanspp,
        ))

    def get_random_perm_excluding(seed: int, exclude_ids: np.ndarray) -> np.ndarray:
        exclude = set(int(x) for x in exclude_ids)
        remaining = np.array([e for e in control_enrolids if int(e) not in exclude], dtype=np.int64)
        rng = np.random.RandomState(seed + 202603)
        return rng.permutation(remaining)

    for seed in seeds:
        match_ids = get_matched_1N(seed)
        disp_ids = sample_stageA_on_restricted_pool(
            control_enrolids, match_ids, dnn_matrix, dnn_enrolids_path, pn_h5,
            case_enrolids, N, stageA_seed_method, seed,
            X_majority_leaf, id_to_pos, verbose=False, use_kmeanspp=use_kmeanspp,
        )
        mix_ids = np.unique(np.concatenate([match_ids, disp_ids]))[:K]
        if len(mix_ids) < K:
            remaining = np.array([e for e in control_enrolids if int(e) not in set(int(x) for x in mix_ids)])
            extra = sample_random_controls(remaining, K - len(mix_ids), seed + 9999)
            mix_ids = np.concatenate([mix_ids, extra])[:K]
        exp3_train = pd.concat([cases, controls[controls["ENROLID"].isin(mix_ids)]], ignore_index=True)
        exp3_path = os.path.join(train_root, f"exp3_mix_s{seed}_train.csv")
        exp3_train.to_csv(exp3_path, index=False)
        print(f"  Saved {exp3_path}")

        perm = get_random_perm_excluding(seed, match_ids)
        extra_ids = perm[:N].astype(np.int64)
        ctrl_exp4 = np.unique(np.concatenate([match_ids, extra_ids]).astype(np.int64))[: 2 * N]
        if len(ctrl_exp4) < 2 * N:
            remaining = np.array([e for e in control_enrolids if int(e) not in set(int(x) for x in ctrl_exp4)])
            extra = sample_random_controls(remaining, 2 * N - len(ctrl_exp4), seed + 7777)
            ctrl_exp4 = np.concatenate([ctrl_exp4, extra])[: 2 * N]
        exp4_train = pd.concat([cases, controls[controls["ENROLID"].isin(ctrl_exp4)]], ignore_index=True)
        exp4_path = os.path.join(train_root, f"exp4_matched_1N_plus_1N_random_s{seed}_train.csv")
        exp4_train.to_csv(exp4_path, index=False)
        print(f"  Saved {exp4_path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # ----------------------------
    # Load parquet + make target if missing
    # ----------------------------
    df = pd.read_parquet(args.parquet_path)
    if TARGET_COL not in df.columns and COST_COL in df.columns:
        thresh = df[COST_COL].quantile(0.98)
        df[TARGET_COL] = (df[COST_COL] >= thresh).astype(int)

    # Columns
    BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
    CAT_COLUMNS = get_cat_columns(df)
    TRUE_NUM_COLUMNS = get_true_num_columns(df, CAT_COLUMNS, BIN_FLAG_COLUMNS)

    exclude_cols = ["ENROLID", TARGET_COL] + [c for c in df.columns if "2018" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    _, _, train_pd, test_pd = train_test_split_enrol(
            df, target_col=TARGET_COL, test_size=0.3, verbose=False,
            random_state=TRAIN_TEST_SEED,
        )
    _, _, val_pd, test_pd = train_test_split_enrol(
        test_pd, target_col=TARGET_COL, test_size=0.5, verbose=False,
        random_state=TRAIN_TEST_SEED,
    )


    if args.generate_trains:
        import h5py
        from public.dnn_matrix_storage import dnn_matrix_storage_exists, ensure_dnn_matrix_npy,dnn_enrolids_npy_path
        pn_h5 = os.path.join(args.distances_dir, "distances_majority_minority_gower.h5")
        dnn_dir = os.path.join(args.distances_dir, f"global_dnn_seed_{TRAIN_TEST_SEED}_gower")
        dnn_enrolids_path = dnn_enrolids_npy_path(dnn_dir)
        for p in [pn_h5, dnn_enrolids_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Train pool / holdout split require distances: {p}")
        ctrl_ids = np.load(dnn_enrolids_path)
        with h5py.File(pn_h5, "r") as f:
            case_ids = f["minority_enrolids"][:]
    
        if not dnn_matrix_storage_exists(dnn_dir):
            raise FileNotFoundError(f"Missing D-N-N matrix: {dnn_dir}/leaf_global_dnn_matrix.npy")
        dnn_matrix = ensure_dnn_matrix_npy(dnn_dir)
        cases = train_pd[train_pd[TARGET_COL] == 1]
        controls = train_pd[train_pd[TARGET_COL] == 0]
        N, K = len(cases), 2 * len(cases)
        control_enrolids = controls["ENROLID"].values.astype(np.int64)
        case_enrolids = cases["ENROLID"].values.astype(np.int64)
        dnn_ids = np.load(dnn_enrolids_path)
        id_to_pos = {int(e): i for i, e in enumerate(dnn_ids)}
        M_pool = args.M_pool if args.M_pool is not None else len(controls) // 2
        print("Generating exp3/exp4 train sets (shared get_matched_1N)...")
        generate_exp3_exp4_trains(train_pd, cases, controls, control_enrolids, case_enrolids, N, K,
            pn_h5, dnn_matrix, dnn_enrolids_path, dnn_ids, id_to_pos,
            args.stageA_seed_method, M_pool, args.use_kmeanspp, None, args.train_root, seeds)

    all_rows = []

    def run_one(tag: str, seed: int, train_path: str):
        train_df = pd.read_csv(train_path)
        out_base = os.path.join(args.outdir, f"{tag}_s{seed}")
        os.makedirs(out_base, exist_ok=True)

        # --- Train unpruned ---
        m_un, prep_un, fn_un, params_un = train_oct_fixed(
            train_df, val_pd, feature_cols, TARGET_COL,
            CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
            depth=args.depth, minbucket=args.minbucket_unpruned, cp=args.cp_unpruned
        )

        # Evaluate to write JSON + HTML + predictions
        suffix_un = f"{tag}_s{seed}_unpruned_d{args.depth}_mb{args.minbucket_unpruned}_cp{args.cp_unpruned}"
        X_test_df = test_pd[feature_cols+["ENROLID"]].copy()
        met_un = evaluate_binary_oct(
            m_un,
            X_test_df=X_test_df,
            y_test=test_pd[TARGET_COL],
            preprocessor=prep_un,
            feature_names=fn_un,
            results_dir=out_base,
            save_suffix=suffix_un,
            X_val_df=val_pd[feature_cols],
            y_val=val_pd[TARGET_COL],
        )

        # Leaf compositions
        lc_train_un = leaf_composition_from_apply(m_un, prep_un, fn_un, train_df[feature_cols], train_df[TARGET_COL])
        lc_test_un  = leaf_composition_from_apply(m_un, prep_un, fn_un, test_pd[feature_cols], test_pd[TARGET_COL])
        lc_train_un.to_csv(os.path.join(out_base, "leaf_train_unpruned.csv"), index=False)
        lc_test_un.to_csv(os.path.join(out_base, "leaf_test_unpruned.csv"), index=False)

        # --- Train pruned ---
        m_pr, prep_pr, fn_pr, params_pr = train_oct_fixed(
            train_df, val_pd, feature_cols, TARGET_COL,
            CAT_COLUMNS, TRUE_NUM_COLUMNS, BIN_FLAG_COLUMNS,
            depth=args.depth, minbucket=args.minbucket_pruned, cp=args.cp_pruned
        )

        suffix_pr = f"{tag}_s{seed}_pruned_d{args.depth}_mb{args.minbucket_pruned}_cp{args.cp_pruned}"
        met_pr = evaluate_binary_oct(
            m_pr,
            X_test_df=X_test_df,
            y_test=test_pd[TARGET_COL],
            preprocessor=prep_pr,
            feature_names=fn_pr,
            results_dir=out_base,
            save_suffix=suffix_pr,
            X_val_df=val_pd[feature_cols],
            y_val=val_pd[TARGET_COL],
        )

        lc_train_pr = leaf_composition_from_apply(m_pr, prep_pr, fn_pr, train_df[feature_cols], train_df[TARGET_COL])
        lc_test_pr  = leaf_composition_from_apply(m_pr, prep_pr, fn_pr, test_pd[feature_cols], test_pd[TARGET_COL])
        lc_train_pr.to_csv(os.path.join(out_base, "leaf_train_pruned.csv"), index=False)
        lc_test_pr.to_csv(os.path.join(out_base, "leaf_test_pruned.csv"), index=False)

        # --- Compare splits using the JSON your evaluator writes ---
        tree_un_path = os.path.join(out_base, f"oct_tree_{suffix_un}.json")
        tree_pr_path = os.path.join(out_base, f"oct_tree_{suffix_pr}.json")

        split_diff_ok = False
        if os.path.exists(tree_un_path) and os.path.exists(tree_pr_path):
            try:
                with open(tree_un_path, "r") as f:
                    t_un = json.load(f)
                with open(tree_pr_path, "r") as f:
                    t_pr = json.load(f)

                s_un = extract_splits_from_tree_json(t_un)
                s_pr = extract_splits_from_tree_json(t_pr)

                s_un.to_csv(os.path.join(out_base, "splits_unpruned.csv"), index=False)
                s_pr.to_csv(os.path.join(out_base, "splits_pruned.csv"), index=False)

                paths_un = set(s_un["path"].astype(str))
                paths_pr = set(s_pr["path"].astype(str))

                pruned_paths = sorted(list(paths_un - paths_pr))
                kept_paths   = sorted(list(paths_un & paths_pr))

                s_un[s_un["path"].isin(pruned_paths)].to_csv(os.path.join(out_base, "splits_pruned_away.csv"), index=False)
                kept = s_un[s_un["path"].isin(kept_paths)].merge(
                    s_pr, on="path", how="inner", suffixes=("_unpruned", "_pruned")
                )
                kept.to_csv(os.path.join(out_base, "splits_kept.csv"), index=False)
                split_diff_ok = True
            except Exception:
                traceback.print_exc()

        # Log row
        row = {
            "tag": tag,
            "seed": seed,
            "train_path": train_path,
            "split_diff_ok": split_diff_ok,
            "unpruned_depth": args.depth,
            "unpruned_minbucket": args.minbucket_unpruned,
            "unpruned_cp": args.cp_unpruned,
            "pruned_depth": args.depth,
            "pruned_minbucket": args.minbucket_pruned,
            "pruned_cp": args.cp_pruned,
            **{f"un_{k}": v for k, v in met_un.items()},
            **{f"pr_{k}": v for k, v in met_pr.items()},
            "un_n_leaves_test": int(lc_test_un["leaf_id"].nunique()),
            "pr_n_leaves_test": int(lc_test_pr["leaf_id"].nunique()),
        }
        all_rows.append(row)
        print(f"[OK] {tag} seed={seed} -> {out_base}")

    for s in seeds:
        exp3_path = os.path.join(args.train_root, f"exp3_mix_s{s}_train.csv")
        exp4_path = os.path.join(args.train_root, f"exp4_matched_1N_plus_1N_random_s{s}_train.csv")
        if not os.path.exists(exp3_path):
            raise FileNotFoundError(exp3_path)
        if not os.path.exists(exp4_path):
            raise FileNotFoundError(exp4_path)

        run_one("exp3_mix", s, exp3_path)
        run_one("exp4_matched_plus_random", s, exp4_path)

    # Write run summary
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(os.path.join(args.outdir, "tree_inspection_summary.csv"), index=False)
    print("\nWrote:", os.path.join(args.outdir, "tree_inspection_summary.csv"))
    print("Done.")


if __name__ == "__main__":
    main()

