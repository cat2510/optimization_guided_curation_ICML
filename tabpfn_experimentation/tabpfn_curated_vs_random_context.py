#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# Repo root (adjust if needed)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

# Your split + column typing utilities (same pattern as your OCT scripts)
from public.model_IAI import train_test_split_enrol, get_bin_flag_columns, get_true_num_columns, get_cat_columns, get_preprocessor_with_impute


# TabPFN (PriorLabs)
from tabpfn import TabPFNClassifier


TRAIN_TEST_SEED_DEFAULT = 123
TARGET_COL_DEFAULT = "highcost_gt_200000"


# ------------------------- leakage guards -------------------------

def assert_disjoint_enrolids(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> None:
    if "ENROLID" not in df_a.columns or "ENROLID" not in df_b.columns:
        raise ValueError("Leakage check requires ENROLID column.")
    a = set(df_a["ENROLID"].astype(np.int64).tolist())
    b = set(df_b["ENROLID"].astype(np.int64).tolist())
    inter = a.intersection(b)
    if inter:
        sample = sorted(list(inter))[:20]
        raise ValueError(f"❌ DATA LEAKAGE: {name_a} overlaps {name_b} on ENROLID. n={len(inter)} sample={sample}")


def assert_context_is_subset_of_train(context_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    ctx = set(context_df["ENROLID"].astype(np.int64).tolist())
    trn = set(train_df["ENROLID"].astype(np.int64).tolist())
    missing = ctx - trn
    if missing:
        sample = sorted(list(missing))[:20]
        raise ValueError(
            f"❌ Context contains ENROLIDs not in TRAIN split. n_missing={len(missing)} sample={sample}\n"
            "Fix: ensure the curated CSV was built strictly from the train split."
        )


# ------------------------- metrics -------------------------

def recall_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_specificity: float = 0.6) -> dict:
    """
    Choose threshold maximizing recall subject to specificity >= target_specificity.
    Returns dict with recall, specificity, threshold (None if infeasible).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # thresholds over unique scores (descending)
    thresholds = np.unique(y_score)[::-1]

    best = {"threshold": None, "specificity": -np.inf, "recall": 0.0}

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        if spec >= target_specificity and rec >= best["recall"]:
            best = {"threshold": float(t), "specificity": float(spec), "recall": float(rec)}

    return best


# ------------------------- feature prep -------------------------

def prepare_feature_cols(df: pd.DataFrame, target_col: str, drop_high_corr: bool = True) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Mirror the MSK/CKD style: drop ENROLID, target, and '2018' columns.
    """
    
    exclude = ["ENROLID", target_col, "annual_cost_2017"] + [c for c in df.columns if "2018" in c] + [c for c in df.columns if c.startswith('highcost_gt_')]
    feature_cols = [c for c in df.columns if c not in exclude]

    # optional: correlation pruning if you're using small feature set
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


def fit_preprocessor_on_train(train_df: pd.DataFrame, feature_cols: list[str], CAT: list[str], NUM: list[str], BIN: list[str]):
    X_train = train_df[feature_cols]
    pre = get_preprocessor_with_impute(X_train, CAT, NUM, binary_cols=BIN, verbose=True)
    X_train_p = pre.fit_transform(X_train)
    return pre, X_train_p


def transform(pre, df: pd.DataFrame, feature_cols: list[str]):
    return pre.transform(df[feature_cols])


# ------------------------- context builders -------------------------

def load_curated_context(csv_path: str) -> pd.DataFrame:
    ctx = pd.read_csv(csv_path)
    if "ENROLID" not in ctx.columns:
        raise ValueError(f"Curated CSV missing ENROLID: {csv_path}")
    return ctx


def make_random_context_matching_counts(train_df: pd.DataFrame, target_col: str, n_min: int, n_maj: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minority = train_df[train_df[target_col] == 1]
    majority = train_df[train_df[target_col] == 0]
    if len(minority) < n_min or len(majority) < n_maj:
        raise ValueError(f"Not enough samples in TRAIN for n_min={n_min}, n_maj={n_maj}")
    idx_min = rng.choice(minority.index.values, size=n_min, replace=False)
    idx_maj = rng.choice(majority.index.values, size=n_maj, replace=False)
    return pd.concat([train_df.loc[idx_min], train_df.loc[idx_maj]], axis=0, ignore_index=True)


def enrolid_index_map(train_df: pd.DataFrame) -> dict[int, int]:
    # mapping enrolid -> row index in train_df (after reset_index)
    tr = train_df.reset_index(drop=True)
    return {int(e): i for i, e in enumerate(tr["ENROLID"].astype(int).tolist())}


# ------------------------- tabpfn runner -------------------------

def  tabpfn_predict_proba(
    X_context, y_context,
    X_test,
    device: str,
    fit_with_cache: bool,
    batch_size: int | None,
) -> np.ndarray:
    """
    Returns proba for class 1 over test.
    """
    if fit_with_cache:
        clf = TabPFNClassifier(device=device, fit_mode="fit_with_cache")
    else:
        clf = TabPFNClassifier(device=device)

    clf.fit(X_context, y_context)

    if batch_size is None:
        proba = clf.predict_proba(X_test)[:, 1]
        return proba

    probs = []
    for i in range(0, X_test.shape[0], batch_size):
        probs.append(clf.predict_proba(X_test[i:i+batch_size])[:, 1])
    return np.concatenate(probs, axis=0)


def evaluate_probs(y_test: np.ndarray, proba: np.ndarray, spec_target: float = 0.6) -> dict:
    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    r = recall_at_specificity(y_test, proba, target_specificity=spec_target)
    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "recall_at_specificity_0.6": float(r["recall"]),
        "achieved_specificity_0.6": float(r["specificity"]) if r["threshold"] is not None else None,
        "threshold_specificity_0.6": float(r["threshold"]) if r["threshold"] is not None else None,
    }


# ------------------------- main experiment -------------------------

def main():
    parser = argparse.ArgumentParser(description="TabPFN: curated vs random context (ratio 1:1), evaluate on fixed test set")
    parser.add_argument("--curated_csv", type=str, required=True,
                        help="Path to curated undersampled CSV (context), must contain ENROLID + target + features")
    parser.add_argument("--data_parquet", type=str, default="./0917_2017_18_with_2017_cost.parquet",
                        help="Path to raw parquet used to reproduce splits")
    parser.add_argument("--target_col", type=str, default=TARGET_COL_DEFAULT)
    parser.add_argument("--seed", type=int, default=TRAIN_TEST_SEED_DEFAULT, help="Train/test split seed")
    parser.add_argument("--n_random_seeds", type=int, default=10, help="How many random contexts to compare")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"], help="TabPFN device")
    parser.add_argument("--fit_with_cache", action="store_true",
                        help="Use TabPFN fit_mode='fit_with_cache' for efficient batched prediction")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size for test prediction. Set to 0 to do one-shot predict_proba.")
    parser.add_argument("--output_dir", type=str, default="tabpfn_context_experiment_out")
    parser.add_argument("--no_drop_high_corr", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    # Spark load (as requested)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("TabPFNContext").getOrCreate()
    df = spark.read.format("parquet").load(args.data_parquet).toPandas()
    # Create cost stratum
    def make_cost_stratum_3class(df):
        cost_stratum = pd.Series(0, index=df.index)
        cost_stratum[(df['highcost_gt_50000'] == 1) & (df['highcost_gt_100000'] == 0)] = 1
        cost_stratum[(df['highcost_gt_100000'] == 1) & (df['highcost_gt_200000'] == 0)] = 2
        cost_stratum[df['highcost_gt_200000'] == 1] = 3
        return cost_stratum
    
    df['cost_stratum_2018'] = make_cost_stratum_3class(df)
    
    # feature col setup (matches your general pattern)
    feature_cols, CAT, NUM, BIN = prepare_feature_cols(df, args.target_col, drop_high_corr=not args.no_drop_high_corr)

    # reproduce split
    _, _, train_pd, test_pd = train_test_split_enrol(df, target_col="cost_stratum_2018", test_size=0.3, verbose=False, random_state=args.seed)
    _, _, val_pd, test_pd = train_test_split_enrol(test_pd, target_col="cost_stratum_2018", test_size=0.5, verbose=False, random_state=args.seed)

    # curated context
    curated = load_curated_context(args.curated_csv)

    # leakage + membership checks
    assert_context_is_subset_of_train(curated, train_pd)
    assert_disjoint_enrolids(curated, val_pd, "curated_context", "val")
    assert_disjoint_enrolids(curated, test_pd, "curated_context", "test")

    # fit preprocessing ONCE on train split
    pre, X_train_p = fit_preprocessor_on_train(train_pd, feature_cols, CAT, NUM, BIN)
    X_test_p = transform(pre, test_pd, feature_cols)
    y_test = test_pd[args.target_col].values.astype(int)

    # ENROLID -> row index in train for fast slicing
    train_reset = train_pd.reset_index(drop=True)
    eid2i = enrolid_index_map(train_pd)

    def slice_train_by_enrolids(enrolids: list[int]):
        idx = []
        for e in enrolids:
            e = int(e)
            if e not in eid2i:
                raise ValueError(f"ENROLID {e} not in train split; check curated/random sampling.")
            idx.append(eid2i[e])
        idx = np.asarray(idx, dtype=int)
        X = X_train_p[idx]
        y = train_reset.loc[idx, args.target_col].values.astype(int)
        return X, y

    # curated counts
    curated_ids = curated["ENROLID"].astype(int).tolist()
    X_cur, y_cur = slice_train_by_enrolids(curated_ids)
    n_min = int((y_cur == 1).sum())
    n_maj = int((y_cur == 0).sum())
    print(f"Curated context size: {len(y_cur)} (min={n_min}, maj={n_maj})")

    batch_size = None if args.batch_size == 0 else int(args.batch_size)

    rows = []

    # ---- Curated run
    t0 = time.perf_counter()
    proba_cur = tabpfn_predict_proba(
        X_cur, y_cur, X_test_p,
        device=args.device,
        fit_with_cache=args.fit_with_cache,
        batch_size=batch_size,
    )
    t1 = time.perf_counter()
    met_cur = evaluate_probs(y_test, proba_cur, spec_target=0.6)
    rows.append({"source": "curated", "seed": None, "runtime_s": float(t1 - t0), **met_cur})
    print(f"[curated] PR-AUC={met_cur['pr_auc']:.4f} AUC={met_cur['auc']:.4f} "
          f"R@Spec0.6={met_cur['recall_at_specificity_0.6']:.4f} time={t1-t0:.1f}s")

    # ---- Random runs
    for s in range(args.n_random_seeds):
        rnd = make_random_context_matching_counts(train_pd, args.target_col, n_min, n_maj, seed=s)
        assert_disjoint_enrolids(rnd, val_pd, f"random_context_seed{s}", "val")
        assert_disjoint_enrolids(rnd, test_pd, f"random_context_seed{s}", "test")

        rnd_ids = rnd["ENROLID"].astype(int).tolist()
        X_r, y_r = slice_train_by_enrolids(rnd_ids)

        t0 = time.perf_counter()
        proba_r = tabpfn_predict_proba(
            X_r, y_r, X_test_p,
            device=args.device,
            fit_with_cache=args.fit_with_cache,
            batch_size=batch_size,
        )
        t1 = time.perf_counter()
        met_r = evaluate_probs(y_test, proba_r, spec_target=0.6)
        rows.append({"source": "random", "seed": s, "runtime_s": float(t1 - t0), **met_r})
        print(f"[random s={s}] PR-AUC={met_r['pr_auc']:.4f} AUC={met_r['auc']:.4f} "
              f"R@Spec0.6={met_r['recall_at_specificity_0.6']:.4f} time={t1-t0:.1f}s")

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "tabpfn_curated_vs_random_context_ratio1.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")

    # summary
    rnd = out[out["source"] == "random"]
    if len(rnd) > 0:
        for m in ["pr_auc", "auc", "recall_at_specificity_0.6"]:
            mu = rnd[m].mean()
            sd = rnd[m].std(ddof=1) if len(rnd) > 1 else 0.0
            print(f"Random {m}: {mu:.4f} ± {sd:.4f}")
        print("Curated:", out[out["source"] == "curated"][["pr_auc","auc","recall_at_specificity_0.6"]].to_dict("records")[0])


if __name__ == "__main__":
    main()
