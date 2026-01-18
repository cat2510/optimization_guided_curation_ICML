import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
def _pct_ci(vals, alpha=0.05):
    if len(vals) == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "se_boot": np.nan, "n_eff": 0}
    vals = np.asarray(vals, dtype=float)
    return {
        "point": float(np.mean(vals)),
        "lo": float(np.quantile(vals, alpha / 2.0)),
        "hi": float(np.quantile(vals, 1 - alpha / 2.0)),
        "se_boot": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n_eff": int(len(vals)),
    }

  
def _safe_auc(y, p, metric="roc"):
    """Return ROC-AUC or PR-AUC; NaN if not defined (needs both classes)."""
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    if metric == "roc":
        return float(roc_auc_score(y, p))
    elif metric == "pr":
        return float(average_precision_score(y, p))
    else:
        raise ValueError("metric must be 'roc' or 'pr'")

def bootstrap_ci_global_metrics(y, pv, ps, B=2000, alpha=0.05, rng=42):
    """
    Paired bootstrap CIs for global ROC-AUC and PR-AUC of both models.
    Uses the same bootstrap resamples for M_v and M_s (recommended).
    Also returns paired CIs for differences (M_s - M_v).
    """
    y = np.asarray(y).astype(int)
    pv = np.asarray(pv).astype(float)
    ps = np.asarray(ps).astype(float)

    n = len(y)
    r = np.random.default_rng(rng)

    roc_v, roc_s, roc_diff = [], [], []
    pr_v, pr_s, pr_diff = [], [], []

    for _ in range(B):
        idx = r.integers(0, n, size=n)
        yb = y[idx]

        # If bootstrap sample has only one class, AUC/PR undefined -> skip
        if np.unique(yb).size < 2:
            continue

        pv_b = pv[idx]
        ps_b = ps[idx]

        rv = roc_auc_score(yb, pv_b)
        rs = roc_auc_score(yb, ps_b)
        av = average_precision_score(yb, pv_b)
        a_s = average_precision_score(yb, ps_b)

        roc_v.append(rv); roc_s.append(rs); roc_diff.append(rs - rv)
        pr_v.append(av);  pr_s.append(a_s); pr_diff.append(a_s - av)

    out = {
        "global_ROC_Mv": _pct_ci(roc_v, alpha),
        "global_ROC_Ms": _pct_ci(roc_s, alpha),
        "global_ROC_diff_Ms_minus_Mv": _pct_ci(roc_diff, alpha),
        "global_PR_Mv": _pct_ci(pr_v, alpha),
        "global_PR_Ms": _pct_ci(pr_s, alpha),
        "global_PR_diff_Ms_minus_Mv": _pct_ci(pr_diff, alpha),
    }
    return out

def _weighted_excess_auc_from_partition(
    leaf_ids,
    y,
    proba,
    metric="roc",
    min_events_per_class=10,
    weights="n",  # "n" (size) or "invvar" (optional: inverse-variance-ish)
):
    """
    Size-weighted (or inverse-variance-ish) mean of (AUC - 0.5) across informative leaves.
    This is the *aggregate* quantity you report, without returning per-leaf tables.

    Informative leaf definition matches the paper text:
      n_pos >= min_events_per_class AND n_neg >= min_events_per_class

    Parameters
    ----------
    leaf_ids : array-like
        Leaf assignment vector defining the partition (anchor model).
    y : array-like
        True binary labels aligned with leaf_ids/proba.
    proba : array-like
        Predicted probabilities of the model being evaluated inside this partition.
    metric : {"roc","pr"}
        ROC-AUC or PR-AUC within each leaf.
    min_events_per_class : int
        Minimum positives and negatives required to include a leaf.
    weights : {"n","invvar"}
        "n": weight by leaf size (standard, matches Eq. (explicit_weighted)).
        "invvar": optional heuristic weight ~ 1 / [pi(1-pi)/n], downweights extreme prevalence.
                 (still uses leaf size implicitly). If you don't need this, keep "n".

    Returns
    -------
    float
        Weighted mean excess AUC over informative leaves; NaN if none.
    """
    leaf_ids = np.asarray(leaf_ids)
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba).astype(float)

    num, den = 0.0, 0.0
    for lid in np.unique(leaf_ids):
        idx = np.where(leaf_ids == lid)[0]
        if idx.size == 0:
            continue

        yL = y[idx]
        pL = proba[idx]

        n = int(idx.size)
        n_pos = int((yL == 1).sum())
        n_neg = n - n_pos

        # informative leaf constraint
        if n_pos < min_events_per_class or n_neg < min_events_per_class:
            continue

        auc = _safe_auc(yL, pL, metric=metric)
        if np.isnan(auc):
            continue

        # weight choice
        if weights == "n":
            w = float(n)
        elif weights == "invvar":
            # heuristic: downweight extreme prevalence leaves where AUC is unstable
            pi = n_pos / n
            # avoid division by 0; add tiny ridge
            var_proxy = (pi * (1 - pi) / max(n, 1)) + 1e-12
            w = float(1.0 / var_proxy)
        else:
            raise ValueError("weights must be 'n' or 'invvar'")

        num += w * (auc - 0.5)
        den += w

    return float(num / den) if den > 0 else np.nan

def bootstrap_ci_weighted_excess(
    leaf_ids,
    y,
    proba,
    metric="roc",
    min_events_per_class=10,
    weights="n",
    B=2000,
    alpha=0.05,
    rng=42,
):
    """
    Bootstrap CI for the aggregate weighted excess AUC:
      - resample test rows with replacement
      - recompute informative leaves and the weighted excess AUC
      - return percentile CI

    Returns dict: {point, lo, hi, se_boot, n_eff}
    """
    leaf_ids = np.asarray(leaf_ids)
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba).astype(float)

    n = len(y)
    r = np.random.default_rng(rng)

    point = _weighted_excess_auc_from_partition(
        leaf_ids=leaf_ids, y=y, proba=proba,
        metric=metric, min_events_per_class=min_events_per_class, weights=weights
    )

    vals = []
    for _ in range(B):
        idx = r.integers(0, n, size=n)
        v = _weighted_excess_auc_from_partition(
            leaf_ids=leaf_ids[idx],
            y=y[idx],
            proba=proba[idx],
            metric=metric,
            min_events_per_class=min_events_per_class,
            weights=weights,
        )
        if not np.isnan(v):
            vals.append(float(v))

    if len(vals) == 0:
        return {"point": float(point), "lo": np.nan, "hi": np.nan, "se_boot": np.nan, "n_eff": 0}

    vals = np.asarray(vals)
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1 - alpha / 2.0))
    se = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"point": float(point), "lo": lo, "hi": hi, "se_boot": se, "n_eff": int(len(vals))}

def symmetric_leaf_evaluation_oct(
    mv_pred_path: str,          # vanilla model predictions CSV
    ms_pred_path: str,          # curated/balanced model predictions CSV
    y_test,                     # aligned labels
    leaf_col: str = "leaf_assignment",
    proba_col: str = "predicted_proba",
    enrolid_col: str | None = None,
    min_events_per_class: int = 10,
    # uncertainty
    B: int = 2000,
    alpha: float = 0.05,
    rng: int = 42,
    # weighting
    weights: str = "n",         # "n" (standard) or "invvar" (optional)
):
    """
    Symmetric aggregate evaluation between M_v (vanilla) and M_s (sampled/curated),
    reporting only the *aggregate size-weighted excess* metrics + bootstrap CIs.

    Computes:
      - Global ROC/PR for each model on the aligned test set
      - Excess AUC_s|v (within M_v partition): weighted mean(AUC_s|v(leaf) - 0.5) over informative leaves
      - Excess AUC_v|s (within M_s partition): weighted mean(AUC_v|s(leaf) - 0.5) over informative leaves
      - Symmetric excess: average of the two directions
      - Bootstrap percentile CIs for each excess metric (and symmetric excess)

    Notes:
      - Informative leaves require >= min_events_per_class positives and negatives.
      - weights="n" matches the explicit formula your advisor suggested.
    """
    mv = pd.read_csv(mv_pred_path)
    ms = pd.read_csv(ms_pred_path)

    for name, df, path in [("M_v", mv, mv_pred_path), ("M_s", ms, ms_pred_path)]:
        if leaf_col not in df.columns:
            raise ValueError(f"{name} predictions missing '{leaf_col}': {path}")
        if proba_col not in df.columns:
            raise ValueError(f"{name} predictions missing '{proba_col}': {path}")

    y_test = pd.Series(y_test)

    # Align rows
    if enrolid_col is not None:
        if enrolid_col not in mv.columns or enrolid_col not in ms.columns:
            raise ValueError(f"enrolid_col='{enrolid_col}' must exist in both prediction files.")
        mv = mv.set_index(enrolid_col)
        ms = ms.set_index(enrolid_col)

        common = mv.index.intersection(ms.index).intersection(y_test.index)
        if len(common) == 0:
            raise ValueError("No overlapping indices among M_v, M_s, and y_test using enrolid_col alignment.")
        mv = mv.loc[common]
        ms = ms.loc[common]
        y = y_test.loc[common].to_numpy(dtype=int)
    else:
        m = min(len(mv), len(ms), len(y_test))
        if m < len(y_test) or m < len(mv) or m < len(ms):
            print(f"⚠ Using first {m} rows by position (length mismatch).")
        mv = mv.iloc[:m].reset_index(drop=True)
        ms = ms.iloc[:m].reset_index(drop=True)
        y = y_test.iloc[:m].reset_index(drop=True).to_numpy(dtype=int)

    # Extract aligned arrays
    lv = mv[leaf_col].to_numpy()
    ls = ms[leaf_col].to_numpy()
    pv = mv[proba_col].astype(float).to_numpy()
    ps = ms[proba_col].astype(float).to_numpy()

    # Global metrics
    overall = {
        "M_v_global_ROC": _safe_auc(y, pv, "roc"),
        "M_v_global_PR":  _safe_auc(y, pv, "pr"),
        "M_s_global_ROC": _safe_auc(y, ps, "roc"),
        "M_s_global_PR":  _safe_auc(y, ps, "pr"),
        "n_test": int(len(y)),
        "prevalence": float(np.mean(y)),
    }
     # NEW: Global bootstrap CIs (paired bootstrap)
    overall_ci = bootstrap_ci_global_metrics(
        y=y, pv=pv, ps=ps, B=B, alpha=alpha, rng=rng
    )


    # Aggregate excess metrics (point estimates)
    excess_ROC_s_v = _weighted_excess_auc_from_partition(
        leaf_ids=lv, y=y, proba=ps, metric="roc",
        min_events_per_class=min_events_per_class, weights=weights
    )
    excess_ROC_v_s = _weighted_excess_auc_from_partition(
        leaf_ids=ls, y=y, proba=pv, metric="roc",
        min_events_per_class=min_events_per_class, weights=weights
    )
    scores = {
        "excess_ROC_s|v": float(excess_ROC_s_v),
        "excess_ROC_v|s": float(excess_ROC_v_s),
        "sym_excess_ROC": float(np.nanmean([excess_ROC_s_v, excess_ROC_v_s])),
    }

    # Coverage diagnostics (fraction of test points in informative leaves by each partition)
    def _informative_mask(leaf_ids_vec):
        mask = np.zeros(len(y), dtype=bool)
        for lid in np.unique(leaf_ids_vec):
            idx = np.where(leaf_ids_vec == lid)[0]
            yL = y[idx]
            n_pos = int((yL == 1).sum())
            n_neg = int((yL == 0).sum())
            if n_pos >= min_events_per_class and n_neg >= min_events_per_class:
                mask[idx] = True
        return mask

    inf_v = _informative_mask(lv)
    inf_s = _informative_mask(ls)
    scores["coverage_informative_v"] = float(inf_v.mean())
    scores["coverage_informative_s"] = float(inf_s.mean())

    # Bootstrap CIs for aggregate excess metrics
    ci = {}
    ci["excess_ROC_s|v"] = bootstrap_ci_weighted_excess(
        leaf_ids=lv, y=y, proba=ps, metric="roc",
        min_events_per_class=min_events_per_class, weights=weights,
        B=B, alpha=alpha, rng=rng
    )
    ci["excess_ROC_v|s"] = bootstrap_ci_weighted_excess(
        leaf_ids=ls, y=y, proba=pv, metric="roc",
        min_events_per_class=min_events_per_class, weights=weights,
        B=B, alpha=alpha, rng=rng + 1
    )

  
    # Summary table (aggregate-only)
    summary_table = pd.DataFrame([
        {"Metric": "Global ROC-AUC", "M_v": overall["M_v_global_ROC"], "M_s": overall["M_s_global_ROC"]},
        {"Metric": "Global PR-AUC",  "M_v": overall["M_v_global_PR"],  "M_s": overall["M_s_global_PR"]},

        {"Metric": r"Weighted excess AUCon informative subset",
         "M_v": scores["excess_ROC_v|s"], "M_s": scores["excess_ROC_s|v"]},
      
        {"Metric": "Coverage in informative leaves (M_v partition)", "M_v": scores["coverage_informative_v"], "M_s": scores["coverage_informative_s"]},
    ])

    return {
        "overall": overall,
        "overall_ci": overall_ci,   # <-- NEW
        "scores": scores,
        "ci": ci,                         # bootstrap CIs for aggregate metrics
        "summary_table": summary_table,    # nice printable aggregate-only table
    }
