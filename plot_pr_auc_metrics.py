#!/usr/bin/env python3
"""
plot_pr_auc_metrics.py
======================
Plot the precision–recall curve and print PR-AUC (average precision) and ROC-AUC
for any binary classifier that outputs continuous scores (probabilities or scores
for the positive class).

Typical sources: OCT with recalibrated leaf probabilities, XGBoost `predict_proba`,
sklearn RandomForest `predict_proba` column 1, etc.

Usage
-----
    python public/plot_pr_auc_metrics.py \\
        --input predictions.parquet \\
        --label-col y_true \\
        --score-col y_score \\
        --output pr_curve.png

    # CSV example
    python public/plot_pr_auc_metrics.py -i scores.csv --label-col outcome --score-col p_case
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Headless-safe default unless the user passes --show (needs a GUI backend).
if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def _load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type {suf!r}; use .csv, .txt, or .parquet")


def _to_binary_labels(y: np.ndarray, positive_label) -> np.ndarray:
    """Map labels to {0, 1} with `positive_label` as class 1."""
    y = np.asarray(y)
    if positive_label is not None:
        return (y == positive_label).astype(np.int8)
    # Infer: expect exactly two unique values
    uniq = np.unique(y[~pd.isna(y)])
    if len(uniq) != 2:
        raise ValueError(
            f"Expected exactly two label values; got {len(uniq)}: {uniq[:10]!r}. "
            "Pass --positive-label explicitly."
        )
    lo, hi = float(np.min(uniq)), float(np.max(uniq))
    if lo == 0.0 and hi == 1.0:
        return y.astype(np.int8)
    # e.g. -1 / 1 -> 0 / 1
    return (y == hi).astype(np.int8)


def compute_pr_roc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return PR-AUC, ROC-AUC, and PR curve arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if np.any(~np.isfinite(y_score)):
        raise ValueError("Scores contain NaN or inf; remove or impute before plotting.")

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both classes in labels (got n_positive={n_pos}, n_negative={n_neg})."
        )

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "prevalence": float(n_pos / len(y_true)),
    }


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    pr_auc: float,
    prevalence: float,
    title: str | None,
    show_baseline: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(recall, precision, color="C0", lw=2.0, label=f"PR curve (AP = {pr_auc:.4f})")
    if show_baseline:
        ax.axhline(
            y=prevalence,
            color="gray",
            ls="--",
            lw=1.2,
            label=f"Chance (prevalence = {prevalence:.4f})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision–recall curve")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot PR curve and print PR-AUC and ROC-AUC from a table of labels and scores."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="CSV or Parquet file with label and score columns",
    )
    p.add_argument(
        "--label-col",
        required=True,
        help="Column name for binary ground-truth labels",
    )
    p.add_argument(
        "--score-col",
        required=True,
        help="Column name for predicted probability (or score) of the positive class",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pr_curve.png"),
        help="Output image path (.png, .pdf, .svg)",
    )
    p.add_argument("--title", default=None, help="Figure title")
    p.add_argument(
        "--positive-label",
        default=None,
        help="Value treated as positive class if labels are not 0/1 (e.g. 1 or True)",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not draw horizontal baseline at class prevalence",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure with an interactive backend (if available)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Raster resolution for bitmap formats")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = _load_table(args.input)
    if args.label_col not in df.columns:
        print(f"error: label column {args.label_col!r} not in columns: {list(df.columns)}", file=sys.stderr)
        return 1
    if args.score_col not in df.columns:
        print(f"error: score column {args.score_col!r} not in columns: {list(df.columns)}", file=sys.stderr)
        return 1

    y_raw = df[args.label_col].values
    y_score = df[args.score_col].values

    drop = pd.isna(y_raw) | pd.isna(y_score)
    n_drop = int(np.sum(drop))
    if n_drop:
        print(f"warning: dropping {n_drop} row(s) with missing label or score", file=sys.stderr)
        y_raw = y_raw[~drop]
        y_score = y_score[~drop]

    try:
        y_true = _to_binary_labels(y_raw, args.positive_label)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        m = compute_pr_roc(y_true, y_score)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"n = {len(y_true):,}")
    print(f"PR-AUC (average precision) = {m['pr_auc']:.6f}")
    print(f"ROC-AUC                      = {m['roc_auc']:.6f}")

    fig = plot_pr_curve(
        m["recall"],
        m["precision"],
        m["pr_auc"],
        m["prevalence"],
        args.title,
        show_baseline=not args.no_baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"saved figure to {args.output.resolve()}")

    if args.show:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
