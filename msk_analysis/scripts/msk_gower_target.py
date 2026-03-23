"""MSK binary cost targets: top-X% from annual_cost_2018_deflated."""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def target_column_name_for_top_pct(top_pct: float) -> str:
    """e.g. 2 -> top_2_pct_cost_2018, 0.5 -> top_0_5_pct_cost_2018."""
    if top_pct == int(top_pct):
        return f"top_{int(top_pct)}_pct_cost_2018"
    return f"top_{str(top_pct).replace('.', '_')}_pct_cost_2018"


def ensure_msk_top_pct_cost_target(
    df: pd.DataFrame,
    target_col: str,
    *,
    cost_col: str = "annual_cost_2018_deflated",
    verbose: bool = True,
) -> None:
    """
    If ``target_col`` is missing, parse ``top_<p>_pct_cost_2018`` and create it from ``cost_col``.
    Threshold: values >= quantile(1 - p/100) are case (1).
    """
    if target_col in df.columns:
        if verbose:
            print(f"  Using existing target: {target_col}")
        return
    m = re.match(r"^top_(\d+(?:_\d+)?)_pct_cost_2018$", target_col)
    if not m:
        raise ValueError(
            f"Column {target_col!r} not in dataframe and name does not match "
            "top_<pct>_pct_cost_2018 (pct may use underscore for decimals, e.g. 0_5)."
        )
    if cost_col not in df.columns:
        raise ValueError(f"Need {cost_col!r} to derive {target_col}")
    pct = float(m.group(1).replace("_", "."))
    thresh = df[cost_col].quantile(1.0 - pct / 100.0)
    df[target_col] = (df[cost_col] >= thresh).astype(int)
    if verbose:
        print(f"  Derived {target_col} (top {pct}%): threshold={thresh:g}")
        print(df[target_col].value_counts())
