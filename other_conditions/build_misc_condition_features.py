from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

def safe_col_name(x: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(x))
    return re.sub(r"_+", "_", s).strip("_")

# ----------------------------
# Spark bootstrap
# ----------------------------
def start_spark(driver_memory: str = "80g", storage_fraction: float = 0.5, num_threads: int = 10) -> SparkSession:
    conf = SparkConf().setAppName("misc_condition_feature_build")
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.memory.storageFraction", str(storage_fraction))
    conf.setMaster(f"local[{num_threads}]")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ----------------------------
# Discovery + schema helpers
# ----------------------------
def list_condition_codes(base_dir: Path) -> List[str]:
    """
    Find all codes that have BOTH <CODE>_claims and <CODE>_enrollment under base_dir.
    Assumes parquet folders (Spark can load directories).
    """
    claims = {p.name.replace("_claims", "") for p in base_dir.glob("*_claims")}
    enroll = {p.name.replace("_enrollment", "") for p in base_dir.glob("*_enrollment")}
    codes = sorted(claims.intersection(enroll))
    return codes


def normalize_icd(code_col: F.Column) -> F.Column:
    """Remove dots and whitespace, uppercase."""
    return F.upper(F.regexp_replace(F.trim(code_col), r"\.", ""))


def get_dx_cols(df_claims: DataFrame) -> List[str]:
    cols = df_claims.columns
    dx = sorted([c for c in cols if re.fullmatch(r"DX\d+", c)])
    if "PDX" in cols:
        dx.append("PDX")
    if not dx:
        raise ValueError("No diagnosis columns found. Expected DX1..DXn and/or PDX.")
    return dx


def pick_cost_column(df_claims: DataFrame) -> str:
    """Prefer NETPAY; fall back to common alternatives."""
    candidates = ["NETPAY", "TOTPAY", "PAY", "ALLOWED", "NETPAYAMT"]
    for c in candidates:
        if c in df_claims.columns:
            return c
    raise ValueError(f"No cost column found among {candidates}. Claims columns: {df_claims.columns}")


def ensure_date(df: DataFrame, col: str) -> DataFrame:
    if dict(df.dtypes).get(col) in ("date", "timestamp"):
        return df
    # common case: string
    return df.withColumn(col, F.to_date(F.col(col)))


# ----------------------------
# Core feature blocks
# ----------------------------
def build_annual_costs_and_labels(
    df_claims: DataFrame,
    baseline_year: int,
    outcome_year: int,
    inflation: Dict[int, float],
    enrolid_col: str = "ENROLID",
    date_col: str = "SVCDATE",
) -> Tuple[DataFrame, Dict[str, float]]:
    """
    Creates deflated annual costs for baseline/outcome years + percentile labels for outcome year.
    Uses RX-only filtering if possible (to match your MSK approach), otherwise uses all claims.
    """
    df_claims = ensure_date(df_claims, date_col)
    cost_col = pick_cost_column(df_claims)

    infl_map = sum([[F.lit(y), F.lit(f)] for y, f in inflation.items()], [])
    infl_expr = F.create_map(*infl_map)

    df = (
        df_claims.withColumn("YEAR", F.year(F.col(date_col)))
                 .filter(F.col("YEAR").isin([baseline_year, outcome_year]))
                 .withColumn("inflation_factor", infl_expr[F.col("YEAR")])
                 .withColumn("COST_DEF", F.col(cost_col) / F.col("inflation_factor"))
                 .drop("inflation_factor")
    )

    # RX-only if we can detect it
    if "CLAIM_TYPE" in df.columns:
        df = df.filter(F.col("CLAIM_TYPE") == "RX")
    elif "RX" in df.columns:
        df = df.filter(F.col("RX").isNotNull())
    # else: fall back to all claims

    annual = (
        df.groupBy(enrolid_col, "YEAR")
          .agg(F.sum("COST_DEF").alias("annual_cost_deflated"))
    )

    base = (
        annual.filter(F.col("YEAR") == baseline_year)
              .select(enrolid_col, F.col("annual_cost_deflated").alias(f"annual_cost_{baseline_year}_deflated"))
    )
    out = (
        annual.filter(F.col("YEAR") == outcome_year)
              .select(enrolid_col, F.col("annual_cost_deflated").alias(f"annual_cost_{outcome_year}_deflated"))
    )

    merged = base.join(out, on=enrolid_col, how="inner")

    out_cost_col = f"annual_cost_{outcome_year}_deflated"
    # Approx quantiles for thresholding (fast)
    p90, p95, p98 = merged.approxQuantile(out_cost_col, [0.90, 0.95, 0.98], 0.001)
    thresholds = {"p90": float(p90), "p95": float(p95), "p98": float(p98)}


    merged = (
        merged.withColumn(f"top_10_pct_cost_{outcome_year}", (F.col(out_cost_col) >= F.lit(p90)).cast("int"))
              .withColumn(f"top_5_pct_cost_{outcome_year}",  (F.col(out_cost_col) >= F.lit(p95)).cast("int"))
              .withColumn(f"top_2_pct_cost_{outcome_year}",  (F.col(out_cost_col) >= F.lit(p98)).cast("int"))
    )
    return merged, thresholds


def top_k_values(df: DataFrame, col: str, k: int) -> List[str]:
    rows = df.groupBy(col).agg(F.count("*").alias("n")).orderBy(F.desc("n")).limit(k).collect()
    return [r[col] for r in rows if r[col] is not None]


def build_icd_onehot(
    df_claims: DataFrame,
    baseline_year: int,
    condition_regex: str,
    k_condition: int = 25,
    k_comorb: int = 25,
    enrolid_col: str = "ENROLID",
    date_col: str = "SVCDATE",
) -> Tuple[DataFrame, List[str], List[str]]:
    """
    Builds:
      - top-K condition ICD3 flags (icd3 = first 3 chars after normalization)
      - top-K comorbidity ICD3 flags excluding condition matches
    """
    df_claims = ensure_date(df_claims, date_col)
    dx_cols = get_dx_cols(df_claims)

    df = (
        df_claims.withColumn("YEAR", F.year(F.col(date_col)))
                 .filter(F.col("YEAR") == baseline_year)
                 .select(enrolid_col, *dx_cols)
    )

    exploded = (
        df.withColumn("raw_code", F.explode(F.array(*[F.col(c) for c in dx_cols])))
          .filter(F.col("raw_code").isNotNull())
          .withColumn("icd_norm", normalize_icd(F.col("raw_code")))
          .withColumn("icd3", F.substring(F.col("icd_norm"), 1, 3))
    )

    cond = exploded.filter(F.col("icd_norm").rlike(condition_regex))
    noncond = exploded.filter(~F.col("icd_norm").rlike(condition_regex))

    top_cond = top_k_values(cond.select("icd3"), "icd3", k_condition)
    top_comorb = top_k_values(noncond.select("icd3"), "icd3", k_comorb)

    def pivot_onehot(src: DataFrame, values: List[str], prefix: str) -> DataFrame:
        if not values:
            return src.select(enrolid_col).distinct()
        return (
            src.filter(F.col("icd3").isin(values))
               .select(enrolid_col, "icd3")
               .dropDuplicates()
               .withColumn("flag", F.lit(1))
               .groupBy(enrolid_col)
               .pivot("icd3", values)
               .agg(F.max("flag"))
               .na.fill(0)
               .select(
                   enrolid_col,
                   *[F.col(v).alias(f"{prefix}{v}") for v in values]
               )
        )

    cond_oh = pivot_onehot(cond, top_cond, prefix="has_cond_icd3_")
    comorb_oh = pivot_onehot(noncond, top_comorb, prefix="has_comorb_icd3_")

    out = cond_oh.join(comorb_oh, on=enrolid_col, how="outer").na.fill(0)
    return out, top_cond, top_comorb


def build_proc_onehot(
    df_claims: DataFrame,
    baseline_year: int,
    condition_regex: str,
    k_proc: int = 25,
    enrolid_col: str = "ENROLID",
    date_col: str = "SVCDATE",
    proc_col: str = "PROCGRP",
) -> Tuple[DataFrame, List[str]]:
    """
    Top-K PROCGRP among baseline-year claims that have a condition dx.
    """
    if proc_col not in df_claims.columns:
        return df_claims.select(enrolid_col).distinct(), []

    df_claims = ensure_date(df_claims, date_col)
    dx_cols = get_dx_cols(df_claims)

    base = (
        df_claims.withColumn("YEAR", F.year(F.col(date_col)))
                 .filter(F.col("YEAR") == baseline_year)
                 .filter(F.col(proc_col).isNotNull())
                 .select(enrolid_col, proc_col, *dx_cols)
    )

    has_cond_dx = None
    for c in dx_cols:
        term = normalize_icd(F.col(c)).rlike(condition_regex)
        has_cond_dx = term if has_cond_dx is None else (has_cond_dx | term)

    cond_claims = base.filter(has_cond_dx)

    top_proc = top_k_values(cond_claims.select(proc_col), proc_col, k_proc)

    if not top_proc:
        return df_claims.select(enrolid_col).distinct(), []

    oh = (
        base.filter(F.col(proc_col).isin(top_proc))
            .select(enrolid_col, proc_col)
            .dropDuplicates()
            .withColumn("flag", F.lit(1))
            .groupBy(enrolid_col)
            .pivot(proc_col, top_proc)
            .agg(F.max("flag"))
            .na.fill(0)
    )

    for p in top_proc:
        old = str(p)
        new = f"has_procgrp_{safe_col_name(old)}"
        if old in oh.columns:
            oh = oh.withColumnRenamed(old, new)
    return oh, top_proc

def build_medication_onehot(
    df_claims: DataFrame,
    df_redbook: DataFrame,
    cohort_enrollees: DataFrame,
    baseline_year: int,
    topk_thercls: int = 50,
    enrolid_col: str = "ENROLID",
    date_col: str = "SVCDATE",
) -> Tuple[DataFrame, Dict[str, str], List[str]]:
    """
    Medication features using Redbook:
      - filter baseline-year RX claims for cohort enrollees
      - top-K THERCLS
      - map THERCLS -> (THERGRP, THRGRDS) from redbook
      - pivot on THERGRP and rename columns using THRGRDS names

    Returns:
      df_med_flags: ENROLID + one-hot columns
      grp_map: {THERGRP_code_str: THRGRDS_name}
      top_thercls: list of THERCLS strings used
    """

    if df_redbook is None:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

    if "THERCLS" not in df_claims.columns:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

    # redbook columns (handle minor schema variation)
    rb_cols = set(df_redbook.columns)
    if "THERGRP" not in rb_cols or ("THRGRDS" not in rb_cols and "THERGRDS" not in rb_cols):
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

    thrgrds_col = "THRGRDS" if "THRGRDS" in rb_cols else "THERGRDS"

    df_claims = ensure_date(df_claims, date_col)

    # RX filter (same spirit as your MSK notebook)
    rx = df_claims.withColumn("YEAR", F.year(F.col(date_col))).filter(F.col("YEAR") == baseline_year)
    if "CLAIM_TYPE" in rx.columns:
        rx = rx.filter(F.col("CLAIM_TYPE") == "RX")

    rx = rx.join(cohort_enrollees.select(enrolid_col).distinct(), on=enrolid_col, how="inner")

    # Use STRING keys to avoid Decimal / type mismatch issues across datasets
    rx = rx.withColumn("THERCLS_S", F.col("THERCLS").cast("string"))

    # Top-K THERCLS by frequency
    top_rows = (
        rx.groupBy("THERCLS_S").agg(F.count("*").alias("n"))
          .orderBy(F.desc("n"))
          .limit(topk_thercls)
          .collect()
    )
    top_thercls = [r["THERCLS_S"] for r in top_rows if r["THERCLS_S"] is not None]
    if not top_thercls:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

    # Prepare redbook mapping (string-typed join keys)
    rb = (
        df_redbook
          .select(
              F.col("THERCLS").cast("string").alias("THERCLS_S"),
              F.col("THERGRP").cast("string").alias("THERGRP_S"),
              F.col(thrgrds_col).cast("string").alias("THRGRDS_S"),
          )
          .filter(F.col("THERCLS_S").isin(top_thercls))
          .dropDuplicates(["THERCLS_S", "THERGRP_S"])
    )

    # Choose one mapping per THERCLS (your notebook used first())
    thercls_to_group = (
        rb.groupBy("THERCLS_S")
          .agg(
              F.first("THERGRP_S").alias("THERGRP_S"),
              F.first("THRGRDS_S").alias("THRGRDS_S"),
          )
    )

    # Join RX claims -> therapeutic groups, then distinct per enrollee/group
    rx_groups = (
        rx.select(enrolid_col, "THERCLS_S")
          .filter(F.col("THERCLS_S").isin(top_thercls))
          .join(thercls_to_group, on="THERCLS_S", how="left")
          .select(enrolid_col, "THERGRP_S", "THRGRDS_S")
          .filter(F.col("THERGRP_S").isNotNull())
          .distinct()
    )

    # Unique THERGRP codes for pivot (as strings)
    grp_pairs = (
        rx_groups.select("THERGRP_S", "THRGRDS_S")
                 .distinct()
                 .orderBy("THERGRP_S")
                 .collect()
    )
    grp_codes = [r["THERGRP_S"] for r in grp_pairs]
    grp_map = {r["THERGRP_S"]: (r["THRGRDS_S"] or r["THERGRP_S"]) for r in grp_pairs}

    if not grp_codes:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, top_thercls

    # Pivot
    df_med = (
        rx_groups.withColumn("flag", F.lit(1))
                 .groupBy(enrolid_col)
                 .pivot("THERGRP_S", grp_codes)
                 .agg(F.max("flag"))
                 .na.fill(0)
    )

    # Rename to readable columns; prefix to avoid collisions
    for code in grp_codes:
        name = grp_map.get(code, code)
        new_col = f"med_has_{safe_col_name(name)}"
        df_med = df_med.withColumnRenamed(str(code), new_col)

    return df_med, grp_map, top_thercls

def add_quarterly_derivatives(df: DataFrame, cost_type: str, year: int) -> DataFrame:
    qcols = [f"{year}Q{q}_{cost_type}_cost_3month" for q in [1, 2, 3, 4]]
    q = [F.coalesce(F.col(c), F.lit(0.0)) for c in qcols]

    df = df.withColumn(f"{cost_type}_cost_deriv_Q1_Q2_{year}", q[1] - q[0]) \
           .withColumn(f"{cost_type}_cost_deriv_Q2_Q3_{year}", q[2] - q[1]) \
           .withColumn(f"{cost_type}_cost_deriv_Q3_Q4_{year}", q[3] - q[2])

    df = df.withColumn(f"{cost_type}_is_increasing_Q1_Q2_{year}", (F.col(f"{cost_type}_cost_deriv_Q1_Q2_{year}") > 0).cast("int")) \
           .withColumn(f"{cost_type}_is_increasing_Q2_Q3_{year}", (F.col(f"{cost_type}_cost_deriv_Q2_Q3_{year}") > 0).cast("int")) \
           .withColumn(f"{cost_type}_is_increasing_Q3_Q4_{year}", (F.col(f"{cost_type}_cost_deriv_Q3_Q4_{year}") > 0).cast("int"))

    df = df.withColumn(
            f"{cost_type}_total_increasing_quarters_{year}",
            F.col(f"{cost_type}_is_increasing_Q1_Q2_{year}") +
            F.col(f"{cost_type}_is_increasing_Q2_Q3_{year}") +
            F.col(f"{cost_type}_is_increasing_Q3_Q4_{year}")
        ) \
        .withColumn(f"{cost_type}_is_consistently_increasing_{year}", (F.col(f"{cost_type}_total_increasing_quarters_{year}") == 3).cast("int")) \
        .withColumn(f"{cost_type}_is_consistently_decreasing_{year}", (F.col(f"{cost_type}_total_increasing_quarters_{year}") == 0).cast("int")) \
        .withColumn(
            f"{cost_type}_avg_quarterly_derivative_{year}",
            (F.col(f"{cost_type}_cost_deriv_Q1_Q2_{year}") +
             F.col(f"{cost_type}_cost_deriv_Q2_Q3_{year}") +
             F.col(f"{cost_type}_cost_deriv_Q3_Q4_{year}")) / F.lit(3.0)
        )
    return df


def add_temporal_stats(df: DataFrame, cost_type: str, year: int) -> DataFrame:
    qcols = [f"{year}Q{q}_{cost_type}_cost_3month" for q in [1, 2, 3, 4]]
    q1, q2, q3, q4 = [F.coalesce(F.col(c), F.lit(0.0)) for c in qcols]

    mean_col = f"{cost_type}_quarterly_mean_{year}"
    var_col = f"{cost_type}_quarterly_variance_{year}"
    std_col = f"{cost_type}_quarterly_std_{year}"

    df = df.withColumn(mean_col, (q1 + q2 + q3 + q4) / F.lit(4.0)) \
           .withColumn(
               var_col,
               (((q1 - F.col(mean_col)) ** 2) +
                ((q2 - F.col(mean_col)) ** 2) +
                ((q3 - F.col(mean_col)) ** 2) +
                ((q4 - F.col(mean_col)) ** 2)) / F.lit(4.0)
           ) \
           .withColumn(std_col, F.sqrt(F.col(var_col)))

    df = df.withColumn(
            f"{cost_type}_quarterly_skewness_{year}",
            F.when(F.col(std_col) > 0,
                   ((((q1 - F.col(mean_col)) / F.col(std_col)) ** 3) +
                    (((q2 - F.col(mean_col)) / F.col(std_col)) ** 3) +
                    (((q3 - F.col(mean_col)) / F.col(std_col)) ** 3) +
                    (((q4 - F.col(mean_col)) / F.col(std_col)) ** 3)) / F.lit(4.0)
                   ).otherwise(F.lit(0.0))
        ) \
        .withColumn(
            f"{cost_type}_quarterly_kurtosis_{year}",
            F.when(F.col(std_col) > 0,
                   ((((q1 - F.col(mean_col)) / F.col(std_col)) ** 4) +
                    (((q2 - F.col(mean_col)) / F.col(std_col)) ** 4) +
                    (((q3 - F.col(mean_col)) / F.col(std_col)) ** 4) +
                    (((q4 - F.col(mean_col)) / F.col(std_col)) ** 4)) / F.lit(4.0) - F.lit(3.0)
                   ).otherwise(F.lit(0.0))
        ) \
        .withColumn(
            f"{cost_type}_quarterly_cv_{year}",
            F.when(F.col(mean_col) > 0, F.col(std_col) / F.col(mean_col)).otherwise(F.lit(0.0))
        ) \
        .withColumn(f"{cost_type}_quarterly_max_{year}", F.greatest(q1, q2, q3, q4)) \
        .withColumn(f"{cost_type}_quarterly_min_{year}", F.least(q1, q2, q3, q4)) \
        .withColumn(f"{cost_type}_quarterly_range_{year}", F.col(f"{cost_type}_quarterly_max_{year}") - F.col(f"{cost_type}_quarterly_min_{year}"))

    # Interpretable buckets
    df = df.withColumn(
            f"{cost_type}_cost_pattern_{year}",
            F.when(F.col(f"{cost_type}_quarterly_skewness_{year}") > 0.5, F.lit("late_heavy"))
             .when(F.col(f"{cost_type}_quarterly_skewness_{year}") < -0.5, F.lit("early_heavy"))
             .otherwise(F.lit("balanced"))
        ) \
        .withColumn(
            f"{cost_type}_cost_stability_{year}",
            F.when(F.col(f"{cost_type}_quarterly_cv_{year}") < 0.3, F.lit("stable"))
             .when(F.col(f"{cost_type}_quarterly_cv_{year}") > 1.0, F.lit("highly_volatile"))
             .otherwise(F.lit("moderate"))
        )

    return df.drop(mean_col, var_col, std_col)


def build_quarterly_cost_features(
    df_claims: DataFrame,
    baseline_year: int,
    inflation: Dict[int, float],
    condition_regex: str,
    comorb_icd3_list: List[str],
    proc_list: List[str],
    enrolid_col: str = "ENROLID",
    date_col: str = "SVCDATE",
    proc_col: str = "PROCGRP",
) -> DataFrame:
    """
    Quarterly baseline-year cost features:
      - direct_condition_cost_3month (dx matches condition)
      - condition_procedure_cost_3month (PROCGRP in top proc list)
      - comorbidity_only_cost_3month (icd3 in top comorb list, AND not condition dx)
    plus annual sums + derivatives + temporal stats for each cost type.
    """
    df_claims = ensure_date(df_claims, date_col)
    dx_cols = get_dx_cols(df_claims)
    cost_col = pick_cost_column(df_claims)

    infl_map = sum([[F.lit(y), F.lit(f)] for y, f in inflation.items()], [])
    infl_expr = F.create_map(*infl_map)

    base = (
        df_claims.withColumn("YEAR", F.year(F.col(date_col)))
                 .filter(F.col("YEAR") == baseline_year)
                 .withColumn("QUARTER", F.quarter(F.col(date_col)))
                 .withColumn("inflation_factor", infl_expr[F.col("YEAR")])
                 .withColumn("COST_DEF", F.col(cost_col) / F.col("inflation_factor"))
                 .drop("inflation_factor")
    )

    # condition dx flag
    has_cond_dx = None
    for c in dx_cols:
        term = normalize_icd(F.col(c)).rlike(condition_regex)
        has_cond_dx = term if has_cond_dx is None else (has_cond_dx | term)

    # comorbidity flag based on ICD3 list
    # (we compute icd3 per claim as "any diagnosis icd3 in list")
    icd3_terms = []
    for c in dx_cols:
        icd3_terms.append(F.substring(normalize_icd(F.col(c)), 1, 3).isin(comorb_icd3_list))
    has_comorb = None
    for t in icd3_terms:
        has_comorb = t if has_comorb is None else (has_comorb | t)

    # procedure flag
    has_proc = F.lit(False)
    if proc_col in base.columns and proc_list:
        has_proc = F.col(proc_col).isin(proc_list)

    typed = (
        base.withColumn("has_cond_dx", has_cond_dx)
            .withColumn("has_cond_proc", has_proc)
            .withColumn("has_comorb", has_comorb)
    )

    quarterly = (
        typed.groupBy(enrolid_col, "QUARTER")
             .agg(
                 F.sum(F.when(F.col("has_cond_dx"), F.col("COST_DEF")).otherwise(0.0)).alias("direct_condition_cost_3month"),
                 F.sum(F.when(F.col("has_cond_proc"), F.col("COST_DEF")).otherwise(0.0)).alias("condition_procedure_cost_3month"),
                 F.sum(F.when(F.col("has_comorb") & ~F.col("has_cond_dx"), F.col("COST_DEF")).otherwise(0.0)).alias("comorbidity_only_cost_3month"),
                 F.count("*").alias("total_claims_3month"),
             )
    )

    quarter_labels = [f"{baseline_year}Q{q}" for q in [1, 2, 3, 4]]
    wide = (
        quarterly.withColumn("YQ", F.concat_ws("Q", F.lit(str(baseline_year)), F.col("QUARTER")))
                .groupBy(enrolid_col)
                .pivot("YQ", quarter_labels)
                .agg(
                    F.first("direct_condition_cost_3month").alias("direct_condition_cost"),
                    F.first("condition_procedure_cost_3month").alias("condition_procedure_cost"),
                    F.first("comorbidity_only_cost_3month").alias("comorbidity_only_cost"),
                    F.first("total_claims_3month").alias("total_claims"),
                )
    )

    # flatten names + fill missing
    fill = {}
    for q in [1, 2, 3, 4]:
        ql = f"{baseline_year}Q{q}"
        ren = {
            f"{ql}_direct_condition_cost": f"{ql}_direct_condition_cost_3month",
            f"{ql}_condition_procedure_cost": f"{ql}_condition_procedure_cost_3month",
            f"{ql}_comorbidity_only_cost": f"{ql}_comorbidity_only_cost_3month",
            f"{ql}_total_claims": f"{ql}_total_claims_3month",
        }
        for old, new in ren.items():
            if old in wide.columns:
                wide = wide.withColumnRenamed(old, new)
        fill[f"{ql}_direct_condition_cost_3month"] = 0.0
        fill[f"{ql}_condition_procedure_cost_3month"] = 0.0
        fill[f"{ql}_comorbidity_only_cost_3month"] = 0.0
        fill[f"{ql}_total_claims_3month"] = 0

    wide = wide.na.fill(fill)

    # annual sums
    def sum_q(prefix: str) -> F.Column:
        return sum([F.coalesce(F.col(f"{baseline_year}Q{q}_{prefix}_3month"), F.lit(0.0)) for q in [1, 2, 3, 4]])

    wide = wide.withColumn("direct_condition_cost_annual", sum_q("direct_condition_cost")) \
               .withColumn("condition_procedure_cost_annual", sum_q("condition_procedure_cost")) \
               .withColumn("comorbidity_only_cost_annual", sum_q("comorbidity_only_cost"))

    # derivatives + stats for each cost type
    for ct in ["direct_condition", "condition_procedure", "comorbidity_only"]:
        wide = add_quarterly_derivatives(wide, cost_type=ct, year=baseline_year)
        wide = add_temporal_stats(wide, cost_type=ct, year=baseline_year)

    # fill pattern/stability strings
    for ct in ["direct_condition", "condition_procedure", "comorbidity_only"]:
        wide = wide.na.fill({
            f"{ct}_cost_pattern_{baseline_year}": "balanced",
            f"{ct}_cost_stability_{baseline_year}": "moderate",
        })

    return wide


def build_demographics(df_enrollment: DataFrame, enrolid_col: str = "ENROLID") -> DataFrame:
    """
    Take first observed values for common enrollment demographics if they exist.
    """
    cand = ["AGEGRP", "SEX", "REGION", "EESTATU"]
    cols = [c for c in cand if c in df_enrollment.columns]
    if not cols:
        return df_enrollment.select(enrolid_col).distinct()
    return df_enrollment.groupBy(enrolid_col).agg(*[F.first(c).alias(c) for c in cols])


# ----------------------------
# End-to-end per condition
# ----------------------------
def build_features_for_condition(
    spark: SparkSession,
    base_dir: Path,
    out_dir: Path,
    code: str,
    baseline_year: int,
    outcome_year: int,
    inflation: Dict[int, float],
    k_cond_icd: int,
    k_comorb_icd: int,
    k_proc: int,
    df_redbook: Optional[DataFrame],
    k_med_thercls: int,
) -> None:
    claims_path = base_dir / f"{code}_claims"
    enroll_path = base_dir / f"{code}_enrollment"
    if not claims_path.exists() or not enroll_path.exists():
        print(f"[SKIP] Missing files for {code}")
        return

    df_claims = spark.read.format("parquet").load(str(claims_path))
    df_enroll = spark.read.format("parquet").load(str(enroll_path))

    # Condition regex: match codes starting with ICD-10 root (dotless)
    # Example: E11 matches E11*, I50 matches I50*, C50 matches C50*
    condition_regex = rf"^{re.escape(code.upper())}"
    print("Build annual cost labels and thresholds")

    # 1) annual cost labels
    costs_labels, thresholds = build_annual_costs_and_labels(
        df_claims, baseline_year, outcome_year, inflation
    )
    print("Build dx onehots (condition + comorb icd3)")
    # 2) dx onehots (condition + comorb icd3)
    icd_onehots, top_cond_icd3, top_comorb_icd3 = build_icd_onehot(
        df_claims, baseline_year, condition_regex, k_condition=k_cond_icd, k_comorb=k_comorb_icd
    )
    print("Build procedures (top-k among condition dx claims)")
    # 3) procedures (top-k among condition dx claims)
    proc_oh, top_proc = build_proc_onehot(
        df_claims, baseline_year, condition_regex, k_proc=k_proc
    )
    top_proc = [str(x) for x in top_proc]

    print("Build quarterly cost features (needs comorb/proc lists)")
    # 4) quarterly + enhanced cost features (needs comorb/proc lists)
    qcost = build_quarterly_cost_features(
        df_claims=df_claims,
        baseline_year=baseline_year,
        inflation=inflation,
        condition_regex=condition_regex,
        comorb_icd3_list=top_comorb_icd3,
        proc_list=top_proc,
    )
    print("Build demographics")
    # 5) demographics
    demo = build_demographics(df_enroll)

    # Merge (inner join on costs_labels to ensure both baseline+outcome costs exist)
    all_enrollees = costs_labels.select("ENROLID").distinct()
    med_flags, med_grp_map, top_thercls = build_medication_onehot(
    df_claims=df_claims,
    df_redbook=df_redbook,
    cohort_enrollees=all_enrollees,
    baseline_year=baseline_year,
    topk_thercls=k_med_thercls)

    print("Merge all features")
    features = (
        all_enrollees
            .join(costs_labels, on="ENROLID", how="inner")
            .join(icd_onehots, on="ENROLID", how="left")
            .join(proc_oh, on="ENROLID", how="left")
            .join(qcost, on="ENROLID", how="left")
            .join(med_flags, on="ENROLID", how="left")   
            .join(demo, on="ENROLID", how="left")
            .na.fill(0)  # safe for numeric indicators/costs; pattern/stability filled earlier
    )
    print("Save features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet"
    features.write.mode("overwrite").parquet(str(out_path))
    print("Save metadata")
    # Save metadata for reproducibility
    meta = {
        "code": code,
        "baseline_year": baseline_year,
        "outcome_year": outcome_year,
        "condition_regex": condition_regex,
        "thresholds": thresholds,
        "top_cond_icd3": top_cond_icd3,
        "top_comorb_icd3": top_comorb_icd3,
        "top_procgrp": top_proc,
        "top_med_thercls": top_thercls,
        "med_thergrp_to_name": med_grp_map,
    }
    with open(out_dir / f"{code}_features_{baseline_year}_{outcome_year}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] {code}: wrote {out_path.name} with {features.count():,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/Users/charles/DATA/misc_conditions")
    parser.add_argument("--out_dir", type=str, default="/Users/cat2510/my_projects/other_conditions/misc_conditions_features")
    parser.add_argument("--baseline_year", type=int, default=2017)
    parser.add_argument("--outcome_year", type=int, default=2018)
    parser.add_argument("--codes", type=str, default="")  # comma-separated override
    parser.add_argument("--k_cond_icd", type=int, default=25)
    parser.add_argument("--k_comorb_icd", type=int, default=25)
    parser.add_argument("--k_proc", type=int, default=25)
    parser.add_argument("--driver_memory", type=str, default="80g")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--add_k_med_thercls", type=int, default=50)
    args = parser.parse_args()

    spark = start_spark(driver_memory=args.driver_memory, num_threads=args.threads)
    df_redbook = None
    rb_path = Path("/Users/Charles/DATA/ckd/redbook")
    if rb_path.exists():
        df_redbook = spark.read.format("parquet").load(str(rb_path)).cache()
        _ = df_redbook.count()  # materialize cache
        print(f"Loaded redbook: /Users/Charles/DATA/ckd/redbook")
    else:
        print(f"[WARN] redbook not found at {args.redbook_dir}; skipping medication features.")

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    # inflation to deflate to baseline year dollars (edit if you change years)
    inflation = {args.baseline_year: 1.0}
    if args.outcome_year == 2018 and args.baseline_year == 2017:
        inflation[2018] = 1.0685946832951803
    else:
        # fallback: no deflation unless you supply your own map
        inflation[args.outcome_year] = 1.0

    if args.codes.strip():
        codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
    else:
        codes = list_condition_codes(base_dir)

    print(f"Found {len(codes)} condition cohorts: {codes}")

    for code in codes:
        build_features_for_condition(
            spark=spark,
            base_dir=base_dir,
            out_dir=out_dir,
            code=code,
            baseline_year=args.baseline_year,
            outcome_year=args.outcome_year,
            inflation=inflation,
            k_cond_icd=args.k_cond_icd,
            k_comorb_icd=args.k_comorb_icd,
            k_proc=args.k_proc,
        )

    spark.stop()


if __name__ == "__main__":
    main()
