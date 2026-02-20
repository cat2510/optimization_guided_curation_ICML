from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


# ----------------------------
# Spark bootstrap
# ----------------------------
def start_spark(driver_memory: str = "80g", storage_fraction: float = 0.5, num_threads: int = 10) -> SparkSession:
    conf = SparkConf().setAppName("augment_features_with_meds")
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.memory.storageFraction", str(storage_fraction))
    conf.setMaster(f"local[{num_threads}]")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ----------------------------
# Helpers
# ----------------------------
def ensure_date(df: DataFrame, col: str) -> DataFrame:
    if dict(df.dtypes).get(col) in ("date", "timestamp"):
        return df
    return df.withColumn(col, F.to_date(F.col(col)))


def safe_col_name(x: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(x))
    return re.sub(r"_+", "_", s).strip("_")


def json_safe(x):
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    return x


def list_condition_codes(base_dir: Path) -> List[str]:
    claims = {p.name.replace("_claims", "") for p in base_dir.glob("*_claims")}
    enroll = {p.name.replace("_enrollment", "") for p in base_dir.glob("*_enrollment")}
    return sorted(claims.intersection(enroll))


# ----------------------------
# Medication features (Redbook)
# ----------------------------
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
    Builds medication flags using Redbook mapping:
      baseline-year RX claims -> top-K THERCLS -> map to THERGRP + THRGRDS -> pivot by THERGRP.
    """
    if df_redbook is None or "THERCLS" not in df_claims.columns:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

    rb_cols = set(df_redbook.columns)
    if "THERGRP" not in rb_cols or ("THRGRDS" not in rb_cols and "THERGRDS" not in rb_cols):
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

    thrgrds_col = "THRGRDS" if "THRGRDS" in rb_cols else "THERGRDS"

    df_claims = ensure_date(df_claims, date_col)

    rx = df_claims.withColumn("YEAR", F.year(F.col(date_col))).filter(F.col("YEAR") == baseline_year)

    # RX filter (best-effort; keeps logic consistent with typical MarketScan layouts)
    if "CLAIM_TYPE" in rx.columns:
        rx = rx.filter(F.col("CLAIM_TYPE") == "RX")

    rx = rx.join(cohort_enrollees.select(enrolid_col).distinct(), on=enrolid_col, how="inner")
    rx = rx.withColumn("THERCLS_S", F.col("THERCLS").cast("string"))

    # top-K THERCLS
    top_rows = (
        rx.groupBy("THERCLS_S").agg(F.count("*").alias("n"))
          .orderBy(F.desc("n"))
          .limit(topk_thercls)
          .collect()
    )
    top_thercls = [r["THERCLS_S"] for r in top_rows if r["THERCLS_S"] is not None]
    if not top_thercls:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, []

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

    thercls_to_group = (
        rb.groupBy("THERCLS_S")
          .agg(
              F.first("THERGRP_S").alias("THERGRP_S"),
              F.first("THRGRDS_S").alias("THRGRDS_S"),
          )
    )

    rx_groups = (
        rx.select(enrolid_col, "THERCLS_S")
          .filter(F.col("THERCLS_S").isin(top_thercls))
          .join(thercls_to_group, on="THERCLS_S", how="left")
          .select(enrolid_col, "THERGRP_S", "THRGRDS_S")
          .filter(F.col("THERGRP_S").isNotNull())
          .distinct()
    )

    grp_pairs = rx_groups.select("THERGRP_S", "THRGRDS_S").distinct().orderBy("THERGRP_S").collect()
    grp_codes = [r["THERGRP_S"] for r in grp_pairs]
    grp_map = {r["THERGRP_S"]: (r["THRGRDS_S"] or r["THERGRP_S"]) for r in grp_pairs}

    if not grp_codes:
        return cohort_enrollees.select(enrolid_col).distinct(), {}, top_thercls

    df_med = (
        rx_groups.withColumn("flag", F.lit(1))
                 .groupBy(enrolid_col)
                 .pivot("THERGRP_S", grp_codes)
                 .agg(F.max("flag"))
                 .na.fill(0)
    )

    for code in grp_codes:
        name = grp_map.get(code, code)
        df_med = df_med.withColumnRenamed(str(code), f"med_has_{safe_col_name(name)}")

    return df_med, grp_map, top_thercls


# ----------------------------
# Augment existing feature parquet
# ----------------------------
def augment_one_condition(
    spark: SparkSession,
    base_dir: Path,
    features_dir: Path,
    out_dir: Path,
    code: str,
    baseline_year: int,
    outcome_year: int,
    df_redbook: Optional[DataFrame],
    k_med_thercls: int,
    overwrite: bool,
) -> None:
    claims_path = base_dir / f"{code}_claims"
    feat_path = features_dir / f"{code}_features_{baseline_year}_{outcome_year}.parquet"
    meta_path = features_dir / f"{code}_features_{baseline_year}_{outcome_year}.meta.json"

    if not claims_path.exists():
        print(f"[SKIP] {code}: missing claims at {claims_path}")
        return
    if not feat_path.exists():
        print(f"[SKIP] {code}: missing features at {feat_path}")
        return

    df_feats = spark.read.parquet(str(feat_path))
    cohort = df_feats.select("ENROLID").distinct()

    df_claims = spark.read.parquet(str(claims_path))

    med_flags, med_grp_map, top_thercls = build_medication_onehot(
        df_claims=df_claims,
        df_redbook=df_redbook,
        cohort_enrollees=cohort,
        baseline_year=baseline_year,
        topk_thercls=k_med_thercls,
    )

    df_aug = (
        df_feats.join(med_flags, on="ENROLID", how="left")
                .na.fill(0)  # safe because med flags are 0/1; won’t affect strings unless you have them as nulls
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        out_path = feat_path
        mode = "overwrite"
    else:
        out_path = out_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.parquet"
        mode = "overwrite"

    df_aug.write.mode(mode).parquet(str(out_path))
    print(f"[OK] {code}: wrote {out_path}")

    # Update metadata (write alongside augmented file if not overwriting)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    meta["med_top_thercls"] = top_thercls
    meta["med_thergrp_to_name"] = med_grp_map
    meta = json_safe(meta)

    if overwrite:
        out_meta = meta_path
    else:
        out_meta = out_dir / f"{code}_features_{baseline_year}_{outcome_year}_with_meds.meta.json"

    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[OK] {code}: wrote {out_meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/Users/charles/DATA/misc_conditions")
    parser.add_argument("--features_dir", type=str, default="/Users/charles/DATA/misc_conditions_features")
    parser.add_argument("--out_dir", type=str, default="/Users/charles/DATA/misc_conditions_features_augmented")
    parser.add_argument("--baseline_year", type=int, default=2017)
    parser.add_argument("--outcome_year", type=int, default=2018)
    parser.add_argument("--codes", type=str, default="")
    parser.add_argument("--redbook_dir", type=str, default="/Users/Charles/DATA/ckd/redbook")
    parser.add_argument("--k_med_thercls", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--driver_memory", type=str, default="80g")
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    spark = start_spark(driver_memory=args.driver_memory, num_threads=args.threads)

    base_dir = Path(args.base_dir)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)

    # Load redbook once
    df_redbook = None
    rb_path = Path(args.redbook_dir)
    if rb_path.exists():
        df_redbook = spark.read.parquet(str(rb_path)).cache()
        _ = df_redbook.count()
        print(f"Loaded redbook: {rb_path}")
    else:
        print(f"[WARN] redbook not found at {rb_path}; med features will be empty.")

    if args.codes.strip():
        codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
    else:
        codes = list_condition_codes(base_dir)

    print(f"Augmenting {len(codes)} cohorts: {codes}")

    for code in codes:
        augment_one_condition(
            spark=spark,
            base_dir=base_dir,
            features_dir=features_dir,
            out_dir=out_dir,
            code=code,
            baseline_year=args.baseline_year,
            outcome_year=args.outcome_year,
            df_redbook=df_redbook,
            k_med_thercls=args.k_med_thercls,
            overwrite=args.overwrite,
        )

    spark.stop()


if __name__ == "__main__":
    main()
