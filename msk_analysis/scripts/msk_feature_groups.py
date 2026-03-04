"""
MSK feature groups for distance precomputation.
Defines granular medical, context, intensity/utilization, and cost features.
Ensures no 2018 leakage.
"""
from __future__ import annotations

from typing import List, Tuple


def _exclude_2018(col: str) -> bool:
    """True if column should be excluded (2018 leakage)."""
    return "2018" in col


def get_granular_medical_columns(
    df,
    bin_flag_cols: List[str],
) -> List[str]:
    """
    Granular medical codes: dx/med indicator columns (mostly binary like has_*, is_*).
    These are the fine-grained medical indicators that can cause tie-heavy Euclidean distances.
    """
    # BIN_FLAG_COLUMNS are the dx/med and condition indicators
    return [c for c in bin_flag_cols if c in df.columns]


def get_context_columns(df, cat_cols: List[str]) -> List[str]:
    """
    Context: demographics, region, plan, economic status, etc.
    One-hot encodable categoricals (AGEGRP, SEX, REGION, EESTATU, etc.).
    """
    return [c for c in cat_cols if c in df.columns and not _exclude_2018(c)]


def get_utilization_columns(df) -> List[str]:
    """
    Intensity/utilization: claims counts, counts of unique dx/med codes, etc.
    Continuous or count columns. Excludes 2018.
    """
    utilization = [
        c for c in df.columns
        if ("claims" in c.lower() or "n_unique" in c.lower() or "n_dx" in c.lower()
            or "n_med" in c.lower() or "util_" in c.lower())
        and not _exclude_2018(c)
    ]
    return [c for c in utilization if c in df.columns]


def get_cost_columns_2017(df) -> List[str]:
    """
    Cost features: 2017 cost summaries. DO NOT use 2018.
    """
    cost = [
        col for col in df.columns
        if ("cost" in col.lower() or "quarterly" in col.lower() or "increasing" in col.lower()
            or "decreasing" in col.lower() or "skewness" in col.lower() or "kurtosis" in col.lower()
            or "cv" in col.lower() or "range" in col.lower())
        and not _exclude_2018(col)
    ]
    return [c for c in cost if c in df.columns]


def get_za_coarse_phenotype_columns(
    df,
    bin_flag_cols: List[str],
    cat_cols: List[str],
    cost_cols: List[str],
    utilization_cols: List[str],
) -> List[str]:
    """
    Z_A: Coarse clinical phenotype for Stage A (k-center dispersion).
    = comorbidity flags/indices + grouped dx/med indicators.
    Bin flag columns with "has_" prefix:
    ['has_Anemia', 'has_Atherosclerotic_Heart_Disease', 'has_Chronic_Pain', 
    'has_Depression_Anxiety', 'has_Gastroesophageal_Reflux_Disease', 'has_General_Health_Check', 
    'has_Hyperlipidemia', 'has_Hypertension', 'has_Long_term_Drug_Therapy', 'has_Obesity', 
    'has_Sleep_Apnea', 'has_Type_2_Diabetes', 'has_Vitamin_D_Deficiency', 
    
    'has_Arthropathies', 'has_Dorsopathies', 'has_Soft_tissue', 'has_Osteopathies', 'has_Other_MSK']

    Explicitly EXCLUDES raw cost and raw intensity totals.
    """
    # Coarse phenotype = has_* (comorbidity, condition flags) - NOT cost-derived binary
    coarse = [
        c for c in bin_flag_cols
        if c in df.columns
        and c.startswith("has_")  # comorbidity/condition flags
        and not _exclude_2018(c)
    ]

    # Exclude any bin flags that are cost-derived (is_increasing, etc.) - they're in cost_cols conceptually
    cost_related_keywords = ["increasing", "decreasing", "cost", "quarterly", "deriv", "skewness", "kurtosis", "cv", "range"]
    coarse = [
        c for c in coarse
        if not any(kw in c.lower() for kw in cost_related_keywords)
    ]
    # Also exclude columns that appear in cost/util
    exclude = set(cost_cols) | set(utilization_cols)
    return [c for c in coarse if c not in exclude]


def get_zb_intensity_context_columns(
    df,
    cat_cols: List[str],
    cost_cols: List[str],
    utilization_cols: List[str],
    bin_flag_cols: List[str],
) -> List[str]:
    """
    Z_B: Nuisance variables for Stage B (matching).
    = utilization/intensity + context. EXCLUDES granular medical codes (has_*).
    Context columns = "categorical/object/string" cols:  
    ['direct_msk_cost_pattern_2017', 'direct_msk_cost_stability_2017', 'msk_procedure_cost_pattern_2017', 
    'msk_procedure_cost_stability_2017', 'comorbidity_only_cost_pattern_2017', 
    'comorbidity_only_cost_stability_2017', 'AGEGRP', 'SEX', 'REGION', 'EESTATU'] 

    Reduces shortcut learning by matching on demographics, utilization, cost intensity
    rather than specific diagnoses.
    """
    # Context (demographics, region, plan, economic)
    
    context = get_context_columns(df, cat_cols)
    # Utilization + cost (intensity proxies)
    util = [c for c in utilization_cols if c in df.columns]
    cost = [c for c in cost_cols if c in df.columns]
    # Exclude granular medical (has_*, condition flags)
    granular = set(bin_flag_cols) & set(df.columns)
    all_cands = set(context) | set(util) | set(cost)
    return [c for c in all_cands if c not in granular]


def validate_no_2018_leakage(feature_cols: List[str]) -> Tuple[bool, List[str]]:
    """Return (ok, offending_cols)."""
    bad = [c for c in feature_cols if _exclude_2018(c)]
    return len(bad) == 0, bad
