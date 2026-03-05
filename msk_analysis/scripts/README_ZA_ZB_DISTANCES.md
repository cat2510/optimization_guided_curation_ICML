# Z_A / Z_B Distance Folders for MSK Two-Stage Sampling

## Overview

Stage A (k-center dispersion) and Stage B (minority-majority matching) can use different feature sets and distance metrics via precomputed artifacts. This avoids tie-heavy Euclidean in sparse binary medical spaces and tail-heavy dispersion in cost-intensity spaces.

## Z_A Presets (Stage A)

| Preset | Features | Metric (default) |
|--------|----------|------------------|
| `ZA_v0_flags_only` | 18 `has_*` flags (current baseline) | Euclidean |
| `ZA_v1_flags_plus_counts` | Flags + comorbidity_count, msk_flag_count, non_msk_comorbidity_count | Gower |
| `ZA_v2_flags_plus_intensity_norm` | ZA_v1 + log1p(intensity), age continuous | Gower |
| `za_coarse_phenotype` | Legacy: same as ZA_v0 | Euclidean |

## Z_B (Stage B)

| Folder | Purpose | Features | Metric |
|--------|---------|----------|--------|
| `precomputed_distances_msk_zb_intensity_context/` | Stage B (matching) | Utilization + context: claims, demographics, cost intensity; excludes granular dx/med | Gower |

### Feature group utilities (scripts/msk_feature_groups.py)
get_granular_medical_columns() – dx/med indicators (BIN_FLAG_COLUMNS)
get_context_columns() – demographics, region, plan, etc.
get_utilization_columns() – claims counts, counts of unique dx/med codes
get_cost_columns_2017() – 2017 cost summaries (no 2018)
get_za_coarse_phenotype_columns() – Z_A: has_* comorbidity flags, no raw cost/intensity
get_zb_intensity_context_columns() – Z_B: utilization + context, no granular medical
validate_no_2018_leakage() – checks features for 2018 leakage


## How to Build

**Legacy (za_coarse_phenotype + zb_intensity_context):**
```bash
cd msk_analysis
python scripts/precompute_msk_distances_za_zb.py --parquet msk_2017_18_full.parquet --seed 123
```

**Z_A variants (recommended for tie-degeneracy fix):**
```bash
python scripts/precompute_msk_ZA_variants.py --za_preset ZA_v1_flags_plus_counts --metric gower --compute_dnn 1 --compute_pn 0 --run_diagnostics 1
python scripts/precompute_msk_ZA_variants.py --za_preset ZA_v2_flags_plus_intensity_norm --metric gower --compute_dnn 1 --run_stageA_overlap_test 1
```

## Artifact Schema (compatible with `two_stage_kcenter_then_match`)

Each folder contains:

- `distances_majority_minority.h5` – P-N distances (controls × cases)
- `global_dnn_seed_123/leaf_global_dnn_matrix.npy` – D-N-N (control-control)
- `global_dnn_seed_123/leaf_global_dnn_enrolids.npy` – ENROLID order

## Tie-Degeneracy Diagnostics

For each PN H5, the precompute script prints:

- Number of unique distance values
- Fraction of top-5 most frequent distances

High tie fraction or very few unique values indicate possible tie degeneracy.

## How to Run Two-Stage with Z_A / Z_B

```bash
cd msk_analysis
python scripts/run_za_zb_stage_ablation.py --stageA ZA_v1_flags_plus_counts --stageB zb_intensity_context
# or legacy:
python scripts/run_za_zb_stage_ablation.py --stageA za_coarse_phenotype --stageB zb_intensity_context
```

Stage A uses DNN matrix from the `--stageA` folder; Stage B uses PN H5 from the `--stageB` folder. Results (including selected_majority_cost2018_median, timestamps) are appended to `za_zb_stage_ablation_results/summary.csv`.
