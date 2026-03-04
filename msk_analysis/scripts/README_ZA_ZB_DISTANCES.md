# Z_A / Z_B Distance Folders for MSK Two-Stage Sampling

## Overview

Stage A (k-center dispersion) and Stage B (minority-majority matching) can use different feature sets and distance metrics via precomputed artifacts. This avoids tie-heavy Euclidean in sparse binary medical spaces and tail-heavy dispersion in cost-intensity spaces.

## New Distance Folders

| Folder | Purpose | Features | Metric |
|--------|---------|----------|--------|
| `precomputed_distances_msk_za_coarse_phenotype/` | Stage A (dispersion) | Coarse clinical phenotype: comorbidity flags (`has_*`), no raw cost/intensity | Euclidean |
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

Run from `msk_analysis`:

```bash
cd msk_analysis
python scripts/precompute_msk_distances_za_zb.py --parquet msk_2017_18_full.parquet --seed 123
```

This creates both folders. Use `--za_only` or `--zb_only` to build just one.

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
python run_za_zb_stage_ablation.py --stageA za_coarse_phenotype --stageB zb_intensity_context
```

Stage A uses DNN matrix from the `--stageA` folder; Stage B uses PN H5 from the `--stageB` folder.
