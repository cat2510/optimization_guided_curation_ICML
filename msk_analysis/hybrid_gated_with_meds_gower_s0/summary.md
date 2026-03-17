# Percentile-Gated Hybrid Evaluation — Summary

## Setup
- **Retraining**: Yes
- **Data**: msk_2017_18_full.parquet; Split: train_test_split_enrol, TRAIN_TEST_SEED=123
- **Sampling**: tfidf_svd_cosine_qcost (DNN + Gower P-N) from distances_dir; ours 1:1 + rnd 1:1
- **Hardness**: Full original train, Gower on all prediction features, k=10
- **Calibration**: Isotonic on validation; applied to both models before ensemble/hybrid

## Gate Selection
- **Percentile reference**: Validation hardness distribution
- **Candidates q**: [50, 40, 30, 20, 15, 10, 5, 2]
- **Chosen q**: 50 (validation ROC-AUC)
- **Gate threshold t_q**: 0.305329

## Results

### Validation
| Method | AUC | PR-AUC |
|--------|-----|--------|
| curated | 0.7398 | 0.0616 |
| random | 0.7702 | 0.0563 |
| avg_ensemble | 0.7885 | 0.0745 |
| hybrid_q50 | 0.7756 | 0.0705 |

### Test
| Method | AUC | PR-AUC |
|--------|-----|--------|
| curated | 0.6947 | 0.0554 |
| random | 0.7509 | 0.0508 |
| avg_ensemble | 0.7560 | 0.0680 |
| hybrid_q50 | 0.7491 | 0.0619 |

## Answers
- **Hybrid vs random baseline on test**: No (hybrid AUC 0.7491 vs random 0.7509)
- **Hard-subset advantage preserved**: See subgroup_metrics.csv and fig_subgroup_auc.png

## Caveats
- Percentile threshold depends on validation hardness distribution
- Possible calibration mismatch between models
- If hybrid does not improve overall or is unstable, report plainly
