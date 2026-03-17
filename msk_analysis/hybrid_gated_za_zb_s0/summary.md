# Percentile-Gated Hybrid Evaluation — Summary

## Setup
- **Retraining**: No (reused aligned predictions)
- **Split**: Same as analyze_hard_negative_boundary_eval (train_test_split_enrol, TRAIN_TEST_SEED=123)
- **Hardness**: Full original train, Gower on all prediction features, k=10
- **Calibration**: Isotonic on validation; applied to both models before ensemble/hybrid

## Gate Selection
- **Percentile reference**: Validation hardness distribution
- **Candidates q**: [50, 40, 30, 20, 15, 10, 5, 2]
- **Chosen q**: 2 (validation ROC-AUC)
- **Gate threshold t_q**: 0.358419

## Results

### Validation
| Method | AUC | PR-AUC |
|--------|-----|--------|
| curated | 0.7395 | 0.0548 |
| random | 0.7805 | 0.0611 |
| avg_ensemble | 0.7893 | 0.0834 |
| hybrid_q2 | 0.7755 | 0.0608 |

### Test
| Method | AUC | PR-AUC |
|--------|-----|--------|
| curated | 0.7401 | 0.0593 |
| random | 0.7788 | 0.0626 |
| avg_ensemble | 0.7923 | 0.0872 |
| hybrid_q2 | 0.7690 | 0.0653 |

## Answers
- **Hybrid vs random baseline on test**: No (hybrid AUC 0.7690 vs random 0.7788)
- **Hard-subset advantage preserved**: See subgroup_metrics.csv and fig_subgroup_auc.png

## Caveats
- Percentile threshold depends on validation hardness distribution
- Possible calibration mismatch between models
- If hybrid does not improve overall or is unstable, report plainly
