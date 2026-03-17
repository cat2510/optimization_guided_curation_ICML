# Percentile-Gated Hybrid — Experiment Setup

## Reference
Based on scripts/analyze_hard_negative_boundary_eval.py.

- Target: top_2_pct_cost_2018
- Split: train_test_split_enrol, TRAIN_TEST_SEED=123
- Hardness: full original train, Gower, k=10
- Gate: t_q from validation hardness; q chosen by validation ROC-AUC
