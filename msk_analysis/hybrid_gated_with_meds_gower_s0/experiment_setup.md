# Percentile-Gated Hybrid — Experiment Setup

## Data & Split
- Parquet: msk_2017_18_full.parquet
- Split: train_test_split_enrol, TRAIN_TEST_SEED=123
- Distances: tfidf_svd_cosine_qcost (from distances_dir)
- Sampling: ours 1:1 (two-stage) + rnd 1:1

## Hardness & Gate
- Hardness: full original train, Gower on prediction features, k=10
- Gate: t_q from validation hardness; q chosen by validation ROC-AUC
