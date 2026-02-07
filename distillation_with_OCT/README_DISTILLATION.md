# OCT→Student Distillation for Imbalanced Binary Classification

This module implements knowledge distillation from an Optimal Classification Tree (OCT) to an XGBoost student model, with a focus on improving minority class recall in imbalanced binary classification tasks.

## Overview

The distillation process uses **soft guidance** (not hard constraints) to transfer knowledge from a trained OCT teacher model to an XGBoost student. The key idea is to blend the teacher's probability predictions with the true labels during training, allowing the student to learn from both the ground truth and the teacher's interpretable decision rules.

## How It Works

### 1. Teacher Probability Interface

The `OCTTeacher` class provides probability predictions from an OCT model. It supports:

- **Loading from saved IAI model**: If you have a saved `OptimalTreeClassifier` (pickle or IAI format)
- **Reconstructing from CSV splits**: If you only have the tree splits in CSV format (node_id, feature, threshold)
- **Leaf empirical probabilities**: For each leaf node, computes the empirical class probability from training data with Laplace smoothing:
  ```
  P(class=1 | leaf) = (count_positive + laplace) / (count_total + 2*laplace)
  ```

The teacher probabilities are computed using the same preprocessing pipeline as the student to ensure consistency.

### 2. Distillation Training

The student (XGBoost) is trained with **soft label blending**:

```
p_soft = (1 - α) * y_true + α * p_teacher
```

Where:
- `α` (alpha) is the distillation strength (0 = no distillation, 1 = only teacher)
- `y_true` is the true binary label (0 or 1)
- `p_teacher` is the teacher's probability for class 1

The student is trained to predict `p_soft` using XGBoost's `binary:logistic` objective, which naturally handles probabilistic targets in [0, 1].

**Key advantages:**
- Soft guidance preserves the teacher's interpretable rules while allowing the student to learn additional patterns
- The blending parameter `α` controls the trade-off between following the teacher vs. learning from data
- Works seamlessly with XGBoost's native training loop

### 3. Minority-Focused Evaluation

The evaluation metrics emphasize minority class performance:

- **PR-AUC (Average Precision)**: Better for imbalanced data than ROC-AUC
- **Minority recall**: Recall for the positive class (minority)
- **G-mean**: Geometric mean of recall and specificity
- **MCC (Matthews Correlation Coefficient)**: Balanced metric that accounts for all confusion matrix entries

Early stopping can be configured to optimize for:
- `pr_auc`: PR-AUC (recommended for imbalanced data)
- `recall`: Minority class recall
- `auc`: ROC-AUC
- `logloss`: Standard log loss

### 4. Calibration Check

Since distillation can affect probability calibration, the module includes a calibration check using:
- **ECE (Expected Calibration Error)**: Measures how well-calibrated the probabilities are
- **Reliability curve**: Visualizes calibration (can be plotted from saved data)

## Usage

### Basic Command

```bash
python train_student.py --distill --alpha 0.3 --teacher oct --early_stop_metric pr_auc
```

### Key Arguments

**Data:**
- `--data`: Path to dataset (parquet or CSV)
- `--target_col`: Target column name (default: `highcost_gt_200000`)
- `--exclude_cols`: Columns to exclude from features

**Teacher (OCT):**
- `--teacher_splits`: Path to OCT splits CSV (required if no saved model)
- `--teacher_model`: Path to saved OCT model (optional, if available)

**Distillation:**
- `--distill`: Enable distillation (required)
- `--alpha`: Distillation strength, 0.0-1.0 (default: 0.3)
- `--temperature`: Temperature scaling (not used in current implementation)

**Training:**
- `--n_estimators`: Maximum number of trees (default: 500)
- `--max_depth`: Tree depth (default: 6)
- `--learning_rate`: Learning rate (default: 0.1)
- `--scale_pos_weight`: XGBoost class weight (auto if None)
- `--early_stop_metric`: Metric for early stopping (default: `pr_auc`)
- `--early_stop_rounds`: Early stopping patience (default: 50)

**Output:**
- `--output_dir`: Output directory (default: `student_distillation_results`)

### Example: Full Training Run

```bash
python train_student.py \
    --data 0917_2017_18_with_2017_cost.parquet \
    --target_col highcost_gt_200000 \
    --distill \
    --alpha 0.3 \
    --teacher_splits two_stage_kcenter_results_global/oct_tree_ckd_best_curated_prauc_splits.csv \
    --early_stop_metric pr_auc \
    --n_estimators 500 \
    --max_depth 6 \
    --output_dir student_results
```

## Output Files

The training script saves:

1. **Model**: `output_dir/models/student_model.pkl`
   - Contains the trained XGBoost model, preprocessor, and feature metadata

2. **Metrics**: `output_dir/metrics/metrics.json`
   - Comprehensive metrics on validation and test sets
   - Includes calibration information

3. **Summary**: `output_dir/metrics/summary.csv`
   - CSV summary of key metrics (AUC, PR-AUC, recall, etc.)

4. **Feature Importance**: `output_dir/metrics/feature_importance.csv`
   - XGBoost feature importance scores

## Hyperparameter Tuning Recommendations

### Distillation Strength (α)

- **α = 0.0**: No distillation, standard XGBoost training
- **α = 0.2-0.4**: Light distillation, recommended starting point
- **α = 0.5-0.7**: Moderate distillation, good if teacher is very strong
- **α = 0.8-1.0**: Heavy distillation, use only if teacher significantly outperforms baseline

**Tuning strategy**: Start with α=0.3, then try 0.2 and 0.4. Monitor validation PR-AUC and minority recall.

### Early Stopping Metric

- **`pr_auc`** (recommended): Best for imbalanced data, focuses on precision-recall trade-off
- **`recall`**: Use if minority recall is the primary concern
- **`auc`**: Use if you care about overall discrimination
- **`logloss`**: Standard metric, but may not optimize for minority class

### XGBoost Hyperparameters

- **`scale_pos_weight`**: Auto-computed as `n_negative / n_positive`. Can be tuned if needed (try 1.5x or 2x the auto value)
- **`max_depth`**: Start with 6, try 4-8 range
- **`learning_rate`**: Start with 0.1, can go lower (0.05) for more trees
- **`n_estimators`**: Set high (500-1000) and rely on early stopping

## Technical Details

### Preprocessing Consistency

The teacher and student use the **same preprocessor** to ensure:
- Identical feature encoding (one-hot, scaling, imputation)
- Consistent column order
- Same handling of missing values

The code includes assertions to verify column alignment between teacher and student.

### Leaf Probability Estimation

When reconstructing from CSV splits, the teacher uses leaf empirical probabilities:

1. Apply tree splits to training data to get leaf assignments
2. For each leaf, count positive and negative samples
3. Apply Laplace smoothing: `p = (pos + 1) / (total + 2)`
4. Use these probabilities for test-time predictions

This approach is robust even with small leaves and provides well-calibrated probabilities.

### Distillation Objective

The student is trained with XGBoost's `binary:logistic` objective on soft labels. This is equivalent to minimizing:

```
L = -[y_soft * log(p) + (1 - y_soft) * log(1 - p)]
```

Where `y_soft` is the blended label and `p` is the student's predicted probability. This naturally handles the probabilistic nature of the soft labels.

## Limitations and Future Work

1. **Temperature scaling**: Currently not implemented. Could be added to soften teacher probabilities further.
2. **Multi-teacher distillation**: Only supports single OCT teacher. Could be extended to ensemble of teachers.
3. **Feature importance alignment**: Could add analysis to compare teacher vs. student feature importance.
4. **Calibration post-processing**: Could add Platt scaling or isotonic regression to improve calibration.

## References

- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Optimal Classification Trees: Bertsimas & Dunn, "Optimal Classification Trees" (2017)
- Imbalanced Learning: He & Garcia, "Learning from Imbalanced Data" (2009)
