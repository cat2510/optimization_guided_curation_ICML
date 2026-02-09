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

### 2. Rule-Based Distillation Training

Instead of blending probabilities (which doesn't help when OCT is less accurate), the student learns from OCT's **interpretable decision rules**:

1. **Rule-based features**: Extract OCT's decision structure as features:
   - **Leaf assignment**: Which leaf each sample falls into (categorical feature)
   - **Rule indicators**: Binary indicators for each split rule (e.g., `oct_rule_1_left`: feature ≤ threshold)
   - **Rule confidence**: Confidence score based on leaf purity and size

2. **Sample weighting**: Weight samples based on rule confidence:
   - Samples where OCT's rules are clear/confident get higher weights
   - Encourages XGBoost to pay attention to interpretable patterns  
   - **Default is boost-only** (`weight_min=1.0`): low-confidence samples keep weight 1.0; only confident (high-purity) leaves get weight > 1. When leaf purity is moderate (e.g. P(class=1) ≈ 0.6), confidence is low and the old downweighting scheme (min 0.5) was shrinking the training signal for most samples; boost-only avoids that.
   - Strategies: `confidence`, `agreement` (OCT-label agreement), `minority_boost`

**Key advantages:**
- XGBoost learns from OCT's interpretable structure, not inaccurate probabilities
- Rule features allow XGBoost to discover additional patterns beyond OCT's rules
- Sample weighting guides learning toward confident interpretable patterns
- XGBoost can still outperform OCT by learning non-linear combinations of rules

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
python train_student.py --distill --use_rule_features --use_sample_weights --weight_strategy confidence --early_stop_metric pr_auc
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
- `--distill`: Enable rule-based distillation
- `--compare`: Run **both** baseline (no distillation) and distilled, then save a side-by-side comparison table
- `--use_rule_features`: Add OCT rule features to XGBoost (default: True)
- `--use_sample_weights`: Use rule-based sample weighting (default: True)
- `--weight_strategy`: Sample weighting strategy - `confidence`, `agreement`, or `minority_boost` (default: `confidence`)
- `--weight_min`, `--weight_max`: Sample weight range (defaults 1.0 and 2.0). Use `weight_min=1.0` so only confident samples are upweighted; avoid values below 1.0 or most data gets downweighted when purity is moderate.
- `--confidence_exponent`: Use confidence**exponent when mapping to weights (default 1.0). Use **&gt; 1** (e.g. 2.0) so only high-confidence samples get a strong boost; low-confidence samples stay near weight 1.0.
- `--rule_feature_scale`: Multiply rule features by this before concatenating (default 1.0). Use **&gt; 1** (e.g. 2.0 or 3.0) so OCT rule columns have larger magnitude and XGBoost uses them more in splits.

**Training:**
- `--n_estimators`: Maximum number of trees (default: 500)
- `--max_depth`: Tree depth (default: 6)
- `--learning_rate`: Learning rate (default: 0.1)
- `--scale_pos_weight`: XGBoost class weight (auto if None)
- `--early_stop_metric`: Metric for early stopping (default: `pr_auc`)
- `--early_stop_rounds`: Early stopping patience (default: 50)

**Output:**
- `--output_dir`: Output directory (default: `student_distillation_results`)

### Comparing baseline vs distilled

To train both XGBoost baseline (no distillation) and distilled on the same data and compare them:

```bash
python train_student.py --compare \
    --data /path/to/data.parquet \
    --target_col your_target \
    --teacher_splits /path/to/oct_tree_*_splits.csv \
    --output_dir comparison_run
```

This writes:
- `metrics/comparison_baseline_vs_distilled.csv` — metric, baseline_val, baseline_test, distilled_val, distilled_test, delta_val, delta_test
- `metrics/metrics_baseline.json`, `metrics/metrics_distilled.json`
- `models/student_baseline.pkl`, `models/student_distilled.pkl`

and prints a comparison table to the console.

### Example: Full Training Run

Run from the **repo root** so that data and teacher paths resolve, or run from `distillation_with_OCT` and use paths relative to the parent (one level up):

```bash
# From repo root (my_projects/):
python distillation_with_OCT/train_student.py \
    --data 0917_2017_18_with_2017_cost.parquet \
    --target_col highcost_gt_200000 \
    --distill \
    --use_rule_features \
    --use_sample_weights \
    --weight_strategy confidence \
    --teacher_splits two_stage_kcenter_results_global/oct_tree_ckd_best_curated_prauc_splits.csv \
    --early_stop_metric pr_auc \
    --n_estimators 500 \
    --max_depth 6 \
    --output_dir distillation_with_OCT/student_results

# From distillation_with_OCT/ (data and folders one level up):
cd distillation_with_OCT
python train_student.py \
    --data ../0917_2017_18_with_2017_cost.parquet \
    --target_col highcost_gt_200000 \
    --distill \
    --use_rule_features \
    --use_sample_weights \
    --weight_strategy confidence \
    --teacher_splits ../two_stage_kcenter_results_global/oct_tree_ckd_best_curated_prauc_splits.csv \
    --early_stop_metric pr_auc \
    --n_estimators 500 \
    --max_depth 6 \
    --output_dir student_results

# From distillation_with_OCT/ you can also use repo-root-relative paths (no ../):
# the script resolves them under the repo root if not found in cwd.
python train_student.py --data 0917_2017_18_with_2017_cost.parquet \
    --teacher_splits two_stage_kcenter_results_global/oct_tree_ckd_best_curated_prauc_splits.csv \
    --target_col highcost_gt_200000 --distill --output_dir student_results
```

**Paths and imports:** The script adds the repo root (parent of `distillation_with_OCT`) to `sys.path`, so `model_nonIAI_utils` and `public.model_IAI` resolve correctly. Relative `--data` and `--teacher_splits` are tried under the repo root if they don’t exist in the current working directory.

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

### Rule-Based Distillation Options

- **`--use_rule_features`**: Add OCT rule features (recommended: True)
  - Gives XGBoost access to OCT's decision structure
  - Allows learning non-linear combinations of rules
  
- **`--use_sample_weights`**: Use rule-based weighting (recommended: True)
  - Focuses learning on samples where OCT is confident
  - Helps preserve interpretable patterns

- **`--weight_strategy`**: Choose weighting strategy
  - **`confidence`** (default): Weight by rule confidence (leaf purity + size)
  - **`agreement`**: Higher weight when OCT prediction agrees with label
  - **`minority_boost`**: Boost minority class samples in confident leaves (good for imbalanced data)

**Tuning strategy**: Start with `confidence` strategy. If minority recall is low, try `minority_boost`. If OCT is very accurate, try `agreement`.

**Making OCT rules affect the student more strongly:**
- **`--rule_feature_scale 2` or `3`**: Scale up the rule-derived features (leaf id, rule indicators, confidence) so they have larger magnitude; XGBoost is more likely to split on them.
- **`--confidence_exponent 2`**: Use a steeper weight curve so only high-confidence (high-purity) leaves get a real boost; moderate-confidence samples stay near weight 1.0.
- **`--weight_max 3`**: Allow confident samples to get up to 3× weight instead of 2×.

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

### How Leaf Probabilities, Rule Features, and Sample Weights Connect

**1. Leaf probabilities (teacher.leaf_probs)**  
Per-leaf empirical P(class=1) from training data (Laplace smoothed). Stored on the teacher; not passed to XGBoost as a raw column.

**2. Rule features (what gets concatenated at 681–697)**  
`extract_oct_rule_features()` builds a DataFrame with:

| Column(s) | Meaning | Uses leaf probs? |
|-----------|--------|-------------------|
| `oct_leaf_id` | Which leaf the sample falls into | No |
| `oct_rule_{id}_left`, `oct_rule_{id}_right` | Binary: did this split go left (≤ thresh) or right (> thresh)? | No |
| **`oct_rule_confidence`** | One number per sample: how “confident” the teacher is for that leaf | **Yes** |

**3. Where leaf probabilities are used**  
Only inside **`oct_rule_confidence`** (in `extract_oct_rule_features`, ~lines 430–454):

- For each sample we know its **leaf_id**.
- We get **p_positive = teacher.leaf_probs[leaf_id]** (leaf probability).
- **Purity** = |p_positive − 0.5| × 2 (0 at 0.5, 1 at 0 or 1).
- **Size confidence** = function of how many training samples fell in that leaf (larger leaf → more stable).
- **oct_rule_confidence** = 0.7 × purity + 0.3 × size_confidence.

So: **leaf probabilities → purity → oct_rule_confidence**. That confidence is both:

- **One of the rule features** concatenated to the student input (so XGBoost sees “how confident the OCT is” as a feature), and  
- **The same scalar** used to define confidence-based sample weights.

**4. Confidence-based sample weights**  
`compute_rule_based_sample_weights(..., weight_strategy='confidence')` calls `extract_oct_rule_features(..., include_rule_confidence=True)` and reads **the same `oct_rule_confidence`** column. It then turns that into a weight per sample:

- **weights = min_weight + (max_weight − min_weight) × confidences**  
  Defaults: **min_weight=1.0, max_weight=2.0** (boost-only). When leaf purity is moderate (e.g. p_positive ≈ 0.6), confidence is low (purity = |p−0.5|×2 is small), so with min_weight=0.5 most samples were downweighted (weight below 1) and the training signal was weakened. With min_weight=1.0, only confident samples get weight above 1.

So: **higher rule confidence → higher sample weight** in the XGBoost loss. The student is trained on **true labels**, with **rule features** (including `oct_rule_confidence`) as extra inputs, and **sample_weights** so that high-confidence (high-purity, reasonably large) leaves count more in the gradient. Leaf probabilities are used only to compute that confidence (and thus both the extra feature and the weights), not as a direct input or soft label.

### Rule-Based Distillation Mechanism

The student is trained with XGBoost's standard `binary:logistic` objective on true labels, but with:

1. **Additional rule features**: OCT's decision structure (leaf assignments, rule indicators) are concatenated with original features
2. **Sample weighting**: Samples in confident OCT leaves get higher weights, encouraging XGBoost to learn from interpretable patterns

This approach:
- Preserves OCT's interpretable rules as features XGBoost can use
- Allows XGBoost to discover additional patterns beyond OCT's simple splits
- Uses sample weighting to guide learning toward confident interpretable patterns
- Doesn't rely on OCT's probabilities (which may be less accurate than XGBoost)

## Does the pipeline change for TabPFN (or other tabular foundation models)?

**Short answer:** The *distillation idea* does not change; the *student training step* does.

| Component | XGBoost | TabPFN / other foundation model |
|-----------|--------|----------------------------------|
| **OCT teacher** | Same | Same (OCTTeacher, splits CSV, preprocessor) |
| **Rule features** | Same | Same — `extract_oct_rule_features()` returns a DataFrame you concatenate with your tabular features |
| **Sample weights** | Passed to `DMatrix(..., weight=...)` | Depends on the API: TabPFN’s zero-shot API doesn’t take weights; if you fine-tune, you’d need a weighted loss where supported |
| **Training** | `train_student_distilled()` uses XGBoost’s fit API | Replace with: build `X = [original_features, rule_features]`, then call TabPFN’s fit/predict (and optionally implement weighted loss) |

So you keep:

1. **Same teacher and rule extraction**  
   `OCTTeacher`, `extract_oct_rule_features()`, and `compute_rule_based_sample_weights()` are model-agnostic. Use them to get rule features and (if you want) weights.

2. **Same feature construction**  
   `X_train_enriched = [X_train_processed, rule_features_train]` (and same for val/test). TabPFN (or any tabular model) just gets more columns.

What you change:

- **Student training**: Instead of `train_student_distilled(...)` (XGBoost), call your model’s API on the enriched features (and labels). For TabPFN, respect its limits (e.g. max 100 features, 1000 train samples) by subsetting or reducing rule features if needed.
- **Sample weights**: Use only if the model supports them (e.g. weighted loss in fine-tuning); otherwise train without weights.

So the pipeline does **not** change significantly in terms of teacher or features; only the “train student” part is swapped for TabPFN (or another tabular foundation model).

---

## Testing the current pipeline

### Option 1: Quick run with your data (few rounds)

Use your real data but shorten training so it finishes quickly:

```bash
cd distillation_with_OCT   # or add path to script
python train_student.py \
  --data /path/to/your.parquet \
  --target_col your_target \
  --distill \
  --teacher_splits /path/to/oct_tree_*_splits.csv \
  --n_estimators 20 \
  --early_stop_rounds 5 \
  --output_dir test_run
```

Check that `test_run/metrics/metrics.json` and `test_run/models/student_model.pkl` exist and that metrics look plausible.

### Option 2: Synthetic data (no real data needed)

From the repo root or `distillation_with_OCT`:

```bash
python test_distillation_smoke.py
```

This script (see below) creates a small synthetic dataset, a dummy OCT splits CSV, fits the preprocessor and OCTTeacher, extracts rule features, and runs XGBoost student training for a few rounds. It verifies that the pipeline runs end-to-end and that rule features and metrics are produced.

### Option 3: Unit-style checks

In a Python shell or notebook:

```python
from oct_distillation import OCTTeacher, extract_oct_rule_features, compute_rule_based_sample_weights
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Minimal data
X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
y = np.random.randint(0, 2, 100)
preprocessor = ColumnTransformer([("num", StandardScaler(), ["a", "b", "c"])])
preprocessor.fit(X)
feature_names = ["a", "b", "c"]

# Dummy splits (one node)
splits_csv = "path/to/oct_tree_*_splits.csv"  # or create a tiny CSV with node_id, feature, threshold
teacher = OCTTeacher(splits_csv=splits_csv, X_train=X, y_train=y, preprocessor=preprocessor, feature_names=feature_names)
rule_feat, meta = extract_oct_rule_features(teacher, X)
weights = compute_rule_based_sample_weights(teacher, X, y, weight_strategy="confidence")
assert rule_feat.shape[0] == len(X)
assert len(weights) == len(X)
```

If these run without errors, the teacher and rule-feature part of the pipeline work; you can then plug in TabPFN (or another model) for the student step as above.

---

## Limitations and Future Work

1. **Temperature scaling**: Currently not implemented. Could be added to soften teacher probabilities further.
2. **Multi-teacher distillation**: Only supports single OCT teacher. Could be extended to ensemble of teachers.
3. **Feature importance alignment**: Could add analysis to compare teacher vs. student feature importance.
4. **Calibration post-processing**: Could add Platt scaling or isotonic regression to improve calibration.

## References

- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Optimal Classification Trees: Bertsimas & Dunn, "Optimal Classification Trees" (2017)
- Imbalanced Learning: He & Garcia, "Learning from Imbalanced Data" (2009)
