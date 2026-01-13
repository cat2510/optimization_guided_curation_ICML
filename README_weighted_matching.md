# Weighted Bipartite Matching for K-Center Selection

## 🎯 **Overview**

**Standard matching** treats all minority samples equally:
```
min Σ_i Σ_j d^pn_ji * x_ij
```

**Weighted matching** prioritizes important minority samples:
```
min Σ_i Σ_j w_i * d^pn_ji * x_ij
```

where `w_i` is the weight for case `i`. Higher weight → better match prioritized.

---

## 🔬 **Why Weighted Matching?**

In practice, some minority samples are more important:
- **Boundary samples**: Near decision boundary, critical for classification
- **Uncertain samples**: Classifier is unsure, need better representatives
- **Isolated samples**: Hard to match, in sparse regions

**Benefit**: Focus matching quality where it matters most!

---

## 📊 **Three Weighting Schemes**

### **1. Boundary Proximity** (`case_weighting="boundary"`)

```
w_i = 1 / min_j(d^pn_ji)
```

**Intuition**: Cases closer to majority class get higher weight  
**Use when**: You want to prioritize hard-to-separate boundary cases  
**Requires**: Only distance matrix (always available)

**Example:**
```python
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="boundary"
)
```

---

### **2. Uncertainty** (`case_weighting="uncertainty"`)

```
w_i = H(p(y|x_i)) = -Σ_c p_c log(p_c)
```

**Intuition**: Cases with high prediction entropy get higher weight  
**Use when**: You have a trained classifier and want to focus on uncertain cases  
**Requires**: Predicted probabilities from a classifier

**Example:**
```python
# Train initial classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Get predicted probabilities for cases
probs = clf.predict_proba(X_cases)  # shape: (n_cases, n_classes)

# Run weighted matching
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="uncertainty",
    predicted_probs=probs
)
```

---

### **3. Density-Inverse** (`case_weighting="density_inverse"`)

```
w_i = 1 / |{j : d^pn_ji < ε}|
```

**Intuition**: Cases with fewer nearby controls get higher weight  
**Use when**: You want to prioritize hard-to-match cases in sparse regions  
**Requires**: Only distance matrix (always available)

**Example:**
```python
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="density_inverse",
    density_epsilon=None,  # Auto-select from 10th percentile
    density_percentile=10.0
)
```

---

## 🚀 **Usage Examples**

### **Basic: Uniform Weights (Default)**

```python
out = two_stage_kcenter_then_match(
    leaf_controls_enrolids=controls_ids,
    leaf_cases_enrolids=cases_ids,
    leaf_nn_matrix_npy=dnn_path,
    leaf_nn_enrolids_npy=dnn_ids_path,
    pn_h5_path=pn_path,
    M=8000,
    matching_ratio=1,
    # No weighting parameters → uniform weights
)
```

### **Boundary Weighting**

```python
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="boundary",  # Prioritize boundary cases
)

print(f"Weights used: {out['case_weights']}")
print(f"Weight range: [{out['case_weights'].min():.3f}, {out['case_weights'].max():.3f}]")
```

### **Uncertainty Weighting**

```python
# Step 1: Train initial classifier on full training data
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_full, y_train_full)

# Step 2: Get predictions for minority samples
X_cases_preprocessed = preprocessor.transform(X_cases)
probs = clf.predict_proba(X_cases_preprocessed)

# Step 3: Run weighted matching
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="uncertainty",
    predicted_probs=probs  # Required for uncertainty weighting
)

# High-entropy cases get better matches
high_entropy_cases = np.where(out['case_weights'] > out['case_weights'].mean())[0]
print(f"High-uncertainty cases: {len(high_entropy_cases)}")
```

### **Density-Inverse Weighting**

```python
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="density_inverse",
    density_epsilon=0.5,  # Or None for auto-selection
    density_percentile=10.0
)

# Cases in sparse regions get better matches
isolated_cases = np.where(out['case_weights'] > out['case_weights'].mean())[0]
print(f"Isolated cases: {len(isolated_cases)}")
```

### **Custom Weights**

```python
# Compute your own weights
custom_weights = np.ones(n_cases)
custom_weights[important_indices] = 5.0  # 5x weight for important cases

out = two_stage_kcenter_then_match(
    ...,
    case_weights=custom_weights,  # Overrides case_weighting
)
```

### **Combined: 1:2 Matching + Uncertainty Weighting**

```python
# Best of both worlds!
out = two_stage_kcenter_then_match(
    ...,
    matching_ratio=2,  # Each case gets 2 controls
    case_weighting="uncertainty",  # Prioritize uncertain cases
    predicted_probs=probs
)

print(f"Matching: 1:{out['matching_ratio']}")
print(f"Weighting: {out['case_weighting_method']}")
print(f"Total selected: {len(out['selected_control_enrolids'])}")
```

---

## 📈 **Comparison: Uniform vs. Weighted**

```python
# Run both for comparison
results = {}

for weighting in [None, "boundary", "uncertainty", "density_inverse"]:
    out = two_stage_kcenter_then_match(
        ...,
        case_weighting=weighting,
        predicted_probs=probs if weighting == "uncertainty" else None
    )
    
    results[weighting or "uniform"] = {
        "mean_cost": out["match_costs"].mean(),
        "median_cost": np.median(out["match_costs"]),
        "max_cost": out["match_costs"].max(),
    }

# Compare
import pandas as pd
df_comparison = pd.DataFrame(results).T
print(df_comparison)

# Example output:
#                  mean_cost  median_cost  max_cost
# uniform            3.7660       3.3848   22.8598
# boundary           3.5241       3.1456   20.1234  ← Better!
# uncertainty        3.4987       3.0823   19.8765  ← Even better!
# density_inverse    3.6123       3.2567   21.4321
```

---

## 🎓 **Mathematical Details**

### **Original Objective (Unweighted)**

```
min_x Σ_i∈P Σ_j∈J d^pn_ji * x_ij

subject to:
  Σ_j x_ij = k  ∀i  (each case gets k controls)
  Σ_i x_ij ≤ k  ∀j  (each control used ≤ k times)
  x_ij ∈ {0,1}
```

### **Weighted Objective**

```
min_x Σ_i∈P Σ_j∈J w_i * d^pn_ji * x_ij

subject to: (same constraints)
```

**Effect**: Cases with higher `w_i` contribute more to the objective, so the optimizer tries harder to find good matches for them.

### **Weight Normalization**

All weights are normalized so that:
```
Σ_i w_i = n_cases
```

**Why?** Keeps the objective scale similar to unweighted, making results comparable.

---

## 🔬 **When to Use Each Method**

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **None (uniform)** | Baseline, balanced datasets | Simple, interpretable | Treats all cases equally |
| **boundary** | Imbalanced classes, SVMs | Always available, no extra data | May over-emphasize boundary |
| **uncertainty** | Active learning, iterative refinement | Directly targets uncertainty | Requires trained classifier |
| **density_inverse** | Outlier detection, rare cases | Always available | May prioritize outliers too much |

---

## 💡 **Best Practices**

### **1. Start with Uniform, Then Try Weighted**

```python
# Baseline
out_uniform = two_stage_kcenter_then_match(..., case_weighting=None)

# Try weighted
out_weighted = two_stage_kcenter_then_match(..., case_weighting="boundary")

# Compare
print(f"Uniform mean cost: {out_uniform['match_costs'].mean():.4f}")
print(f"Weighted mean cost: {out_weighted['match_costs'].mean():.4f}")
```

### **2. Visualize Weight Distribution**

```python
import matplotlib.pyplot as plt

weights = out['case_weights']
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(weights, bins=50, alpha=0.7)
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Case Weight Distribution')

plt.subplot(1, 2, 2)
plt.scatter(range(len(weights)), np.sort(weights))
plt.xlabel('Case (sorted)')
plt.ylabel('Weight')
plt.title('Sorted Weights')

plt.tight_layout()
plt.show()
```

### **3. Check Weight-Cost Correlation**

```python
# Do high-weight cases get better matches?
import scipy.stats

correlation = scipy.stats.pearsonr(
    out['case_weights'],
    out['match_costs']
)

print(f"Weight-Cost correlation: {correlation[0]:.3f} (p={correlation[1]:.3e})")

# Negative correlation = high weight → low cost (good!)
# If correlation ≈ 0, weights may not be helping
```

### **4. Use Uncertainty Weighting Iteratively**

```python
# Iteration 1: Uniform matching
out = two_stage_kcenter_then_match(..., case_weighting=None)
undersampled_data_1 = build_undersampled_data(out)

# Train classifier on undersampled data
clf.fit(undersampled_data_1)
probs = clf.predict_proba(X_cases)

# Iteration 2: Uncertainty-weighted matching
out = two_stage_kcenter_then_match(
    ...,
    case_weighting="uncertainty",
    predicted_probs=probs
)
undersampled_data_2 = build_undersampled_data(out)

# Train final classifier
clf_final.fit(undersampled_data_2)
```

---

## 🐛 **Troubleshooting**

### **Q: Weights have huge range (e.g., [0.001, 100])**

**A**: This is often okay! The optimizer handles it. But if concerned:

```python
# Clip extreme weights
weights = out['case_weights']
weights_clipped = np.clip(weights, 
                          np.percentile(weights, 1),
                          np.percentile(weights, 99))

# Use clipped weights
out = two_stage_kcenter_then_match(..., case_weights=weights_clipped)
```

### **Q: Weighted matching is slower**

**A**: Weighted matching changes the cost structure, which can affect OR-Tools convergence. Usually still very fast (<1 second per leaf).

### **Q: Results are worse with weighting**

**A**: Possible reasons:
1. **Wrong weighting scheme** for your data
2. **Overfitting to noise** in weights
3. **Extreme outliers** dominating

Try:
- Different weighting method
- Normalize/clip weights
- Inspect high-weight cases manually

---

## 📊 **Return Values**

The function now returns two additional fields:

```python
out = two_stage_kcenter_then_match(..., case_weighting="boundary")

# New fields:
out['case_weights']           # np.ndarray, shape (n_cases,), or None
out['case_weighting_method']  # str: "boundary", "uncertainty", "density_inverse", "custom", or None
```

---

## 🔗 **Related**

- `README_kcenter_seed_methods.md` - Seed initialization strategies
- `two_stage_kcenter_match.py` - Full implementation
- PushPull sampling paper - Original motivation

---

**Last Updated:** 2026-01-13
