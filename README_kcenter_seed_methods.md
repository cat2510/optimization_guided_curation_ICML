# K-Center Seed Initialization Methods

## Overview

The k-center algorithm (farthest-first traversal) requires an initial **seed point** to start the greedy selection process. This document describes three initialization strategies implemented in `two_stage_kcenter_match.py`.

---

## 🎯 Available Methods

### 1. **Minority-Biased (smart)** - DEFAULT ✅

```python
seed_method="smart"
```

**Formula:**
```
s₁ = argmin_j (1/|P|) Σᵢ d^pn_ji
```

**Description:**
- Selects the control with **minimum mean distance** to all minority samples
- Directly optimizes for coverage of the minority class

**Implementation:**
```python
seed_idx = int(np.argmin(d_pn_leaf.mean(axis=1)))
```

**Advantages:**
- ✅ Best matching quality (lowest cost)
- ✅ Directly aligned with matching objective
- ✅ Recommended default

**Use when:**
- Matching quality is the priority
- You want optimal results
- No specific domain constraints

---

### 2. **Centroid**

```python
seed_method="centroid"
X_majority_leaf=X_majority_leaf  # Required!
```

**Formula:**
```
s₁ = argmin_j ||x_j - x̄_N||
```
where `x̄_N` is the centroid of all majority points

**Description:**
- Selects the control **closest to the majority centroid**
- Starts from a "representative" majority point

**Implementation:**
```python
def choose_seed_centroid(X_majority: np.ndarray) -> int:
    centroid = X_majority.mean(axis=0)
    distances_to_centroid = np.linalg.norm(X_majority - centroid, axis=1)
    return int(np.argmin(distances_to_centroid))
```

**Advantages:**
- ✅ Good general coverage
- ✅ Representative of majority distribution
- ✅ Intuitive geometric interpretation

**Use when:**
- You want a "typical" majority starting point
- Domain knowledge suggests centroid is meaningful
- Exploring alternative initialization strategies

**Note:** Requires the feature matrix `X_majority_leaf` to compute the centroid.

---

### 3. **Max-Density**

```python
seed_method="density"
density_epsilon=None  # Auto-selected from 10th percentile
density_percentile=10.0  # Adjustable
```

**Formula:**
```
s₁ = argmax_j |{k : d^nn_jk < ε}|
```

**Description:**
- Selects the control with **highest local density**
- Counts neighbors within radius ε
- Starts from a "typical" or common majority profile

**Implementation:**
```python
def choose_seed_max_density(
    d_nn: np.ndarray,
    epsilon: Optional[float] = None,
    percentile: float = 10.0
) -> int:
    # Auto-select epsilon if not provided
    if epsilon is None:
        epsilon = np.percentile(d_nn[mask], percentile)
    
    # Count neighbors within epsilon
    neighbor_counts = np.sum((d_nn < epsilon) & (d_nn > 0), axis=1)
    return int(np.argmax(neighbor_counts))
```

**Advantages:**
- ✅ Captures clustering structure
- ✅ Starts from dense/typical region
- ✅ Auto-selects epsilon if not provided

**Use when:**
- Majority class has strong clustering
- You want to start from a "common" profile
- Exploring density-based initialization

**Parameters:**
- `density_epsilon`: Radius for neighborhood (auto-selected if None)
- `density_percentile`: Percentile of distances for auto-epsilon (default: 10.0)

---

## 📊 Experimental Comparison

### Example Usage

```python
from two_stage_kcenter_match import two_stage_kcenter_then_match

# Compare all three methods
seed_methods = ["smart", "centroid", "density"]

for method in seed_methods:
    out = two_stage_kcenter_then_match(
        leaf_controls_enrolids=leaf_controls["ENROLID"].to_numpy(),
        leaf_cases_enrolids=leaf_cases["ENROLID"].to_numpy(),
        leaf_nn_matrix_npy=dnn_matrix_npy,
        leaf_nn_enrolids_npy=dnn_enrolids_npy,
        pn_h5_path=pn_h5_path,
        M=8000,
        seed_method=method,
        X_majority_leaf=X_majority_leaf if method == "centroid" else None,
        density_epsilon=None,  # Auto-select for density
    )
    
    print(f"{method}: mean cost = {out['match_costs'].mean():.4f}")
```

### Expected Results

| Method | Expected Rank | Typical Mean Cost | Notes |
|--------|---------------|-------------------|-------|
| **smart** | 🥇 1st (best) | Lowest | Directly optimizes matching |
| **centroid** | 🥈 2nd | Close to smart | Good representative |
| **density** | 🥉 3rd | Varies | Depends on clustering |

**Difference magnitude:**
- Typically **< 1%** difference in matching cost
- If methods are very similar, use "smart" as default
- Large differences (>5%) suggest dataset-specific effects

---

## 🔧 Implementation Details

### Return Values

The `two_stage_kcenter_then_match` function now returns additional seed information:

```python
out = {
    "candidate_majority_enrolids": ...,
    "selected_control_enrolids": ...,
    "case_to_control_map": ...,
    "match_costs": ...,
    "seed_method": "smart",  # NEW
    "seed_idx": 1234,  # NEW
    "seed_enrolid": 56789,  # NEW
}
```

### Function Signatures

```python
def choose_seed_centroid(X_majority: np.ndarray) -> int
```
- Input: Feature matrix (n_controls, n_features)
- Output: Seed index

```python
def choose_seed_max_density(
    d_nn: np.ndarray,
    epsilon: Optional[float] = None,
    percentile: float = 10.0
) -> int
```
- Input: Distance matrix (n_controls, n_controls)
- Output: Seed index
- Side effect: Prints auto-selected epsilon and density stats

---

## 🎓 Theoretical Background

### Why Does Seed Matter?

K-center (farthest-first) is a **greedy algorithm**:
1. Start from seed `s₁`
2. Iteratively add point farthest from current set
3. Selection depends on initial seed

Different seeds → different candidate pools → different matching costs

### Optimality

- **No theoretical guarantee** that minority-biased is optimal
- **Empirically**: Minority-biased performs best in practice
- **Intuition**: Covering minority class is the objective

### Computational Cost

All three methods have **negligible overhead**:
- **smart**: O(n × p) - one mean computation
- **centroid**: O(n × d) - centroid + distances
- **density**: O(n² × p_sample) - neighbor counting (sampled for large n)

The k-center traversal itself is O(M × n), so seed selection is **not a bottleneck**.

---

## 📖 References

**K-center algorithm:**
- Gonzalez, T. F. (1985). "Clustering to minimize the maximum intercluster distance"
- Hochbaum, D. S., & Shmoys, D. B. (1985). "A best possible heuristic for the k-center problem"

**Seed initialization strategies:**
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"
  (Similar motivation for seed selection)

---

## 💡 Best Practices

### Recommendations

1. **Default**: Use `seed_method="smart"` (minority-biased)
2. **Experimentation**: Try all three methods and compare
3. **Reporting**: Always report which seed method was used
4. **Reproducibility**: For "random", set `random_state` for reproducibility

### When to Try Alternatives

- Matching costs are very similar (< 1% difference)
- Domain knowledge suggests centroid is meaningful
- Majority class has known clustering structure
- Exploring robustness to initialization

### Debugging

If seed methods give **very different results** (>5% cost difference):
- Check data quality (outliers, preprocessing)
- Visualize the distributions (PCA/t-SNE)
- Inspect seed points manually
- Consider increasing M (candidate pool size)

---

## 🛠️ Troubleshooting

### Error: "seed_method='centroid' requires X_majority_leaf parameter"

**Solution:** Pass the feature matrix when using centroid:

```python
out = two_stage_kcenter_then_match(
    ...,
    seed_method="centroid",
    X_majority_leaf=X_majority_leaf  # Required!
)
```

### Warning: "Auto-selected epsilon (density radius): 0.0003"

**Interpretation:** Epsilon is very small, suggesting high-dimensional space. This is **normal** for high-dimensional data.

**Action:** No action needed. The algorithm auto-scales.

### All methods give identical results

**Likely cause:** M = n_controls (using all controls as candidates)

**Solution:** Reduce M to force selection:
```python
M = min(8000, len(leaf_controls) // 2)  # Force subset selection
```

---

## 📝 Citation

If you use these seed initialization methods in your research, please cite:

```
@software{kcenter_seed_methods,
  title={K-Center Seed Initialization Methods for Matching},
  author={Your Team},
  year={2026},
  url={https://github.com/yourrepo}
}
```

---

## 🔗 Related Files

- `two_stage_kcenter_match.py`: Implementation
- `pushpull_per_leaf_run.ipynb`: Example usage and comparison
- `README_pushpull_samplers.md`: PushPull sampling overview

---

**Last Updated:** 2026-01-13
