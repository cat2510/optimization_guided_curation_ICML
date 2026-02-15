# two_stage_kcenter_match.py
# Includes bin-quota constrained matching for representativeness control.
import numpy as np
import h5py
from typing import Dict, Tuple, List, Optional

# ----------------------------
# Loading utilities
# ----------------------------

def load_nn_memmap(nn_matrix_npy: str, nn_enrolids_npy: str):
    """
    Loads majority-majority distances as a memmap and enrolids (row/col order).
    """
    d_nn = np.load(nn_matrix_npy, mmap_mode="r")  # (n, n) float32
    maj_ids = np.load(nn_enrolids_npy)            # (n,) int
    return d_nn, maj_ids

def load_pn_hdf5(pn_h5_path: str):
    """
    Loads majority-minority distances from HDF5.
    Must contain datasets:
        distances: (n_majority, n_minority)
        majority_enrolids: (n_majority,)
        minority_enrolids: (n_minority,)
    Returns: (h5_file_handle, distances_dataset, maj_ids, min_ids)
    Note: caller must close h5_file_handle.
    """
    f = h5py.File(pn_h5_path, "r")
    d_pn = f["distances"]
    maj_ids = f["majority_enrolids"][:]
    min_ids = f["minority_enrolids"][:]
    return f, d_pn, maj_ids, min_ids

def build_id_to_index(ids: np.ndarray) -> Dict[int, int]:
    return {int(x): i for i, x in enumerate(ids)}

def align_indices(requested_ids: np.ndarray, id_to_index: Dict[int, int]) -> np.ndarray:
    """
    Convert requested enrolids to matrix indices using dict mapping.
    """
    idx = np.empty(len(requested_ids), dtype=np.int64)
    for t, eid in enumerate(requested_ids):
        idx[t] = id_to_index[int(eid)]
    return idx

# ----------------------------
# Stage A: Farthest-first k-center
# ----------------------------
def farthest_first_adaptive_pool(
    d_nn,                 # (nN, nN) memmap/ndarray
    d_pn_leaf,            # (nN, nP) ndarray float32
    seed_idx: int,
    M_cap: int,           # hard cap, e.g. min(8000, nN)
    tau: float | None = None,     # radius threshold for Option A
    plateau_eps: float = 0.01,    # 1% relative improvement
    plateau_window: int = 100,    # check every 100 additions
    M_min: int | None = None      # default baseline if not provided
):
    nN = d_nn.shape[0]
    nP = d_pn_leaf.shape[1]
    if M_min is None:
        M_min = nP  #max(2*nP, int(np.ceil(0.1*nN)))

    # state for k-center
    selected = []
    min_dist = np.array(d_nn[seed_idx, :], dtype=np.float32)
    min_dist[seed_idx] = -np.inf

    # state for preview metric
    # min distance from each positive to current pool
    min_to_pool = d_pn_leaf[seed_idx, :].copy()  # shape (nP,)

    selected.append(seed_idx)
    C_hist = [float(min_to_pool.mean())]

    while len(selected) < M_cap:
        # stopping checks (only after reaching M_min)
        if len(selected) >= M_min:
            # Option A: radius
            if tau is not None:
                R = float(np.max(min_dist[min_dist != -np.inf]))
                if R <= tau:
                    break

            # Option B: plateau of C_hat
            if len(selected) % plateau_window == 0:
                C_now = float(min_to_pool.mean())
                C_prev = C_hist[-1]
                rel_impr = (C_prev - C_now) / max(C_prev, 1e-12)
                C_hist.append(C_now)
                if rel_impr <= plateau_eps:
                    break

        # farthest-first step
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)

        # update k-center distances
        row = np.array(d_nn[nxt, :], dtype=np.float32)
        min_dist = np.minimum(min_dist, row)
        min_dist[nxt] = -np.inf

        # update preview distances to positives
        min_to_pool = np.minimum(min_to_pool, d_pn_leaf[nxt, :])

    return selected, float(min_to_pool.mean())

def farthest_first_kcenter_indices(d_nn, M: int, seed_idx: int) -> List[int]:
    """
    Farthest-first k-center on majority space using precomputed d_nn.
    d_nn: (n, n) memmap/ndarray float32 symmetric
    M: number of centers to pick
    seed_idx: starting index

    Returns list of selected indices (length M).
    """
    n = d_nn.shape[0]
    if d_nn.shape[1] != n:
        raise ValueError("d_nn must be square")

    selected = np.empty(M, dtype=np.int64)
    selected[0] = seed_idx

    # min_dist[x] = distance from x to nearest selected center so far
    min_dist = np.array(d_nn[seed_idx, :], dtype=np.float32)
    # mark selected nodes as -inf so they won't be selected again
    min_dist[seed_idx] = -np.inf

    for t in range(1, M):
        nxt = int(np.argmax(min_dist))
        selected[t] = nxt
        # update min_dist with new center
        row = np.array(d_nn[nxt, :], dtype=np.float32)
        min_dist = np.minimum(min_dist, row)
        min_dist[nxt] = -np.inf

    return selected.tolist()

def choose_seed_closest_to_positives_from_pn(
    d_pn_rows_for_leaf: np.ndarray
) -> int:
    """
    Seed = majority index (within leaf ordering) with smallest mean distance to positives.
    d_pn_rows_for_leaf: (n_majority_leaf, n_pos_leaf) float32/float64
    """
    mean_d = d_pn_rows_for_leaf.mean(axis=1)
    return int(np.argmin(mean_d))

def choose_seed_random(
    n_controls: int,
    random_state: Optional[int] = None
) -> int:
    """
    Select a random control as the initial seed for k-center.
    n_controls: number of controls available
    random_state: random seed for reproducibility
    
    Returns: random index in [0, n_controls)
    """
    rng = np.random.RandomState(random_state)
    return int(rng.randint(0, n_controls))

def choose_seed_centroid(
    X_majority: np.ndarray
) -> int:
    """
    Seed = majority point closest to the centroid of all majority points.
    
    This initialization selects a "central" representative control.
    
    Parameters
    ----------
    X_majority : np.ndarray, shape (n_controls, n_features)
        Feature matrix of majority samples
    
    Returns
    -------
    int
        Index of control closest to the majority centroid
    """
    # Compute centroid
    centroid = X_majority.mean(axis=0)
    
    # Find point closest to centroid
    distances_to_centroid = np.linalg.norm(X_majority - centroid, axis=1)
    return int(np.argmin(distances_to_centroid))


def choose_seed_max_density(
    d_nn: np.ndarray,
    epsilon: Optional[float] = None,
    percentile: float = 10.0
) -> int:
    """
    Seed = majority point with highest local density (most neighbors within radius epsilon).
    
    This initialization selects a control from a dense region, which may be more
    representative of the majority class distribution.
    
    Parameters
    ----------
    d_nn : np.ndarray, shape (n_controls, n_controls)
        Pairwise distance matrix between majority samples
    epsilon : float, optional
        Radius for counting neighbors. If None, uses percentile-based threshold.
    percentile : float, default=10.0
        If epsilon is None, compute epsilon as this percentile of all pairwise distances.
        Lower percentile = stricter neighborhood (fewer neighbors required).
    
    Returns
    -------
    int
        Index of control with maximum local density
    """
    n = d_nn.shape[0]
    
    # Auto-select epsilon if not provided
    if epsilon is None:
        # Sample distances to avoid loading entire matrix (for large memmap)
        if hasattr(d_nn, 'shape') and d_nn.shape[0] > 1000:
            # For large matrices, sample to estimate percentile
            sample_size = min(1000, n)
            sample_idx = np.random.choice(n, size=sample_size, replace=False)
            sample_dists = []
            for i in sample_idx:
                sample_dists.extend(d_nn[i, sample_idx].ravel())
            epsilon = float(np.percentile(sample_dists, percentile))
        else:
            # For small matrices, use all distances (excluding diagonal)
            mask = ~np.eye(n, dtype=bool)
            epsilon = float(np.percentile(d_nn[mask], percentile))
        
        print(f"    Auto-selected epsilon (density radius): {epsilon:.4f} ({percentile}th percentile)")
    
    # Count neighbors within epsilon for each point
    # neighbor_counts[i] = number of points within epsilon of point i (excluding self)
    neighbor_counts = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        # Count neighbors (excluding self: distance = 0)
        neighbors = (d_nn[i, :] < epsilon) & (d_nn[i, :] > 0)
        neighbor_counts[i] = np.sum(neighbors)
    
    # Return point with maximum density
    max_density_idx = int(np.argmax(neighbor_counts))
    max_density = neighbor_counts[max_density_idx]
    
    print(f"    Max density: {max_density} neighbors within epsilon={epsilon:.4f}")
    print(f"    Mean density: {neighbor_counts.mean():.1f} neighbors")
    
    return max_density_idx

# ----------------------------
# Case Weighting Schemes
# ----------------------------

def compute_case_weights_boundary(
    d_pn_leaf: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Boundary proximity weighting: w_i = 1 / min_j(d^pn_ji)
    
    Cases closer to the majority class boundary get higher weight.
    Intuition: Hard-to-separate cases near the boundary are more important.
    
    Parameters
    ----------
    d_pn_leaf : np.ndarray, shape (n_controls, n_cases)
        Distance matrix from controls to cases
    normalize : bool, default=True
        Whether to normalize weights to sum to n_cases (for interpretability)
    
    Returns
    -------
    weights : np.ndarray, shape (n_cases,)
        Weight for each case. Higher weight = more important.
    """
    # Distance to nearest majority point
    min_dist_to_majority = d_pn_leaf.min(axis=0)  # (n_cases,)
    
    # Inverse distance (closer = higher weight)
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (min_dist_to_majority + 1e-8)
    
    if normalize:
        # Normalize so sum(weights) = n_cases
        # This keeps the total objective scale similar to unweighted
        weights = weights * (len(weights) / weights.sum())
    
    return weights.astype(np.float32)


def compute_case_weights_density_inverse(
    d_pn_leaf: np.ndarray,
    epsilon: Optional[float] = None,
    percentile: float = 10.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Density-inverse weighting: w_i = 1 / |{j : d^pn_ji < ε}|
    
    Cases with fewer nearby controls get higher weight.
    Intuition: Hard-to-match cases in sparse regions need more attention.
    
    Parameters
    ----------
    d_pn_leaf : np.ndarray, shape (n_controls, n_cases)
        Distance matrix from controls to cases
    epsilon : float, optional
        Radius for counting neighbors. If None, auto-computed from percentile.
    percentile : float, default=10.0
        Percentile of distances to use as epsilon if not provided
    normalize : bool, default=True
        Whether to normalize weights to sum to n_cases
    
    Returns
    -------
    weights : np.ndarray, shape (n_cases,)
        Weight for each case. Higher weight = fewer nearby controls.
    """
    n_controls, n_cases = d_pn_leaf.shape
    
    # Auto-select epsilon if not provided
    if epsilon is None:
        epsilon = float(np.percentile(d_pn_leaf, percentile))
        print(f"    Auto-selected epsilon (density radius): {epsilon:.4f} ({percentile}th percentile)")
    
    # Count controls within epsilon for each case
    neighbor_counts = np.sum(d_pn_leaf < epsilon, axis=0)  # (n_cases,)
    
    # Inverse density (fewer neighbors = higher weight)
    # Add small constant to avoid division by zero
    weights = 1.0 / (neighbor_counts + 1.0)
    
    if normalize:
        weights = weights * (len(weights) / weights.sum())
    
    print(f"    Density-inverse: min={neighbor_counts.max()} neighbors, "
          f"max={neighbor_counts.min()} neighbors, mean={neighbor_counts.mean():.1f}")
    
    return weights.astype(np.float32)


# ----------------------------
# Bin-Quota Helpers (imported from quota_helpers.py)
# ----------------------------
try:
    from public.quota_helpers import (
        compute_proximity_scores,
        deterministic_binning,
        compute_bin_quotas,
        select_population_subset_indices,
        compute_cutpoints_from_population,
        assign_bins_from_cutpoints,
        cap_quotas_at_pool_counts,
        collapse_to_active_bins,
    )
except ImportError:
    from quota_helpers import (
        compute_proximity_scores,
        deterministic_binning,
        compute_bin_quotas,
        select_population_subset_indices,
        compute_cutpoints_from_population,
        assign_bins_from_cutpoints,
        cap_quotas_at_pool_counts,
        collapse_to_active_bins,
    )


# ----------------------------
# Stage B: 1-to-1 min-cost assignment
# ----------------------------

def solve_min_cost_assignment_adaptive_topk(
    cost_matrix: np.ndarray,
    topk_start: int = 50,
    topk_max: Optional[int] = None,
    topk_growth: int = 2,
    cost_scale: int = 10000,
    matching_ratio: int = 1,  # 1:k matching support
    case_weights: Optional[np.ndarray] = None  # NEW: Weighted bipartite matching
) -> Tuple[np.ndarray | List[np.ndarray], np.ndarray | List[np.ndarray]]:
    """
    Solve 1-to-k assignment: each case matched to k DISTINCT controls.
    Supports weighted bipartite matching to prioritize important cases.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Shape (n_cases, n_controls), float cost for each case-control pair
    topk_start : int, default=50
        Initial number of top-K cheapest controls per case to consider
    topk_max : int, optional
        Maximum K to try before giving up. If None, uses n_controls
    topk_growth : int, default=2
        Multiplicative factor to grow K if infeasible
    cost_scale : int, default=10000
        Scale factor to convert float costs to integers for OR-Tools
    matching_ratio : int, default=1
        Number of distinct controls to match per case (k in 1:k matching)
        - matching_ratio=1: 1:1 matching (returns arrays)
        - matching_ratio>1: 1:k matching (returns lists of arrays)
    case_weights : np.ndarray, optional, shape (n_cases,)
        Weight for each case. Higher weight = prioritize matching quality for this case.
        If None, uniform weights (all cases treated equally).
        Modified objective: min Σ_i Σ_j w_i * d_ij * x_ij

    Returns
    -------
    match_ctrl_for_case : np.ndarray or List[np.ndarray]
        If matching_ratio=1: array of length n_cases with control indices
        If matching_ratio>1: list of length n_cases, where element i is an array
                            of matching_ratio control indices for case i
    matched_costs : np.ndarray or List[np.ndarray]
        If matching_ratio=1: array of length n_cases with costs
        If matching_ratio>1: list of length n_cases, where element i is an array
                            of matching_ratio costs for case i

    Notes
    -----
    Min-cost flow network structure:
    - 1:1 matching: Each control can be used once globally
    - 1:k matching: Controls can be reused across cases (up to k times)
      Each case->control edge still has capacity 1 to ensure distinctness per case
    
    Weighted matching:
    - If case_weights provided, costs are multiplied element-wise by weights
    - Higher weight → higher cost penalty → better match prioritized
    - Weights are applied BEFORE scaling to integers
    """
    try:
        from ortools.graph.python import min_cost_flow
    except Exception:
        # older ortools fallback
        from ortools.graph import pywrapgraph as min_cost_flow  # type: ignore

    n_cases, n_controls = cost_matrix.shape
    
    # Validate matching_ratio
    if matching_ratio < 1:
        raise ValueError(f"matching_ratio must be >= 1, got {matching_ratio}")
    if matching_ratio > n_controls:
        raise ValueError(f"matching_ratio={matching_ratio} > n_controls={n_controls}. Not enough controls.")
    
    # Apply case weights if provided
    if case_weights is not None:
        if len(case_weights) != n_cases:
            raise ValueError(f"case_weights length ({len(case_weights)}) != n_cases ({n_cases})")
        # Multiply each row by its weight: cost[i,j] *= w[i]
        # Broadcasting: (n_cases, 1) * (n_cases, n_controls)
        cost_matrix = cost_matrix * case_weights[:, np.newaxis]
    
    if topk_max is None:
        topk_max = n_controls

    # If topk_start is None -> use ALL edges for exact optimal matching
    if topk_start is None:
        K = n_controls
        topk_max = n_controls
    else:
        K = min(topk_start, n_controls)

    while True:
        # Build arcs
        # Node ids:
        #   source = 0
        #   cases  = 1..n_cases
        #   ctrls  = 1+n_cases .. n_cases+n_controls
        #   sink   = n_cases+n_controls+1
        source = 0
        case_offset = 1
        ctrl_offset = 1 + n_cases
        sink = 1 + n_cases + n_controls

        # OR-Tools min cost flow object
        mcf = min_cost_flow.SimpleMinCostFlow()

        # Supplies/demands (modified for 1:k matching)
        # Source supplies k units per case
        mcf.set_node_supply(source, n_cases * matching_ratio)
        # Sink demands k units per case
        mcf.set_node_supply(sink, -n_cases * matching_ratio)
        # other nodes default to 0

        # Source -> cases (each case receives k units)
        for i in range(n_cases):
            mcf.add_arc_with_capacity_and_unit_cost(
                source, case_offset + i,
                matching_ratio,  # capacity = k
                0
            )

        # Controls -> sink (each control can be matched up to k times)
        for j in range(n_controls):
            mcf.add_arc_with_capacity_and_unit_cost(
                ctrl_offset + j, sink,
                matching_ratio,  # capacity = k (reusable across cases)
                0
            )

        # Case -> controls (top-K edges per case)
        # CRITICAL: Each edge has capacity = 1 to ensure distinctness per case
        for i in range(n_cases):
            row = cost_matrix[i]
            if K < n_controls:
                cand = np.argpartition(row, K-1)[:K]
            else:
                cand = np.arange(n_controls)
            # Add edges with integer costs
            for j in cand:
                c = int(row[j] * cost_scale)
                if c < 0:
                    c = 0
                mcf.add_arc_with_capacity_and_unit_cost(
                    case_offset + i,
                    ctrl_offset + int(j),
                    1,  # capacity = 1 (ensures each control used at most once per case)
                    c
                )

        status = mcf.solve()
        if status == min_cost_flow.SimpleMinCostFlow.OPTIMAL:
            # Recover assignment: find arcs case->control with flow=1
            if matching_ratio == 1:
                # 1:1 matching - return arrays for backward compatibility
                match_ctrl = np.full(n_cases, -1, dtype=np.int64)
                match_cost = np.full(n_cases, np.nan, dtype=np.float32)

                for a in range(mcf.num_arcs()):
                    tail = mcf.tail(a)
                    head = mcf.head(a)
                    flow = mcf.flow(a)

                    # case->control arc has tail in cases range and head in controls range
                    if flow == 1 and (case_offset <= tail < case_offset + n_cases) and (ctrl_offset <= head < ctrl_offset + n_controls):
                        i = tail - case_offset
                        j = head - ctrl_offset
                        match_ctrl[i] = j
                        match_cost[i] = float(cost_matrix[i, j])

                if np.any(match_ctrl < 0):
                    raise RuntimeError("Optimal flow found but incomplete matching.")

                return match_ctrl, match_cost
            
            else:
                # 1:k matching - return lists of arrays
                match_ctrl = [[] for _ in range(n_cases)]
                match_cost = [[] for _ in range(n_cases)]

                for a in range(mcf.num_arcs()):
                    tail = mcf.tail(a)
                    head = mcf.head(a)
                    flow = mcf.flow(a)

                    # case->control arc has tail in cases range and head in controls range
                    if flow == 1 and (case_offset <= tail < case_offset + n_cases) and (ctrl_offset <= head < ctrl_offset + n_controls):
                        i = tail - case_offset
                        j = head - ctrl_offset
                        match_ctrl[i].append(j)
                        match_cost[i].append(float(cost_matrix[i, j]))

                # Verify each case has exactly k matches
                for i in range(n_cases):
                    if len(match_ctrl[i]) != matching_ratio:
                        raise RuntimeError(
                            f"Optimal flow found but case {i} has {len(match_ctrl[i])} matches, "
                            f"expected {matching_ratio}"
                        )

                # Convert lists to arrays
                match_ctrl_arrays = [np.array(m, dtype=np.int64) for m in match_ctrl]
                match_cost_arrays = [np.array(c, dtype=np.float32) for c in match_cost]

                return match_ctrl_arrays, match_cost_arrays

        # infeasible or not solved
        if K >= topk_max:
            if matching_ratio > 1:
                raise RuntimeError(
                    f"Min-cost assignment infeasible even with K={K} for 1:{matching_ratio} matching. "
                    f"Check that n_controls ({n_controls}) is sufficient and indices align."
                )
            else:
                raise RuntimeError(
                    f"Min-cost assignment infeasible even with K={K}. "
                    f"Check that n_controls >= n_cases and indices align."
                )
        K = min(topk_max, K * topk_growth)

# ----------------------------
# Stage B: Quota-constrained min-cost flow
# ----------------------------

def solve_min_cost_assignment_with_quotas(
    cost_matrix: np.ndarray,
    bin_assignments: np.ndarray,
    quotas: List[int],
    K_per_bin: int = 25,
    K_growth: int = 2,
    max_K_per_bin: Optional[int] = None,
    cost_scale: int = 10000,
    case_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    1:1 min-cost assignment with bin-quota constraints.

    Network topology:
        source -> case_i          (cap=1, cost=0)
        case_i -> ctrl_j          (cap=1, cost=d[i,j])  [per-bin sparsified]
        ctrl_j -> bin_node[bin(j)] (cap=1, cost=0)
        bin_node[t] -> sink       (cap=quotas[t], cost=0)

    Parameters
    ----------
    cost_matrix : (n_cases, n_pool) float
    bin_assignments : (n_pool,) int  -- bin index per pool control
    quotas : list[int]  -- required count per bin, sum = n_cases
    K_per_bin : int  -- initial top-K edges per case per bin
    K_growth : int  -- multiplicative factor on retry
    max_K_per_bin : int or None
    cost_scale : int  -- float-to-int scale for OR-Tools
    case_weights : (n_cases,) float or None

    Returns
    -------
    match_ctrl : (n_cases,) int64
    match_cost : (n_cases,) float32  (weighted if case_weights given)
    diagnostics : dict
    """
    try:
        from ortools.graph.python import min_cost_flow
    except Exception:
        from ortools.graph import pywrapgraph as min_cost_flow  # type: ignore

    n_cases, n_pool = cost_matrix.shape
    T = len(quotas)

    if sum(quotas) != n_cases:
        raise ValueError(f"sum(quotas)={sum(quotas)} != n_cases={n_cases}")

    # Apply case weights (multiply rows)
    if case_weights is not None:
        if len(case_weights) != n_cases:
            raise ValueError(
                f"case_weights length {len(case_weights)} != n_cases {n_cases}"
            )
        cost_matrix = cost_matrix * case_weights[:, np.newaxis]

    # Per-bin member lists (indices sorted for determinism)
    bin_members: List[List[int]] = [[] for _ in range(T)]
    for j in range(n_pool):
        bin_members[int(bin_assignments[j])].append(j)

    for t in range(T):
        if quotas[t] > len(bin_members[t]):
            raise ValueError(
                f"Quota for bin {t} ({quotas[t]}) > bin size "
                f"({len(bin_members[t])})"
            )

    max_bin_sz = max((len(bm) for bm in bin_members), default=n_pool)
    eff_max_K = max_K_per_bin if max_K_per_bin is not None else max_bin_sz
    K_bins = [min(K_per_bin, len(bm)) for bm in bin_members]

    retries = 0
    K_schedule = [list(K_bins)]

    while True:
        source = 0
        case_off = 1
        ctrl_off = 1 + n_cases
        bin_off = 1 + n_cases + n_pool
        sink = 1 + n_cases + n_pool + T

        mcf = min_cost_flow.SimpleMinCostFlow()
        mcf.set_node_supply(source, n_cases)
        mcf.set_node_supply(sink, -n_cases)

        # source -> cases
        for i in range(n_cases):
            mcf.add_arc_with_capacity_and_unit_cost(
                source, case_off + i, 1, 0
            )

        # cases -> controls  (per-bin top-K sparsification)
        for i in range(n_cases):
            row = cost_matrix[i]
            for t in range(T):
                if quotas[t] == 0:
                    continue
                members = bin_members[t]
                if not members:
                    continue
                K_t = K_bins[t]
                member_arr = np.array(members, dtype=np.int64)
                if K_t < len(members):
                    member_costs = row[member_arr]
                    top_idx = np.argpartition(member_costs, K_t - 1)[:K_t]
                    cands = member_arr[top_idx]
                else:
                    cands = member_arr
                for j_val in cands:
                    j = int(j_val)
                    c = int(float(row[j]) * cost_scale)
                    if c < 0:
                        c = 0
                    mcf.add_arc_with_capacity_and_unit_cost(
                        case_off + i, ctrl_off + j, 1, c
                    )

        # controls -> bin nodes
        for j in range(n_pool):
            mcf.add_arc_with_capacity_and_unit_cost(
                ctrl_off + j, bin_off + int(bin_assignments[j]), 1, 0
            )

        # bin nodes -> sink
        for t in range(T):
            if quotas[t] > 0:
                mcf.add_arc_with_capacity_and_unit_cost(
                    bin_off + t, sink, quotas[t], 0
                )

        status = mcf.solve()
        if status == min_cost_flow.SimpleMinCostFlow.OPTIMAL:
            match_ctrl = np.full(n_cases, -1, dtype=np.int64)
            match_cost = np.full(n_cases, np.nan, dtype=np.float32)

            for a in range(mcf.num_arcs()):
                tail = mcf.tail(a)
                head = mcf.head(a)
                if (mcf.flow(a) == 1
                        and case_off <= tail < case_off + n_cases
                        and ctrl_off <= head < ctrl_off + n_pool):
                    i = tail - case_off
                    j = head - ctrl_off
                    match_ctrl[i] = j
                    match_cost[i] = float(cost_matrix[i, j])

            if np.any(match_ctrl < 0):
                raise RuntimeError(
                    "Optimal flow but incomplete quota matching."
                )

            return match_ctrl, match_cost, {
                "K_per_bin_final": list(K_bins),
                "K_per_bin_schedule": K_schedule,
                "retries": retries,
            }

        # Infeasible -- grow K_per_bin
        retries += 1
        grew = False
        for t in range(T):
            new_K = min(len(bin_members[t]), K_bins[t] * K_growth, eff_max_K)
            if new_K > K_bins[t]:
                K_bins[t] = new_K
                grew = True
        K_schedule.append(list(K_bins))

        if not grew:
            raise RuntimeError(
                f"Quota-constrained flow infeasible at max K. "
                f"quotas={quotas}, "
                f"bin_sizes={[len(bm) for bm in bin_members]}"
            )


# ----------------------------
# Full two-stage runner
# ----------------------------
def two_stage_kcenter_then_match(
    leaf_controls_enrolids: np.ndarray,
    leaf_cases_enrolids: np.ndarray,
    leaf_nn_matrix_npy: str,
    leaf_nn_enrolids_npy: str,
    pn_h5_path: str,
    M: int = 8000,
    use_adaptive_pool: bool = True,
    tau: float | None = None,
    plateau_eps: float = 0.01,
    force_nearest_per_case: bool = True,
    force_topm: int = 1,
    assignment_topk_start: int | None = None,  # None => exact full edges
    seed_method: str = "smart",  # "smart", "random", "centroid", or "density"
    random_state: Optional[int] = None,  # for reproducibility when seed_method="random"
    X_majority_leaf: Optional[np.ndarray] = None,  # Required for "centroid" method
    density_epsilon: Optional[float] = None,  # Optional for "density" method
    density_percentile: float = 10.0,  # For auto-selecting epsilon in "density"
    matching_ratio: int = 1,  # 1:k matching support
    case_weighting: Optional[str] = None,  # NEW: "boundary", "uncertainty", "density_inverse", or None
    quota_cfg: Optional[Dict] = None,  # Bin-quota constraints config (see below)
    debug_alignment: bool = False,  # Print reorder/pn alignment diagnostics
) -> Dict[str, object]:
    """
    Two-stage k-center matching: select diverse candidate controls, then optimally assign.
    
    Parameters
    ----------
    leaf_controls_enrolids : np.ndarray
        ENROLID array for majority samples in this leaf
    leaf_cases_enrolids : np.ndarray
        ENROLID array for minority samples in this leaf
    leaf_nn_matrix_npy : str
        Path to .npy file containing control-control distance matrix (n x n)
    leaf_nn_enrolids_npy : str
        Path to .npy file containing ENROLIDs for control-control matrix
    pn_h5_path : str
        Path to HDF5 file containing majority-minority distances
    M : int, default=8000
        Size of candidate pool (k-center will select M diverse controls)
    use_adaptive_pool : bool, default=True
        If True, use adaptive pool size with stopping criteria (may select < M candidates)
        If False, use fixed k-center with exactly M candidates
    tau : float or None, default=None
        Quality threshold for adaptive pool. If None and use_adaptive_pool=True,
        automatically computed as 95th percentile of best possible distance per case.
        Ignored if use_adaptive_pool=False.
    plateau_eps : float, default=0.01
        Plateau detection threshold for adaptive pool (1% = 0.01)
    force_nearest_per_case : bool, default=True
        Whether to force include nearest control(s) for each case
    force_topm : int, default=1
        How many nearest controls per case to force include
    assignment_topk_start : int or None, default=None
        If None, use exact assignment; otherwise adaptive sparse matching
    seed_method : str, default="smart"
        Initial seed selection method:
        - "smart": select control with minimum mean distance to all cases (minority-biased)
        - "random": select random control as initial seed
        - "centroid": select control closest to majority centroid
        - "density": select control with maximum local density
    random_state : int or None, default=None
        Random seed for reproducibility when seed_method="random"
    X_majority_leaf : np.ndarray, optional
        Feature matrix for majority samples in this leaf (n_controls, n_features).
        Required if seed_method="centroid".
    density_epsilon : float, optional
        Radius for density calculation in "density" method.
        If None, auto-computed from percentile of distances.
    density_percentile : float, default=10.0
        Percentile for auto-selecting density epsilon (only used if density_epsilon is None)
    matching_ratio : int, default=1
        Number of distinct controls to match per case (k in 1:k matching)
        - matching_ratio=1: 1:1 matching (each case gets 1 control)
        - matching_ratio>1: 1:k matching (each case gets k controls, reusable across cases)
    case_weighting : str, optional
        Method for computing case weights. Options:
        - None: Uniform weights (all cases treated equally)
        - "boundary": Inverse distance to nearest control (closer = higher weight)
        - "uncertainty": Shannon entropy of predicted probabilities (requires predicted_probs)
        - "density_inverse": Inverse local density (fewer neighbors = higher weight)
    quota_cfg : dict or None, default=None
        Bin-quota constraints config.  If None or ``{"enabled": False}``, the
        original unconstrained matching runs unchanged.  Keys when enabled::

    Returns
    -------
    dict with keys:
        - candidate_majority_enrolids: all M selected candidates
        - selected_control_enrolids: final matched controls
            If matching_ratio=1: array of n_cases ENROLIDs
            If matching_ratio>1: concatenated array of all selected controls (with duplicates)
        - case_to_control_map: dict mapping case ENROLID -> matched control ENROLID(s)
            If matching_ratio=1: int
            If matching_ratio>1: list of ints
        - match_costs: array of matching costs
            If matching_ratio=1: array of n_cases costs
            If matching_ratio>1: concatenated array of all costs
        - matching_ratio: int, the k value used
        - seed_method: str, the seed method used
        - seed_idx: int, the seed index selected
        - seed_enrolid: int, the seed ENROLID selected
        - quota_diagnostics: dict (only present when quotas enabled)
            Contains target_quotas, achieved_counts, per_bin_b_stats,
            matched_cost_quantiles, K_retries, etc.
    """

    # ---- Load leaf d^nn ----
    d_nn = np.load(leaf_nn_matrix_npy, mmap_mode="r")
    dnn_ids = np.load(leaf_nn_enrolids_npy)

    if d_nn.shape[0] != d_nn.shape[1]:
        raise ValueError("Leaf d_nn must be square.")
    if d_nn.shape[0] != len(dnn_ids):
        raise ValueError("Leaf d_nn and enrolids length mismatch.")

    # Reorder leaf_controls to match d_nn ordering if necessary
    # We want: row i of d_nn corresponds to leaf_controls_enrolids[i]
    _debug_alignment = debug_alignment
    if not np.array_equal(leaf_controls_enrolids, dnn_ids):
        id2pos = {int(e): i for i, e in enumerate(leaf_controls_enrolids)}
        perm = np.array([id2pos[int(e)] for e in dnn_ids], dtype=np.int64)
        leaf_controls_enrolids = leaf_controls_enrolids[perm]
        # IMPORTANT: your d_pn extraction later uses leaf_controls_enrolids,
        # so this keeps everything aligned.
        if _debug_alignment:
            is_identity = np.array_equal(perm, np.arange(len(perm)))
            print(f"  [debug_alignment] Reorder applied: permutation is identity = {is_identity}")
            if not is_identity:
                n_moved = int(np.sum(perm != np.arange(len(perm))))
                print(f"  [debug_alignment]   {n_moved}/{len(perm)} positions differ from identity")
            # Verify post-reorder alignment
            assert np.array_equal(leaf_controls_enrolids, dnn_ids), (
                "BUG: leaf_controls_enrolids != dnn_ids after reorder"
            )
            print(f"  [debug_alignment]   Post-reorder leaf_controls == dnn_ids: OK")
    else:
        if _debug_alignment:
            print(f"  [debug_alignment] leaf_controls already matches dnn_ids (no reorder needed)")

    # ---- Load d^pn ----
    f_pn, d_pn, pn_maj_ids, pn_min_ids = load_pn_hdf5(pn_h5_path)
    pn_maj_id2idx = build_id_to_index(pn_maj_ids)
    pn_min_id2idx = build_id_to_index(pn_min_ids)

    if _debug_alignment:
        # Verify pn_rows uses the *reordered* control list
        ctrl_in_pn = sum(1 for e in leaf_controls_enrolids if int(e) in pn_maj_id2idx)
        case_in_pn = sum(1 for e in leaf_cases_enrolids if int(e) in pn_min_id2idx)
        print(f"  [debug_alignment] pn lookup: {ctrl_in_pn}/{len(leaf_controls_enrolids)} controls found in pn_maj, "
              f"{case_in_pn}/{len(leaf_cases_enrolids)} cases found in pn_min")

    try:
        pn_rows = np.array([pn_maj_id2idx[int(e)] for e in leaf_controls_enrolids], dtype=np.int64)
        pn_cols = np.array([pn_min_id2idx[int(e)] for e in leaf_cases_enrolids], dtype=np.int64)

        # HDF5 fancy indexing requires sorted indices
        # Sort indices, extract data, then restore original order
        rows_sort_idx = np.argsort(pn_rows)
        cols_sort_idx = np.argsort(pn_cols)
        
        pn_rows_sorted = pn_rows[rows_sort_idx]
        pn_cols_sorted = pn_cols[cols_sort_idx]
        
        # Extract sorted submatrix
        d_pn_sorted = np.array(d_pn[pn_rows_sorted, :][:, pn_cols_sorted], dtype=np.float32)
        
        # Restore original order
        rows_unsort_idx = np.argsort(rows_sort_idx)
        cols_unsort_idx = np.argsort(cols_sort_idx)
        d_pn_leaf = d_pn_sorted[rows_unsort_idx, :][:, cols_unsort_idx]

        # Choose initial seed based on method
        print(f"  Seed selection method: '{seed_method}'")
        
        if seed_method == "random":
            seed_idx = choose_seed_random(d_nn.shape[0], random_state=random_state)
            print(f"    Random seed selected: index {seed_idx}")
            
        elif seed_method == "smart":
            # Minority-biased: closest to all cases
            seed_idx = int(np.argmin(d_pn_leaf.mean(axis=1)))
            mean_dist_to_cases = float(d_pn_leaf[seed_idx, :].mean())
            print(f"    Smart seed selected: index {seed_idx} (mean dist to cases: {mean_dist_to_cases:.4f})")
            
        elif seed_method == "centroid":
            # Centroid-based: closest to majority centroid
            if X_majority_leaf is None:
                raise ValueError("seed_method='centroid' requires X_majority_leaf parameter")
            if X_majority_leaf.shape[0] != d_nn.shape[0]:
                raise ValueError(f"X_majority_leaf shape {X_majority_leaf.shape} doesn't match d_nn shape {d_nn.shape}")
            
            seed_idx = choose_seed_centroid(X_majority_leaf)
            mean_dist_to_cases = float(d_pn_leaf[seed_idx, :].mean())
            print(f"    Centroid seed selected: index {seed_idx} (mean dist to cases: {mean_dist_to_cases:.4f})")
            
        elif seed_method == "density":
            # Density-based: highest local density
            seed_idx = choose_seed_max_density(
                d_nn,
                epsilon=density_epsilon,
                percentile=density_percentile
            )
            mean_dist_to_cases = float(d_pn_leaf[seed_idx, :].mean())
            print(f"    Density seed selected: index {seed_idx} (mean dist to cases: {mean_dist_to_cases:.4f})")
            
        else:
            raise ValueError(
                f"Unknown seed_method: '{seed_method}'. "
                f"Must be 'smart', 'random', 'centroid', or 'density'."
            )
        
        M_eff = min(M, d_nn.shape[0])

        # Choose between fixed or adaptive pool
        if use_adaptive_pool:
            # Auto-compute tau if not provided
            if tau is None:
                b = d_pn_leaf.min(axis=0)  # per-case best possible control in full leaf
                tau = float(np.percentile(b, 95))  # 95th percentile
                print(f"  Auto-computed tau (95th percentile of best distances): {tau:.4f}")
            else:
                print(f"  Using provided tau: {tau:.4f}")
            
            cand_idx, C_hat = farthest_first_adaptive_pool(
                d_nn=d_nn,
                d_pn_leaf=d_pn_leaf,
                seed_idx=seed_idx,
                M_cap=M_eff,
                tau=tau,
                plateau_eps=plateau_eps,
            )
            print(f"  Adaptive pool stopped at {len(cand_idx)} candidates (max cost: {C_hat:.4f})")
        else:
            cand_idx = farthest_first_kcenter_indices(d_nn, M_eff, seed_idx)
        cand_set = set(cand_idx)

        if force_nearest_per_case:
            if force_topm == 1:
                nn_each = np.argmin(d_pn_leaf, axis=0)
                cand_set.update(map(int, nn_each))
            else:
                idxs = np.argpartition(d_pn_leaf, force_topm-1, axis=0)[:force_topm, :].ravel()
                cand_set.update(map(int, idxs))

        # Trim to M_eff deterministically
        if len(cand_set) > M_eff:
            trimmed = []
            for j in cand_idx:
                if j in cand_set:
                    trimmed.append(j)
                if len(trimmed) >= M_eff:
                    break
            cand_set = set(trimmed)

        cand_idx_final = sorted(list(cand_set))
        candidate_majority_enrolids = leaf_controls_enrolids[cand_idx_final]

        # ============================================================================
        # Compute case weights (if requested)
        # ============================================================================
        weights_to_use = None
        
        if case_weighting == "boundary":
            weights_to_use = compute_case_weights_boundary(d_pn_leaf, normalize=True)
            print(f"    Boundary weights: min={weights_to_use.min():.3f}, max={weights_to_use.max():.3f}, mean={weights_to_use.mean():.3f}")
            
        elif case_weighting == "density_inverse":
            weights_to_use = compute_case_weights_density_inverse(
                d_pn_leaf, 
                epsilon=density_epsilon,
                percentile=density_percentile,
                normalize=True
            )
            print(f"    Density-inverse weights: min={weights_to_use.min():.3f}, max={weights_to_use.max():.3f}, mean={weights_to_use.mean():.3f}")
            
        
        # ================================================================
        # Stage B: Min-cost flow matching
        # ================================================================
        quota_diagnostics = None

        if quota_cfg is not None and quota_cfg.get("enabled", False):
            # -- Bin-quota constrained matching (1:1 only) --
            if matching_ratio != 1:
                raise ValueError(
                    "Bin-quota constraints only supported for "
                    f"matching_ratio=1 (got {matching_ratio})"
                )

            # ---- Config ----
            q_quantiles = quota_cfg.get(
                "quantiles", [0, .2, .4, .6, .8, 1.0]
            )
            T = len(q_quantiles) - 1
            n_target = len(leaf_cases_enrolids)
            q_mode = quota_cfg.get("mode", "pool_mass")
            q_lambda = quota_cfg.get("lambda", 0.0)
            binning = quota_cfg.get("binning", "population")

            # ---- B1: proximity scores on POOL ----
            d_pn_pool = d_pn_leaf[cand_idx_final, :]   # (M, n_cases)
            b_J = compute_proximity_scores(d_pn_pool)

            if binning == "population":
                # ============================================================
                # POPULATION-BASED binning & quotas
                # Cutpoints come from a large deterministic population subset;
                # quotas are proportional to population bin counts (not pool).
                # ============================================================
                pop_S = quota_cfg.get("pop_S", 50000)
                pop_subset_mode = quota_cfg.get(
                    "pop_subset", "sorted_prefix"
                )

                # (1) Deterministic population subset
                pop_idx = select_population_subset_indices(
                    leaf_controls_enrolids,
                    S=pop_S,
                    mode=pop_subset_mode,
                )

                # (2) Cutpoints from population proximity distribution
                cutpoints, b_pop = compute_cutpoints_from_population(
                    d_pn_leaf, pop_idx, q_quantiles,
                )

                # (3) Bin population and pool using SAME cutpoints
                pop_bins_raw = assign_bins_from_cutpoints(
                    b_pop, cutpoints
                )
                pop_bin_counts_raw = [
                    int(np.sum(pop_bins_raw == t)) for t in range(T)
                ]
                pool_bins_raw = assign_bins_from_cutpoints(
                    b_J, cutpoints
                )
                pool_bin_counts_raw = [
                    int(np.sum(pool_bins_raw == t)) for t in range(T)
                ]

                # (4) Quotas from POPULATION counts (proportional)
                #     compute_bin_quotas caps at the reference counts,
                #     but we also need to cap at pool counts afterwards.
                if q_mode == "tilted":
                    bin_midpoints = []
                    for t_raw in range(T):
                        mask = pop_bins_raw == t_raw
                        if mask.any():
                            v = b_pop[mask]
                            bin_midpoints.append(
                                float((v.min() + v.max()) / 2.0)
                            )
                        else:
                            bin_midpoints.append(0.0)
                    quotas_raw = compute_bin_quotas(
                        pop_bin_counts_raw, n_target,
                        mode=q_mode, lamda=q_lambda,
                        bin_midpoints=bin_midpoints,
                    )
                else:
                    quotas_raw = compute_bin_quotas(
                        pop_bin_counts_raw, n_target,
                        mode=q_mode,
                    )

                # Cap quotas at pool capacity per bin
                quotas_capped = cap_quotas_at_pool_counts(
                    quotas_raw, pool_bin_counts_raw
                )

                # (5) Collapse bins with zero pool members
                (pool_bins, pool_bin_counts, pop_bin_counts,
                 quotas, active_bins) = collapse_to_active_bins(
                    pool_bins_raw, pool_bin_counts_raw,
                    pop_bin_counts_raw, quotas_capped, T,
                )
                T_active = len(active_bins)
                bin_assignments = pool_bins

                print(
                    f"  Bin-quota matching (population-based): "
                    f"T_active={T_active}, quotas={quotas}, "
                    f"pop_per_bin={pop_bin_counts}, "
                    f"pool_per_bin={pool_bin_counts}"
                )

            elif binning == "pool":
                # ============================================================
                # POOL-BASED binning & quotas  (legacy behavior)
                # Cutpoints from pool b_J quantiles, quotas from pool counts.
                # ============================================================
                q_T = quota_cfg.get("T", 5)
                bin_assignments, cutpoints, pool_bin_counts, active_bins = \
                    deterministic_binning(
                        b_J, T=q_T, quantiles=q_quantiles
                    )
                T_active = len(active_bins)
                pop_bin_counts = pool_bin_counts   # same for pool mode

                if q_mode == "tilted":
                    bin_midpoints = []
                    for t in range(T_active):
                        v = b_J[bin_assignments == t]
                        bin_midpoints.append(
                            float((v.min() + v.max()) / 2.0)
                        )
                    quotas = compute_bin_quotas(
                        pool_bin_counts, n_target,
                        mode=q_mode, lamda=q_lambda,
                        bin_midpoints=bin_midpoints,
                    )
                else:
                    quotas = compute_bin_quotas(
                        pool_bin_counts, n_target,
                        mode=q_mode,
                    )

                pop_S = None
                pop_subset_mode = None

                print(
                    f"  Bin-quota matching (pool-based): "
                    f"T_active={T_active}, quotas={quotas}, "
                    f"pool_per_bin={pool_bin_counts}"
                )

            else:
                raise ValueError(
                    f"Unknown binning mode '{binning}'. "
                    f"Use 'population' or 'pool'."
                )

            # ---- Solve with per-bin sparsification ----
            cost = d_pn_leaf[cand_idx_final, :].T  # (n_cases, M)
            match_ctrl_local, match_costs, solve_diag = \
                solve_min_cost_assignment_with_quotas(
                    cost_matrix=cost,
                    bin_assignments=bin_assignments,
                    quotas=quotas,
                    K_per_bin=quota_cfg.get("K_per_bin", 25),
                    K_growth=quota_cfg.get("K_growth", 2),
                    max_K_per_bin=quota_cfg.get(
                        "max_K_per_bin", None
                    ),
                    case_weights=weights_to_use,
                )

            # ---- Diagnostics ----
            matched_bins = bin_assignments[match_ctrl_local]
            achieved = [
                int(np.sum(matched_bins == t))
                for t in range(T_active)
            ]
            matched_b = b_J[match_ctrl_local]
            per_bin_stats = {}
            for t in range(T_active):
                mask_t = matched_bins == t
                if mask_t.any():
                    v = matched_b[mask_t]
                    per_bin_stats[t] = {
                        "min": float(v.min()),
                        "median": float(np.median(v)),
                        "max": float(v.max()),
                    }
                else:
                    per_bin_stats[t] = {
                        "min": None, "median": None,
                        "max": None,
                    }
            dq = np.quantile(
                match_costs, [0, 0.25, 0.5, 0.75, 1.0]
            )
            quota_diagnostics = {
                "binning_source": binning,
                "T_active": T_active,
                "target_quotas": quotas,
                "achieved_counts": achieved,
                "pop_bin_counts": pop_bin_counts,
                "pool_bin_counts": pool_bin_counts,
                "cutpoints": (
                    cutpoints.tolist()
                    if hasattr(cutpoints, "tolist")
                    else list(cutpoints)
                ),
                "active_bins": active_bins,
                "K_per_bin_final": solve_diag["K_per_bin_final"],
                "K_retries": solve_diag["retries"],
                "per_bin_b_stats": per_bin_stats,
                "matched_cost_quantiles": {
                    "min": float(dq[0]),
                    "q25": float(dq[1]),
                    "median": float(dq[2]),
                    "q75": float(dq[3]),
                    "max": float(dq[4]),
                },
                "mode": q_mode,
            }
            if binning == "population":
                quota_diagnostics["pop_S"] = len(pop_idx)
                quota_diagnostics["pop_subset_mode"] = (
                    pop_subset_mode
                )

            print(
                f"  Quota solve done "
                f"(retries={solve_diag['retries']})"
            )
            for t in range(T_active):
                print(
                    f"    Bin {t}: target={quotas[t]}, "
                    f"achieved={achieved[t]}, "
                    f"pool={pool_bin_counts[t]}"
                )

            selected_control_enrolids = \
                candidate_majority_enrolids[match_ctrl_local]
            case_to_control = {
                int(ci): int(cj)
                for ci, cj in zip(
                    leaf_cases_enrolids, selected_control_enrolids
                )
            }

        else:
            # -- Original (unconstrained) matching path --
            cost = d_pn_leaf[cand_idx_final, :].T  # (n_cases, M)
            match_ctrl_local, match_costs = \
                solve_min_cost_assignment_adaptive_topk(
                    cost_matrix=cost,
                    topk_start=assignment_topk_start,
                    matching_ratio=matching_ratio,
                    case_weights=weights_to_use,
                )

            if matching_ratio == 1:
                selected_control_enrolids = \
                    candidate_majority_enrolids[match_ctrl_local]
                case_to_control = {
                    int(ci): int(cj)
                    for ci, cj in zip(
                        leaf_cases_enrolids, selected_control_enrolids
                    )
                }
            else:
                selected_control_indices_local = np.concatenate(
                    match_ctrl_local
                )
                selected_control_enrolids = \
                    candidate_majority_enrolids[
                        selected_control_indices_local
                    ]
                match_costs = np.concatenate(match_costs)
                case_to_control = {}
                for i, case_enrolid in enumerate(leaf_cases_enrolids):
                    matched_ctrls = \
                        candidate_majority_enrolids[match_ctrl_local[i]]
                    case_to_control[int(case_enrolid)] = \
                        matched_ctrls.tolist()

        result = {
            "candidate_majority_enrolids": candidate_majority_enrolids,
            "selected_control_enrolids": selected_control_enrolids,
            "case_to_control_map": case_to_control,
            "match_costs": match_costs,
            "matching_ratio": matching_ratio,
            "seed_method": seed_method,
            "seed_idx": seed_idx,
            "seed_enrolid": int(leaf_controls_enrolids[seed_idx]),
            "case_weights": weights_to_use,
            "case_weighting_method": case_weighting,
        }
        if quota_diagnostics is not None:
            result["quota_diagnostics"] = quota_diagnostics
        return result

    finally:
        f_pn.close()

