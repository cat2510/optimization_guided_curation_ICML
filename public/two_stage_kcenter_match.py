# two_stage_kcenter_match.py
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


def load_nn(nn_matrix_path: str, nn_enrolids_path: str):
    """
    Load majority-majority distances from .npy (memmap) or .h5 (HDF5 compressed).
    Use HDF5 for large matrices to save disk (~3-5x smaller with gzip).
    Returns (d_nn, maj_ids) where d_nn supports d_nn[i,:] slicing.
    """
    maj_ids = np.load(nn_enrolids_path)
    if nn_matrix_path.endswith('.h5') or nn_matrix_path.endswith('.hdf5'):
        f = h5py.File(nn_matrix_path, "r")
        d_dset = f["distances"]
        # Wrap so we can close file when done; support shape + __getitem__
        class _H5DnnView:
            def __getitem__(self, key):
                return np.asarray(d_dset[key], dtype=np.float32)
            @property
            def shape(self):
                return d_dset.shape
        d_nn = _H5DnnView()
        d_nn._h5_file = f  # keep file open
        return d_nn, maj_ids
    else:
        d_nn = np.load(nn_matrix_path, mmap_mode="r")
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


def kmeanspp_metric_indices(d_nn, M: int, seed_idx: int, rng: np.random.RandomState):
    """
    k-means++ (D^2) seeding using a precomputed distance matrix d_nn.
    Picks next center with prob proportional to (distance to nearest selected)^2.

    Complexity: O(M*n) distance reads/updates, same order as farthest-first.
    Deterministic given rng seed.
    """
    n = d_nn.shape[0]
    if d_nn.shape[1] != n:
        raise ValueError("d_nn must be square")
    M = min(M, n)
    selected = np.empty(M, dtype=np.int64)
    selected[0] = seed_idx

    # distance to nearest selected
    min_dist = np.array(d_nn[seed_idx, :], dtype=np.float32)
    min_dist[seed_idx] = 0.0

    for t in range(1, M):
        w = (min_dist ** 2).astype(np.float64)
        w[selected[:t]] = 0.0  # prevent reselection
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            # degenerate fallback: choose any unselected
            cand = np.setdiff1d(np.arange(n), selected[:t], assume_unique=False)
            nxt = int(rng.choice(cand))
        else:
            w /= s
            nxt = int(rng.choice(n, p=w))

        selected[t] = nxt
        row = np.array(d_nn[nxt, :], dtype=np.float32)
        min_dist = np.minimum(min_dist, row)
        min_dist[nxt] = 0.0

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
    random_state: Optional[int] = 123
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
# Full two-stage runner
# ----------------------------
def two_stage_kcenter_then_match(
    leaf_controls_enrolids: np.ndarray,
    leaf_cases_enrolids: np.ndarray,
    leaf_nn_matrix_npy: str,
    leaf_nn_enrolids_npy: str,
    pn_h5_path: str,
    M: int = 8000,
    use_kmeanspp: bool = False,  # If True, use k-means++ seeding
    use_adaptive_pool: bool = True,
    tau: float | None = None,
    plateau_eps: float = 0.01,
    force_nearest_per_case: bool = True,
    force_topm: int = 1,
    assignment_topk_start: int | None = None,  # None => exact full edges
    seed_method: str = "smart",  # "smart", "random", "centroid", or "density"
    random_state: Optional[int] = 123,  # for reproducibility when seed_method="random"
    X_majority_leaf: Optional[np.ndarray] = None,  # Required for "centroid" method
    density_epsilon: Optional[float] = None,  # Optional for "density" method
    density_percentile: float = 10.0,  # For auto-selecting epsilon in "density"
    matching_ratio: int = 1,  # 1:k matching support
    case_weighting: Optional[str] = None, 
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
    """

    # ---- Load leaf d^nn (.npy or .h5) ----
    d_nn, dnn_ids = load_nn(leaf_nn_matrix_npy, leaf_nn_enrolids_npy)

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
        elif use_kmeanspp:
            rng = np.random.RandomState(random_state + 1337)  # offset avoids coupling to other random uses
            cand_idx = kmeanspp_metric_indices(d_nn, M_eff, seed_idx, rng)
            print(f"  K-means++ pool selected {len(cand_idx)} candidates")
        else:
            cand_idx = farthest_first_kcenter_indices(d_nn, M_eff, seed_idx)
            print(f"  Farthest-first pool selected {len(cand_idx)} candidates")
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
        return result
    finally:
        f_pn.close()

