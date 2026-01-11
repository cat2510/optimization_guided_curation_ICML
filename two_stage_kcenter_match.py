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

# ----------------------------
# Stage B: 1-to-1 min-cost assignment
# ----------------------------

def solve_min_cost_assignment_adaptive_topk(cost_matrix: np.ndarray,
                                           topk_start: int = 50,
                                           topk_max: Optional[int] = None,
                                           topk_growth: int = 2,
                                           cost_scale: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve 1-to-1 assignment: each case matched to a UNIQUE control.

    cost_matrix: shape (n_cases, n_controls), float
    We build a min-cost flow with edges from each case to its top-K cheapest controls.
    If infeasible, increase K multiplicatively until feasible.
    If true optimal is desired, set assignment_topk_start = topk_max = None

    Returns:
      match_ctrl_for_case: array length n_cases with control index in [0..n_controls-1]
      matched_costs: array length n_cases with original float costs
    """
    try:
        from ortools.graph.python import min_cost_flow
    except Exception:
        # older ortools fallback
        from ortools.graph import pywrapgraph as min_cost_flow  # type: ignore

    n_cases, n_controls = cost_matrix.shape
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

        # Supplies (use snake_case for newer OR-Tools versions)
        mcf.set_node_supply(source, n_cases)
        mcf.set_node_supply(sink, -n_cases)
        # other nodes default to 0

        # Source -> cases
        for i in range(n_cases):
            mcf.add_arc_with_capacity_and_unit_cost(source, case_offset + i, 1, 0)

        # Controls -> sink
        for j in range(n_controls):
            mcf.add_arc_with_capacity_and_unit_cost(ctrl_offset + j, sink, 1, 0)

        # Case -> controls (top-K edges per case)
        # Use argpartition for speed
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
                mcf.add_arc_with_capacity_and_unit_cost(case_offset + i, ctrl_offset + int(j), 1, c)

        status = mcf.solve()
        if status == min_cost_flow.SimpleMinCostFlow.OPTIMAL:
            # Recover assignment: find arcs case->control with flow=1
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
                # Should not happen if optimal with full supply satisfied, but guard anyway
                raise RuntimeError("Optimal flow found but incomplete matching.")

            return match_ctrl, match_cost

        # infeasible or not solved
        if K >= topk_max:
            raise RuntimeError(f"Min-cost assignment infeasible even with K={K}. "
                               "Check that n_controls >= n_cases and indices align.")
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
    force_nearest_per_case: bool = True,
    force_topm: int = 1,
    assignment_topk_start: int | None = None,  # None => exact full edges
    seed_method: str = "smart",  # "smart" (closest to cases) or "random"
    random_state: Optional[int] = None,  # for reproducibility when seed_method="random"
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
    force_nearest_per_case : bool, default=True
        Whether to force include nearest control(s) for each case
    force_topm : int, default=1
        How many nearest controls per case to force include
    assignment_topk_start : int or None, default=None
        If None, use exact assignment; otherwise adaptive sparse matching
    seed_method : str, default="smart"
        Initial seed selection method:
        - "smart": select control with minimum mean distance to all cases (optimal)
        - "random": select random control as initial seed
    random_state : int or None, default=None
        Random seed for reproducibility when seed_method="random"
    
    Returns
    -------
    dict with keys:
        - candidate_majority_enrolids: all M selected candidates
        - selected_control_enrolids: final matched controls (one per case)
        - case_to_control_map: dict mapping case ENROLID -> matched control ENROLID
        - match_costs: array of matching costs (distance per case)
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
    if not np.array_equal(leaf_controls_enrolids, dnn_ids):
        id2pos = {int(e): i for i, e in enumerate(leaf_controls_enrolids)}
        perm = np.array([id2pos[int(e)] for e in dnn_ids], dtype=np.int64)
        leaf_controls_enrolids = leaf_controls_enrolids[perm]
        # IMPORTANT: your d_pn extraction later uses leaf_controls_enrolids,
        # so this keeps everything aligned.

    # ---- Load d^pn ----
    f_pn = h5py.File(pn_h5_path, "r")
    d_pn = f_pn["distances"]
    pn_maj_ids = f_pn["majority_enrolids"][:]
    pn_min_ids = f_pn["minority_enrolids"][:]
    pn_maj_id2idx = {int(x): i for i, x in enumerate(pn_maj_ids)}
    pn_min_id2idx = {int(x): i for i, x in enumerate(pn_min_ids)}

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
        if seed_method == "random":
            seed_idx = choose_seed_random(d_nn.shape[0], random_state=random_state)
        elif seed_method == "smart":
            seed_idx = int(np.argmin(d_pn_leaf.mean(axis=1)))
        else:
            raise ValueError(f"Unknown seed_method: {seed_method}. Must be 'smart' or 'random'.")
        
        M_eff = min(M, d_nn.shape[0])

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

        # Stage B exact matching (topk_start=None => full graph exact)
        cost = d_pn_leaf[cand_idx_final, :].T  # (n_cases, n_candidates)
        match_ctrl_local, match_costs = solve_min_cost_assignment_adaptive_topk(
            cost_matrix=cost,
            topk_start=assignment_topk_start,  # None => exact
        )

        selected_control_enrolids = candidate_majority_enrolids[match_ctrl_local]
        case_to_control = {int(ci): int(cj) for ci, cj in zip(leaf_cases_enrolids, selected_control_enrolids)}

        return {
            "candidate_majority_enrolids": candidate_majority_enrolids,
            "selected_control_enrolids": selected_control_enrolids,
            "case_to_control_map": case_to_control,
            "match_costs": match_costs,
        }

    finally:
        f_pn.close()
