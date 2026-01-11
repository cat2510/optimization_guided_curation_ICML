"""
Extended version of two_stage_kcenter_match.py with flexible matching ratios.

Key addition: Support for 1:k matching (each case matched to k distinct controls)

Usage:
    # 1:1 matching (original)
    matches, costs = solve_min_cost_assignment_adaptive_topk(cost_matrix, matching_ratio=1)
    
    # 1:2 matching (each case gets 2 controls)
    matches, costs = solve_min_cost_assignment_adaptive_topk(cost_matrix, matching_ratio=2)
    
    # Returns:
    #   matches: list of arrays, where matches[i] contains k control indices for case i
    #   costs: list of arrays, where costs[i] contains k costs for case i
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


def solve_min_cost_assignment_adaptive_topk(
    cost_matrix: np.ndarray,
    topk_start: int = 50,
    topk_max: Optional[int] = None,
    topk_growth: int = 2,
    cost_scale: int = 10000,
    matching_ratio: int = 1  # NEW: 1:k matching
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Solve 1-to-k assignment: each case matched to k DISTINCT controls.
    
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
        - matching_ratio=1: 1:1 matching (original behavior)
        - matching_ratio=2: 1:2 matching (each case gets 2 controls)
        - matching_ratio=k: 1:k matching (each case gets k controls)
    
    Returns
    -------
    match_ctrl_for_case : list of np.ndarray
        List of length n_cases, where match_ctrl_for_case[i] is an array of
        matching_ratio control indices matched to case i
    matched_costs : list of np.ndarray
        List of length n_cases, where matched_costs[i] is an array of
        matching_ratio costs for case i
    
    Notes
    -----
    Min-cost flow network structure for 1:k matching:
    - Source node supplies: n_cases * k flow units
    - Each case node: receives k flow units from source, sends k to controls
    - Each control node: can receive up to k flow units (reusable)
    - Each case->control edge: capacity 1 (ensures distinctness per case)
    - Sink node demands: n_cases * k flow units
    
    Examples
    --------
    >>> # 1:1 matching (original)
    >>> cost = np.array([[1, 2, 3], [4, 5, 6]])  # 2 cases, 3 controls
    >>> matches, costs = solve_min_cost_assignment_adaptive_topk(cost, matching_ratio=1)
    >>> # matches[0] = [0], matches[1] = [1]  (each case gets 1 control)
    
    >>> # 1:2 matching
    >>> matches, costs = solve_min_cost_assignment_adaptive_topk(cost, matching_ratio=2)
    >>> # matches[0] = [0, 1], matches[1] = [1, 2]  (each case gets 2 controls)
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
                matching_ratio,  # capacity = k (reusable)
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
            # For 1:k matching, each case will have k arcs with flow=1
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

        # Infeasible → try increasing K
        if K >= min(topk_max, n_controls):
            raise RuntimeError(
                f"Min-cost assignment infeasible even with K={K}. "
                f"Check that n_controls >= n_cases * matching_ratio and indices align."
            )

        K = min(topk_max, K * topk_growth)


def solve_min_cost_assignment_1to1(
    cost_matrix: np.ndarray,
    topk_start: int = 50,
    topk_max: Optional[int] = None,
    topk_growth: int = 2,
    cost_scale: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for 1:1 matching (backward compatible with original).
    
    Returns arrays instead of lists for backward compatibility.
    """
    matches, costs = solve_min_cost_assignment_adaptive_topk(
        cost_matrix, topk_start, topk_max, topk_growth, cost_scale, matching_ratio=1
    )
    # Convert list of arrays to single arrays (each has length 1)
    match_ctrl = np.array([m[0] for m in matches], dtype=np.int64)
    match_cost = np.array([c[0] for c in costs], dtype=np.float32)
    return match_ctrl, match_cost


def solve_min_cost_assignment_1tok(
    cost_matrix: np.ndarray,
    k: int,
    topk_start: int = 50,
    topk_max: Optional[int] = None,
    topk_growth: int = 2,
    cost_scale: int = 10000
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convenience wrapper for 1:k matching with explicit k parameter.
    
    Parameters
    ----------
    cost_matrix : np.ndarray
        Shape (n_cases, n_controls)
    k : int
        Number of controls to match per case
    
    Returns
    -------
    matches : list of np.ndarray
        matches[i] contains k control indices for case i
    costs : list of np.ndarray
        costs[i] contains k costs for case i
    
    Examples
    --------
    >>> # Each case gets 2 controls
    >>> matches, costs = solve_min_cost_assignment_1tok(cost_matrix, k=2)
    >>> print(matches[0])  # [3, 7] (case 0 matched to controls 3 and 7)
    >>> print(costs[0])    # [1.2, 1.5] (corresponding costs)
    """
    return solve_min_cost_assignment_adaptive_topk(
        cost_matrix, topk_start, topk_max, topk_growth, cost_scale, matching_ratio=k
    )


# Example usage
if __name__ == "__main__":
    # Create example cost matrix
    np.random.seed(42)
    n_cases = 5
    n_controls = 10
    cost_matrix = np.random.rand(n_cases, n_controls) * 10
    
    print("Cost Matrix:")
    print(cost_matrix)
    print()
    
    # 1:1 matching (original behavior)
    print("="*60)
    print("1:1 MATCHING (each case gets 1 control)")
    print("="*60)
    matches_1to1, costs_1to1 = solve_min_cost_assignment_1to1(cost_matrix)
    for i in range(n_cases):
        print(f"Case {i} -> Control {matches_1to1[i]}, Cost: {costs_1to1[i]:.4f}")
    print()
    
    # 1:2 matching
    print("="*60)
    print("1:2 MATCHING (each case gets 2 controls)")
    print("="*60)
    matches_1to2, costs_1to2 = solve_min_cost_assignment_1tok(cost_matrix, k=2)
    for i in range(n_cases):
        print(f"Case {i} -> Controls {matches_1to2[i]}, Costs: {costs_1to2[i]}")
    print()
    
    # 1:3 matching
    print("="*60)
    print("1:3 MATCHING (each case gets 3 controls)")
    print("="*60)
    matches_1to3, costs_1to3 = solve_min_cost_assignment_1tok(cost_matrix, k=3)
    for i in range(n_cases):
        print(f"Case {i} -> Controls {matches_1to3[i]}, Costs: {costs_1to3[i]}")
    
    # Verify distinctness: each case has k unique controls
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    for i in range(n_cases):
        unique_2 = len(set(matches_1to2[i]))
        unique_3 = len(set(matches_1to3[i]))
        print(f"Case {i}: 1:2 unique={unique_2}/2, 1:3 unique={unique_3}/3")

