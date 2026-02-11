"""
Quota helper functions for bin-quota constrained min-cost flow matching.

Extracted from two_stage_kcenter_match.py for modularity.
Contains:
  - Proximity score computation (batched, memmap-safe)
  - Deterministic binning (pool-based or population-based)
  - Quota computation (pool_mass / tilted)
  - Population subset selection (deterministic, no randomness)
  - Cutpoint computation from population
  - Bin assignment from external cutpoints
  - Quota capping at pool capacity
"""
import numpy as np
from typing import Dict, Tuple, List, Optional


# ---------------------------------------------------------------------------
# Proximity scores
# ---------------------------------------------------------------------------

def compute_proximity_scores(
    d_pn_pool: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Compute proximity score b(j) = min_{i in P} d_pn[j, i] for each pool
    member j.  Iterates over positives in batches (column-wise), so it works
    even when d_pn_pool is a memmap or HDF5 dataset.

    Parameters
    ----------
    d_pn_pool : array-like, shape (n_pool, n_cases)
    batch_size : int

    Returns
    -------
    b_J : np.ndarray of float64, shape (n_pool,)
    """
    n_pool = d_pn_pool.shape[0]
    n_cases = d_pn_pool.shape[1]
    b_J = np.full(n_pool, np.inf, dtype=np.float64)
    for start in range(0, n_cases, batch_size):
        end = min(start + batch_size, n_cases)
        batch_min = np.asarray(d_pn_pool[:, start:end]).min(axis=1)
        np.minimum(b_J, batch_min, out=b_J)
    return b_J


# ---------------------------------------------------------------------------
# Pool-based (legacy) binning
# ---------------------------------------------------------------------------

def deterministic_binning(
    b_J: np.ndarray,
    T: int = 5,
    quantiles: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Assign each pool member to a proximity bin deterministically.

    Rule: bin(j) = smallest t such that b_J[j] <= cutpoints[t+1].
    Empty bins are collapsed to contiguous indices.

    Parameters
    ----------
    b_J : np.ndarray, shape (n_pool,)
    T : int  --  target number of bins (ignored if quantiles given)
    quantiles : list[float], length T+1, e.g. [0, .2, .4, .6, .8, 1.0]

    Returns
    -------
    bin_assignments : np.ndarray int32 (n_pool,)  in 0..T_active-1
    cutpoints : np.ndarray  (T+1,)
    bin_counts : list[int]  per active bin
    active_bins : list[int] original bin indices that are non-empty
    """
    if quantiles is not None:
        T = len(quantiles) - 1
    else:
        quantiles = [i / T for i in range(T + 1)]

    if len(quantiles) != T + 1:
        raise ValueError(
            f"quantiles length must be T+1={T + 1}, got {len(quantiles)}"
        )

    cutpoints = np.quantile(b_J, quantiles)

    # searchsorted(cutpoints[1:], v, 'left') -> smallest t with cutpoints[t+1] >= v
    raw_bins = np.searchsorted(cutpoints[1:], b_J, side="left").astype(np.int32)
    raw_bins = np.clip(raw_bins, 0, T - 1)

    raw_counts = [int(np.sum(raw_bins == t)) for t in range(T)]
    active_bins = [t for t in range(T) if raw_counts[t] > 0]

    if len(active_bins) < T:
        remap = {old: new for new, old in enumerate(active_bins)}
        bin_assignments = np.array(
            [remap[int(raw_bins[j])] for j in range(len(b_J))], dtype=np.int32
        )
        bin_counts = [raw_counts[t] for t in active_bins]
    else:
        bin_assignments = raw_bins
        bin_counts = raw_counts
        active_bins = list(range(T))

    return bin_assignments, cutpoints, bin_counts, active_bins


# ---------------------------------------------------------------------------
# Quota computation
# ---------------------------------------------------------------------------

def compute_bin_quotas(
    bin_counts: List[int],
    n_target: int,
    mode: str = "tilted",
    lamda: float = 0.0,
    bin_midpoints: Optional[List[float]] = None,
) -> List[int]:
    """
    Compute integer quotas m_t summing to n_target, one per active bin.

    Parameters
    ----------
    bin_counts : list[int]
        Reference counts per bin (population or pool).
    n_target : int  (= n_cases for 1:1)
    mode : 'pool_mass' | 'tilted'
    lamda : float >= 0  (tilt param, only for 'tilted')
    bin_midpoints : list[float]  (midpoint of b-range per bin, for 'tilted')

    Returns
    -------
    quotas : list[int]  summing to n_target, each <= bin_counts[t]
    """
    T_bins = len(bin_counts)
    total_pool = sum(bin_counts)

    if total_pool < n_target:
        raise ValueError(
            f"Pool size ({total_pool}) < n_target ({n_target})."
        )

    if mode == "pool_mass":
        raw = [n_target * c / total_pool for c in bin_counts]
    elif mode == "tilted":
        if bin_midpoints is None:
            raise ValueError("bin_midpoints required for 'tilted' mode")
        raw_weights = [
            c * float(np.exp(-lamda * m))
            for c, m in zip(bin_counts, bin_midpoints)
        ]
        total_w = sum(raw_weights)
        if total_w < 1e-15:
            raw = [n_target * c / total_pool for c in bin_counts]
        else:
            raw = [n_target * w / total_w for w in raw_weights]
    else:
        raise ValueError(f"Unknown quota mode: '{mode}'.")

    # Deterministic rounding: floor + largest-remainder
    floors = [int(np.floor(r)) for r in raw]
    remainder = n_target - sum(floors)
    fracs = [(raw[t] - floors[t], t) for t in range(T_bins)]
    fracs.sort(key=lambda x: (-x[0], x[1]))
    for k in range(int(remainder)):
        floors[fracs[k][1]] += 1

    quotas = floors

    # Cap at bin_counts and redistribute excess
    excess_total = 0
    for t in range(T_bins):
        if quotas[t] > bin_counts[t]:
            excess_total += quotas[t] - bin_counts[t]
            quotas[t] = bin_counts[t]

    while excess_total > 0:
        candidates = [
            (bin_counts[t] - quotas[t], t)
            for t in range(T_bins)
            if quotas[t] < bin_counts[t]
        ]
        if not candidates:
            raise RuntimeError(
                f"Cannot satisfy quotas: bin_counts={bin_counts}, "
                f"n_target={n_target}"
            )
        candidates.sort(key=lambda x: (-x[0], x[1]))
        quotas[candidates[0][1]] += 1
        excess_total -= 1

    if sum(quotas) != n_target:
        raise RuntimeError(
            f"Quota sum {sum(quotas)} != n_target {n_target}."
        )
    return quotas


# ---------------------------------------------------------------------------
# Population-based binning helpers  (NEW)
# ---------------------------------------------------------------------------

def select_population_subset_indices(
    leaf_controls_enrolids: np.ndarray,
    S: int = 50000,
    mode: str = "sorted_prefix",
) -> np.ndarray:
    """
    Return row-indices (into leaf_controls_enrolids / d_pn_leaf) of a
    deterministic population subset of size min(S, N).

    Parameters
    ----------
    leaf_controls_enrolids : (N,) int array
    S : int  -- target subset size
    mode : 'sorted_prefix' | 'fixed_stride'
        sorted_prefix : sort enrolids ascending, take the first S.
        fixed_stride  : sort enrolids ascending, take every ceil(N/S)-th.

    Returns
    -------
    pop_idx : np.ndarray int64, shape (min(S, N),)
        Indices into the original array (NOT the enrolids themselves).
    """
    N = len(leaf_controls_enrolids)
    S = min(S, N)

    # Deterministic ordering by enrolid value
    sorted_order = np.argsort(leaf_controls_enrolids)

    if mode == "sorted_prefix":
        return sorted_order[:S].copy()
    elif mode == "fixed_stride":
        stride = max(1, N // S)
        return sorted_order[::stride][:S].copy()
    else:
        raise ValueError(
            f"Unknown pop_subset mode: '{mode}'. "
            f"Use 'sorted_prefix' or 'fixed_stride'."
        )


def compute_cutpoints_from_population(
    d_pn_leaf: np.ndarray,
    pop_idx: np.ndarray,
    quantiles: List[float],
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bin cutpoints from the proximity-score distribution of a
    population subset.

    Parameters
    ----------
    d_pn_leaf : (n_controls, n_cases) array
    pop_idx   : (S,) int -- row indices of the population subset
    quantiles : list[float], length T+1
    batch_size : int -- column batch size for proximity computation

    Returns
    -------
    cutpoints : np.ndarray (T+1,)
    b_pop     : np.ndarray float64 (S,)  -- proximity scores of pop members
    """
    b_pop = compute_proximity_scores(d_pn_leaf[pop_idx, :], batch_size=batch_size)
    cutpoints = np.quantile(b_pop, quantiles)
    return cutpoints, b_pop


def assign_bins_from_cutpoints(
    values: np.ndarray,
    cutpoints: np.ndarray,
) -> np.ndarray:
    """
    Assign each value to a bin using pre-computed cutpoints.

    Rule: bin(j) = smallest t such that values[j] <= cutpoints[t+1].

    Parameters
    ----------
    values    : (n,) float
    cutpoints : (T+1,) float  -- bin edges from quantile computation

    Returns
    -------
    raw_bins : np.ndarray int32 (n,)  in 0..T-1  (NOT collapsed)
    """
    T = len(cutpoints) - 1
    raw_bins = np.searchsorted(cutpoints[1:], values, side="left").astype(np.int32)
    raw_bins = np.clip(raw_bins, 0, T - 1)
    return raw_bins


def cap_quotas_at_pool_counts(
    quotas: List[int],
    pool_counts: List[int],
) -> List[int]:
    """
    Cap each quota at the pool bin count and deterministically redistribute
    the excess to bins with remaining pool capacity.

    Parameters
    ----------
    quotas      : list[int]  -- initial quotas (from population mass)
    pool_counts : list[int]  -- how many pool members are in each bin

    Returns
    -------
    capped : list[int]  -- quotas with same sum, each <= pool_counts[t]
    """
    n_target = sum(quotas)
    T = len(quotas)
    capped = list(quotas)

    excess = 0
    for t in range(T):
        if capped[t] > pool_counts[t]:
            excess += capped[t] - pool_counts[t]
            capped[t] = pool_counts[t]

    while excess > 0:
        # Find bins with remaining pool room, sorted by most room first,
        # then smaller index for deterministic tie-breaking.
        candidates = [
            (pool_counts[t] - capped[t], t)
            for t in range(T)
            if capped[t] < pool_counts[t]
        ]
        if not candidates:
            raise RuntimeError(
                f"Cannot redistribute excess quotas: pool_counts={pool_counts}, "
                f"quotas={quotas}, excess={excess}"
            )
        candidates.sort(key=lambda x: (-x[0], x[1]))
        capped[candidates[0][1]] += 1
        excess -= 1

    if sum(capped) != n_target:
        raise RuntimeError(
            f"Capped quota sum {sum(capped)} != n_target {n_target}."
        )
    return capped


def collapse_to_active_bins(
    pool_bins_raw: np.ndarray,
    pool_bin_counts_raw: List[int],
    pop_bin_counts_raw: List[int],
    quotas_raw: List[int],
    T: int,
) -> Tuple[np.ndarray, List[int], List[int], List[int], List[int]]:
    """
    Collapse bins where the pool has zero members (irrelevant to solver).
    Remaps remaining bins to contiguous 0..T_active-1.

    Returns
    -------
    pool_bins     : (n_pool,) int32 -- remapped
    pool_counts   : list[int]       -- active only
    pop_counts    : list[int]       -- active only
    quotas        : list[int]       -- active only
    active_bins   : list[int]       -- original indices of kept bins
    """
    active_bins = [t for t in range(T) if pool_bin_counts_raw[t] > 0]
    remap = {old: new for new, old in enumerate(active_bins)}

    pool_bins = np.array(
        [remap[int(pool_bins_raw[j])] for j in range(len(pool_bins_raw))],
        dtype=np.int32,
    )
    pool_counts = [pool_bin_counts_raw[t] for t in active_bins]
    pop_counts = [pop_bin_counts_raw[t] for t in active_bins]
    quotas = [quotas_raw[t] for t in active_bins]

    return pool_bins, pool_counts, pop_counts, quotas, active_bins
