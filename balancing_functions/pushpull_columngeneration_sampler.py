#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Column-generation-based Push–Pull sampler.

This module implements a *continuous* (LP-relaxed) version of the bi-objective
distance–dispersion (push–pull) model with a simple column generation loop.

Design choices (compared to the previous MILP-based implementation):

- We do NOT use node pruning (K_case, K_outer) for the majority controls.
  All majority controls are, in principle, eligible to enter via pricing.
- The restricted master problem (RMP) is solved as a linear program (LP)
  using OR-Tools' GLOP solver. All variables are in [0, 1].
- Column generation is performed on the *control* selection variables.
- At the end, we round by selecting the k controls with the largest s_j.

This is intentionally "clean" and self-contained. It does NOT depend on the
LexicographicCaseControlResampler base class; you can wrap it if you want
to integrate with your existing balancing_functions package.

The public entry point is `PushPullSamplerCG.solve_pushpull_cg(...)`, which
takes pre-split cases / controls DataFrames and returns an undersampled DataFrame.
"""

from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from ortools.linear_solver import pywraplp


class PushPullSamplerCG:
    """
    Column-generation-based push–pull sampler.

    We solve a *relaxed* version of the push–pull formulation:

        min   w * (1/|P|) sum_{i,j} d_pn(i,j) * a_ij
            - (1-w) * (1/|E|) sum_{(j,k) in E} d_nn(j,k) * z_jk

    s.t.  sum_j a_ij = 1                ∀ i
          a_ij <= s_j                   ∀ i,j # this means that s_j >= max_i a_ij so it's like a convex surrogate for using control j
          z_jk <= s_j, z_jk <= s_k      ∀ (j,k)
          sum_j s_j = k
          0 <= a_ij,z_jk,s_j <= 1

    - P = set of minority (cases)
    - N = set of majority (controls)
    - We keep E as all unordered pairs among active controls in the RMP.

    Column generation:
    - Restricted master problem (RMP) has an active subset J ⊂ N.
    - We solve the LP over variables (s_j, a_ij, z_jk) for j in J.
    - Pricing scans all candidates j ∈ N \ J and computes the marginal
      improvement (approximate reduced cost) of adding j.

    IMPORTANT:
        For simplicity, pricing here is implemented in a *primal* way:
        for each candidate control j, we compute the objective value if
        we:
            - add j with s_j = 1,
            - allow assignments a_ij to re-choose their closest control
              among (J ∪ {j}),
            - and add dispersion edges between j and all controls in J.

        This is NOT exact dual-based pricing, but it preserves the
        intuition of column generation without requiring duals.
    """

    def __init__(
        self,
        w: float = 0.5,
        max_iter: int = 5,
        init_size: int = 10,
        random_state: Optional[int] = 0,
        distance_metric: str = "euclidean",
    ):
        assert 0.0 <= w <= 1.0, "w must be in [0,1]"
        self.w = w
        self.max_iter = max_iter
        self.init_size = init_size
        self.random_state = random_state
        self.distance_metric = distance_metric

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_X(
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert df to a numeric feature matrix X, excluding specified columns.
        Returns (X, feature_names).
        """
        if exclude_cols is None:
            exclude_cols = []

        cols = [c for c in df.columns if c not in exclude_cols]
        X = df[cols].to_numpy(dtype=float)
        return X, cols

    # ------------------------------------------------------------------
    # Master problem (RMP)
    # ------------------------------------------------------------------
    def _solve_master_lp(
        self,
        D_pn_active: np.ndarray,
        D_nn_active: np.ndarray,
        k: int,
        w: float,
        verbose: bool = False,
    ) -> Dict:
        """
        Solve LP-relaxed push–pull RMP on the current active set.

        Parameters
        ----------
        D_pn_active : (P, C) array
            Distances between cases and active controls.
        D_nn_active : (C, C) array
            Distances between active controls.
        k : int
            Target number of controls to select (sum_j s_j = k).
        w : float
            Weight on pull term.
        """
        P, C = D_pn_active.shape

        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            raise RuntimeError("Failed to create GLOP solver")

        # Variables
        s = [solver.NumVar(0.0, 1.0, f"s[{j}]") for j in range(C)]
        a = [[solver.NumVar(0.0, 1.0, f"a[{i},{j}]") for j in range(C)]
             for i in range(P)]
        # Only j<k pairs
        z = {}
        for j in range(C):
            for k2 in range(j + 1, C):
                z[(j, k2)] = solver.NumVar(0.0, 1.0, f"z[{j},{k2}]")

        # Constraints
        # assignment
        for i in range(P):
            ct = solver.Sum(a[i][j] for j in range(C))
            solver.Add(ct == 1.0)
            for j in range(C):
                solver.Add(a[i][j] <= s[j])

        # dispersion activation
        for (j, k2), z_var in z.items():
            solver.Add(z_var <= s[j])
            solver.Add(z_var <= s[k2])

        # cardinality: upper bound (RMP constraint)
        # Use <= k instead of == k to allow feasibility during column generation
        solver.Add(solver.Sum(s) <= float(k))

        # Objective: w*f1 - (1-w)*f2
        # (simple unnormalized, averaged by P and |E|)
        f1 = solver.Sum(D_pn_active[i, j] * a[i][j]
                        for i in range(P) for j in range(C)) / float(P)

        if len(z) > 0:
            f2 = solver.Sum(D_nn_active[j, k2] * z[(j, k2)]
                            for (j, k2) in z.keys()) / float(len(z))
        else:
            f2 = 0.0

        objective = w * f1 - (1.0 - w) * f2
        solver.Minimize(objective)

        status = solver.Solve()
        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            raise RuntimeError(f"RMP LP failed with status {status}")

        if verbose:
            print("[RMP] status:", status)
            print("[RMP] objective:", solver.Objective().Value())

        s_val = np.array([var.solution_value() for var in s])
        a_val = np.array([[var.solution_value() for var in row] for row in a])
        # we do not currently need z_val explicitly

        return {
            "objective": solver.Objective().Value(),
            "s": s_val,
            "a": a_val,
        }

    # ------------------------------------------------------------------
    # Pricing: primal "what if we add j" evaluation
    # ------------------------------------------------------------------
    def _pricing_scan(
        self,
        X_cases: np.ndarray,
        X_controls: np.ndarray,
        active_idx: np.ndarray,
        current_assignment_dist: np.ndarray,
        current_f1: float,
        current_f2: float,
        w: float,
        k_target: int,
        rng: np.random.Generator,
        candidate_subset: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[int], float, float, float]:
        """
        Scan all (or a random subset of) controls not in active_idx, and
        approximate the improvement from adding each candidate j.

        Parameters
        ----------
        current_assignment_dist : (P,) array
            d_i = distance from case i to its currently assigned control.
        current_f1 : float
            Current average pull term value.
        current_f2 : float
            Current average dispersion term value.
        candidate_subset : int, optional
            If not None, randomly subsample this many candidates to price;
            useful for very large N.

        Returns
        -------
        best_j : int or None
            Index in [0, N_controls) of best candidate to add; None if no
            improving candidate is found.
        best_obj : float
            Objective value AFTER adding best_j (approximate).
        best_f1, best_f2 : floats
            Corresponding decomposed terms.
        """
        N_controls = X_controls.shape[0]
        all_idx = np.arange(N_controls, dtype=int)
        mask_in = np.zeros(N_controls, dtype=bool)
        mask_in[active_idx] = True
        candidate_idx = all_idx[~mask_in]

        if candidate_subset is not None and candidate_subset < len(candidate_idx):
            candidate_idx = rng.choice(candidate_idx, size=candidate_subset, replace=False)

        if verbose:
            print(f"[Pricing] Evaluating {len(candidate_idx)} candidates")

        P = X_cases.shape[0]
        C = len(active_idx)

        # Precompute pairwise distances among active controls for f2
        if C > 1:
            D_nn_active = cdist(X_controls[active_idx], X_controls[active_idx], metric=self.distance_metric)
            # current f2 is average over pairs
            # we trust caller's current_f2, but we recompute number of pairs:
            num_pairs = C * (C - 1) / 2.0
        else:
            D_nn_active = None
            num_pairs = 0.0

        best_j = None
        best_obj = np.inf
        best_f1 = current_f1
        best_f2_out = current_f2

        # For each candidate j, compute:
        #   - new assignment distances d_i' = min(d_i, ||x_i - x_j||)
        #   - new f1' = avg(d_i')
        #   - new f2' = average pairwise distance among active ∪ {j}
        for j in candidate_idx:
            # distances from all cases to candidate control j
            d_pn_new = cdist(X_cases, X_controls[j:j+1, :], metric=self.distance_metric).reshape(P)
            d_new_all = np.minimum(current_assignment_dist, d_pn_new)
            f1_new = d_new_all.mean()

            # dispersion: distances from candidate j to active controls
            if C > 0:
                d_nn_new = cdist(X_controls[active_idx], X_controls[j:j+1, :],
                                 metric=self.distance_metric).reshape(C)
                # total sum of pairwise distances among active set:
                if C > 1:
                    sum_pairs_active = D_nn_active[np.triu_indices(C, k=1)].sum()
                else:
                    sum_pairs_active = 0.0
                # add distances between j and each active control
                sum_pairs_new = sum_pairs_active + d_nn_new.sum()
                num_pairs_new = num_pairs + C  # j paired with each of C existing controls
                f2_new = sum_pairs_new / num_pairs_new
            else:
                # no dispersion if j is the first control
                f2_new = 0.0

            obj_new = w * f1_new - (1.0 - w) * f2_new

            if obj_new < best_obj - 1e-9:
                best_obj = obj_new
                best_j = int(j)
                best_f1 = float(f1_new)
                best_f2_out = float(f2_new)

        return best_j, best_obj, best_f1, best_f2_out

    # ------------------------------------------------------------------
    # High-level CG driver
    # ------------------------------------------------------------------
    def solve_pushpull_cg(
        self,
        df_cases: pd.DataFrame,
        df_controls: pd.DataFrame,
        exclude_cols_matching: Optional[List[str]],
        final_ratio: float,
        verbose: bool = True,
        candidate_subset_per_iter: Optional[int] = None,
        warm_start_active_idx: Optional[np.ndarray] = None,   # <-- NEW

    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Column generation driver for push–pull sampling.

        Parameters
        ----------
        df_cases, df_controls : DataFrame
            Minority and majority data.
        exclude_cols_matching : list[str] or None
            Columns to exclude from distance computations (IDs, labels, etc.).
        final_ratio : float
            Target majority:minority ratio (k = final_ratio * |cases|).
        candidate_subset_per_iter : int or None
            If not None, random subset size for pricing at each iteration
            (stochastic approximate CG).

        Returns
        -------
        undersampled : DataFrame
            Concatenation of all cases and selected controls.
        info : dict
            Diagnostic information: active control indices, objective trace, etc.
        """
        rng = np.random.default_rng(self.random_state)

        X_cases, _ = self._extract_X(df_cases, exclude_cols=exclude_cols_matching)
        X_controls, _ = self._extract_X(df_controls, exclude_cols=exclude_cols_matching)

        P = X_cases.shape[0]
        N_controls = X_controls.shape[0]
        k_target = int(round(final_ratio * P))
        k_target = max(1, min(k_target, N_controls))

        if verbose:
            print(f"[PushPullCG] |cases|={P}, |controls|={N_controls}, k_target={k_target}")

       # --- Initialization: warm start or random seed set ---
        if warm_start_active_idx is not None:
            # Use the provided active set
            active_idx = np.array(warm_start_active_idx, dtype=int)

            # Safety: remove invalid indices (should not happen)
            active_idx = active_idx[(active_idx >= 0) & (active_idx < N_controls)]
            active_idx = np.unique(active_idx)

            if verbose:
                print(f"[Init] Warm-start active controls = {len(active_idx)} (k_target={k_target})")

        else:
            # Default random initialization
            init_size = min(self.init_size, N_controls)
            active_idx = rng.choice(np.arange(N_controls), size=init_size, replace=False)
            active_idx = np.unique(active_idx)

            if verbose:
                print(f"[Init] Random active controls = {len(active_idx)} (k_target={k_target})")

        obj_trace = []
        f1_trace = []
        f2_trace = []

        # --- Main CG loop (primal-style pricing) ---
        # Continue until we have enough controls OR max_iter reached OR no improvements
        max_iter_effective = max(self.max_iter, k_target - self.init_size + 10)  # Ensure we can reach k_target
        for it in range(max_iter_effective):
            C = len(active_idx)
            if C == 0:
                # Should not happen with our initialization
                raise RuntimeError("Empty active set in CG loop")

            # Distances between cases and active controls
            D_pn_active = cdist(X_cases, X_controls[active_idx], metric=self.distance_metric)
            # Check for invalid values
            if np.any(np.isnan(D_pn_active)) or np.any(np.isinf(D_pn_active)):
                raise RuntimeError("Distance matrix D_pn_active contains NaN or Inf values")
            
            # Distances among active controls
            if C > 1:
                D_nn_active = cdist(X_controls[active_idx], X_controls[active_idx],
                                    metric=self.distance_metric)
                if np.any(np.isnan(D_nn_active)) or np.any(np.isinf(D_nn_active)):
                    raise RuntimeError("Distance matrix D_nn_active contains NaN or Inf values")
            else:
                D_nn_active = np.zeros((C, C), dtype=float)

            # Solve RMP LP on active set
            rmp = self._solve_master_lp(D_pn_active, D_nn_active, k=k_target, w=self.w, verbose=verbose)
            obj_val = rmp["objective"]
            obj_trace.append(obj_val)

            # Derive "current assignment" distances from LP solution:
            # for each case, we compute expected distance under a_ij, but for
            # the primal pricing we approximate by nearest active control.
            # (If LP solution is close to integral, these coincide.)
            assign_probs = rmp["a"]          # (P,C)
            # expected cost per i:
            current_assignment_dist = (assign_probs * D_pn_active).sum(axis=1)
            current_f1 = current_assignment_dist.mean()

            # current dispersion term (approx)
            if C > 1:
                pair_mask = np.triu(np.ones((C, C), dtype=bool), k=1)
                sum_pairs = D_nn_active[pair_mask].sum()
                num_pairs = pair_mask.sum()
                current_f2 = sum_pairs / num_pairs
            else:
                current_f2 = 0.0

            f1_trace.append(current_f1)
            f2_trace.append(current_f2)

            if verbose:
                print(f"[Iter {it}] RMP obj={obj_val:.4f}, f1={current_f1:.4f}, f2={current_f2:.4f}, |J|={C}")

            # Stop early if we've already reached k_target active controls;
            # adding more will force some s_j to drop below 1 in the LP, but
            # our final rounding will just take top-k anyway.
            if C >= k_target and it > 0:
                if verbose:
                    print("[CG] Active set size reached k_target; stopping CG iterations.")
                break

            # Pricing step: look for a new control j that reduces objective
            best_j, best_obj, best_f1, best_f2 = self._pricing_scan(
                X_cases=X_cases,
                X_controls=X_controls,
                active_idx=active_idx,
                current_assignment_dist=current_assignment_dist,
                current_f1=current_f1,
                current_f2=current_f2,
                w=self.w,
                k_target=k_target,
                rng=rng,
                candidate_subset=candidate_subset_per_iter,
                verbose=verbose,
            )

            # If no improving column found, but we still need more controls, add a random one
            if best_j is None:
                if verbose:
                    print("[CG] No improving column found in pricing; stopping.")
                break

            if verbose:
                print(f"[Pricing] Adding control j={best_j} (approx new obj={best_obj:.4f})")

            # Add best_j to active set
            active_idx = np.unique(np.concatenate([active_idx, np.array([best_j], dtype=int)]))

        # --- Final rounding: pick top-k_target controls by s_j value ---
        # Re-solve RMP one last time on the final active set, then use s-values.
        C_final = len(active_idx)
        D_pn_active = cdist(X_cases, X_controls[active_idx], metric=self.distance_metric)
        if C_final > 1:
            D_nn_active = cdist(X_controls[active_idx], X_controls[active_idx],
                                metric=self.distance_metric)
        else:
            D_nn_active = np.zeros((C_final, C_final), dtype=float)

        rmp_final = self._solve_master_lp(D_pn_active, D_nn_active, k=k_target, w=self.w, verbose=verbose)
        s_final = rmp_final["s"]  # length C_final

        # Take indices of k_target largest s_j (or all if fewer available)
        k_select = min(k_target, C_final)
        order = np.argsort(-s_final)
        chosen_local = order[:k_select]
        chosen_controls_idx = active_idx[chosen_local]

        if verbose:
            print(f"[Final] Selected {len(chosen_controls_idx)} controls out of {N_controls}")

        undersampled = pd.concat(
            [df_cases, df_controls.iloc[chosen_controls_idx]],
            ignore_index=True,
        )

        info = {
            "chosen_controls_idx": chosen_controls_idx,
            "active_idx": active_idx,
            "obj_trace": obj_trace,
            "f1_trace": f1_trace,
            "f2_trace": f2_trace,
            "final_lp_obj": rmp_final["objective"],
        }

        return undersampled, info
