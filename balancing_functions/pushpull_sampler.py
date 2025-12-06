#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bilevel undersampling module:
 - Inner problem: distance-based selection (p-median style)
 - Outer problem: diversity maximization (facility-location)

Inherits from LexicographicCaseControlResampler to reuse
all preprocessing and solver infrastructure.

ADDED ALSO : 
 - Biobjective Double Facility sampler with simple scaled objective (just 1/max(f) etc) --> for extreme point scaling see DoubleFacilitySampler
 - Biobjective Push Pull sampler with simple scaled objective (just 1/max(f) etc)
"""

import numpy as np
import pandas as pd
from tqdm import trange
from scipy.spatial.distance import cdist
from ortools.linear_solver import pywraplp
from sklearn.metrics import pairwise_distances

# import parent class
from balancing_functions.optimal_match_lexicographic import LexicographicCaseControlResampler
import time
from tqdm import tqdm


def evaluate_terms_case_and_coverage(P, C, NN_case, D_pn, a, NN_ctrl, D_nn, r):
    """Compute raw term_1 and term_2 under the current MILP solution."""
    # Case–control term
    term1 = 0.0
    for i in range(P):
        for j in NN_case[i]:
            term1 += D_pn[i, j] * a[i][j].solution_value()

    # Coverage/diversity term
    term2 = 0.0
    for c in range(C):
        for j in NN_ctrl[c]:
            term2 += D_nn[c, j] * r[c][j].solution_value()

    return term1, term2

class PushPullSampler(LexicographicCaseControlResampler):
    """
    One-stage multi-objective with extreme point scaling of cost terms 
    """   

    def prune_nodes_distance_stratified(
        self, 
        X_cases, 
        X_controls, 
        df_controls,
        top_k_per_case=50,
        K_outer=2000,
        stratified=False,       # <-- NEW flag
    ):
        """
        Distance-based node pruning.
        If stratified=True, we preserve approximate global proportions of strata.
        If stratified=False, we return the K_outer closest nodes overall.
        """
        # ---- Step 1: Case→Control expansion (top-k per case) ----
        D = cdist(X_cases, X_controls)
        nearest = np.argpartition(D, top_k_per_case, axis=1)[:, :top_k_per_case]
        candidate_indices = sorted(set(nearest.flatten()))  # list of original control indices
        
        # ---- Step 2: Compute minimal distance score for these candidates ----
        D_sub = cdist(X_cases, X_controls[candidate_indices])
        min_dist = D_sub.min(axis=0)

        # ---- Step 3: Build dfC with RESET INDEX ----
        dfC = df_controls.iloc[candidate_indices].copy()
        dfC = dfC.reset_index(drop=False)  # keep original index
        dfC["min_dist"] = min_dist
        # Sort all candidates purely by minimal distance
        df_sorted = dfC.sort_values("min_dist")
        K_eff = min(K_outer, len(df_sorted))
        selected = dfC.nsmallest(K_eff, "min_dist").index.tolist()
        return selected
    
    def _topk_prune_case_to_control(self,X_cases, X_controls, L):
        """
        Returns list of lists:
            NN_case[i] = array of L nearest control indices for case i.
        """
        D = pairwise_distances(X_cases, X_controls, metric="euclidean")
        # Take smallest L distances per case
        nearest = np.argpartition(D, L, axis=1)[:, :L]
        # Optional: sort each L-block for stability
        sorted_idx = np.argsort(D[np.arange(D.shape[0])[:,None], nearest], axis=1)
        NN_case = nearest[np.arange(nearest.shape[0])[:,None], sorted_idx]

        return NN_case, D

    def compute_pushpull_extreme_points(self, X_cases, X_controls, candidate_indices, top_k, verbose=True):
        """
        Compute true extreme points (f1_min, f1_max, f2_min, f2_max)
        by solving 4 MILPs with fixed objective directions.
        """

        # Helper: solve push-pull MILP with objective = f1 or f2 only
        def solve_for_objective(maximize_f1=False, maximize_f2=False):
            return self.solve_pushpull_MILP(
                X_cases, X_controls, candidate_indices,
                w=None,       # ignore weights (special mode)
                top_k=top_k,
                objective_mode=("f1max" if maximize_f1 else
                                "f1min" if not maximize_f1 and not maximize_f2 else
                                "f2max" if maximize_f2 else
                                "f2min"),
                return_only_objective=True,
                verbose=verbose
            )

        f1_min = solve_for_objective(maximize_f1=False)
        f1_max = solve_for_objective(maximize_f1=True)

        f2_min = solve_for_objective(maximize_f2=False)
        f2_max = solve_for_objective(maximize_f2=True)

        return {
            "f1_min": float(f1_min),
            "f1_max": float(f1_max),
            "f2_min": float(f2_min),
            "f2_max": float(f2_max),
        }
  


    def _topk_farthest_control_pairs(self, X_C, L_pairs=20, verbose=True):
        """
        For dispersion: build a sparse set of FAR pairs (j,k) among controls.

        Parameters
        ----------
        X_C : np.ndarray
            Feature matrix for candidate controls (C x d)
        L_pairs : int
            Number of far neighbors to keep per control.

        Returns
        -------
        pairs : list[(int,int)]
            Unique (j,k) index pairs with j < k.
        D_nn : np.ndarray
            Full control-control distance matrix (C x C).
        """
        from sklearn.metrics import pairwise_distances

        C = X_C.shape[0]
        D_nn = pairwise_distances(X_C, X_C, metric="euclidean")

        pairs_set = set()
        # for each control j, find L_pairs farthest others
        for j in range(C):
            row = D_nn[j]
            # indices of L largest distances (excluding j itself)
            # argpartition on -row yields largest L indices
            L_eff = min(L_pairs, C - 1)
            far_idx = np.argpartition(-row, L_eff)[:L_eff]
            # drop self if present
            far_idx = far_idx[far_idx != j]
            for k in far_idx:
                j_min, j_max = (j, k) if j < k else (k, j)
                pairs_set.add((j_min, j_max))

        pairs = sorted(list(pairs_set))
        if verbose:
            print(f"[Dispersion] Using {len(pairs)} far pairs among {C} controls (L_pairs={L_pairs})")

        return pairs, D_nn
    
    
    def solve_pushpull_MILP(
        self,
        X_cases,
        X_controls,
        candidate_indices,
        final_ratio,
        top_k_case_ctrl,
        L_pairs,
        objective_mode="weighted",   # "f1min", "f1max", "f2min", "f2max", "weighted"
        w=0.5,
        ext=None,                    # required only for weighted case
        return_only_objective=False,
        verbose=True,
    ):
        """
        Raw Push–Pull MILP solver.
        Supports 5 modes:
            f1min, f1max, f2min, f2max → extreme points
            weighted → normalized weighted-sum using ext

        This is the underlying solver.
        """

        from ortools.linear_solver import pywraplp
        
        # Prepare data
        X_C = X_controls[candidate_indices]
        P = X_cases.shape[0]
        C = len(candidate_indices)
        k = int(np.clip(final_ratio * P, 1, C))

        # Nearest neighbors: case → control
        NN_case, D_pn = self._topk_prune_case_to_control(X_cases, X_C, top_k_case_ctrl)

        # Farthest pairs: control ↔ control
        pairs, D_nn = self._topk_farthest_control_pairs(X_C, L_pairs=L_pairs, verbose=verbose)

        # Build SCIP model
        solver = pywraplp.Solver.CreateSolver("SCIP")

        # Selection
        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]

        # Assignment (binary recommended)
        a = [
            { j: solver.BoolVar(f"a[{i},{j}]") for j in NN_case[i] }
            for i in range(P)
        ]

        # Diversity var
        z = { (j,k): solver.BoolVar(f"z[{j},{k}]") for (j,k) in pairs }

        # Constraints
        solver.Add(sum(s) == k)

        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for (j,k), z_var in z.items():
            solver.Add(z_var <= s[j])
            solver.Add(z_var <= s[k])

        # Raw objective components
        f1 = solver.Sum(D_pn[i,j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(D_nn[j,k] * z[(j,k)] for (j,k) in pairs)

        # --- Objective selection ---
        if objective_mode == "f1min":
            solver.Minimize(f1)

        elif objective_mode == "f1max":
            solver.Maximize(f1)

        elif objective_mode == "f2min":
            solver.Minimize(f2)

        elif objective_mode == "f2max":
            solver.Maximize(f2)

        elif objective_mode == "weighted":
            assert ext is not None, "ext must be provided for weighted objective"
            f1_min, f1_max = ext["f1_min"], ext["f1_max"]
            f2_min, f2_max = ext["f2_min"], ext["f2_max"]
            range_f1 = max(f1_max - f1_min, 1e-8)
            range_f2 = max(f2_max - f2_min, 1e-8)

            tilde_f1 = (f1 - f1_min) / range_f1
            tilde_f2 = (f2 - f2_min) / range_f2

            solver.Minimize(w * tilde_f1 + (1 - w) * tilde_f2)

        else:
            raise ValueError("Invalid objective_mode")

        if verbose:
            print("[MILP-Diverse] Variables:", solver.NumVariables())
            print("[MILP-Diverse] Constraints:", solver.NumConstraints())

        solver.SetTimeLimit(1200000)   # 300 sec = 5 minutes

        # SCIP internal parameters (seconds)
        solver.SetSolverSpecificParametersAsString(r"""
        timing/clocktype = 1
        limits/time = 1200
        limits/softtime = 1200
        display/verblevel = 4
        """)
        status = solver.Solve()

        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            raise RuntimeError(f"Unified Diverse MILP failed with status {status}")

        if return_only_objective:
            return float(solver.Objective().Value())

        selected = [candidate_indices[j] for j in range(C) if s[j].solution_value() > 0.5]

        return {
            "selected": selected,
            "f1": float(solver.Value(f1)),
            "f2": float(solver.Value(f2)),
            "status": int(status),
        }

    def compute_pushpull_extreme_points(
        self, X_cases, X_controls, candidate_indices,
        final_ratio, top_k_case_ctrl, L_pairs,
        verbose=True
    ):
        """
        Compute extreme points:
            f1_min, f1_max, f2_min, f2_max
        using 4 MILP solves of the push–pull model.
        """

        f1_min = self.solve_pushpull_MILP(
            X_cases, X_controls, candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            objective_mode="f1min",
            return_only_objective=True,
            verbose=verbose
        )

        f1_max = self.solve_pushpull_MILP(
            X_cases, X_controls, candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            objective_mode="f1max",
            return_only_objective=True,
            verbose=verbose
        )

        f2_min = self.solve_pushpull_MILP(
            X_cases, X_controls, candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            objective_mode="f2min",
            return_only_objective=True,
            verbose=verbose
        )

        f2_max = self.solve_pushpull_MILP(
            X_cases, X_controls, candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            objective_mode="f2max",
            return_only_objective=True,
            verbose=verbose
        )

        return {
            "f1_min": float(f1_min),
            "f1_max": float(f1_max),
            "f2_min": float(f2_min),
            "f2_max": float(f2_max),
        }

    def solve_pushpull_normalized_MILP(
        self,
        X_cases,
        X_controls,
        candidate_indices,
        ext,                # dict with f1_min, f1_max, f2_min, f2_max
        final_ratio: float,
        w: float = 0.5,
        top_k_case_ctrl: int = 20,
        L_pairs: int = 20,
        verbose: bool = True,
    ):
        """
        Unified MILP with:
        - Term 1: case→control distance (closeness)
        - Term 2: pairwise control–control distance (diversity / dispersion)

        We select k = final_ratio * P controls from candidate_indices.
        """
        from ortools.linear_solver import pywraplp
                # Unpack normalization constants
        f1_min, f1_max = ext["f1_min"], ext["f1_max"]
        f2_min, f2_max = ext["f2_min"], ext["f2_max"]
        range_f1 = max(f1_max - f1_min, 1e-8)
        range_f2 = max(f2_max - f2_min, 1e-8)


        # Restrict controls to candidate pool
        X_C = X_controls[candidate_indices]
        P = X_cases.shape[0]
        C = len(candidate_indices)

        # Number of controls to select
        k = int(np.clip(final_ratio * P, 1, C))
        if verbose:
            print(f"[Unified-Diverse] P={P}, C={C}, final_ratio={final_ratio} → k={k} controls")

        # --- Case→Control: nearest neighbors for assignment a[i,j] ---
        NN_case, D_pn = self._topk_prune_case_to_control(
            X_cases, X_C, top_k_case_ctrl
        )  # D_pn shape (P, C_eff == C)

        # --- Control→Control: farthest pairs for z[j,k] ---
        pairs, D_nn = self._topk_farthest_control_pairs(
            X_C, L_pairs=L_pairs, verbose=verbose
        )

        # Build solver
        solver = pywraplp.Solver.CreateSolver("SCIP")

        # Selection vars: s[j] ∈ {0,1}
        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]

        # Case assignment vars: a[i][j] ∈ [0,1] only for j in NN_case[i]
        # Use BoolVar for tight relaxation (as you just discovered)
        a = [{j: solver.BoolVar(f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]

        # Pairwise co-selection vars: z[(j,k)] ∈ [0,1]
        z = { (j,k): solver.BoolVar(f"z[{j},{k}]") for (j,k) in pairs }

        # --- Constraints ---

        # Total selected controls
        solver.Add(sum(s) == k)

        # Case coverage: each case assigned to exactly one selected control
        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        # Pairwise linkage: z[j,k] ≤ s[j], z[j,k] ≤ s[k]
        for (j, k), z_var in z.items():
            solver.Add(z_var <= s[j])
            solver.Add(z_var <= s[k])

       
        # Raw objectives:
        f1 = solver.Sum(D_pn[i,j] * a[i][j] 
                        for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(D_nn[j, k] * z[(j, k)] for (j, k) in pairs)


        # Normalized objectives
        tilde_f1 = (f1 - f1_min) / range_f1
        tilde_f2 = (f2 - f2_min) / range_f2

        obj = w * tilde_f1 + (1 - w) * tilde_f2
        solver.Minimize(obj)



        if verbose:
            print("[MILP-Diverse] Variables:", solver.NumVariables())
            print("[MILP-Diverse] Constraints:", solver.NumConstraints())

        solver.SetTimeLimit(1200000)   # 300 sec = 5 minutes

        # SCIP internal parameters (seconds)
        solver.SetSolverSpecificParametersAsString(r"""
        timing/clocktype = 1
        limits/time = 1200
        limits/softtime = 1200
        display/verblevel = 4
        """)
        status = solver.Solve()

        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            raise RuntimeError(f"Unified Diverse MILP failed with status {status}")

        if verbose:
            # ---- CASE TERM ----
            raw_cases = 0.0
            for i in range(P):
                for j in NN_case[i]:
                    raw_cases += D_pn[i, j] * a[i][j].solution_value()

            # ---- DISPERSION TERM ----
            raw_disp = 0.0
            for (j, k) in pairs:
                raw_disp += D_nn[j, k] * z[(j, k)].solution_value()

            print("[Objective decomposition]")
            print(f"  term_cases_raw    = {raw_cases:.4f}")
            print(f"  term_disp_raw     = {raw_disp:.4f}")
            print(f"  weighted_obj      = {solver.Objective().Value():.4f}")
                # Return selected controls (candidate_indices indexing)
        selected_controls = [
            candidate_indices[j] for j in range(C) if s[j].solution_value() > 0.5
        ]

        return {
                "selected": selected_controls,
                "raw_objectives": {
                    "f1": float(solver.Value(f1)),
                    "f2": float(solver.Value(f2)),
                }
            }

    def solve_pushpull_normalized_MILP(
        self,
        df_cases,
        df_controls,
        exclude_cols_matching,
        final_ratio,
        w, # weight in the multi-objective cost
        K_factor, # inner pruning ratio
        top_k_case_ctrl,
        L_pairs,
        verbose=True
    ):
        """
        Full wrapper for normalized push–pull sampling.
        1. Preprocess cases/controls into feature space.
        2. Prune controls using distance-based strategy.
        3. Compute extreme points (min/max f1,f2).
        4. Solve weighted normalized MILP to select control subset.
        5. Return undersampled dataset.
        """

        X_cases, X_controls = self.get_preprocessed_control_case_features(
            df_cases, df_controls, exclude_cols_matching, verbose=verbose
        )
        P = X_cases.shape[0]

        # Optional: inner node pruning (e.g., keep K_factor * P closest controls globally)
        # Or use your existing prune_nodes_by_distance function
        K_outer = int(K_factor * P)
        K_outer = min(K_outer, X_controls.shape[0])
        if verbose:
            print(f"[INNER] Selecting {K_outer} candidate controls out of {X_controls.shape[0]}")

        candidate_indices = self.prune_nodes_distance(
           X_cases, X_controls, df_controls,top_k_per_case=top_k_case_ctrl, K_outer = K_outer
)
        candidate_indices = list(map(int, candidate_indices))
        if verbose:
            print(f"[INNER] → Kept {len(candidate_indices)} controls")

        ext = self.compute_pushpull_extreme_points(
            X_cases, X_controls, candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            verbose=verbose
        )

        result = self.solve_pushpull_MILP(
            X_cases, X_controls, candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            objective_mode="weighted",
            w=w,
            ext=ext,
            verbose=verbose
        )
        selected_ctrls = result["selected"]

        # --- Step 5: Build undersampled dataset ---
        undersampled = pd.concat(
            [df_cases, df_controls.iloc[selected_ctrls]],
            ignore_index=True
        )

        return undersampled, result, ext
