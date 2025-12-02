#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bilevel undersampling module:
 - Inner problem: distance-based selection (p-median style)
 - Outer problem: diversity maximization (facility-location)

Inherits from LexicographicCaseControlResampler to reuse
all preprocessing and solver infrastructure.
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

def compute_extreme_points_double_facility(
        solve_fn,   # your MILP solver but without scaling or w
        X_cases, X_controls, candidate_indices,
        top_k_case_ctrl, top_k_ctrl_ctrl, final_ratio,
        L_pairs=20,  # for f3 (pairwise diversity)
    ):
    results = {}

    # ---- 1) f1_min: minimize case-control distance
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        L_pairs=L_pairs,
        force_direction="min_f1",
    )
    results["f1_min"], _, _ = sol["raw"]

    # ---- 2) f1_max: maximize case-control distance
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        L_pairs=L_pairs,
        force_direction="max_f1",
    )
    results["f1_max"], _, _ = sol["raw"]

    # ---- 3) f2_min: minimize coverage term
    sol = solve_fn(
        X_cases, X_controls, candidate_indices, final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        L_pairs=L_pairs,
        force_direction="min_f2",
    )
    _, results["f2_min"], _ = sol["raw"]

    # ---- 4) f2_max: maximize coverage
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        L_pairs=L_pairs,
        force_direction="max_f2",
    )
    _, results["f2_max"], _ = sol["raw"]

    # ---- 5) f3_min: minimize pairwise diversity (minimize distances)
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        L_pairs=L_pairs,
        force_direction="min_f3",
    )
    _, _, results["f3_min"] = sol["raw"]

    # ---- 6) f3_max: maximize pairwise diversity (maximize distances)
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        L_pairs=L_pairs,
        force_direction="max_f3",
    )
    _, _, results["f3_max"] = sol["raw"]

    return results

def evaluate_terms_case_and_coverage(P, C, NN_case, D_pn, a, NN_ctrl, S_nn, r, pairs=None, D_pairs=None, z=None):
    """Compute raw term_1, term_2, and term_3 under the current MILP solution."""
    # Case–control term
    term1 = 0.0
    for i in range(P):
        for j in NN_case[i]:
            term1 += D_pn[i, j] * a[i][j].solution_value()

    # Coverage/diversity term (similarity-based)
    term2 = 0.0
    for c in range(C):
        for j in NN_ctrl[c]:
            term2 += S_nn[c, j] * r[c][j].solution_value()

    # Pairwise diversity term (distance-based)
    term3 = 0.0
    if pairs is not None and D_pairs is not None and z is not None:
        for (j, k) in pairs:
            term3 += D_pairs[j, k] * z[(j, k)].solution_value()

    return term1, term2, term3

class TriobjectiveSampler(LexicographicCaseControlResampler):

    def prune_nodes_distance(
        self, 
        X_cases, 
        X_controls, 
        df_controls,
        top_k_per_case=50,
        K_outer=2000,
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
        selected = dfC.nsmallest(K_eff, "min_dist").index.tolist() # or df_sorted.iloc[:K_eff]["index"].tolist()?
        return selected
    
    def _topk_prune_control_to_control(self,X_C, L):
        """
        Returns list:
            NN_ctrl[c] = array of L nearest control indices for control c.
        """
        D_nn = pairwise_distances(X_C, X_C, metric="euclidean")
        nearest = np.argpartition(D_nn, L, axis=1)[:, :L]

        # Sort neighbors for stability
        sorted_idx = np.argsort(D_nn[np.arange(D_nn.shape[0])[:,None], nearest], axis=1)
        NN_ctrl = nearest[np.arange(nearest.shape[0])[:,None], sorted_idx]

        return NN_ctrl, D_nn

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

    def _topk_farthest_control_pairs(self, X_C, L_pairs=20, verbose=True):
        """
        For dispersion: build a sparse set of FAR pairs (j,k) among controls.
        This is used for the pairwise diversity term (f3).

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


    def solve_triobjective_MILP_raw(
        self,
        X_cases, X_controls, candidate_indices,
        final_ratio,
        top_k_case_ctrl, top_k_ctrl_ctrl,
        L_pairs=20,
        force_direction=None    # "min_f1", "max_f1", "min_f2", "max_f2", "min_f3", "max_f3"
    ):
        """
        Solve the MILP once with a *single-objective* (f1, f2, or f3),
        return raw f1, f2, f3 values and selected controls.

        NOTE: w is NOT used here. This function is ONLY for the
        extreme point calibrations in normalization.
        """

        P = X_cases.shape[0]
        X_C = X_controls[candidate_indices]
        C = len(candidate_indices)
        k = int(final_ratio * P)

        # --- Pruning ---
        NN_case, D_pn = self._topk_prune_case_to_control(
            X_cases, X_C, top_k_case_ctrl
        )
        NN_ctrl, D_nn = self._topk_prune_control_to_control(
            X_C, top_k_ctrl_ctrl
        )

        sigma = D_nn.std() + 1e-8
        S_nn = np.exp(-D_nn / sigma)

        # --- Pairwise diversity: farthest pairs ---
        pairs, D_pairs = self._topk_farthest_control_pairs(X_C, L_pairs=L_pairs, verbose=False)

        # --- Build MILP ---
        solver = pywraplp.Solver.CreateSolver("SCIP")

        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]
        a = [{j: solver.NumVar(0,1,f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]
        r = [{j: solver.NumVar(0,1,f"r[{c},{j}]") for j in NN_ctrl[c]} for c in range(C)]
        z = {(j, k): solver.NumVar(0, 1, f"z[{j},{k}]") for (j, k) in pairs}

        solver.Add(sum(s) == k)

        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for c in range(C):
            solver.Add(sum(r[c][j] for j in NN_ctrl[c]) == 1)
            for j in NN_ctrl[c]:
                solver.Add(r[c][j] <= s[j])

        # Pairwise linkage: z[j,k] ≤ s[j], z[j,k] ≤ s[k]
        for (j, k), z_var in z.items():
            solver.Add(z_var <= s[j])
            solver.Add(z_var <= s[k])

        # --- Build f1, f2, f3 ---
        f1 = solver.Sum(D_pn[i,j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(S_nn[c,j] * r[c][j] for c in range(C) for j in NN_ctrl[c])
        f3 = solver.Sum(D_pairs[j, k] * z[(j, k)] for (j, k) in pairs)

        # --- Choose objective for extreme-point solves ---
        if force_direction == "min_f1":
            obj = f1
        elif force_direction == "max_f1":
            obj = -f1
        elif force_direction == "min_f2":
            obj = f2
        elif force_direction == "max_f2":
            obj = -f2
        elif force_direction == "min_f3":
            obj = f3
        elif force_direction == "max_f3":
            obj = -f3
        else:
            raise ValueError("force_direction must be in {min_f1, max_f1, min_f2, max_f2, min_f3, max_f3}")

        solver.Minimize(obj)
        solver.SetTimeLimit(3000)
        solver.Solve()

        # --- Evaluate raw f1, f2, f3 (using fixed helper) ---
        raw_f1, raw_f2, raw_f3 = evaluate_terms_case_and_coverage(
            P, C,
            NN_case, D_pn, a,
            NN_ctrl, S_nn, r,
            pairs=pairs, D_pairs=D_pairs, z=z
        )

        selected = [
            candidate_indices[j] for j in range(C)
            if s[j].solution_value() > 0.5
        ]

        return {
            "selected": selected,
            "raw": (raw_f1, raw_f2, raw_f3)
        }

    def solve_triobjective_MILP(
        self,
        X_cases, X_controls, candidate_indices,
        w, final_ratio,
        top_k_case_ctrl, top_k_ctrl_ctrl,
        ext,                    # dictionary with f1_min, f1_max, f2_min, f2_max, f3_min, f3_max
        L_pairs=20,
        v=0.0,                  # weight for f3 (diversity): v=0 means no f3, v=1 means only f3
        verbose=True
    ):
        """
        Solve the MILP with three objectives:
            minimize  (1-v) * [w * \tilde f1(x) - (1-w) * \tilde f2(x)]  -  v * \tilde f3(x)
        
        where:
        - f1: case-control distance (minimize)
        - f2: control-control coverage/similarity (maximize, so minimize -f2)
        - f3: pairwise control-control diversity (maximize, so minimize -f3)
        
        Parameters:
        - w: weight for f1 vs f2 (when v=0, this is the original double facility)
        - v: weight for f3 (diversity term). v=0 gives original double facility, v>0 adds diversity
        """

        # ----------------------------------------------------------
        # Unpack normalization ranges
        # ----------------------------------------------------------
        f1_min = ext["f1_min"]
        f1_max = ext["f1_max"]
        f2_min = ext["f2_min"]
        f2_max = ext["f2_max"]
        f3_min = ext["f3_min"]
        f3_max = ext["f3_max"]

        range_f1 = max(f1_max - f1_min, 1e-8)
        range_f2 = max(f2_max - f2_min, 1e-8)
        range_f3 = max(f3_max - f3_min, 1e-8)

        P = X_cases.shape[0]
        X_C = X_controls[candidate_indices]
        C = len(candidate_indices)
        k = int(final_ratio * P)

        # ----------------------------------------------------------
        # Local pruning
        # ----------------------------------------------------------
        NN_case, D_pn = self._topk_prune_case_to_control(X_cases, X_C, top_k_case_ctrl)
        NN_ctrl, D_nn = self._topk_prune_control_to_control(X_C, top_k_ctrl_ctrl)

        sigma = D_nn.std() + 1e-8
        S_nn = np.exp(-D_nn / sigma)

        # --- Pairwise diversity: farthest pairs ---
        pairs, D_pairs = self._topk_farthest_control_pairs(X_C, L_pairs=L_pairs, verbose=False)

        # ----------------------------------------------------------
        # MILP model
        # ----------------------------------------------------------
        solver = pywraplp.Solver.CreateSolver("SCIP")

        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]
        a = [{j: solver.NumVar(0,1,f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]
        r = [{j: solver.NumVar(0,1,f"r[{c},{j}]") for j in NN_ctrl[c]} for c in range(C)]
        z = {(j, k): solver.NumVar(0, 1, f"z[{j},{k}]") for (j, k) in pairs}

        solver.Add(sum(s) == k)

        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for c in range(C):
            solver.Add(sum(r[c][j] for j in NN_ctrl[c]) == 1)
            for j in NN_ctrl[c]:
                solver.Add(r[c][j] <= s[j])

        # Pairwise linkage: z[j,k] ≤ s[j], z[j,k] ≤ s[k]
        for (j, k), z_var in z.items():
            solver.Add(z_var <= s[j])
            solver.Add(z_var <= s[k])

       # ----------------------------------------------------------
        # Raw objective terms f1, f2, f3
        # ----------------------------------------------------------
        f1 = solver.Sum(D_pn[i, j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(S_nn[c, j] * r[c][j] for c in range(C) for j in NN_ctrl[c])
        f3 = solver.Sum(D_pairs[j, k] * z[(j, k)] for (j, k) in pairs)

        # ----------------------------------------------------------
        # Normalized objective
        # ----------------------------------------------------------
        tilde_f1 = (f1 - f1_min) / range_f1
        tilde_f2 = (f2 - f2_min) / range_f2
        tilde_f3 = (f3 - f3_min) / range_f3

        # Combined objective: (1-v) * [w*f1 - (1-w)*f2] - v * f3
        # Note: we maximize f2 and f3, so we minimize -f2 and -f3
        obj_double_facility = w * tilde_f1 - (1 - w) * tilde_f2
        obj = (1 - v) * obj_double_facility - v * tilde_f3
        
        solver.Minimize(obj)

        solver.SetTimeLimit(3000)
        status = solver.Solve()

        if verbose:
            print("Solve status:", status)
            print("Normalized objective =", solver.Objective().Value())
            print(f"  w={w}, v={v}")

        # ----------------------------------------------------------
        # Return selected controls *and* f1, f2, f3
        # ----------------------------------------------------------
        return {
            "selected": [candidate_indices[j] for j in range(C)
                        if s[j].solution_value() > 0.5],
            "raw": evaluate_terms_case_and_coverage(
                        P, C,
                        NN_case, D_pn, a,
                        NN_ctrl, S_nn, r,
                        pairs=pairs, D_pairs=D_pairs, z=z
                )
        }
    
    def triobjective_milp_wrapper(
        self,
        df_cases,
        df_controls,
        exclude_cols_matching,
        final_ratio=1.0,
        w=0.5,
        v=0.0,                  # weight for f3 (diversity): v=0 means no diversity term
        top_k_case_ctrl=20,
        top_k_ctrl_ctrl=20,
        L_pairs=20,             # number of far pairs per control for f3
        K_factor=3.0,
        verbose=True
    ):
        """
        Wrapper for triple-objective optimization:
        - f1: case-control distance (minimize)
        - f2: control-control coverage/similarity (maximize)
        - f3: pairwise control-control diversity (maximize)
        
        Parameters:
        - w: weight for f1 vs f2 (0=only f2, 1=only f1)
        - v: weight for f3 diversity term (0=no diversity, 1=only diversity)
        """
        # ===== 1. Preprocess =====
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            df_cases, df_controls, exclude_cols_matching, verbose=verbose
        )
        P = X_cases.shape[0]

        # ===== 2. Prune =====
        K_outer = min(int(K_factor * P), X_controls.shape[0])

        if verbose:
            print(f"[INNER] Pruning to ~{K_outer} controls")

        candidate_indices = self.prune_nodes_distance(
            X_cases, X_controls, df_controls,
            top_k_per_case=50,         
            K_outer=K_outer        
            )
        candidate_indices = list(map(int, candidate_indices))

        if verbose:
            print(f"[INNER] → Kept {len(candidate_indices)} controls")

        # ===== 3. Compute extreme points =====
        if verbose:
            print("[EXTREME POINTS] Computing f1_min, f1_max, f2_min, f2_max, f3_min, f3_max...")

        ext = compute_extreme_points_double_facility(
            solve_fn=self.solve_double_facility_raw,
            X_cases=X_cases,
            X_controls=X_controls,
            candidate_indices=candidate_indices,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            final_ratio=final_ratio,
            L_pairs=L_pairs,
        )
        if verbose:
            print("[EXTREME POINTS]: ", ext)

        # ===== 4. Solve normalized objective =====
        sol = self.solve_double_facility_normalized_MILP(
            X_cases, X_controls, candidate_indices,
            w=w,
            v=v,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            L_pairs=L_pairs,
            ext=ext,
            verbose=verbose
        )
        selected = sol["selected"]
        raw_f1, raw_f2, raw_f3 = sol["raw"]
       
        # ===== 5. Return training set =====
        return pd.concat(
            [df_cases, df_controls.iloc[selected]], ignore_index=True
        ), {"w": w, "v": v, "f1": raw_f1, "f2": raw_f2, "f3": raw_f3}