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
    ):
    results = {}

    # ---- 1) f1_min: minimize case-control distance (w=1, minimize)
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        force_direction="min_f1",
    )
    results["f1_min"], _ = sol["raw"]
    print("Solved for F1 min")
  
    # ---- 2) f1_max: maximize case-control distance
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        force_direction="max_f1",
    )
    results["f1_max"], _ = sol["raw"]
    print("Solved for F1 max")

    # ---- 3) f2_min: minimize coverage term (w=0)
    sol = solve_fn(
        X_cases, X_controls, candidate_indices, final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        force_direction="min_f2",
    )
    _, results["f2_min"] = sol["raw"]
    print("Solved for F2 min")

    # ---- 4) f2_max: maximize coverage
    sol = solve_fn(
        X_cases, X_controls, candidate_indices,
        final_ratio=final_ratio,
        top_k_case_ctrl=top_k_case_ctrl,
        top_k_ctrl_ctrl=top_k_ctrl_ctrl,
        force_direction="max_f2",
    )
    _, results["f2_max"] = sol["raw"]
    print("Solved for F2 max")

    return results

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

class DoubleFacilitySampler(LexicographicCaseControlResampler):
    """
    Two-stage undersampling:
      1) Inner: minimize distance between majority and minority sets (select K candidate negatives)
      2) Outer: maximize diversity among those K candidates (select final k subset)
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

        # ==========================================
        # OPTIONAL STRATIFIED PRUNING
        # ==========================================

        if stratified:
            # ---- Step 4: Compute proportional quotas from the candidate pool ----
            p = dfC["cost_stratum_2018"].value_counts(normalize=True).to_dict()
            strata = sorted(p.keys())

            # Quotas sum to K_outer
            quota = {s: int(round(p[s] * K_outer)) for s in strata}

            # Fix rounding drift
            drift = K_outer - sum(quota.values())
            if drift != 0:
                s0 = max(quota, key=lambda s: quota[s])
                quota[s0] += drift
            selected = []
            # ---- Step 5: Stratified minimal-distance selection ----
            for s, g in dfC.groupby("cost_stratum_2018"):
                q = quota.get(s, 0)
                g_sorted = g.sort_values("min_dist")
                q = min(q, len(g_sorted))   # cannot exceed available in this stratum
                selected.extend(g_sorted.iloc[:q]["index"].tolist())  # original indices

        # ==========================================
        # NON-STRATIFIED PRUNING
        # ==========================================
        else:
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


    def solve_double_facility_raw(
        self,
        X_cases, X_controls, candidate_indices,
        final_ratio,
        top_k_case_ctrl, top_k_ctrl_ctrl,
        force_direction=None    # "min_f1", "max_f1", "min_f2", "max_f2"
    ):
        """
        Solve the MILP once with a *single-objective* (f1 or f2),
        return raw f1,f2 values and selected controls.

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

        # --- Build MILP ---
        solver = pywraplp.Solver.CreateSolver("CLP") # RELAXED VERSION TO SPEED UP

        s = [solver.NumVar(0,1,f"s[{j}]") for j in range(C)]# RELAXED VERSION TO SPEED UP
        a = [{j: solver.NumVar(0,1,f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]
        r = [{j: solver.NumVar(0,1,f"r[{c},{j}]") for j in NN_ctrl[c]} for c in range(C)]

        solver.Add(sum(s) == k)

        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for c in range(C):
            solver.Add(sum(r[c][j] for j in NN_ctrl[c]) == 1)
            for j in NN_ctrl[c]:
                solver.Add(r[c][j] <= s[j])

        # --- Build f1, f2 ---
        f1 = solver.Sum(D_pn[i,j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(S_nn[c,j] * r[c][j] for c in range(C) for j in NN_ctrl[c])

        # --- Choose objective for extreme-point solves ---
        if force_direction == "min_f1":
            obj = f1
        elif force_direction == "max_f1":
            obj = -f1
        elif force_direction == "min_f2":
            obj = f2
        elif force_direction == "max_f2":
            obj = -f2
        else:
            raise ValueError("force_direction must be in {min_f1, max_f1, min_f2, max_f2}")

        solver.Minimize(obj)
        solver.SetTimeLimit(1200000)   # 300 sec = 5 minutes 

        
        solver.Solve()

        # --- Evaluate raw f1,f2 (using fixed helper) ---
        raw_f1, raw_f2 = evaluate_terms_case_and_coverage(
            P, C,
            NN_case, D_pn, a,
            NN_ctrl, S_nn, r
        )

        #selected = [
       #     candidate_indices[j] for j in range(C)
       #     if s[j].solution_value() > 0.5
       # ]

        return {
            #"selected": selected,
            "raw": (raw_f1, raw_f2)
        }

    def solve_double_facility_normalized_MILP(
        self,
        X_cases, X_controls, candidate_indices,
        w, final_ratio,
        top_k_case_ctrl, top_k_ctrl_ctrl,
        ext,                    # dictionary with f1_min, f1_max, f2_min, f2_max
        verbose=True
    ):
        """
        Solve the MILP:
            minimize  w * \tilde f1(x)  -  (1-w) * \tilde f2(x)
        where f1 and f2 are the raw objective terms.

        This is the CORRECT normalization-based weighted-sum scalarization.
        Every w gives a different MILP solve.
        """

        # ----------------------------------------------------------
        # Unpack normalization ranges
        # ----------------------------------------------------------
        f1_min = ext["f1_min"]
        f1_max = ext["f1_max"]
        f2_min = ext["f2_min"]
        f2_max = ext["f2_max"]

        range_f1 = max(f1_max - f1_min, 1e-8)
        range_f2 = max(f2_max - f2_min, 1e-8)

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

        # ----------------------------------------------------------
        # MILP model
        # ----------------------------------------------------------
        solver = pywraplp.Solver.CreateSolver("SCIP")

        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]
        #a = [{j: solver.NumVar(0,1,f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]
        #r = [{j: solver.NumVar(0,1,f"r[{c},{j}]") for j in NN_ctrl[c]} for c in range(C)]
        a = [{j: solver.BoolVar(f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]
        r = [{j: solver.BoolVar(f"r[{c},{j}]") for j in NN_ctrl[c]} for c in range(C)]

        solver.Add(sum(s) == k)

        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for c in range(C):
            solver.Add(sum(r[c][j] for j in NN_ctrl[c]) == 1)
            for j in NN_ctrl[c]:
                solver.Add(r[c][j] <= s[j])

       # ----------------------------------------------------------
        # Raw objective terms f1 and f2
        # ----------------------------------------------------------
        f1 = solver.Sum(D_pn[i, j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(S_nn[c, j] * r[c][j] for c in range(C) for j in NN_ctrl[c])

        # ----------------------------------------------------------
        # Normalized objective
        # ----------------------------------------------------------
        tilde_f1 = (f1 - f1_min) / range_f1
        tilde_f2 = (f2 - f2_min) / range_f2

        obj = w * tilde_f1 - (1 - w) * tilde_f2
        # NEW: Add "tie-break" as Secondary objective: among all equivalent ones, pick the lexicographically smallest subset of controls.
        if final_ratio==1.0:
            print("Added tie break term into objective...")
            epsilon = 1e-4 / C
            tie_break = solver.Sum(j * s[j] for j in range(C))
            obj += epsilon * tie_break

        solver.Minimize(obj)

                # OR-Tools wrapper time limit (ms)
        solver.SetTimeLimit(2400000)   # 40 min

        # SCIP internal parameters (seconds)
        solver.SetSolverSpecificParametersAsString(r"""
        timing/clocktype = 1
        limits/time = 2400
        limits/softtime = 2400
        display/verblevel = 5
        display/freq = 1
        """)
        #presolving/maxrounds = 0
        #presolving/maxrestarts = 0

        #separating/maxrounds = 0
        status = solver.Solve()

        if verbose:
            print("Solve status:", status)
            print("Normalized objective =", solver.Objective().Value())

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(f"Solve failed with status {status}")
        # ----------------------------------------------------------
        # Return selected controls *and* f1,f2
        # ----------------------------------------------------------
        return {
            "selected": [candidate_indices[j] for j in range(C)
                        if s[j].solution_value() > 0.5],
            "raw": evaluate_terms_case_and_coverage(
                        P, C,
                        NN_case, D_pn, a,
                        NN_ctrl, S_nn, r
                )
        }
    
    def biobjective_double_facility_wrapper(
        self,
        df_cases,
        df_controls,
        exclude_cols_matching,
        w_range,
        results_dir = None,
        final_ratio=1.0,
        top_k_case_ctrl=20,
        top_k_ctrl_ctrl=20,
        K_factor=3.0,
        stratified=False,
        verbose=True,
    ):
        # ===== 1. Preprocess =====
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            df_cases, df_controls, exclude_cols_matching, verbose=verbose
        )
        P = X_cases.shape[0]

        # ===== 2. Prune =====
        K_outer = min(int(K_factor * P), X_controls.shape[0])

        if verbose:
            print(f"[INNER] Pruning to ~{K_outer} controls")

        candidate_indices = self.prune_nodes_distance_stratified(
            X_cases, X_controls, df_controls,
            top_k_per_case=top_k_case_ctrl,         
            K_outer=K_outer,
            stratified=stratified
        )
        candidate_indices = list(map(int, candidate_indices))

        if verbose:
            print(f"[INNER] → Kept {len(candidate_indices)} controls")

        # ===== 3. Compute extreme points =====
        if verbose:
            print("[EXTREME POINTS] Computing f1_min, f1_max, f2_min, f2_max...")

        ext = compute_extreme_points_double_facility(
            solve_fn=self.solve_double_facility_raw,
            X_cases=X_cases,
            X_controls=X_controls,
            candidate_indices=candidate_indices,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            final_ratio=final_ratio,
        )
        if verbose:
            print("[EXTREME POINTS]: ", ext)
            
        records = []  
        
        for w in w_range:
            print("="*40)
            print(f" Running double-facility sampling w={w:.2f}")
            print("="*40)
            # ===== 4. Solve normalized objective =====
            sol = self.solve_double_facility_normalized_MILP(
                X_cases, X_controls, candidate_indices,
                w=w,
                final_ratio=final_ratio,
                top_k_case_ctrl=top_k_case_ctrl,
                top_k_ctrl_ctrl=top_k_ctrl_ctrl,
                ext=ext,
                verbose=verbose
            )
            selected = sol["selected"]
            raw_f1, raw_f2 = sol["raw"]
            undersampled_training_data = pd.concat([df_cases, df_controls.iloc[selected]], ignore_index=True)

            records.append({"w": w, "f1": raw_f1, "f2": raw_f2})
            print("Final shape at w",w, " ",undersampled_training_data.shape)
            print(undersampled_training_data["cost_stratum_2018"].value_counts())
            if results_dir:
                save_path = f"{results_dir}/undersampled_w_{w:.2f}.csv"
                undersampled_training_data.to_csv(save_path, index=False)
                print("Undersampled train data saved to", save_path)
            else:
                return undersampled_training_data,pd.DataFrame(records)
        return pd.DataFrame(records)
