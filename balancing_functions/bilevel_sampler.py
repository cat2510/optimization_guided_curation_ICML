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

class BilevelResampler(LexicographicCaseControlResampler):
    """
    Two-stage undersampling: select_candidate_controls,select_diverse_subset,bilevel_undersample
      1) Inner: minimize distance between majority and minority sets (select K candidate negatives)
      2) Outer: maximize diversity among those K candidates (select final k subset)
    
    One-stage biobjective with simple scaling of cost terms (for extreme point scaling, see doublefacilitysampler .py)
      1) Push Pull
      2) Double Facility
    """   
    def select_candidate_controls(
        self,
        X_cases,
        X_controls,
        K: int,
        verbose: bool = True,
        top_k_factor: float = 10.0,
        disable_lexicographic: bool = False,
        return_assignments: bool = True,
    ):
        """
        Inner problem with Euclidean top-K pruning:
        - Build candidate (i,j) pairs by taking top-K nearest controls per case.
        - Solve p-median-style MIP restricted to those pairs only.

        Returns
        -------
        selected_controls : list[int]
            Indices (w.r.t. X_controls) of selected controls (size K').
        """

        n_cases, n_controls = X_cases.shape[0], X_controls.shape[0]
        if n_cases == 0 or n_controls == 0:
            raise RuntimeError("Empty cases or controls.")

        # Distances
        D = cdist(X_cases, X_controls, metric="euclidean")
        # ---------------------------
        # Candidate pair construction
        # ---------------------------
        if disable_lexicographic:
            # All pairs (may be huge)
            candidate_pairs = [(i, j) for i in range(n_cases) for j in range(n_controls)]
        else:
            candidate_pairs = []
            # per-case top-k (like your original)
            k = int(max(1, min(int(top_k_factor), n_controls)))
            kth = k - 1
            for i in range(n_cases):
                row = D[i]
                top_k_idx = np.argpartition(row, kth)[:k]
                # stabilize by sorting the k by true distance
                top_k_idx = top_k_idx[np.argsort(row[top_k_idx])]
                # ensure at least one candidate
                if top_k_idx.size == 0:
                    j_star = int(np.argmin(row))
                    top_k_idx = np.array([j_star], dtype=int)
                candidate_pairs.extend((i, int(j)) for j in top_k_idx)

        if len(candidate_pairs) == 0:
            # Extreme case: fall back to a single nearest neighbor per case
            candidate_pairs = [(i, int(np.argmin(D[i]))) for i in range(n_cases)]

        cases_in_candidates = sorted(set(i for (i, _) in candidate_pairs))
        ctrls_in_candidates = sorted(set(j for (_, j) in candidate_pairs))

        # Cap K to feasible unique controls present
        
        K_eff = min(K, len(ctrls_in_candidates))
        if verbose:
            total_possible = n_cases * n_controls
            kept_pct = 100.0 * len(candidate_pairs) / total_possible
            print(f"[Inner] Candidate pruning: {len(candidate_pairs):,}/{total_possible:,} "
                f"pairs kept ({kept_pct:.2f}%). Unique controls in candidates={len(ctrls_in_candidates)}. "
                f"K requested={K} → K used={K_eff}.")

        # ---------------------------
        # MIP over candidate pairs
        # ---------------------------
        solver = pywraplp.Solver.CreateSolver("SCIP")
        # y for controls that appear in candidates only
        y = {j: solver.BoolVar(f"y_{j}") for j in ctrls_in_candidates}
        # z only for candidate pairs
        z = {(i, j): solver.NumVar(0, 1, f"z_{i}_{j}") for (i, j) in candidate_pairs}

        # Each case must be assigned to exactly one selected control among its candidates
        for i in cases_in_candidates:
            pairs_i = [(ii, j) for (ii, j) in candidate_pairs if ii == i]
            # If somehow empty (shouldn't happen), add globally nearest control
            if not pairs_i:
                j_star = int(np.argmin(D[i]))
                pairs_i = [(i, j_star)]
                if (i, j_star) not in z:
                    z[(i, j_star)] = solver.NumVar(0, 1, f"z_{i}_{j_star}")
                    if j_star not in y:
                        y[j_star] = solver.BoolVar(f"y_{j_star}")
                    candidate_pairs.append((i, j_star))
                    ctrls_in_candidates = sorted(set(ctrls_in_candidates + [j_star]))

            solver.Add(solver.Sum(z[(ii, j)] for (ii, j) in pairs_i) == 1)
            for (_, j) in pairs_i:
                solver.Add(z[(i, j)] <= y[j])

        # Select exactly K_eff controls
        solver.Add(solver.Sum(y[j] for j in ctrls_in_candidates) == K_eff)

        # Objective: minimize total distance
        objective = solver.Sum(D[i, j] * z[(i, j)] for (i, j) in candidate_pairs)
        solver.Minimize(objective)
        solver.SetTimeLimit(300000)
        status = solver.Solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            raise RuntimeError("Inner optimization (candidate selection) failed.")

        selected_controls = [j for j in ctrls_in_candidates if y[j].solution_value() > 0.5]

        if verbose:
            obj_value = sum(D[i, j] * z[(i, j)].solution_value() for (i, j) in candidate_pairs)
            print(f"[Inner] Selected {len(selected_controls)} controls (objective={obj_value:.3f})")

        # ===== NEW: extract assignments for each case =====
        if return_assignments:
            assignments = []
            for (i, j), var in z.items():
                if var.solution_value() > 0.5:
                    assignments.append({"case_idx": i, "control_idx": j})

            if verbose:
                print(f"[Inner] Returned {len(assignments)} assignments.")

            return selected_controls, assignments

        # default: old behavior
        return selected_controls

    # -------------------------------------------------------------
    # Outer problem: diversity maximization (facility-location)
    # -------------------------------------------------------------
    def select_diverse_subset(self, X_controls, candidate_indices, k: int,
                              metric: str = "euclidean", verbose: bool = True):
        """
        Select a diverse subset S ⊆ C using greedy facility-location.

        Parameters
        ----------
        X_controls : np.ndarray
            Feature matrix for all majority samples
        candidate_indices : list[int]
            Indices of candidate controls from inner problem
        k : int
            Final number of negatives to retain
        """
        X_C = X_controls[candidate_indices]
        D = pairwise_distances(X_C, X_C, metric=metric)
        # Convert distance to similarity (bounded in [0,1])
        sim = np.exp(-D / (D.std() + 1e-8))

        nC = len(candidate_indices)
        selected = []
        covered = np.zeros(nC)

        for _ in trange(k, desc="[Outer] Selecting diverse subset", disable=not verbose):
            gains = np.full(nC, -np.inf)
            for j in range(nC):
                if j in selected:
                    continue
                new_cover = np.maximum(covered, sim[:, j])
                gains[j] = new_cover.sum() - covered.sum()
            best = int(np.argmax(gains))
            selected.append(best)
            covered = np.maximum(covered, sim[:, best])

        if verbose:
            print(f"[Outer] Selected {len(selected)} diverse controls out of {nC} candidates.")

        return [candidate_indices[idx] for idx in selected]


    def bilevel_undersample(
        self,
        df_cases: pd.DataFrame,
        df_controls: pd.DataFrame,
        exclude_cols_matching,
        K_factor: float = 3.0,
        final_ratio: float = 1.0,
        verbose: bool = True,
        top_k_factor: float = 10.0,
        disable_lexicographic: bool = False,
    ):
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            df_cases, df_controls, exclude_cols_matching, verbose=verbose
        )

        n_cases = X_cases.shape[0]
        K = int(np.clip(K_factor * n_cases, 1, X_controls.shape[0]))
        k = int(np.clip(final_ratio * n_cases, 1, K))

        if verbose:
            print(f"[Setup] {n_cases} cases | {X_controls.shape[0]} controls | "
                f"K={K} (inner) | k={k} (outer) | top_k_factor={top_k_factor} | "
                f"disable_lexicographic={disable_lexicographic}")

        # --- Inner with pruning
        candidate_ctrls, inner_assignments  = self.select_candidate_controls(
            X_cases, X_controls, K,
            verbose=verbose,
            top_k_factor=top_k_factor,
            disable_lexicographic=disable_lexicographic,return_assignments=True
        )
        df_assign = pd.DataFrame(inner_assignments)
        df_assign["case_uid"] = df_cases.iloc[df_assign["case_idx"]][self.uid_col].values
        df_assign["control_uid"] = df_controls.iloc[df_assign["control_idx"]][self.uid_col].values

        df_assign.to_csv("inner_assignment_solution_with_uids.csv", index=False)
        # Convert control indices → ENROLIDs
        candidate_ctrl_uids = df_controls.iloc[candidate_ctrls][self.uid_col].values

        df_candidate_ctrls = pd.DataFrame({
            "control_idx": candidate_ctrls,          # optional
            "control_uid": candidate_ctrl_uids       # correct UID
        })

        df_candidate_ctrls.to_csv(
            f"inner_candidate_ctrls_K{K}.csv",
            index=False
        )
        # --- Outer diversity (unchanged)
        diverse_ctrls = self.select_diverse_subset(
            X_controls, candidate_ctrls, k, verbose=verbose
        )

        undersampled_df = pd.concat(
            [df_cases, df_controls.iloc[diverse_ctrls]], ignore_index=True
        )
        undersampled_df.to_csv(f"undersampled_bilevel_K{K}.csv", index=False)

        if verbose:
            print(f"[Result] Final dataset: {len(df_cases)} positives + "
                f"{len(diverse_ctrls)} negatives = {len(undersampled_df)} total")
        return undersampled_df
    

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
            selected = dfC.nsmallest(K_eff, "min_dist").index.tolist()
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

    def optimize_double_facility_simple_scaling(
        self,
        X_cases,
        X_controls,
        df_controls,
        candidate_indices,
        final_ratio: float,
        w: float = 0.5,
        top_k_case_ctrl: int = 20,
        top_k_ctrl_ctrl: int = 50,
        stratified=False,
        verbose=True,
    ):
        X_C = X_controls[candidate_indices]
        P = X_cases.shape[0]
        C = len(candidate_indices)

        # Number of controls to select
        k = int(np.clip(final_ratio * P, 1, C))
        if verbose:
            print(f"[Unified] P={P}, C={C}, final_ratio={final_ratio} → k={k} controls")

        # Pruning
        NN_case, D_pn = self._topk_prune_case_to_control(
            X_cases, X_C, top_k_case_ctrl
        )
        NN_ctrl, D_nn = self._topk_prune_control_to_control(
            X_C, top_k_ctrl_ctrl
        )

        # Facility-location similarity
        sigma = D_nn.std() + 1e-8
        S_nn = np.exp(-D_nn / sigma)

        # Build solver
        solver = pywraplp.Solver.CreateSolver("SCIP")

        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]
        a = [{j: solver.NumVar(0, 1, f"a[{i},{j}]") for j in NN_case[i]}
            for i in range(P)]
        r = [{j: solver.NumVar(0, 1, f"r[{c},{j}]") for j in NN_ctrl[c]}
            for c in range(C)]

        # Constraints
        solver.Add(sum(s) == k)

        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for c in range(C):
            solver.Add(sum(r[c][j] for j in NN_ctrl[c]) == 1)
            for j in NN_ctrl[c]:
                solver.Add(r[c][j] <= s[j])

        # Case/coverage terms
        term_cases = solver.Sum(
            D_pn[i, j] * a[i][j] for i in range(P) for j in NN_case[i]
        )
        term_coverage = solver.Sum(
            S_nn[c, j] * r[c][j] for c in range(C) for j in NN_ctrl[c]
        )

        # Scaling
        scale_cases = 1.0 / max(P, 1)
        total_edges = sum(len(NN_ctrl[c]) for c in range(C))
        scale_cov   = 1.0 / max(total_edges, 1)

        term_cases_scaled = scale_cases * term_cases
        term_cov_scaled   = scale_cov   * term_coverage

        solver.Minimize(w * term_cases_scaled - (1 - w) * term_cov_scaled)

        solver.SetTimeLimit(1200000)   # 300 sec = 5 minutes

        # SCIP internal parameters (seconds)
        solver.SetSolverSpecificParametersAsString(r"""
        timing/clocktype = 1
        limits/time = 1200
        limits/softtime = 1200

        presolving/maxrounds = 0
        presolving/maxrestarts = 0

        separating/maxrounds = 0

        display/verblevel = 4
        """)
        status = solver.Solve()

        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            raise RuntimeError("Unified MILP failed")

        # ----------------------------------------------------
        # ★★★ Evaluate raw terms using your helper function ★★★
        # ----------------------------------------------------
        raw_cases, raw_cov = evaluate_terms_case_and_coverage(
            P, C,
            NN_case, D_pn, a,
            NN_ctrl, S_nn, r
        )

        if verbose:
            print("[Objective decomposition]")
            print(f"  term_cases_raw    = {raw_cases:.4f}")
            print(f"  term_coverage_raw = {raw_cov:.4f}")
            print(f"  term_cases_scaled = {scale_cases * raw_cases:.4f}")
            print(f"  term_cov_scaled   = {scale_cov * raw_cov:.4f}")
            print(f"  weighted_obj      = {solver.Objective().Value():.4f}")

        # Selected controls
        return [candidate_indices[j] for j in range(C) if s[j].solution_value() > 0.5]

    def double_facility_wrapper(
        self,
        df_cases,
        df_controls,
        exclude_cols_matching,
        final_ratio=1.0,
        w=0.5,
        top_k_case_ctrl=20,
        top_k_ctrl_ctrl=20,
        K_factor=3.0,        # INNER PRUNING size
        stratified=False,
        verbose=True
    ):

        # 1. Preprocessing
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            df_cases, df_controls, exclude_cols_matching, verbose=verbose
        )

        P = X_cases.shape[0]

        # ---------------------------------------------------
        # 2. INNER PRUNING = NODE PRUNING (reduce controls)
        # ---------------------------------------------------
        K_outer = int(K_factor * P)   # e.g. 3 × 660 = 1980
        K_outer = min(K_outer, X_controls.shape[0])

        if verbose:
            print(f"[INNER] Selecting {K_outer} candidate controls out of {X_controls.shape[0]}")

        candidate_indices = self.prune_nodes_distance_stratified(
           X_cases, X_controls, df_controls, top_k_per_case= top_k_case_ctrl,K_outer=K_outer, stratified=stratified)
        candidate_indices = list(map(int, candidate_indices))
        if verbose:
            print(f"[INNER] → Kept {len(candidate_indices)} controls")

        # ---------------------------------------------------
        # 3. OUTER PRUNING + Unified MILP
        # ---------------------------------------------------
        selected_ctrls = self.optimize_double_facility_simple_scaling(
            X_cases,
            X_controls,
            df_controls,
            candidate_indices,       # pruned nodes
            final_ratio,
            w=w,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            stratified=stratified,
            verbose=verbose
        )

        return pd.concat(
            [df_cases, df_controls.iloc[selected_ctrls]], ignore_index=True)


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


    def optimize_push_pull_simple_scaling(
        self,
        X_cases,
        X_controls,
        df_controls,
        candidate_indices,
        final_ratio: float,
        w: float = 0.5,
        top_k_case_ctrl: int = 20,
        L_pairs: int = 20,
        stratified: bool = False,
        verbose: bool = True,
    ):
        """
        Unified MILP with:
        - Term 1: case→control distance (closeness)
        - Term 2: pairwise control–control distance (diversity / dispersion)

        We select k = final_ratio * P controls from candidate_indices.
        """
        from ortools.linear_solver import pywraplp

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
        a = [
            {j: solver.NumVar(0, 1, f"a[{i},{j}]") for j in NN_case[i]}
            for i in range(P)
        ]

        # Pairwise co-selection vars: z[(j,k)] ∈ [0,1]
        z = { (j,k): solver.NumVar(0, 1, f"z[{j},{k}]") for (j,k) in pairs }

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

        # ---- Optional: Stratum quota constraints (upper bounds) ----
        if stratified:
            full_strata = df_controls["cost_stratum_2018"].values
            unique_full, counts_full = np.unique(full_strata, return_counts=True)
            p = {u: counts_full[i] / len(full_strata) for i, u in enumerate(unique_full)}
            target = {s_key: int(round(p[s_key] * k)) for s_key in p}

            if verbose:
                print("Global stratum proportions:", p)
                print("Quota targets (approx counts among k):", target)

            strata_pruned = df_controls.iloc[candidate_indices]["cost_stratum_2018"].values
            stratum_sets = {s_key: [] for s_key in target}
            for j in range(C):
                s_val = strata_pruned[j]
                if s_val in stratum_sets:
                    stratum_sets[s_val].append(j)

            band = 1.2  # +20% slack
            for s_key, tgt in target.items():
                if len(stratum_sets[s_key]) == 0:
                    continue
                ub = int(band * tgt)
                ub = min(ub, len(stratum_sets[s_key]))
                solver.Add(sum(s[j] for j in stratum_sets[s_key]) <= ub)

            if verbose:
                print("Candidate counts per stratum:",
                    {s_key: len(stratum_sets[s_key]) for s_key in target})

        # --- Objective: closeness (D_pn) vs diversity (D_nn) ---
        # Term 1: sum of case→control distances
        raw_term_cases = solver.Sum(
            D_pn[i, j] * a[i][j] for i in range(P) for j in NN_case[i]
        )

        # Term 2: sum of pairwise distances among selected controls
        raw_term_pairs = solver.Sum(
            D_nn[j, k] * z[(j, k)] for (j, k) in pairs
        )

        # ---- SIMPLE SCALING TO MATCH MAGNITUDE ----
        # average distance per case
        scale_cases = 1.0 / max(P, 1)
        # average distance per pair
        scale_pairs = 1.0 / max(len(pairs), 1)

        term_cases = scale_cases * raw_term_cases
        term_pairs = scale_pairs * raw_term_pairs

        obj = w * term_cases - (1.0 - w) * term_pairs
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

        presolving/maxrounds = 0
        presolving/maxrestarts = 0

        separating/maxrounds = 0

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
            print(f"  term_cases_scaled = {(scale_cases * raw_cases):.4f}")
            print(f"  term_disp_scaled  = {(scale_pairs * raw_disp):.4f}")
            print(f"  weighted_obj      = {solver.Objective().Value():.4f}")
                # Return selected controls (candidate_indices indexing)
        selected_controls = [
            candidate_indices[j] for j in range(C) if s[j].solution_value() > 0.5
        ]

        return selected_controls


    def biobjective_push_pull_wrapper(
        self,
        df_cases: pd.DataFrame,
        df_controls: pd.DataFrame,
        exclude_cols_matching,
        final_ratio=1.0,
        w=0.5,
        top_k_case_ctrl=20,
        L_pairs=10,
        K_factor=3.0,   # for inner node pruning if you have it
        stratified=False,
        verbose=True,
    ):
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

        candidate_indices = self.prune_nodes_distance_stratified(
           X_cases, X_controls, df_controls,top_k_per_case=top_k_case_ctrl, K_outer = K_outer, stratified=stratified
)
        candidate_indices = list(map(int, candidate_indices))
        if verbose:
            print(f"[INNER] → Kept {len(candidate_indices)} controls")

        selected_ctrls = self.optimize_push_pull_simple_scaling(
            X_cases,
            X_controls,
            df_controls,
            candidate_indices,
            final_ratio=final_ratio,
            w=w,
            top_k_case_ctrl=top_k_case_ctrl,
            L_pairs=L_pairs,
            stratified=stratified,
            verbose=verbose,
        )

        undersampled = pd.concat(
            [df_cases, df_controls.iloc[selected_ctrls]], ignore_index=True
        )
        return undersampled
