#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiobjective Push Pull sampler with extreme-point scaled objective (f(x)-f_min/f_max-f_min)
"""

import numpy as np
import pandas as pd
from tqdm import trange
from scipy.spatial.distance import cdist
from ortools.linear_solver import pywraplp
from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer
from typing import Optional, List
from model_pipeline import get_preprocessor, get_bin_flag_columns
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


class BaseMultiObjectiveSampler:
    """
    Base class for multi-objective samplers that provides shared utility functions.
    This is a standalone class that includes the necessary preprocessing functionality
    extracted from LexicographicCaseControlResampler, plus shared pruning methods.
    Provides common functionality for both PushPullSampler and DoubleFacilitySampler.
    """
    
    def __init__(
        self,
        bin_edges: Optional[np.ndarray] = None,
        impute_strategy: str = "mean",
        uid_col: str = "uid",
        random_state: int = 42,
        binary_group: str = "high_cost_2018",
    ):
        """
        Initialize base sampler with preprocessing configuration.
        
        Parameters
        ----------
        bin_edges : Optional[np.ndarray]
            Bin edges for feature binning (not used in multi-objective samplers but kept for compatibility)
        impute_strategy : str
            Strategy for imputing missing values ("mean", "median", "most_frequent")
        uid_col : str
            Name of the unique identifier column
        random_state : int
            Random seed for reproducibility
        binary_group : str
            Name of the binary grouping column to exclude from features
        """
        self.bin_edges = bin_edges if bin_edges is not None else np.linspace(0, 1, 11)
        self.bin_labels = [
            f"{a:.2f}-{b:.2f}" for a, b in zip(self.bin_edges[:-1], self.bin_edges[1:])
        ]
        self.impute_strategy = impute_strategy
        self.uid_col = uid_col
        self.random_state = random_state
        self.binary_group = binary_group
    
    def get_preprocessed_control_case_features(
        self,
        cases: pd.DataFrame,
        controls: pd.DataFrame,
        exclude_cols_matching: List[str],
        verbose: bool = False
    ) -> tuple:
        """
        Use preprocessing pipeline to preserve the feature engineering approach.
        Return X_control, X_cases. Double checked for alignment between df_controls and X_controls.
        """
        # Step 1: Feature categorization logic
        drop_cols = [self.uid_col, self.binary_group] + exclude_cols_matching
        all_cols = [c for c in cases.columns if c not in drop_cols]
        # Combine datasets for consistent preprocessing
        combined_df = pd.concat([cases[all_cols], controls[all_cols]], ignore_index=True)
        
        # Step 2: Handle missing values first
        # Separate numeric and categorical columns
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Impute numeric columns
        if numeric_cols:
            imputer = SimpleImputer(strategy=self.impute_strategy)
            combined_df[numeric_cols] = imputer.fit_transform(combined_df[numeric_cols])
        
        # Impute categorical columns  
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            combined_df[categorical_cols] = cat_imputer.fit_transform(combined_df[categorical_cols])
        
        # Intersect with your three feature pools
        bin_feats = get_bin_flag_columns(combined_df) 
        num_feats = [c for c in numeric_cols if c not in bin_feats]

        if verbose:
            print(f">>> Distance computation will use {len(all_cols)} features")
            print(f"{len(categorical_cols)} Categorical features (one-hot encoded):", categorical_cols)
            print(f"{len(num_feats)} Numeric features (normalized):", num_feats)
            print(f"{len(bin_feats)} Binary features (unchanged):", bin_feats)

        # Step 3: Build your sophisticated preprocessor
        preprocessor = get_preprocessor(
            df=combined_df,
            categorical_cols=categorical_cols,
            numeric_cols=num_feats,
            verbose=False
        )
        
        X_combined = preprocessor.fit_transform(combined_df)
        n_cases = len(cases)
        X_cases = X_combined[:n_cases]
        X_controls = X_combined[n_cases:]
        if verbose:
            print(f"X_cases has shape: {X_cases.shape}, X_controls has shape: {X_controls.shape}")
        
        return X_cases, X_controls

    def prune_nodes_distance(
        self,
        X_cases,
        X_controls,
        top_k_per_case=50,
        K_outer=2000,
    ):
        """
        Purely feature-space pruning.
        
        Algorithm:
        1. For each case, find top_k_per_case nearest controls.
        2. Collect all unique candidate controls that appear in any case's top-k.
        3. For each candidate, compute its minimum distance to any case.
        4. Return the K_outer candidates with smallest minimum distances.
        """

        # ---- Step 1: construct case→control distance matrix ----
        print(f"X_cases shape: {X_cases.shape}")
        print(f"X_controls shape: {X_controls.shape}")
        D = cdist(X_cases, X_controls)          # shape: (P, C)
        P, C = D.shape

        # ---- Step 2: top-k nearest controls for each case ----
        # Handle edge case where top_k_per_case >= C
        top_k_eff = min(top_k_per_case, C)
        if top_k_eff == C:
            print("All controls are candidates, no need for argpartition")
            nearest = np.arange(C).reshape(1, -1).repeat(P, axis=0)
        else:
            nearest = np.argpartition(D, top_k_eff, axis=1)[:, :top_k_eff]

        # Candidate control positions
        candidate_pos = np.unique(nearest.ravel())     # ⟵ indices in 0..C−1

        # ---- Step 3: compute each candidate's minimal distance to any case ----
        # Instead of recomputing cdist, slice D:
        min_dist = D[:, candidate_pos].min(axis=0)

        # ---- Step 4: retain K_outer best candidates ----
        # Sort candidate positions by their min_dist
        order = np.argsort(min_dist)
        K_eff = min(K_outer, len(candidate_pos))

        selected_pos = candidate_pos[order[:K_eff]]

        return selected_pos.tolist()

    def _topk_prune_case_to_control(self, X_cases, X_controls, L):
        """
        For each case i, find L nearest controls j.
        Returns:
            NN_case[i] = length-L array of control indices
            D          = full distance matrix (P x C)

        NEW Jan 7, 2026: Patched version that handles L >= C case (all controls).
        """
        D = pairwise_distances(X_cases, X_controls, metric="euclidean")
        P, C = D.shape
        
        L_eff = min(L, C)
        
        # If L >= C, we want all controls, so just use all indices
        if L_eff >= C:
            # Return all control indices for each case
            nearest = np.arange(C).reshape(1, -1).repeat(P, axis=0)
        else:
            # Use argpartition for partial selection
            nearest = np.argpartition(D, L_eff, axis=1)[:, :L_eff]
        
        # Sort each neighbor list by distance for stability
        sorted_idx = np.argsort(D[np.arange(P)[:, None], nearest], axis=1)
        NN_case = nearest[np.arange(P)[:, None], sorted_idx]
        
        return NN_case, D




class PushPullSampler(BaseMultiObjectiveSampler):
    """
    One-stage multi-objective with extreme point scaling of cost terms 
    """   
    def _topk_farthest_control_pairs(self, X_C, L_pairs=20, verbose=True):
        """
        For dispersion: build a sparse set of FAR pairs (j,k) among controls.
        NEW Jan 7, 2026: Patched version that handles the case where we want all control-control pairs.

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
        
        # If L_pairs >= C - 1, we want all pairs (each control paired with all others)
        if L_pairs >= C - 1:
            # Generate all unique pairs (j, k) where j < k
            for j in range(C):
                for k in range(j + 1, C):
                    pairs_set.add((j, k))
        else:
            # Use original logic for partial selection
            for j in range(C):
                row = D_nn[j]
                # indices of L largest distances (excluding j itself)
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
        
        The weighted mode uses extreme-point normalization:
            tilde_f1 = (f1 - f1_min) / (f1_max - f1_min)
            tilde_f2 = (f2 - f2_min) / (f2_max - f2_min)
        Both terms are normalized to [0, 1], making them comparable.
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
        if verbose:
            print(f"[EDGE PRUNING] → Each case ({np.array(NN_case).shape[0]} cases) has {np.array(NN_case).shape[1]} control candidates")
            print(f"[EDGE PRUNING] → Kept {np.array(pairs).shape} control-control pairs")

        # Build SCIP model
        solver = pywraplp.Solver.CreateSolver("SCIP")

        # Selection
        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]

        # Assignment (binary recommended)
        a = [
            { j: solver.BoolVar(f"a[{i},{j}]") for j in NN_case[i] }
            for i in range(P)
        ]

        # Pairwise co-selection vars: z[(j_idx,k_idx)] ∈ [0,1]
        z = { (j_idx, k_idx): solver.NumVar(0, 1, f"z[{j_idx},{k_idx}]") for (j_idx, k_idx) in pairs }

        # Constraints
        solver.Add(sum(s) == k)

        for i in range(P):
            # each case assigned to at least one control
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1) 
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        for (j_idx, k_idx), z_var in z.items():
            solver.Add(z_var <= s[j_idx])
            solver.Add(z_var <= s[k_idx])


        # Raw objective components
        f1 = solver.Sum(D_pn[i,j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(D_nn[j_idx, k_idx] * z[(j_idx, k_idx)] for (j_idx, k_idx) in pairs)

        # --- Objective selection ---
        if verbose:
            print("Solving for ", objective_mode)

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

            # Both terms are normalized to [0, 1] via extreme-point normalization,
            # so they should be on the same scale and comparable.
            solver.Minimize(w * tilde_f1 - (1 - w) * tilde_f2)

        else:
            raise ValueError("Invalid objective_mode")

        if verbose:
            print("[MILP-Diverse] Variables:", solver.NumVariables())
            print("[MILP-Diverse] Constraints:", solver.NumConstraints())

        solver.SetTimeLimit(2400000)   # 300 sec = 5 minutes

        # SCIP internal parameters (seconds)
        solver.SetSolverSpecificParametersAsString(r"""
        timing/clocktype = 1
        limits/time = 2400
        limits/softtime = 2400
        display/verblevel = 4
        """)
        status = solver.Solve()

        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            raise RuntimeError(f"Unified Diverse MILP failed with status {status}")

        if return_only_objective:
            if verbose:
                print(f"  Computed {objective_mode} objective = {solver.Objective().Value():.4f}")
            return float(solver.Objective().Value())

        selected = [candidate_indices[j] for j in range(C) if s[j].solution_value() > 0.5]
        f1_val = sum(D_pn[i, j] * a[i][j].solution_value()
             for i in range(P)
             for j in NN_case[i])

        f2_val = sum(D_nn[j_idx, k_idx] * z[(j_idx, k_idx)].solution_value()
                    for (j_idx, k_idx) in pairs)

        if verbose:
            print("[Optimal cost decomposition]")
            print(f"  distance min-maj   = {f1_val:.4f}")
            print(f"  distance maj-maj   = {f2_val:.4f}")
            print(f"  weighted objective = {solver.Objective().Value():.4f}")
 

        return {
            "selected": selected,
            "f1": float(f1_val),
            "f2": float(f2_val),
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
        df_cases,
        df_controls,
        exclude_cols_matching,
        final_ratio,
        w, # weight in the multi-objective cost
        K_factor, # inner pruning ratio
        top_k_case_ctrl,
        L_pairs,
        ext=None,  # Optional: pre-computed extreme points
        verbose=True
    ):
        """
        Full wrapper for normalized push–pull sampling.
        1. Preprocess cases/controls into feature space.
        2. Prune controls using distance-based strategy.
        3. Compute extreme points (min/max f1,f2) or use provided ones.
        4. Solve weighted normalized MILP to select control subset.
        5. Return undersampled dataset.
        
        Parameters
        ----------
        ext : dict, optional
            Pre-computed extreme points dict with keys: f1_min, f1_max, f2_min, f2_max.
            If provided, skips extreme point computation.
            
        Note: The weighted objective uses extreme-point normalization to ensure both
        terms (f1 and f2) are on the same [0, 1] scale, making them comparable.
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
            print(f"[K_factor * |minority|] Node pruning {K_outer} candidate controls out of {X_controls.shape[0]}")

        candidate_indices = self.prune_nodes_distance(
           X_cases, X_controls, top_k_per_case=top_k_case_ctrl, K_outer = K_outer)
        candidate_indices = list(map(int, candidate_indices))
        if verbose:
            print(f"[NODE PRUNING] → Kept {len(candidate_indices)} controls")

        # Compute extreme points only if not provided
        if ext is None:
            if verbose:
                print("[EXTREME POINTS] Computing extreme points (this may take a while)...")
            ext = self.compute_pushpull_extreme_points(
                X_cases, X_controls, candidate_indices,
                final_ratio=final_ratio,
                top_k_case_ctrl=top_k_case_ctrl,
                L_pairs=L_pairs,
                verbose=verbose
            )
        else:
            if verbose:
                print("[EXTREME POINTS] Using provided extreme points")
                print(f"  f1 in [{ext['f1_min']:.4f}, {ext['f1_max']:.4f}], "
                      f"f2 in [{ext['f2_min']:.4f}, {ext['f2_max']:.4f}]")

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
