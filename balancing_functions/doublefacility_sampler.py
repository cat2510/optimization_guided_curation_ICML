#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Double-facility undersampling with extreme-point normalization.

Idea:
- f1: distance case→control (we want to MINIMIZE)
- f2: coverage/diversity term built via control→control "facility" assignments (we want to MAXIMIZE)
- Use extreme points (min/max f1, min/max f2) to normalize, then solve
    min_x  w * \tilde f1(x) - (1 - w) * \tilde f2(x)
where \tilde f1, \tilde f2 are normalized to [~0, 1].
"""
import numpy as np
import pandas as pd
from tqdm import trange,tqdm
import time
from typing import Dict, Optional
from scipy.spatial.distance import cdist
from ortools.linear_solver import pywraplp
from sklearn.metrics import pairwise_distances
# import base class from pushpull_sampler
from balancing_functions.pushpull_sampler import BaseMultiObjectiveSampler


def evaluate_terms_case_and_coverage(P, C, NN_case, D_pn, a, NN_ctrl, S_nn, r):
    """
    Compute raw f1 and f2 under the current MILP solution.

    f1: sum over cases i and assigned controls j of D_pn[i,j] * a[i,j]
    f2: sum over controls c and their "representative" controls j of S_nn[c,j] * r[c,j]
    """
    # Case–control term
    term1 = 0.0
    for i in range(P):
        for j in NN_case[i]:
            term1 += D_pn[i, j] * a[i][j].solution_value()

    # Coverage/diversity term (facility-style)
    term2 = 0.0
    for c in range(C):
        for j in NN_ctrl[c]:
            term2 += S_nn[c, j] * r[c][j].solution_value()

    return term1, term2


class DoubleFacilitySampler(BaseMultiObjectiveSampler):
    """
    Double-facility undersampling:

    - Each minority case is assigned to exactly one selected majority control (f1 term).
    - Each majority control is assigned to exactly one selected "facility" control
      among its nearest neighbors (f2 term via S_nn). Selected controls play the role
      of cluster centers / facilities for other controls.

    Multi-objective:
        minimize f1 (closeness case→control)
        maximize f2 (coverage/diversity via facility centers)

    We normalize via extreme points and then solve a weighted objective.
    """

    # ------------------------------------------------------------------
    # 1. Nearest-neighbor structures for f2 (f1 uses shared method from base class)
    # ------------------------------------------------------------------
    def _topk_prune_control_to_control(self, X_C, L):
        """
        For each control c, find L nearest (in feature space) controls.
        Returns:
            NN_ctrl[c] = length-L array of neighbor controls
            D_nn       = full C x C matrix of distances
        """
        D_nn = pairwise_distances(X_C, X_C, metric="euclidean")
        C = D_nn.shape[0]

        L_eff = min(L, C - 1)
        nearest = np.argpartition(D_nn, L_eff, axis=1)[:, :L_eff]

        # sort for stability
        sorted_idx = np.argsort(D_nn[np.arange(C)[:, None], nearest], axis=1)
        NN_ctrl = nearest[np.arange(C)[:, None], sorted_idx]

        return NN_ctrl, D_nn

    # ------------------------------------------------------------------
    # 2. Low-level MILP solver (single model, multiple objective modes)
    # ------------------------------------------------------------------
    def solve_double_facility_MILP(
        self,
        X_cases,
        X_controls,
        candidate_indices,
        final_ratio: float,
        top_k_case_ctrl: int,
        top_k_ctrl_ctrl: int,
        objective_mode: str = "weighted",  # "f1min", "f1max", "f2min", "f2max", "weighted"
        w: float = 0.5,
        ext=None,  # required only for "weighted"
        return_only_objective: bool = False,
        verbose: bool = True,
    ):
        """
        Core MILP for double-facility sampler.

        Variables:
            s[j]   ∈ {0,1}  – select control j as a facility
            a[i,j] ∈ {0,1}  – assign case i to control j
            r[c,j] ∈ {0,1}  – assign control c to facility j

        Constraints:
            ∑_j s[j] = k
            ∑_j a[i,j] = 1         for each case i
            a[i,j] ≤ s[j]
            ∑_j r[c,j] = 1         for each control c
            r[c,j] ≤ s[j]

        Raw terms:
            f1 = Σ_{i,j} D_pn[i,j] a[i,j]
            f2 = Σ_{c,j} S_nn[c,j] r[c,j],   S_nn = exp(-D_nn / σ)

        objective_mode:
            - "f1min": minimize f1
            - "f1max": maximize f1
            - "f2min": minimize f2
            - "f2max": maximize f2
            - "weighted": minimize w*tilde_f1 - (1-w)*tilde_f2
                          with tilde_f1, tilde_f2 normalized via ext.
        """

        # Subset controls to candidate pool
        X_C = X_controls[candidate_indices]
        P = X_cases.shape[0]
        C = len(candidate_indices)
        k = int(np.clip(final_ratio * P, 1, C))

        # Case→control NN (for f1)
        NN_case, D_pn = self._topk_prune_case_to_control(
            X_cases, X_C, top_k_case_ctrl
        )

        # Control→control NN (for f2 via facilities)
        NN_ctrl, D_nn = self._topk_prune_control_to_control(
            X_C, top_k_ctrl_ctrl
        )

        sigma = D_nn.std() + 1e-8
        S_nn = np.exp(-D_nn / sigma)

        if verbose:
            print(f"[EDGE PRUNING] cases={P}, controls={C}")
            print(
                f"[EDGE PRUNING] Each case has {np.array(NN_case).shape[1]} control candidates"
            )
            print(
                f"[EDGE PRUNING] Each control has {np.array(NN_ctrl).shape[1]} facility candidates"
            )

        solver = pywraplp.Solver.CreateSolver("SCIP")

        # Selection vars
        s = [solver.BoolVar(f"s[{j}]") for j in range(C)]

        # Assign each case to one selected control
        a = [{j: solver.BoolVar(f"a[{i},{j}]") for j in NN_case[i]} for i in range(P)]

        # Assign each control to one selected facility control
        r = [{j: solver.BoolVar(f"r[{c},{j}]") for j in NN_ctrl[c]} for c in range(C)]

        # Total number of selected controls
        solver.Add(sum(s) == k)

        # Case coverage constraints
        for i in range(P):
            solver.Add(sum(a[i][j] for j in NN_case[i]) == 1)
            for j in NN_case[i]:
                solver.Add(a[i][j] <= s[j])

        # Control coverage constraints
        for c in range(C):
            solver.Add(sum(r[c][j] for j in NN_ctrl[c]) == 1)
            for j in NN_ctrl[c]:
                solver.Add(r[c][j] <= s[j])

        # Raw objective expressions
        f1 = solver.Sum(D_pn[i, j] * a[i][j] for i in range(P) for j in NN_case[i])
        f2 = solver.Sum(S_nn[c, j] * r[c][j] for c in range(C) for j in NN_ctrl[c])

        # Objective selection
        if verbose:
            print("Solving for objective_mode:", objective_mode)

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

            # We want small f1 and large f2
            solver.Minimize(w * tilde_f1 - (1.0 - w) * tilde_f2)

        else:
            raise ValueError("Invalid objective_mode")

        if verbose:
            print("[MILP-DoubleFacility] Variables:", solver.NumVariables())
            print("[MILP-DoubleFacility] Constraints:", solver.NumConstraints())

        # Time limits (ms for OR-Tools wrapper, seconds for SCIP)
        solver.SetTimeLimit(2400000)  # 2400 s = 40 minutes
        solver.SetSolverSpecificParametersAsString(
            r"""
        timing/clocktype = 1
        limits/time = 2400
        limits/softtime = 2400
        display/verblevel = 4
        """
        )

        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(f"Double-facility MILP failed with status {status}")

        if return_only_objective:
            if verbose:
                print(
                    f"  Computed {objective_mode} objective = {solver.Objective().Value():.4f}"
                )
            return float(solver.Objective().Value())

        # Decode solution
        selected = [
            candidate_indices[j] for j in range(C) if s[j].solution_value() > 0.5
        ]

        f1_val, f2_val = evaluate_terms_case_and_coverage(
            P, C, NN_case, D_pn, a, NN_ctrl, S_nn, r
        )

        if verbose:
            print("[Optimal cost decomposition]")
            print(f"  f1 (case→control) = {f1_val:.4f}")
            print(f"  f2 (coverage)     = {f2_val:.4f}")
            print(f"  objective value   = {solver.Objective().Value():.4f}")

        return {
            "selected": selected,
            "f1": float(f1_val),
            "f2": float(f2_val),
            "status": int(status),
        }

    # ------------------------------------------------------------------
    # 3. Extreme points: f1_min, f1_max, f2_min, f2_max
    # ------------------------------------------------------------------
    def compute_double_facility_extreme_points(
        self,
        X_cases,
        X_controls,
        candidate_indices,
        final_ratio: float,
        top_k_case_ctrl: int,
        top_k_ctrl_ctrl: int,
        verbose: bool = True,
    ):
        """
        Compute extreme points for normalization by solving the same MILP
        with single-objective modes.
        """

        f1_min = self.solve_double_facility_MILP(
            X_cases,
            X_controls,
            candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            objective_mode="f1min",
            return_only_objective=True,
            verbose=verbose,
        )

        f1_max = self.solve_double_facility_MILP(
            X_cases,
            X_controls,
            candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            objective_mode="f1max",
            return_only_objective=True,
            verbose=verbose,
        )

        f2_min = self.solve_double_facility_MILP(
            X_cases,
            X_controls,
            candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            objective_mode="f2min",
            return_only_objective=True,
            verbose=verbose,
        )

        f2_max = self.solve_double_facility_MILP(
            X_cases,
            X_controls,
            candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            objective_mode="f2max",
            return_only_objective=True,
            verbose=verbose,
        )

        if verbose:
            print(
                f"[EXTREME POINTS] f1 in [{f1_min:.4f}, {f1_max:.4f}], "
                f"f2 in [{f2_min:.4f}, {f2_max:.4f}]"
            )

        return {
            "f1_min": float(f1_min),
            "f1_max": float(f1_max),
            "f2_min": float(f2_min),
            "f2_max": float(f2_max),
        }

    # ------------------------------------------------------------------
    # 4. High-level wrapper on (df_cases, df_controls)
    # ------------------------------------------------------------------
    def solve_double_facility_normalized_MILP(
        self,
        df_cases: pd.DataFrame,
        df_controls: pd.DataFrame,
        exclude_cols_matching,
        final_ratio: float,
        w: float,
        K_factor: float,
        top_k_case_ctrl: int,
        top_k_ctrl_ctrl: int,
        ext: Optional[Dict[str, float]] = None,  # Optional: pre-computed extreme points
        verbose: bool = True,
    ):
        """
        Full *pipeline* wrapper (matches the structure of PushPullSampler.solve_pushpull_normalized_MILP):

        1. Preprocess cases/controls into feature space via
           get_preprocessed_control_case_features (from base class).
        2. Node pruning using prune_nodes_distance with K_factor * |minority|.
        3. Compute extreme points (f1_min, f1_max, f2_min, f2_max) or use provided ones.
        4. Solve normalized weighted MILP for the given w.
        5. Return (undersampled_df, result_dict, ext_dict).
        
        Parameters
        ----------
        ext : dict, optional
            Pre-computed extreme points dict with keys: f1_min, f1_max, f2_min, f2_max.
            If provided, skips extreme point computation.
        """

        # 1) Preprocess
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            df_cases, df_controls, exclude_cols_matching, verbose=verbose
        )
        P = X_cases.shape[0]

        # 2) Node pruning
        K_outer = int(K_factor * P)
        K_outer = min(K_outer, X_controls.shape[0])
        if verbose:
            print(
                f"[K_factor * |minority|] Node pruning {K_outer} candidate controls out of {X_controls.shape[0]}"
            )

        candidate_indices = self.prune_nodes_distance(
            X_cases,
            X_controls,
            top_k_per_case=top_k_case_ctrl,
            K_outer=K_outer,
        )
        candidate_indices = list(map(int, candidate_indices))
        if verbose:
            print(f"[NODE PRUNING] → Kept {len(candidate_indices)} controls")

        # 3) Extreme points for this ratio (compute only if not provided)
        if ext is None:
            if verbose:
                print("[EXTREME POINTS] Computing extreme points (this may take a while)...")
            ext = self.compute_double_facility_extreme_points(
                X_cases,
                X_controls,
                candidate_indices,
                final_ratio=final_ratio,
                top_k_case_ctrl=top_k_case_ctrl,
                top_k_ctrl_ctrl=top_k_ctrl_ctrl,
                verbose=verbose,
            )
        else:
            if verbose:
                print("[EXTREME POINTS] Using provided extreme points")
                print(f"  f1 in [{ext['f1_min']:.4f}, {ext['f1_max']:.4f}], "
                      f"f2 in [{ext['f2_min']:.4f}, {ext['f2_max']:.4f}]")

        # 4) Weighted normalized solve
        result = self.solve_double_facility_MILP(
            X_cases,
            X_controls,
            candidate_indices,
            final_ratio=final_ratio,
            top_k_case_ctrl=top_k_case_ctrl,
            top_k_ctrl_ctrl=top_k_ctrl_ctrl,
            objective_mode="weighted",
            w=w,
            ext=ext,
            verbose=verbose,
        )
        selected_ctrls = result["selected"]

        # 5) Build undersampled dataset
        undersampled = pd.concat(
            [df_cases, df_controls.iloc[selected_ctrls]], ignore_index=True
        )

        return undersampled, result, ext
