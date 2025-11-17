# Enhanced version of your match_control.py with optimization-based matching

import pandas as pd
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Union, Any
from collections import Counter
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
from model_pipeline import get_preprocessor, get_bin_flag_columns
from ortools.linear_solver import pywraplp

def sort_features_by_density(X, feature_names):
    """
    Sort features from most dense (fewest zeros) to most sparse (most zeros)
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    feature_names : list
        Feature column names
        
    Returns:
    --------
    list : Column indices sorted by density (most dense first)
    """
    density_scores = []
    for col_idx in range(X.shape[1]):
        non_zero_count = np.sum(X[:, col_idx] != 0)
        density_score = non_zero_count / X.shape[0]  # Proportion of non-zero values
        density_scores.append((density_score, col_idx, feature_names[col_idx]))
    
    # Sort by density (descending - most dense first)
    sorted_features = sorted(density_scores, key=lambda x: x[0], reverse=True)
    
    return [x[1] for x in sorted_features]  # Return column indices

def lexicographic_sort_patients(X, sorted_feature_indices):
    """
    Sort patients lexicographically by feature density
    - First by most dense feature (non-zero comes before zero)  
    - Then by second most dense feature, etc.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    sorted_feature_indices : list
        Feature indices sorted by density
        
    Returns:
    --------
    list : Patient indices sorted lexicographically
    """
    sort_keys = []
    for patient_idx in range(X.shape[0]):
        # Create lexicographic key: tuple of (is_nonzero_feat1, is_nonzero_feat2, ...)
        # Use negative values so non-zero (True=-1) comes before zero (False=0)
        key = tuple(-int(X[patient_idx, feat_idx] != 0) 
                   for feat_idx in sorted_feature_indices)
        sort_keys.append((key, patient_idx))
    
    # Sort lexicographically and return patient indices
    sorted_patients = sorted(sort_keys, key=lambda x: x[0])
    return [x[1] for x in sorted_patients]


class LexicographicCaseControlResampler:
    def __init__(
        self,
        bin_edges: Optional[np.ndarray] = None,
        impute_strategy: str = "mean",
        uid_col: str = "uid",
        random_state: int = 42,
        binary_group: str = "high_cost_2018",
        matching_method: str = "ortools",  # "optimization", "ortools", or "knn"
        solver: str = "GUROBI"  # "CLARABEL" etc see solver_choice.py
    ):
        self.bin_edges = bin_edges if bin_edges is not None else np.linspace(0, 1, 11)
        self.bin_labels = [
            f"{a:.2f}-{b:.2f}" for a, b in zip(self.bin_edges[:-1], self.bin_edges[1:])
        ]
        self.impute_strategy = impute_strategy
        self.uid_col = uid_col
        self.random_state = random_state
        self.binary_group = binary_group
        self.matching_method = matching_method  # "optimization", "ortools", or "knn"
        self.solver = solver

    def _solve_optimization_with_pairs(
        self, dfA, dfB,
        candidate_pairs,
        distances_dict,
        target_controls,
        target_cases,
        max_controls_per_case,
        verbose,
        inverse_matching=False
    ):
        """
        Returns:
            matched_df : pd.DataFrame
            match_map  : dict control_uid -> case_uid
            match_idx  : dict control_index -> case_index
        """

        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Variables
        z = { (i, j): solver.BoolVar(f"z_{i}_{j}") for (i, j) in candidate_pairs }

        cases_in_candidates = set(i for i, _ in candidate_pairs)
        controls_in_candidates = set(j for _, j in candidate_pairs)

        if verbose:
            print(f"Optimization scope: {len(cases_in_candidates)} cases, "
                f"{len(controls_in_candidates)} controls, "
                f"{len(candidate_pairs)} pairs")

        # Control uniqueness
        for j in controls_in_candidates:
            solver.Add(
                solver.Sum(z[i, j] for (i, jj) in candidate_pairs if jj == j) <= 1
            )

        # Case degrees
        deg_case = { i:0 for i in cases_in_candidates }
        for (i, _) in candidate_pairs:
            deg_case[i] += 1

        feasible_from_cases = sum(min(deg_case[i], max_controls_per_case) for i in cases_in_candidates)
        feasible_from_controls = len(controls_in_candidates)
        feasible_from_edges = len(candidate_pairs)

        feasible_cap = min(feasible_from_cases, feasible_from_controls, feasible_from_edges)

        # per-case target
        n_cases = max(1, len(cases_in_candidates))
        desired_per_case = min(
            max_controls_per_case,
            int(round(target_controls / n_cases))
        )

        # Bounds for each case
        for i in cases_in_candidates:
            if deg_case[i] == 0:
                continue
            solver.Add(
                solver.Sum(z[i, j] for (ii, j) in candidate_pairs if ii == i) >= min(desired_per_case, deg_case[i])
            )
            solver.Add(
                solver.Sum(z[i, j] for (ii, j) in candidate_pairs if ii == i) <= max_controls_per_case
            )

        # Total matches
        target_total = min(target_controls, feasible_cap)
        solver.Add(solver.Sum(z.values()) == target_total)

        # Objective
        obj_terms = [distances_dict[(i, j)] * z[(i, j)] for (i, j) in candidate_pairs]

        if inverse_matching:
            if verbose: print("Maximizing distance objective")
            solver.Maximize(solver.Sum(obj_terms))
        else:
            if verbose: print("Minimizing distance objective")
            solver.Minimize(solver.Sum(obj_terms))

        solver.SetTimeLimit(300000)
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(f"Optimization failed, status = {status}")

        # --------------------------
        # Extract matches (NEW!)
        # --------------------------
        match_map = {}      # control_uid  -> case_uid
        match_map_idx = {}  # control_j    -> case_i

        matched_cases = set()
        matched_controls = set()

        for (i, j), var in z.items():
            if var.solution_value() > 0.5:
                matched_cases.add(i)
                matched_controls.add(j)
                match_map_idx[j] = i

        # Build match_map with UIDs
        for ctrl_j, case_i in match_map_idx.items():
            case_uid = dfA.iloc[case_i][self.uid_col]
            ctrl_uid = dfB.iloc[ctrl_j][self.uid_col]
            match_map[ctrl_uid] = case_uid

        # Build final DF
        df_cases_matched = dfA.iloc[list(matched_cases)]
        df_controls_matched = dfB.iloc[list(matched_controls)]

        matched_df = pd.concat([df_cases_matched, df_controls_matched], ignore_index=True)

        if verbose:
            obj_val = sum(distances_dict[(i,j)] * z[(i,j)].solution_value()
                        for (i,j) in candidate_pairs)
            print(f"Optimization successful: {len(matched_cases)} cases, "
                f"{len(matched_controls)} controls; objective={obj_val:.3f}")

        return matched_df, match_map, match_map_idx


    def get_preprocessed_control_case_features(
        self,
        cases: pd.DataFrame,
        controls: pd.DataFrame,
        exclude_cols_matching: List[str],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Use preprocessing pipeline to preserve the feature engineering approach.
        Return X_control, X_cases
        """
        # Step 1: Your existing feature categorization logic
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

        # Step 2: Build your sophisticated preprocessor
        preprocessor = get_preprocessor(
            df=combined_df,
            categorical_cols=categorical_cols,
            numeric_cols=num_feats,verbose=False
        )
        
        X_combined = preprocessor.fit_transform(combined_df)
        n_cases = len(cases)
        X_cases = X_combined[:n_cases]
        X_controls = X_combined[n_cases:]
        if verbose:
            print(f"After preprocessing: {X_cases.shape[1]} features")
        
        return X_cases, X_controls

    def get_lexicographic_candidates(self, X_cases, X_controls, 
                                   top_k_factor=3.0, 
                                   similarity_threshold=0.7,
                                   inverse_matching=False):
        """
        Traditional: Sort similarity descending + Select TOP K most similar sparsity patterns
        Inverse: Sort similarity ascending + Select Top K least similar sparsity patterns
        """
        X_combined = np.vstack([X_cases, X_controls])
        sorted_feature_indices = sort_features_by_density(X_combined, range(X_combined.shape[1]))
        
        case_signatures = []
        for case_idx in range(len(X_cases)):
            signature = tuple(int(X_cases[case_idx, feat_idx] != 0) 
                            for feat_idx in sorted_feature_indices)
            case_signatures.append((case_idx, signature))
        
        control_signatures = []
        for control_idx in range(len(X_controls)):
            signature = tuple(int(X_controls[control_idx, feat_idx] != 0) 
                            for feat_idx in sorted_feature_indices)
            control_signatures.append((control_idx, signature))
        
        candidate_pairs = []
        for case_idx, case_sig in case_signatures:
            similarities = []
            for control_idx, control_sig in control_signatures:
                matches = sum(a == b for a, b in zip(case_sig, control_sig))
                similarity = matches / len(case_sig)
                similarities.append((control_idx, similarity))
            
            max_candidates = int(len(control_signatures) * top_k_factor / len(case_signatures))

            if inverse_matching:
                # Sort ascending: [0.1, 0.2, 0.3, ..., 0.9, 1.0]
                similarities.sort(key=lambda x: x[1], reverse=False)
                
                #print(f"Similarities: {similarities[:5]}")
                
                # Take BOTTOM K (first max_candidates after sorting ascending)
                for control_idx, sim in similarities[:max_candidates]:
                    candidate_pairs.append((case_idx, control_idx))
                  
            else:
                # Original logic: most similar controls
                similarities.sort(key=lambda x: x[1], reverse=True)
                #print(f"Similarities: {similarities[:5]}")
                for control_idx, sim in similarities[:max_candidates]:
                    if sim >= similarity_threshold:
                        candidate_pairs.append((case_idx, control_idx))
    
        return candidate_pairs


    def ortools_optimization_matching(self, dfA, dfB, target_controls, target_cases, 
                                    exclude_cols_matching, verbose=True, max_controls_per_case=1,
                                    top_k_factor=10.0, similarity_threshold=0.7,
                                    disable_lexicographic=False, inverse_matching=False):

        # 1) Features
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            dfA, dfB, exclude_cols_matching, verbose
        )

        # Early guards
        if (X_cases is None) or (X_controls is None) or (len(X_cases) == 0) or (len(X_controls) == 0):
            raise RuntimeError("Empty cases or controls after preprocessing.")

        # 2) Distances
        D = cdist(X_cases, X_controls, metric="euclidean")

        # 3) Candidate selection
        if not disable_lexicographic:
            # Euclidean top-K pruning per case
            candidate_pairs = []  #  initialize
            k = int(top_k_factor)
            k = max(1, min(k, len(X_controls)))  # cap to [1, n_controls]
            kth = k - 1

            for i in range(len(X_cases)):
                # protect against degenerate row (all nan/inf) by falling back to full row if needed
                row = D[i]
                # np.argpartition requires 0 <= kth < len(row)
                top_k_idx = np.argpartition(row, kth)[:k]
                # stabilize by true distance order among the k
                top_k_idx = top_k_idx[np.argsort(row[top_k_idx])]
                candidate_pairs.extend((i, j) for j in top_k_idx)

            total_possible = len(X_cases) * len(X_controls)

            # Build a dict the solver expects
            distances_dict = {(i, j): float(D[i, j]) for (i, j) in candidate_pairs}

            if verbose:
                kept_pct = (len(candidate_pairs) / total_possible) * 100.0
                if len(candidate_pairs) > 0:
                    dv = list(distances_dict.values())
                    print(f"[Mode: Euclidean top-K] Retained {len(candidate_pairs):,}/{total_possible:,} "
                        f"pairs ({kept_pct:.2f}%) — k={k} per case")
                    print(f"Distance range for sorted pairs: [{min(dv):.3f}, {max(dv):.3f}]")
                else:
                    print("[Mode: Euclidean top-K] Retained 0 candidate pairs — increase top_k_factor.")

            # 4) Solve
            return self._solve_optimization_with_pairs(
                dfA, dfB, candidate_pairs, distances_dict,
                target_controls, target_cases, max_controls_per_case, verbose,
                inverse_matching=inverse_matching
            )

        else:
            # No pruning: all pairs
            candidate_pairs = [(i, j) for i in range(len(X_cases)) for j in range(len(X_controls))]
            if verbose:
                print(f"Disabled sorting, full match. Pairs: {len(candidate_pairs):,}")
                print(f"Distance range for {len(candidate_pairs)} pairs: [{D.min():.3f}, {D.max():.3f}]")

            distances_dict = {(i, j): float(D[i, j]) for (i, j) in candidate_pairs}

            return self._solve_optimization_with_pairs(
                dfA, dfB, candidate_pairs, distances_dict,
                target_controls, target_cases, max_controls_per_case, verbose,
                inverse_matching=inverse_matching
            )
