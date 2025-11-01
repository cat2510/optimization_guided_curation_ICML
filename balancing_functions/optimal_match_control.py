# Enhanced version of your match_control.py with optimization-based matching

import pandas as pd
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Union, Any
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from model_pipeline import get_preprocessor, get_bin_flag_columns
from ortools.linear_solver import pywraplp

class EnhancedRiskBinnedCaseControlResampler:
    """
    Enhanced version of the match_control.py RiskBinnedCaseControlResampler with optimization-based matching.
    """
    
    def __init__(
        self,
        bin_edges: Optional[np.ndarray] = None,
        impute_strategy: str = "median",
        uid_col: str = "uid",
        random_state: int = 42,
        proba_col: str = "risk_score",
        binary_group: str = "true_class",
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
        self.proba_col = proba_col
        self.matching_method = matching_method  # "optimization", "ortools", or "knn"
        self.solver = solver

    def assign_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["risk_bin"] = pd.cut(
            df[self.proba_col],
            bins=self.bin_edges,
            labels=self.bin_labels,
            include_lowest=True,
            right=False,
        )
        return df

    
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
        drop_cols = [self.uid_col, self.binary_group, self.proba_col, "risk_bin"] + exclude_cols_matching
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
            print("    Categorical features (one-hot encoded):", categorical_cols)
            print("    Numeric   features (normalized):", num_feats)
            print("    Binary    features (unchanged):", bin_feats)

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

    def ortools_optimization_matching(
        self,
        dfA: pd.DataFrame,
        dfB: pd.DataFrame, 
        target_controls: int,
        target_cases: int,
        exclude_cols_matching: List[str],
        verbose: bool = True,
        max_controls_per_case: int = 1  # NEW parameter passed directly from bin strategy

    ) -> pd.DataFrame:
        """
        Optimization-based matching using Google's OR-Tools.
        dfA is the case group, dfB is the control group.
        
        Parameters:
        -----------
        target_cases: Number of cases to match
        target_controls: Number of controls to match (total)
        max_controls_per_case: Maximum number of controls per case
        """
        print(f"=== ENTERING ortools_optimization_matching ===")
        # Get preprocessed features for distance calculation
        X_cases, X_controls = self.get_preprocessed_control_case_features(
            dfA, dfB, exclude_cols_matching, verbose)
        distances = cdist(X_cases, X_controls, metric='euclidean')
        if verbose:
            print(f"Distance matrix shape: {distances.shape}")
            print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
            print(f"Target cases: {target_cases}, Target controls: {target_controls}")
            print(f"Max controls per case: {max_controls_per_case}")
        
        
        m_cases, n_controls = distances.shape
        target_cases = min(target_cases, m_cases)
        target_controls = min(target_controls, n_controls)
    
        solver = pywraplp.Solver.CreateSolver('SCIP')
        # Create variables - only for valid assignments
        z = {}
        for i in range(m_cases):
            for j in range(n_controls):
                z[i, j] = solver.BoolVar(f'z_{i}_{j}')
        
        if verbose:
            print(f"Created {len(z)} variables out of {m_cases * n_controls} possible")
        
        # Define constraints
        # 1. Each case gets matched to at least 1 control (if available)
        for i in range(target_cases):
            case_matches = [z[i, j] for j in range(n_controls) if (i, j) in z]
            if case_matches:  # Only add constraint if there are possible matches
                solver.Add(solver.Sum(case_matches) >= 1)
    
        # 2. Each case gets at most max_controls_per_case controls
        for i in range(m_cases):
            case_matches = [z[i, j] for j in range(n_controls) if (i, j) in z]
            if case_matches:  # Only add constraint if there are possible matches
                solver.Add(solver.Sum(case_matches) <= max_controls_per_case)
        
        # 3. Each control matched to at most 1 case
        for j in range(n_controls):
            control_matches = [z[i, j] for i in range(m_cases) if (i, j) in z]
            if control_matches:  # Only add constraint if there are possible matches
                solver.Add(solver.Sum(control_matches) <= 1)
        
        # 4. Total matches constraint
        total_matches = [z[i, j] for i, j in z]
        solver.Add(solver.Sum(total_matches) >= min(target_cases, m_cases))
        
        # Objective: minimize total distance
        objective_terms = [distances[i, j] * z[i, j] for i, j in z]
        solver.Minimize(solver.Sum(objective_terms))
        
        # Set time limit and solve
        solver.SetTimeLimit(300000)  # 5 minutes (300,000 milliseconds)
        start_time = time.time()
        if verbose:
            print("Starting optimization...")
        
        status = solver.Solve()
        
        solve_time = time.time() - start_time
        if verbose:
            print(f"Optimization completed in {solve_time:.2f} seconds")
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract matches
            matched_case_indices = set()
            matched_control_indices = set()
            
            for i, j in z:
                if z[i, j].solution_value() > 0.5:
                    matched_case_indices.add(i)
                    matched_control_indices.add(j)
            
            matched_cases = dfA.iloc[list(matched_case_indices)]
            matched_controls = dfB.iloc[list(matched_control_indices)]
            
            matched = pd.concat([matched_cases, matched_controls], ignore_index=True)
            matched = matched.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            if verbose:
                obj_value = sum(distances[i, j] * z[i, j].solution_value() for i, j in z)
                print(f"Optimization successful: {len(matched_cases)} cases, {len(matched_controls)} controls")
                print(f"Objective value: {obj_value:.3f}")
            
            return matched
        else:
            print(f"Optimization failed: {status}, falling back to KNN matching")
            return self.knn_matching(
            dfA, dfB, target_controls, target_cases, exclude_cols_matching, verbose
        )        
           
    def match_case_control(
        self, 
        df: pd.DataFrame, 
        target_controls: int, 
        target_cases: int, 
        exclude_cols_matching: List[str] = [],
        verbose: bool = False,
        max_controls_per_case: int = 1  # Default to 1:1 matching

    ) -> pd.DataFrame:
        """
        Enhanced match_case_control that can use either optimization or KNN.
        """
        cases = df[df[self.binary_group] == 1].reset_index(drop=True)
        controls = df[df[self.binary_group] == 0].reset_index(drop=True)
        
        # Handle empty groups
        if len(cases) == 0:
            return controls.sample(n=min(target_controls, len(controls)), random_state=self.random_state)
        if len(controls) == 0:
            return cases.sample(n=min(target_cases, len(cases)), random_state=self.random_state)
        print("Using matching method: ", self.matching_method)
        # Choose matching method
        if self.matching_method in ["optimization", "ortools"]:
            return self.ortools_optimization_matching(
                cases, controls, target_controls, target_cases, exclude_cols_matching, verbose,max_controls_per_case
            )
        else:
            return self.knn_matching(
                cases, controls, target_controls, target_cases, exclude_cols_matching, verbose
            )
    def knn_matching(
        self, cases: pd.DataFrame, controls: pd.DataFrame, target_controls: int, target_cases: int, 
        exclude_cols_matching: List[str]=[], verbose=False
    ) -> pd.DataFrame:
        """
        1:1 matching: match target_cases cases to target_controls controls.
        Total matched patients will be limited to target_cases + target_controls.
        """
        
        X_cases, X_controls = self.get_preprocessed_control_case_features(cases, controls, exclude_cols_matching, verbose)
        
        # -------- distance: control → nearest case ----------------------
        nbrs_cases = NearestNeighbors(
            n_neighbors=min(len(cases), len(controls))).fit(X_cases)
        d_ctrl, _ = nbrs_cases.kneighbors(X_controls)
        min_d_ctrl = d_ctrl.min(axis=1)  # length == len(controls)

        # keep the controls with the *smallest* of these distances
        pick_idx_ctrl = np.argsort(min_d_ctrl)[:min(target_controls, len(controls))]
        matched_ctrl = controls.iloc[pick_idx_ctrl]

        # -------- distance: case → nearest control ----------------------
        nbrs_ctrl = NearestNeighbors(
            n_neighbors=min(len(cases), len(controls))).fit(X_controls)
        d_case, _ = nbrs_ctrl.kneighbors(X_cases)
        min_d_case = d_case.min(axis=1)  # length == len(cases)

        pick_idx_case = np.argsort(min_d_case)[:min(target_cases, len(cases))]
        matched_case = cases.iloc[pick_idx_case]

        # -------- assemble & shuffle ------------------------------------
        matched = (
            pd.concat([matched_case, matched_ctrl], ignore_index=True)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )
        
        if verbose:
            print(f"KNN matching: {len(matched_case)} cases and {len(matched_ctrl)} controls")
            
        return matched.copy()

    def apply_sampler_by_bin(
        self, 
        df: pd.DataFrame,
        exclude_cols_matching=[],
        bin_specific_strategies: Optional[Dict[str, Dict]] = None,
        verbose: bool = False,
        target_bin_size: Optional[int] = None  # Optional explicit size for all bins
    ):
        if target_bin_size is not None:
            bin_target = target_bin_size
        else:
            bin_target = int(df["risk_bin"].value_counts().min())# Use mean bin size as default

        if verbose:
            print(f"Target bin size: {bin_target}")

        removed_ids, parts = {}, []
        
        # Your existing preprocessing setup
        do_not_match = exclude_cols_matching + [self.uid_col, self.binary_group, self.proba_col]
        
        if verbose:
            print(f"→ We will not match on: {do_not_match}")

        for b, bin_df in df.groupby("risk_bin", dropna=True):
            if verbose:
                print(f"\n--- Processing bin {b} (n={len(bin_df)}) ---")

            # Get bin-specific strategy if provided
            bin_strategy = bin_specific_strategies.get(b, {}) if bin_specific_strategies else {}
            
            # Temporarily override matching method for this bin if specified
            original_method = self.matching_method
            if "matching_method" in bin_strategy:
                self.matching_method = bin_strategy["matching_method"]

            n_cases = (bin_df[self.binary_group] == 1).sum()
            n_ctrl = (bin_df[self.binary_group] == 0).sum()

            # Determine targets using bin-specific or global strategy
            max_ctrl_per_case = bin_strategy.get("max_ctrl_per_case", 1) 
            preserve_all_cases = bin_strategy.get("preserve_all_cases", False)
            if preserve_all_cases:
                # High-risk bin strategy: keep all cases, match appropriate controls
                n_keep_case = n_cases
                n_keep_ctrl = min(n_ctrl, max_ctrl_per_case * n_cases)
                if verbose:
                    print(f"Preserve all cases strategy: keeping {n_keep_case} cases, {n_keep_ctrl} controls")
            else:
               #  Try to get half cases, half controls
                n_keep_case = min(n_cases, bin_target )#// 2
                n_keep_ctrl = min(n_ctrl, bin_target ) #- n_keep_case

            # Perform matching
            matched = self.match_case_control(
                bin_df, 
                target_controls=n_keep_ctrl, 
                target_cases=n_keep_case,
                exclude_cols_matching=do_not_match,
                verbose=verbose,
                max_controls_per_case=max_ctrl_per_case
            )

            # Restore original matching method
            self.matching_method = original_method
            # Restore excluded columns that were dropped during matching
            excluded_cols_to_restore = [col for col in do_not_match if col in bin_df.columns and col not in matched.columns]
            if excluded_cols_to_restore:
                # Get the excluded columns for the matched IDs
                matched_ids = matched[self.uid_col]
                excluded_data = bin_df[bin_df[self.uid_col].isin(matched_ids)][excluded_cols_to_restore + [self.uid_col]]
                
                # Merge back the excluded columns
                matched = matched.merge(excluded_data, on=self.uid_col, how='left')

            # Track removed IDs
            removed_ids[b] = list(
                set(bin_df[self.uid_col]) - set(matched[self.uid_col])
            )
            parts.append(matched.assign(risk_bin=b))

            if verbose:
                print(f"Bin {b}: {len(bin_df)} → {len(matched)} "
                      f"(cases: {n_cases} → {(matched[self.binary_group] == 1).sum()}, "
                      f"controls: {n_ctrl} → {(matched[self.binary_group] == 0).sum()})")

        return pd.concat(parts).reset_index(drop=True), removed_ids


def get_high_cost_detection_config(primary_matching_method="ortools"):
    """
    Returns optimized configuration for high-cost detection in CKD patients.
    """
    # Bin-specific strategies prioritizing recall in high-risk bins
    bin_strategies = {
        # Low-risk bins: aggressive downsampling
        "0.00-0.10": {
            "preserve_all_cases": True,
            "matching_method": primary_matching_method,
            "max_ctrl_per_case": 1
        },
        "0.10-0.20": {
            "preserve_all_cases": True,
            "matching_method": primary_matching_method, 
            "max_ctrl_per_case": 1
        },
        "0.20-0.30": {
            "preserve_all_cases": True,
            "matching_method": primary_matching_method,
            "max_ctrl_per_case": 1
        },
        
        # Medium-risk bins: moderate sampling
        "0.30-0.40": {
            "matching_method": primary_matching_method,
            "preserve_all_cases": True,

            "max_ctrl_per_case": 1
        },
        "0.40-0.50": {
            "matching_method": primary_matching_method,
            "preserve_all_cases": True,

            "max_ctrl_per_case": 1
        },
        "0.50-0.60": {
            "matching_method": primary_matching_method,
            "preserve_all_cases": True,
            "max_ctrl_per_case": 1
        },
        "0.60-0.70": {
            "matching_method": primary_matching_method, 
            "preserve_all_cases": True,
            "max_ctrl_per_case": 1
        },
        
        # High-risk bins: preserve minority class
        "0.70-0.80": {
            "matching_method": primary_matching_method,
            "preserve_all_cases": True,
            "max_ctrl_per_case": 1
        },
        "0.80-0.90": {
            "matching_method": primary_matching_method,
            "preserve_all_cases": True,
            "max_ctrl_per_case": 1
        },
        "0.90-1.00": {
            "matching_method": primary_matching_method,
            "preserve_all_cases": True,
            "max_ctrl_per_case": 1  # Allow more controls per case in highest risk
        }
    }
    
    return bin_strategies
   
    