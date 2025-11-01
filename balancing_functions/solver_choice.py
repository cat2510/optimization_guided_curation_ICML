import cvxpy as cp
import numpy as np
from typing import Optional, Dict, Any
import warnings

class OptimizedMatchingSolver:
    """
    Solver configuration optimized for your matching optimization problem.
    Prioritizes performance and reliability over being "free".
    """
    
    def __init__(self, preferred_solver: str = "auto"):
        self.preferred_solver = preferred_solver
        self.available_solvers = self._check_available_solvers()
        self.chosen_solver = self._select_best_solver()
        
    def _check_available_solvers(self) -> Dict[str, bool]:
        """Check which solvers are actually installed."""
        solvers = {}
        
        # Commercial solvers (best performance)
        try:
            cp.Problem(cp.Minimize(0), []).solve(solver=cp.GUROBI, verbose=False)
            solvers['GUROBI'] = True
        except:
            solvers['GUROBI'] = False
            
        try:
            cp.Problem(cp.Minimize(0), []).solve(solver=cp.MOSEK, verbose=False)
            solvers['MOSEK'] = True
        except:
            solvers['MOSEK'] = False
            
        # Good free solvers for integer problems
        try:
            cp.Problem(cp.Minimize(0), []).solve(solver=cp.SCIP, verbose=False)
            solvers['SCIP'] = True
        except:
            solvers['SCIP'] = False
            
        # Fallback options
        try:
            cp.Problem(cp.Minimize(0), []).solve(solver=cp.CBC, verbose=False)
            solvers['CBC'] = True
        except:
            solvers['CBC'] = False
            
        try:
            cp.Problem(cp.Minimize(0), []).solve(solver=cp.CLARABEL, verbose=False)
            solvers['CLARABEL'] = True
        except:
            solvers['CLARABEL'] = False
            
        return solvers
    
    def _select_best_solver(self) -> str:
        """Select the best available solver for matching problems."""
        if self.preferred_solver != "auto":
            if self.available_solvers.get(self.preferred_solver, False):
                return self.preferred_solver
            else:
                warnings.warn(f"Preferred solver {self.preferred_solver} not available, auto-selecting...")
        
        # Priority order for matching optimization
        solver_priority = [
            'GUROBI',    # Best: commercial, very fast
            'MOSEK',     # Excellent: commercial, fast  
            'SCIP',      # Good: free, designed for integer problems
            'CBC',       # OK: free, basic integer solver
            'CLARABEL'   # Last resort: continuous solver, poor for integer
        ]
        
        for solver in solver_priority:
            if self.available_solvers.get(solver, False):
                if solver in ['CLARABEL'] and any(self.available_solvers.get(s, False) for s in solver_priority[:-1]):
                    continue  # Skip CLARABEL if better options exist
                return solver
        
        raise RuntimeError("No suitable solver found for integer optimization!")
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Return information about the chosen solver."""
        performance_ranking = {
            'GUROBI': {'speed': 10, 'reliability': 10, 'cost': 'Free for academic'},
            'MOSEK': {'speed': 9, 'reliability': 10, 'cost': 'Free for academic'},
            'SCIP': {'speed': 6, 'reliability': 8, 'cost': 'Free'},
            'CBC': {'speed': 4, 'reliability': 6, 'cost': 'Free'},
            'CLARABEL': {'speed': 2, 'reliability': 3, 'cost': 'Free', 'note': 'Not designed for integer problems'}
        }
        
        return {
            'chosen_solver': self.chosen_solver,
            'available_solvers': self.available_solvers,
            'performance': performance_ranking.get(self.chosen_solver, {}),
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get specific recommendations based on chosen solver."""
        if self.chosen_solver == 'GUROBI':
            return "Excellent choice! Gurobi will give you the fastest, most reliable results."
        elif self.chosen_solver == 'MOSEK':
            return "Great choice! MOSEK is excellent for your matching optimization."
        elif self.chosen_solver == 'SCIP':
            return "Good free alternative. Performance will be slower than commercial solvers but reliable."
        elif self.chosen_solver == 'CBC':
            return "Basic free solver. Consider upgrading to GUROBI/MOSEK for better performance."
        elif self.chosen_solver == 'CLARABEL':
            return "⚠️  CLARABEL is not optimal for integer problems. Strongly recommend installing GUROBI (free academic license)."
        else:
            return "Unknown solver configuration."

    def solve_matching_problem(
        self, 
        distance_matrix: np.ndarray,
        max_matches_per_A: int = 2,
        max_matches_per_B: int = 2,
        min_total_matches: int = 0,
        force_one_to_one: bool = False,
        verbose: bool = False
    ):
        """
        Solve the matching optimization with the best available solver.
        """
        m, n = distance_matrix.shape
        
        # Binary decision variables
        z = cp.Variable((m, n), boolean=True)
        
        # Objective: minimize total distance
        objective = cp.Minimize(cp.sum(cp.multiply(distance_matrix, z)))
        
        # Constraints
        constraints = []
        
        if force_one_to_one:
            # 1:1 matching
            for i in range(m):
                constraints.append(cp.sum(z[i, :]) == 1)
            for j in range(n):
                constraints.append(cp.sum(z[:, j]) <= 1)
        else:
            # Flexible matching
            for i in range(m):
                constraints.append(cp.sum(z[i, :]) <= max_matches_per_A)
            for j in range(n):
                constraints.append(cp.sum(z[:, j]) <= max_matches_per_B)
            if min_total_matches > 0:
                constraints.append(cp.sum(z) >= min_total_matches)
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        # Solver-specific configurations
        solver_params = self._get_solver_params(verbose)
        
        try:
            if self.chosen_solver == 'GUROBI':
                result = problem.solve(solver=cp.GUROBI, verbose=verbose, **solver_params)
            elif self.chosen_solver == 'MOSEK':
                result = problem.solve(solver=cp.MOSEK, verbose=verbose, **solver_params)
            elif self.chosen_solver == 'SCIP':
                result = problem.solve(solver=cp.SCIP, verbose=verbose, **solver_params)
            elif self.chosen_solver == 'CBC':
                result = problem.solve(solver=cp.CBC, verbose=verbose, **solver_params)
            elif self.chosen_solver == 'CLARABEL':
                # CLARABEL doesn't handle integer well, so relax and round
                warnings.warn("Using CLARABEL for integer problem - results may be suboptimal")
                z_relaxed = cp.Variable((m, n), nonneg=True)
                constraints_relaxed = [c for c in constraints] + [z_relaxed <= 1]
                problem_relaxed = cp.Problem(
                    cp.Minimize(cp.sum(cp.multiply(distance_matrix, z_relaxed))),
                    constraints_relaxed
                )
                result = problem_relaxed.solve(solver=cp.CLARABEL, verbose=verbose)
                if problem_relaxed.status == cp.OPTIMAL:
                    # Round to nearest integer
                    z.value = np.round(z_relaxed.value).astype(int)
            else:
                result = problem.solve(verbose=verbose)
            
            if problem.status != cp.OPTIMAL:
                if verbose:
                    print(f"Warning: Optimization status: {problem.status}")
                return float('inf'), np.zeros((m, n))
            
            optimal_cost = problem.value
            z_optimal = (z.value > 0.5).astype(int)
            
            if verbose:
                print(f"Solver: {self.chosen_solver}")
                print(f"Optimal cost: {optimal_cost:.3f}")
                print(f"Total matches: {z_optimal.sum()}")
            
            return optimal_cost, z_optimal
            
        except Exception as e:
            if verbose:
                print(f"Optimization failed with {self.chosen_solver}: {e}")
            return float('inf'), np.zeros((m, n))
    
    def _get_solver_params(self, verbose: bool) -> Dict:
        """Get solver-specific parameters for better performance."""
        if self.chosen_solver == 'GUROBI':
            return {
                'TimeLimit': 300,  # 5 minute time limit
                'MIPGap': 0.01,    # 1% optimality gap tolerance  
                'Threads': -1,     # Use all available cores
            }
        elif self.chosen_solver == 'MOSEK':
            return {
                'MSK_DPAR_MIO_MAX_TIME': 300.0,
                'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
            }
        elif self.chosen_solver == 'SCIP':
            return {
                'limits/time': 300,
                'limits/gap': 0.01,
            }
        else:
            return {}





# Integration with your existing code
"""class EnhancedRiskBinnedMatcherWithProperSolver:
    
    def __init__(self, solver_preference: str = "auto", **kwargs):
        # Initialize solver manager with 
        self.solver_manager = OptimizedMatchingSolver(solver_preference)
        
        # Print solver info once
        info = self.solver_manager.get_solver_info()
        print(f"🔧 Using solver: {info['chosen_solver']}")
        if 'note' in info['performance']:
            print(f"⚠️  {info['performance']['note']}")
        
        # Your existing initialization
        self.bin_edges = kwargs.get('bin_edges', np.linspace(0, 1, 11))
        # ... rest of your init
    
    def optimization_matching(self, dfA, dfB, target_controls, target_cases, exclude_cols, verbose=False):
        # distances = compute_distance_matrix(dfA, dfB, exclude_cols)
        
        # Use the optimized solver instead of hardcoded CLARABEL
        optimal_cost, z_matrix = self.solver_manager.solve_matching_problem(
            distance_matrix=distances,  # your computed distances
            max_matches_per_A=3,
            max_matches_per_B=1,
            min_total_matches=min(target_cases, len(dfA)),
            verbose=verbose
        )
        
        # Your existing logic to extract matched individuals...
        return matched_dataframe
    """


