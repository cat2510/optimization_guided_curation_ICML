def select_candidate_controls(self, X_cases, X_controls, K, verbose=True):
    """
    Inner problem: select K majority samples minimizing distance to minority samples.
    Solves a p-median/facility-location problem using OR-Tools MIP.
    """
    n_cases, n_controls = X_cases.shape[0], X_controls.shape[0]
    D = cdist(X_cases, X_controls, metric="euclidean")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    y = [solver.BoolVar(f"y_{j}") for j in range(n_controls)]
    z = [[solver.NumVar(0, 1, f"z_{i}_{j}") for j in range(n_controls)] for i in range(n_cases)]

    # Each case assigned to exactly one selected control
    for i in range(n_cases):
        solver.Add(solver.Sum(z[i][j] for j in range(n_controls)) == 1)
        for j in range(n_controls):
            solver.Add(z[i][j] <= y[j])

    # Select exactly K controls
    solver.Add(solver.Sum(y) == K)

    # Objective: minimize total distance
    objective = solver.Sum(D[i, j] * z[i][j] for i in range(n_cases) for j in range(n_controls))
    solver.Minimize(objective)
    solver.SetTimeLimit(300000)
    status = solver.Solve()

    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        raise RuntimeError("Inner optimization failed.")

    selected_controls = [j for j in range(n_controls) if y[j].solution_value() > 0.5]
    if verbose:
        print(f"Selected {len(selected_controls)} controls as candidates.")
    return selected_controls
def select_diverse_subset(self, X_controls, candidate_indices, k, metric="euclidean"):
    """
    Outer problem: select a diverse subset S ⊆ C using greedy facility-location.
    """
    from sklearn.metrics import pairwise_distances
    from tqdm import trange

    X_C = X_controls[candidate_indices]
    # Convert distance to similarity (bounded in [0,1])
    D = pairwise_distances(X_C, X_C, metric=metric)
    sim = np.exp(-D / (D.std() + 1e-8))

    nC = len(candidate_indices)
    selected = []
    covered = np.zeros(nC)

    for _ in trange(k, desc="Selecting diverse subset"):
        gains = np.zeros(nC)
        for j in range(nC):
            if j in selected:
                gains[j] = -np.inf
            else:
                new_cover = np.maximum(covered, sim[:, j])
                gains[j] = new_cover.sum() - covered.sum()
        best = np.argmax(gains)
        selected.append(best)
        covered = np.maximum(covered, sim[:, best])

    return [candidate_indices[idx] for idx in selected]
