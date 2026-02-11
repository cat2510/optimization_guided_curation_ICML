"""
Tests for bin-quota constrained min-cost flow matching.

Run with:  python -m pytest public/test_quota_matching.py -v
"""
import numpy as np
import pytest
import sys
import os

# Ensure parent directory is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from public.two_stage_kcenter_match import (
    compute_proximity_scores,
    deterministic_binning,
    compute_bin_quotas,
    solve_min_cost_assignment_adaptive_topk,
    solve_min_cost_assignment_with_quotas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_cost(n_cases=20, n_controls=200, seed=42):
    """Build a synthetic cost matrix from 2-D Euclidean distances (fixed seed)."""
    rng = np.random.RandomState(seed)
    X_cases = rng.randn(n_cases, 2).astype(np.float32)
    X_controls = rng.randn(n_controls, 2).astype(np.float32)
    # d_pn: (n_controls, n_cases)  -- same layout as real code
    diff = X_controls[:, np.newaxis, :] - X_cases[np.newaxis, :, :]
    d_pn = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
    return d_pn  # (n_controls, n_cases)


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

class TestProximityScores:
    def test_basic(self):
        d = np.array([[1.0, 2.0], [3.0, 0.5], [2.0, 2.0]], dtype=np.float32)
        b = compute_proximity_scores(d)
        np.testing.assert_allclose(b, [1.0, 0.5, 2.0])

    def test_batch_consistency(self):
        rng = np.random.RandomState(0)
        d = rng.rand(100, 10).astype(np.float32)
        b1 = compute_proximity_scores(d, batch_size=3)
        b2 = compute_proximity_scores(d, batch_size=100)
        np.testing.assert_allclose(b1, b2)


class TestBinning:
    def test_basic_3bins(self):
        b = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
        bins, cuts, counts, active = deterministic_binning(b, T=3)
        assert len(active) <= 3
        assert sum(counts) == len(b)
        assert set(bins) == set(range(len(active)))

    def test_all_equal(self):
        b = np.ones(50)
        bins, cuts, counts, active = deterministic_binning(b, T=5)
        # All identical -> single active bin
        assert len(active) == 1
        assert counts == [50]
        assert np.all(bins == 0)

    def test_determinism(self):
        b = np.random.RandomState(7).rand(200)
        r1 = deterministic_binning(b, T=4)
        r2 = deterministic_binning(b, T=4)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_custom_quantiles(self):
        b = np.arange(100, dtype=np.float64)
        # Use explicit quantiles for T=4
        bins, _, counts, active = deterministic_binning(
            b, quantiles=[0, 0.25, 0.5, 0.75, 1.0]
        )
        assert len(active) == 4
        assert sum(counts) == 100


class TestQuotas:
    def test_pool_mass_sums(self):
        counts = [40, 60, 100]
        q = compute_bin_quotas(counts, n_target=20, mode="pool_mass")
        assert sum(q) == 20
        assert all(q[t] <= counts[t] for t in range(3))

    def test_tilted_sums(self):
        counts = [40, 60, 100]
        mids = [0.1, 0.5, 1.0]
        q = compute_bin_quotas(
            counts, 20, mode="tilted", lamda=1.0, bin_midpoints=mids
        )
        assert sum(q) == 20
        assert all(q[t] <= counts[t] for t in range(3))

    def test_cap_redistribute(self):
        # bin 0 has only 2 members but would get > 2 proportionally
        counts = [2, 50, 50]
        q = compute_bin_quotas(counts, n_target=20, mode="pool_mass")
        assert sum(q) == 20
        assert q[0] <= 2

    def test_exact_edge(self):
        # All in one bin
        counts = [100]
        q = compute_bin_quotas(counts, n_target=50, mode="pool_mass")
        assert q == [50]


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestQuotaSolver:
    def test_basic_quota_matching(self):
        d_pn = make_synthetic_cost(n_cases=20, n_controls=200, seed=42)
        cost = d_pn.T  # (20, 200)

        b_J = compute_proximity_scores(d_pn)
        bins, _, bin_counts, _ = deterministic_binning(b_J, T=3)
        quotas = compute_bin_quotas(bin_counts, n_target=20, mode="pool_mass")

        m, c, diag = solve_min_cost_assignment_with_quotas(
            cost_matrix=cost,
            bin_assignments=bins,
            quotas=quotas,
            K_per_bin=25,
        )

        # All cases matched
        assert len(m) == 20
        assert np.all(m >= 0)
        # Distinct controls
        assert len(set(m.tolist())) == 20

        # Bin quotas exactly met
        matched_bins = bins[m]
        for t, q in enumerate(quotas):
            assert int(np.sum(matched_bins == t)) == q, (
                f"Bin {t}: expected {q}, got {int(np.sum(matched_bins == t))}"
            )

    def test_determinism(self):
        d_pn = make_synthetic_cost(n_cases=20, n_controls=200, seed=42)
        cost = d_pn.T
        b_J = compute_proximity_scores(d_pn)
        bins, _, counts, _ = deterministic_binning(b_J, T=3)
        quotas = compute_bin_quotas(counts, 20, mode="pool_mass")

        m1, c1, _ = solve_min_cost_assignment_with_quotas(
            cost, bins, quotas, K_per_bin=25
        )
        m2, c2, _ = solve_min_cost_assignment_with_quotas(
            cost, bins, quotas, K_per_bin=25
        )

        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(c1, c2)

    def test_small_K_works(self):
        """Even with K_per_bin=1, solver succeeds (via retries if needed)."""
        d_pn = make_synthetic_cost(n_cases=20, n_controls=25, seed=42)
        cost = d_pn.T  # (20, 25)
        b_J = compute_proximity_scores(d_pn)
        bins, _, counts, _ = deterministic_binning(b_J, T=3)
        quotas = compute_bin_quotas(counts, 20, mode="pool_mass")

        m, c, diag = solve_min_cost_assignment_with_quotas(
            cost, bins, quotas, K_per_bin=1, K_growth=2
        )
        # Verify diagnostics structure
        assert "retries" in diag
        assert "K_per_bin_final" in diag
        assert "K_per_bin_schedule" in diag
        assert diag["retries"] >= 0
        assert len(m) == 20
        # Still satisfies quotas
        matched_bins = bins[m]
        for t, q in enumerate(quotas):
            assert int(np.sum(matched_bins == t)) == q

    def test_tilted_mode(self):
        d_pn = make_synthetic_cost(n_cases=20, n_controls=200, seed=42)
        cost = d_pn.T
        b_J = compute_proximity_scores(d_pn)
        bins, _, counts, active = deterministic_binning(b_J, T=3)
        T_active = len(active)

        mids = []
        for t in range(T_active):
            v = b_J[bins == t]
            mids.append(float((v.min() + v.max()) / 2))

        quotas = compute_bin_quotas(
            counts, 20, mode="tilted", lamda=1.0, bin_midpoints=mids
        )

        m, c, _ = solve_min_cost_assignment_with_quotas(
            cost, bins, quotas, K_per_bin=50
        )
        matched_bins = bins[m]
        for t, q in enumerate(quotas):
            assert int(np.sum(matched_bins == t)) == q


class TestBackwardCompatibility:
    def test_baseline_unchanged(self):
        """Existing solver produces identical results when called twice."""
        d_pn = make_synthetic_cost(n_cases=20, n_controls=200, seed=42)
        cost = d_pn.T
        m1, c1 = solve_min_cost_assignment_adaptive_topk(
            cost, topk_start=None
        )
        m2, c2 = solve_min_cost_assignment_adaptive_topk(
            cost, topk_start=None
        )
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(c1, c2)

    def test_baseline_with_sparse(self):
        """Sparse (topk_start=50) produces valid matching."""
        d_pn = make_synthetic_cost(n_cases=20, n_controls=200, seed=42)
        cost = d_pn.T
        m, c = solve_min_cost_assignment_adaptive_topk(
            cost, topk_start=50
        )
        assert len(m) == 20
        assert np.all(m >= 0)
        assert len(set(m.tolist())) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
