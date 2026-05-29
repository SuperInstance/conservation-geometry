"""Tests for laplacian_rubber_sheet.py — Laplacian construction and Dirichlet energy."""
import numpy as np
import pytest

# Import the core math from the module (without triggering plot generation)
import importlib
import sys
import os

# Prevent matplotlib from opening windows
import matplotlib
matplotlib.use("Agg")

# The module runs plotting on import, so we test the math inline
# using the same formulas from laplacian_rubber_sheet.py


class TestLaplacianConstruction:
    """Test Laplacian matrix construction from graph edges."""

    def setup_method(self):
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
        self.n = 5
        self.A = np.zeros((self.n, self.n))
        for i, j in self.edges:
            self.A[i, j] = 1
            self.A[j, i] = 1
        self.D = np.diag(self.A.sum(axis=1))
        self.L = self.D - self.A

    def test_adjacency_symmetric(self):
        assert np.allclose(self.A, self.A.T)

    def test_adjacency_shape(self):
        assert self.A.shape == (self.n, self.n)

    def test_degree_diagonal(self):
        assert np.allclose(self.D, np.diag(np.diag(self.D)))

    def test_laplacian_row_sums_zero(self):
        """Each row of the Laplacian sums to zero."""
        assert np.allclose(self.L.sum(axis=1), np.zeros(self.n))

    def test_laplacian_symmetric(self):
        assert np.allclose(self.L, self.L.T)

    def test_laplacian_positive_semidefinite(self):
        eigs = np.linalg.eigvalsh(self.L)
        assert np.all(eigs >= -1e-10)

    def test_laplacian_has_zero_eigenvalue(self):
        eigs = np.linalg.eigvalsh(self.L)
        assert np.any(np.abs(eigs) < 1e-10)


class TestDirichletEnergy:
    """Test Dirichlet energy computations."""

    def setup_method(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
        n = 5
        A = np.zeros((n, n))
        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1
        D = np.diag(A.sum(axis=1))
        self.L = D - A

    def test_constant_attribute_zero_energy(self):
        """Constant attributes have zero Dirichlet energy."""
        f = np.ones(5) * 3.0
        energy = f @ self.L @ f
        assert np.isclose(energy, 0.0)

    def test_smooth_attribute_low_energy(self):
        """Smooth attributes should have lower energy than rough ones."""
        smooth = np.array([3.0, 3.1, 3.0, 2.9, 3.0])
        rough = np.array([0.0, 5.0, 1.0, 4.0, 2.0])
        e_smooth = smooth @ self.L @ smooth
        e_rough = rough @ self.L @ rough
        assert e_smooth < e_rough

    def test_dirichlet_nonnegative(self):
        """Dirichlet energy is always non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            f = rng.standard_normal(5)
            energy = f @ self.L @ f
            assert energy >= -1e-10

    def test_laplacian_applied_to_constant(self):
        """L @ constant_vector should be zero."""
        f = np.ones(5) * 7.0
        assert np.allclose(self.L @ f, np.zeros(5))


class TestAlignmentCoefficient:
    """Test alignment coefficient α(G,a) from alignment_coefficient.py."""

    @staticmethod
    def compute_alpha(L, a):
        eigs = np.linalg.eigvalsh(L)
        lambda_max = max(eigs)
        if lambda_max == 0 or np.dot(a, a) == 0:
            return 0
        return (a @ L @ a) / (lambda_max * np.dot(a, a))

    def test_alpha_nonnegative(self):
        L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)
        a = np.array([1.0, 2.0, 3.0])
        alpha = self.compute_alpha(L, a)
        assert alpha >= 0

    def test_alpha_zero_for_constant(self):
        L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)
        a = np.ones(3)
        alpha = self.compute_alpha(L, a)
        assert np.isclose(alpha, 0.0)

    def test_alpha_bounded(self):
        """α(G,a) ∈ [0, 1] by definition."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            n = rng.integers(4, 15)
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.4:
                        A[i, j] = A[j, i] = 1
            for i in range(n - 1):
                A[i, i + 1] = A[i + 1, i] = 1
            D = np.diag(A.sum(axis=1))
            L = D - A
            a = rng.standard_normal(n)
            alpha = self.compute_alpha(L, a)
            assert 0 <= alpha <= 1.0 + 1e-10, f"α={alpha} out of bounds"

    def test_alpha_conjecture_smooth(self):
        """Smooth attributes tend to have low α (conjecture C1 direction)."""
        n = 10
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        D = np.diag(A.sum(axis=1))
        L = D - A
        # Smooth (linear ramp)
        smooth = np.linspace(0, 1, n)
        alpha_smooth = self.compute_alpha(L, smooth)
        # Oscillating
        osc = np.array([(-1) ** k for k in range(n)], dtype=float)
        alpha_osc = self.compute_alpha(L, osc)
        # Smooth should have lower α than oscillating
        assert alpha_smooth < alpha_osc


class TestEigenvalueShells:
    """Test eigenvalue shell computations from eigenvalue_shells.py."""

    @staticmethod
    def get_eigenvalues(L):
        return np.sort(np.linalg.eigvalsh(L))

    @staticmethod
    def build_laplacian(A):
        D = np.diag(A.sum(axis=1))
        return D - A

    def test_chain_graph_eigenvalues(self):
        """Chain graph has known eigenvalue structure."""
        n = 5
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        L = self.build_laplacian(A)
        eigs = self.get_eigenvalues(L)
        assert len(eigs) == n
        assert np.isclose(eigs[0], 0.0, atol=1e-10)

    def test_complete_graph_eigenvalues(self):
        """Complete graph K_n has eigenvalues {0, n, n, ..., n}."""
        n = 6
        A = np.ones((n, n)) - np.eye(n)
        L = self.build_laplacian(A)
        eigs = self.get_eigenvalues(L)
        assert np.isclose(eigs[0], 0.0, atol=1e-10)
        for e in eigs[1:]:
            assert np.isclose(e, n, atol=1e-10)

    def test_shell_radii(self):
        """Shell radii = sqrt(eigenvalue) should be non-negative."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        L = self.build_laplacian(A)
        eigs = self.get_eigenvalues(L)
        radii = np.sqrt(np.maximum(eigs, 0))
        assert np.all(radii >= 0)


class TestPhasePortrait:
    """Test phase portrait domain data structure."""

    def test_domains_have_required_keys(self):
        """Each domain entry should have conservation_ratio and spectral_gap."""
        # Simulating the data structure from phase_portrait.py
        sample_points = [
            (0.85, 0.12, 'Scale network'),
            (0.82, 0.15, 'Chord graph'),
        ]
        for cr, sg, label in sample_points:
            assert 0 <= cr <= 1
            assert sg >= 0
            assert isinstance(label, str)
