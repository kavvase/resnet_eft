"""Tests for GaussianExpectation (P1 priority tests).

These tests verify the correctness of E2, Epp, E2s calculations.
"""

import math

import pytest
import torch

from resnet_eft import GaussianExpectation, Params
from resnet_eft.backend import allclose
from resnet_eft.core_types import ActivationSpec


def random_psd_matrix(N: int, dtype=torch.float64) -> torch.Tensor:
    """Generate a random positive semi-definite matrix."""
    A = torch.randn(N, N, dtype=dtype)
    return A @ A.T + torch.eye(N, dtype=dtype)


class TestE2ReLU:
    """Tests for E2 with ReLU activation."""

    def test_E2_relu_analytic(self):
        """Test ReLU E2 matches Cho & Saul (2009) formula."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        # Test case: q1 = q2 = 1, rho = 0.5
        q1, q2 = 1.0, 1.0
        rho = 0.5

        # Create simple 2x2 kernel
        K0 = torch.tensor(
            [[q1, rho * math.sqrt(q1 * q2)], [rho * math.sqrt(q1 * q2), q2]], dtype=torch.float64
        )

        E2 = gauss.E2_pairwise(K0)

        # Cho & Saul formula
        theta = math.acos(rho)
        E2_analytic = (
            math.sqrt(q1 * q2)
            / (2 * math.pi)
            * (math.sin(theta) + (math.pi - theta) * math.cos(theta))
        )

        assert abs(E2[0, 1].item() - E2_analytic) < 1e-6

    def test_E2_relu_diagonal(self):
        """Test ReLU E2 diagonal: E[ReLU(φ)²] = q/2."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        q = 2.0
        K0 = torch.tensor([[q]], dtype=torch.float64)

        E2 = gauss.E2_pairwise(K0)

        # E[ReLU(φ)²] = q/2 for φ ~ N(0, q)
        expected = q / 2
        assert abs(E2[0, 0].item() - expected) < 1e-6

    def test_E2_symmetry(self):
        """Test E2 is symmetric."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        K0 = random_psd_matrix(5)
        E2 = gauss.E2_pairwise(K0)

        assert allclose(E2, E2.T)


class TestEppReLU:
    """Tests for Epp with ReLU activation."""

    def test_Epp_relu_formula(self):
        """Test ReLU Epp = (π - θ) / (2π)."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        q = 1.0
        rho = 0.5
        K0 = torch.tensor([[q, rho * q], [rho * q, q]], dtype=torch.float64)

        Epp = gauss.E_sigma_prime_prime(K0)

        # Formula: (π - θ) / (2π)
        theta = math.acos(rho)
        expected = (math.pi - theta) / (2 * math.pi)

        assert abs(Epp[0, 1].item() - expected) < 1e-6

    def test_Epp_relu_boundary_rho1(self):
        """Test Epp at ρ=1: should be 0.5."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        # ρ = 1 means perfectly correlated
        q = 1.0
        K0 = torch.tensor([[q, q], [q, q]], dtype=torch.float64)

        Epp = gauss.E_sigma_prime_prime(K0)

        # At ρ=1, θ=0, so Epp = (π - 0) / (2π) = 0.5
        # Note: eps_rho clips ρ away from 1, so tolerance is relaxed
        assert abs(Epp[0, 1].item() - 0.5) < 1e-3

    def test_Epp_relu_boundary_rho0(self):
        """Test Epp at ρ=0: should be 0.25."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        # ρ = 0 means uncorrelated
        q = 1.0
        K0 = torch.tensor([[q, 0.0], [0.0, q]], dtype=torch.float64)

        Epp = gauss.E_sigma_prime_prime(K0)

        # At ρ=0, θ=π/2, so Epp = (π - π/2) / (2π) = 0.25
        assert abs(Epp[0, 1].item() - 0.25) < 1e-4

    def test_Epp_symmetry(self):
        """Test Epp is symmetric."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        K0 = random_psd_matrix(5)
        Epp = gauss.E_sigma_prime_prime(K0)

        assert allclose(Epp, Epp.T)


class TestE2sReLU:
    """Tests for E2s (⟨σ''σ⟩) with ReLU activation."""

    def test_E2s_relu_positive(self):
        """Test E2s is non-negative for ReLU."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        K0 = random_psd_matrix(4)
        E2s = gauss.E_sigma_dprime_sigma(K0)

        # E2s should be non-negative (it's a product of positive things)
        assert (E2s >= -1e-10).all()


class TestE2Erf:
    """Tests for E2 with erf activation."""

    def test_E2_erf_analytic(self):
        """Test erf E2 matches Williams (1996) formula."""
        params = Params(act="erf", Cw=1.0)
        gauss = GaussianExpectation(params)

        q = 1.0
        rho = 0.5
        K0 = torch.tensor([[q, rho * q], [rho * q, q]], dtype=torch.float64)

        E2 = gauss.E2_pairwise(K0)

        # Williams formula for standard erf (scale=1)
        # E[erf(φ₁)erf(φ₂)] = (2/π) arcsin(2ρq / ((1+2q)(1+2q)))
        # Actually: (2/π) arcsin(2ρ√(q₁q₂/s⁴) / √((1+2q₁/s²)(1+2q₂/s²)))
        s = 1.0  # input_scale
        num = 2 * rho * math.sqrt(q * q / s**4)
        den = math.sqrt((1 + 2 * q / s**2) * (1 + 2 * q / s**2))
        expected = (2 / math.pi) * math.asin(num / den)

        assert abs(E2[0, 1].item() - expected) < 1e-5

    def test_E2_erf_diagonal(self):
        """Test erf E2 diagonal: E[erf(φ)²]."""
        params = Params(act="erf", Cw=1.0)
        gauss = GaussianExpectation(params)

        q = 1.0
        K0 = torch.tensor([[q]], dtype=torch.float64)

        E2 = gauss.E2_pairwise(K0)

        # For diagonal (ρ=1), the formula gives a specific value
        # E[erf(φ)²] where φ ~ N(0, q)
        # = (2/π) arcsin(2q / (1+2q))
        s = 1.0
        expected = (2 / math.pi) * math.asin(2 * q / s**2 / (1 + 2 * q / s**2))

        assert abs(E2[0, 0].item() - expected) < 1e-5


class TestE4:
    """Tests for E4 (4-point expectation)."""

    def test_E4_shape(self):
        """Test E4 has correct shape."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, gh_order=16)
        gauss = GaussianExpectation(params)

        N = 3
        K0 = random_psd_matrix(N)
        E4 = gauss.E4_pairwise(K0)

        assert E4.shape == (N, N, N, N)

    def test_E4_relu_diagonal(self):
        """Test E4 diagonal element for ReLU."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, gh_order=32)
        gauss = GaussianExpectation(params)

        q = 1.0
        K0 = torch.tensor([[q]], dtype=torch.float64)

        E4 = gauss.E4_pairwise(K0)

        # E[ReLU(φ)⁴] for φ ~ N(0, 1)
        # = E[φ⁴ × 1_{φ>0}] = (1/2) × 3 = 3/2 × q² (for q=1)
        # Actually: E[ReLU(φ)⁴] = 3q²/2 for φ ~ N(0, q)
        expected = 3 * q**2 / 2

        assert abs(E4[0, 0, 0, 0].item() - expected) < 0.1  # Tolerance for numerical integration

    def test_E4_symmetry(self):
        """Test E4 has expected symmetries."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, gh_order=16)
        gauss = GaussianExpectation(params)

        N = 2
        K0 = random_psd_matrix(N)
        E4 = gauss.E4_pairwise(K0)

        # E4[i,j,k,l] should be symmetric under permutations of (i,j,k,l)
        # that preserve the Gaussian structure
        # Note: Numerical integration has ~2% error, so use relative tolerance
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for m in range(N):
                        # E4[i,j,k,m] = E4[k,m,i,j] (pair swap)
                        val1 = E4[i, j, k, m].item()
                        val2 = E4[k, m, i, j].item()
                        rel_err = abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
                        assert (
                            rel_err < 0.08  # MC noise can cause ~5-7% variation
                        ), f"E4 symmetry failed at ({i},{j},{k},{m}): {val1} vs {val2}"

    def test_E4_too_large_N_raises(self):
        """Test E4 raises for N > 10."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        K0 = random_psd_matrix(11)

        with pytest.raises(ValueError, match="too expensive"):
            gauss.E4_pairwise(K0)


class TestE4MC:
    """Tests for E4 Monte Carlo computation."""

    def test_E4_mc_shape(self):
        """Test E4_mc has correct shape."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0)
        gauss = GaussianExpectation(params)

        N = 4
        K0 = random_psd_matrix(N)
        E4 = gauss.E4_pairwise_mc(K0, n_samples=1000, seed=42)

        assert E4.shape == (N, N, N, N)

    def test_E4_mc_vs_gh_small_N(self):
        """Test E4_mc matches GH for small N."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, gh_order=32)
        gauss = GaussianExpectation(params)

        N = 3
        K0 = random_psd_matrix(N)

        E4_gh = gauss.E4_pairwise(K0)
        E4_mc = gauss.E4_pairwise_mc(K0, n_samples=50000, seed=42)

        # MC should match GH within ~1% for 50k samples
        max_diff = torch.abs(E4_gh - E4_mc).max().item()
        assert max_diff < 0.02, f"E4_mc differs from E4_gh by {max_diff}"

    def test_E4_mc_symmetry(self):
        """Test E4_mc has expected symmetries."""
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)

        N = 3
        K0 = random_psd_matrix(N)
        E4 = gauss.E4_pairwise_mc(K0, n_samples=10000, seed=42)

        # E4[i,j,k,l] = E4[k,l,i,j] (pair swap)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for m in range(N):
                        val1 = E4[i, j, k, m].item()
                        val2 = E4[k, m, i, j].item()
                        rel_err = abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
                        assert rel_err < 0.1, f"Symmetry failed: {val1} vs {val2}"

    def test_E4_mc_reproducible_with_seed(self):
        """Test E4_mc gives same result with same seed."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0)
        gauss = GaussianExpectation(params)

        K0 = random_psd_matrix(3)

        E4_1 = gauss.E4_pairwise_mc(K0, n_samples=1000, seed=123)
        E4_2 = gauss.E4_pairwise_mc(K0, n_samples=1000, seed=123)

        assert allclose(E4_1, E4_2)

    def test_E4_mc_large_N(self):
        """Test E4_mc works for large N where GH would fail."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0)
        gauss = GaussianExpectation(params)

        N = 20  # Too large for GH
        K0 = random_psd_matrix(N)

        # Should not raise
        E4 = gauss.E4_pairwise_mc(K0, n_samples=1000, seed=42)
        assert E4.shape == (N, N, N, N)


class TestV4SlicesMC:
    """Tests for V4 slice computation via MC."""

    def test_V4_slices_keys(self):
        """Test compute_V4_slices_mc returns correct keys."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0)
        gauss = GaussianExpectation(params)

        K0 = random_psd_matrix(5)
        slices = gauss.compute_V4_slices_mc(K0, Cw=params.Cw, n_samples=1000, seed=42)

        assert set(slices.keys()) == {"diag_diag", "cross_diag", "diag_cross_L", "diag_cross_R"}

    def test_V4_slices_shape(self):
        """Test V4 slices have correct shape."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0)
        gauss = GaussianExpectation(params)

        N = 5
        K0 = random_psd_matrix(N)
        slices = gauss.compute_V4_slices_mc(K0, Cw=params.Cw, n_samples=1000, seed=42)

        for key, val in slices.items():
            assert val.shape == (N, N), f"{key} has wrong shape: {val.shape}"

    def test_V4_slices_vs_full_tensor(self):
        """Test V4 slices are consistent with GH-based full tensor for small N.

        Uses GH-based E4 (exact) as reference since it's deterministic.
        MC slices should match GH within statistical tolerance.
        """
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, gh_order=32)
        gauss = GaussianExpectation(params)

        N = 3  # Small N where GH is fast
        K0 = random_psd_matrix(N)
        Cw = params.Cw

        # Reference: GH-based V4 (deterministic)
        E4_gh = gauss.E4_pairwise(K0)
        E2_gh = gauss.E2_pairwise(K0)
        V4_gh = Cw**2 * (E4_gh - torch.einsum("ij,kl->ijkl", E2_gh, E2_gh))

        # Test: MC-based slices (high samples for accuracy)
        slices = gauss.compute_V4_slices_mc(K0, Cw=Cw, n_samples=200000, seed=42)

        # Compare - MC should match GH within ~5% for 200k samples
        rel_tol = 0.10  # 10% tolerance for V4 (small connected quantity)
        abs_tol = 0.005

        max_rel_err = 0.0
        for i in range(N):
            for j in range(N):
                v4_ref = V4_gh[i, i, j, j].item()
                slice_val = slices["diag_diag"][i, j].item()

                if abs(v4_ref) > abs_tol:
                    rel_err = abs(slice_val - v4_ref) / abs(v4_ref)
                    max_rel_err = max(max_rel_err, rel_err)
                    assert rel_err < rel_tol, f"diag_diag at ({i},{j}): rel_err={rel_err:.2%}"

        # Also verify cross_diag uses the same M4 but different subtraction
        for i in range(N):
            for j in range(N):
                v4_ref = V4_gh[i, j, i, j].item()
                slice_val = slices["cross_diag"][i, j].item()

                if abs(v4_ref) > abs_tol:
                    rel_err = abs(slice_val - v4_ref) / abs(v4_ref)
                    assert rel_err < rel_tol, f"cross_diag at ({i},{j}): rel_err={rel_err:.2%}"

    def test_V4_slices_large_N(self):
        """Test V4 slices work for large N."""
        params = Params(act=ActivationSpec.tanh(), Cw=1.0)
        gauss = GaussianExpectation(params)

        N = 100  # Large N
        K0 = random_psd_matrix(N)

        # Should complete quickly and not run out of memory
        slices = gauss.compute_V4_slices_mc(K0, Cw=params.Cw, n_samples=1000, seed=42)

        assert slices["diag_diag"].shape == (N, N)
