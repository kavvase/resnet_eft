"""Tests for ChiOp (P0 priority tests).

These tests verify that the pair-space implementation of χ is correct.
They are critical for all downstream computations.
"""

import pytest
import torch

from resnet_eft import GaussianExpectation, Params
from resnet_eft.backend import allclose, einsum, zeros
from resnet_eft.chi_op import ChiOp
from resnet_eft.core_types import ActivationSpec


def random_psd_matrix(N: int, dtype=torch.float64) -> torch.Tensor:
    """Generate a random positive semi-definite matrix."""
    A = torch.randn(N, N, dtype=dtype)
    return A @ A.T + torch.eye(N, dtype=dtype)


def random_matrix(N: int, M: int | None = None, dtype=torch.float64) -> torch.Tensor:
    """Generate a random matrix."""
    if M is None:
        M = N
    return torch.randn(N, M, dtype=dtype)


def build_chi_tensor_naive(chi_op: ChiOp) -> torch.Tensor:
    """Build full χ[x₁,x₂,y₁,y₂] tensor naively for testing.

    χ(x₁,x₂;y₁,y₂) = (Cw/2) × {
        [δ(x₁-y₁)δ(x₂-y₂) + δ(x₁-y₂)δ(x₂-y₁)] × Epp[x₁,x₂]
      + δ(x₁-y₁)δ(x₁-y₂) × E2s[x₁,x₂]
      + δ(x₂-y₁)δ(x₂-y₂) × Es2[x₁,x₂]
    }
    """
    N = chi_op.Epp.shape[0]
    chi = zeros((N, N, N, N), dtype=chi_op.Epp.dtype)
    coeff = chi_op.coeff

    for x1 in range(N):
        for x2 in range(N):
            for y1 in range(N):
                for y2 in range(N):
                    val = 0.0

                    # Term 1: δ(x₁-y₁)δ(x₂-y₂)
                    if x1 == y1 and x2 == y2:
                        val = val + chi_op.Epp[x1, x2].item()

                    # Term 2: δ(x₁-y₂)δ(x₂-y₁)
                    if x1 == y2 and x2 == y1:
                        val = val + chi_op.Epp[x1, x2].item()

                    # Term 3: δ(x₁-y₁)δ(x₁-y₂)
                    if x1 == y1 and x1 == y2:
                        val = val + chi_op.E2s[x1, x2].item()

                    # Term 4: δ(x₂-y₁)δ(x₂-y₂)
                    if x2 == y1 and x2 == y2:
                        val = val + chi_op.Es2[x1, x2].item()

                    chi[x1, x2, y1, y2] = coeff * val

    return chi


class TestChiOpApplyPair:
    """P0 tests: apply_pair matches naive einsum."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_apply_pair_matches_einsum(self, N: int):
        """Test that apply_pair matches naive einsum implementation."""
        K0 = random_psd_matrix(N)
        A = random_matrix(N, N)

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Efficient apply_pair
        result_op = chi_op.apply_pair(A)

        # Naive einsum
        chi_tensor = build_chi_tensor_naive(chi_op)
        result_naive = einsum("ijkl,kl->ij", chi_tensor, A)

        assert allclose(result_op, result_naive, rtol=1e-5)

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_apply_pair_T_matches_einsum(self, N: int):
        """Test that apply_pair_T matches naive einsum transpose."""
        K0 = random_psd_matrix(N)
        A = random_matrix(N, N)

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Efficient apply_pair_T
        result_op = chi_op.apply_pair_T(A)

        # Naive einsum (transpose)
        chi_tensor = build_chi_tensor_naive(chi_op)
        result_naive = einsum("ijkl,ij->kl", chi_tensor, A)

        assert allclose(result_op, result_naive, rtol=1e-5)

    @pytest.mark.parametrize(
        "act",
        [
            ActivationSpec.relu(mode="exact"),
            ActivationSpec.relu(mode="smooth"),
            "erf",
            "tanh",
        ],
    )
    def test_apply_pair_different_activations(self, act):
        """Test apply_pair with different activations."""
        N = 3
        K0 = random_psd_matrix(N)
        A = random_matrix(N, N)

        params = Params(act=act, Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        result_op = chi_op.apply_pair(A)
        chi_tensor = build_chi_tensor_naive(chi_op)
        result_naive = einsum("ijkl,kl->ij", chi_tensor, A)

        assert allclose(result_op, result_naive, rtol=1e-5)


class TestChiOpSymmetry:
    """Tests for χ symmetry properties."""

    def test_is_symmetric_relu(self):
        """Test that χ is symmetric for ReLU."""
        N = 3
        K0 = random_psd_matrix(N)

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # ReLU should give symmetric χ (E2s = Es2.T)
        assert chi_op.is_symmetric()

    def test_epp_is_symmetric(self):
        """Test that Epp is symmetric."""
        N = 3
        K0 = random_psd_matrix(N)

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        assert allclose(chi_op.Epp, chi_op.Epp.T)


class TestChiOpLinearAlgebra:
    """Tests for χ linear algebra properties."""

    def test_apply_pair_linearity(self):
        """Test that apply_pair is linear."""
        N = 3
        K0 = random_psd_matrix(N)
        A = random_matrix(N, N)
        B = random_matrix(N, N)
        alpha = 2.5

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Test linearity: χ(αA + B) = αχ(A) + χ(B)
        lhs = chi_op.apply_pair(alpha * A + B)
        rhs = alpha * chi_op.apply_pair(A) + chi_op.apply_pair(B)

        assert allclose(lhs, rhs, rtol=1e-5)
