"""Gaussian expectation calculations for neural network statistics.

This module computes expectations of activation function outputs
when the pre-activations follow a Gaussian distribution.

Key expectations:
- E2: ⟨σ(φᵢ)σ(φⱼ)⟩
- E4: ⟨σ(φᵢ)σ(φⱼ)σ(φₖ)σ(φₗ)⟩
- Epp: ⟨σ'(φᵢ)σ'(φⱼ)⟩
- E2s: ⟨σ''(φᵢ)σ(φⱼ)⟩
"""

from __future__ import annotations

import itertools
from math import pi
from math import sqrt as math_sqrt
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from resnet_eft.backend import (
    PI,
    arccos,
    arcsin,
    cholesky_safe,
    clip,
    cos,
    diagonal,
    sin,
    sqrt,
    zeros,
)
from resnet_eft.chi_op import ChiOp

if TYPE_CHECKING:
    from resnet_eft.core_types import Params


class GaussianExpectation:
    """Compute Gaussian expectations of activation function outputs.

    Given φ ~ N(0, K₀), computes various expectations of σ(φ).

    Attributes:
        params: Computation parameters
    """

    def __init__(self, params: Params) -> None:
        """Initialize with computation parameters.

        Args:
            params: Computation parameters (activation, Cw, etc.)
        """
        self.params = params
        self._gh_cache: tuple[np.ndarray, np.ndarray] | None = None

    def _get_gh_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Get Gauss-Hermite quadrature points and weights."""
        if self._gh_cache is None:
            self._gh_cache = np.polynomial.hermite.hermgauss(self.params.gh_order)
        return self._gh_cache

    # =========================================================================
    # Activation functions and their derivatives
    # =========================================================================

    def sigma(self, phi: Tensor) -> Tensor:
        """Apply activation function.

        Args:
            phi: Pre-activation values

        Returns:
            Activated values
        """
        act_name = self.params.act_name
        scale = self.params.act_input_scale
        beta = self.params.act_smoothing_beta

        if act_name == "relu":
            return torch.relu(phi)
        elif act_name == "softplus":
            # softplus(βx)/β → ReLU as β → ∞
            beta = beta if beta is not None else 10.0
            return torch.nn.functional.softplus(phi * beta) / beta
        elif act_name == "gelu":
            return torch.nn.functional.gelu(phi)
        elif act_name == "erf":
            return torch.erf(phi / scale)
        elif act_name == "tanh":
            return torch.tanh(phi / scale)
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    def sigma_prime(self, phi: Tensor) -> Tensor:
        """Compute activation derivative σ'(φ).

        Args:
            phi: Pre-activation values

        Returns:
            Derivative values
        """
        act_name = self.params.act_name
        scale = self.params.act_input_scale
        beta = self.params.act_smoothing_beta

        if act_name == "relu":
            return (phi > 0).float()
        elif act_name == "softplus":
            # d/dφ [softplus(βφ)/β] = sigmoid(βφ)
            beta = beta if beta is not None else 10.0
            return torch.sigmoid(phi * beta)
        elif act_name == "gelu":
            # d/dφ GELU(φ) = Φ(φ) + φ × φ'(φ) where Φ is CDF, φ' is PDF
            # Approximation: 0.5 * (1 + tanh(√(2/π) * (φ + 0.044715 * φ³)))
            # + φ * d/dφ[...]
            # Use numerical for simplicity
            cdf = 0.5 * (1 + torch.erf(phi / math_sqrt(2)))
            pdf = torch.exp(-0.5 * phi**2) / math_sqrt(2 * pi)
            return cdf + phi * pdf
        elif act_name == "erf":
            # d/dφ erf(φ/s) = (2/√π) × (1/s) × exp(-(φ/s)²)
            return (2 / math_sqrt(pi)) / scale * torch.exp(-((phi / scale) ** 2))
        elif act_name == "tanh":
            # d/dφ tanh(φ/s) = (1/s) × sech²(φ/s)
            t = torch.tanh(phi / scale)
            return (1 - t**2) / scale
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    # =========================================================================
    # E2: 2-point expectation ⟨σ(φᵢ)σ(φⱼ)⟩
    # =========================================================================

    def E2_pairwise(self, K0: Tensor) -> Tensor:
        """Compute E[σ(φᵢ)σ(φⱼ)] for all pairs.

        Args:
            K0: Kernel matrix, shape (N, N)

        Returns:
            Expectation matrix, shape (N, N)
        """
        act_name = self.params.act_name

        # Extract q (diagonal = variances) and compute ρ (correlation)
        q = diagonal(K0)
        q1 = q[:, None]
        q2 = q[None, :]
        rho = self._get_rho(q1, q2, K0)

        if act_name == "relu":
            return self._E2_relu(q1, q2, rho)
        elif act_name == "erf":
            return self._E2_erf(q1, q2, rho)
        elif act_name in ("tanh", "softplus", "gelu"):
            return self._E2_numerical(q1, q2, rho)
        else:
            return self._E2_numerical(q1, q2, rho)

    def _E2_relu(self, q1: Tensor, q2: Tensor, rho: Tensor) -> Tensor:
        """ReLU: Analytic formula from Cho & Saul (2009).

        E[ReLU(φ₁)ReLU(φ₂)] = √(q₁q₂)/(2π) × [sin(θ) + (π-θ)cos(θ)]
        where θ = arccos(ρ)
        """
        theta = arccos(clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho))
        return sqrt(q1 * q2) / (2 * PI) * (sin(theta) + (PI - theta) * cos(theta))

    def _E2_erf(self, q1: Tensor, q2: Tensor, rho: Tensor) -> Tensor:
        """erf: Analytic formula from Williams (1996).

        For σ(u) = erf(u/s), we have:
        E[erf(φ₁/s)erf(φ₂/s)] = (2/π) arcsin(2ρ√(q₁q₂/s⁴) / √((1+2q₁/s²)(1+2q₂/s²)))
        """
        s = self.params.act_input_scale
        q1_eff = q1 / (s**2)
        q2_eff = q2 / (s**2)

        num = 2 * rho * sqrt(q1_eff * q2_eff)
        den = sqrt((1 + 2 * q1_eff) * (1 + 2 * q2_eff))

        return (2 / PI) * arcsin(clip(num / den, -1 + self.params.eps_rho, 1 - self.params.eps_rho))

    def _E2_numerical(self, q1: Tensor, q2: Tensor, rho: Tensor) -> Tensor:
        """Numerical 2D Gauss-Hermite integration."""
        gh_x, gh_w = self._get_gh_points()

        result = torch.zeros_like(rho)
        for _i, (xi, wi) in enumerate(zip(gh_x, gh_w)):
            for _j, (xj, wj) in enumerate(zip(gh_x, gh_w)):
                # Transform to correlated Gaussian
                # φ₁ = √(2q₁) × x
                # φ₂ = √(2q₂) × (ρx + √(1-ρ²)y)
                phi1 = sqrt(2 * q1) * xi
                rho_clamped = clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho)
                phi2 = sqrt(2 * q2) * (rho_clamped * xi + sqrt(1 - rho_clamped**2) * xj)

                result = result + wi * wj * self.sigma(phi1) * self.sigma(phi2) / pi

        return result

    # =========================================================================
    # E4: 4-point expectation ⟨σ(φᵢ)σ(φⱼ)σ(φₖ)σ(φₗ)⟩
    # =========================================================================

    def E4_pairwise(self, K0: Tensor) -> Tensor:
        """Compute E[σ(φᵢ)σ(φⱼ)σ(φₖ)σ(φₗ)] for all quadruples.

        Warning: This is O(N⁴ × gh_order⁴) and expensive for large N.
        Recommended for N ≤ 6 only.

        Args:
            K0: Kernel matrix, shape (N, N)

        Returns:
            4-point expectation tensor, shape (N, N, N, N)
        """
        N = K0.shape[0]
        if N > 10:
            raise ValueError(f"E4_pairwise is too expensive for N={N} > 10")

        gh_x, gh_w = self._get_gh_points()
        n_gh = len(gh_x)

        # Create meshgrid of GH points: shape (n_gh, n_gh, n_gh, n_gh, 4)
        gh_x_t = torch.tensor(gh_x, dtype=K0.dtype, device=K0.device)
        gh_w_t = torch.tensor(gh_w, dtype=K0.dtype, device=K0.device)

        # Precompute weight products: w1*w2*w3*w4 for all combinations
        # Shape: (n_gh, n_gh, n_gh, n_gh)
        w_prod = (
            gh_w_t[:, None, None, None]
            * gh_w_t[None, :, None, None]
            * gh_w_t[None, None, :, None]
            * gh_w_t[None, None, None, :]
        )

        # Create z vectors: shape (n_gh, n_gh, n_gh, n_gh, 4)
        z_grid = torch.stack(
            torch.meshgrid(gh_x_t, gh_x_t, gh_x_t, gh_x_t, indexing="ij"), dim=-1
        )  # (n_gh, n_gh, n_gh, n_gh, 4)

        result = zeros((N, N, N, N), dtype=K0.dtype)

        for indices in itertools.product(range(N), repeat=4):
            i, j, k, m = indices

            # Build 4-point covariance submatrix
            idx = [i, j, k, m]
            K_sub = K0[idx, :][:, idx]

            # Cholesky for transformation
            L = cholesky_safe(K_sub, jitter=1e-6)

            # Transform all z vectors at once: phi = √2 * L @ z
            # z_grid: (n_gh^4, 4), L: (4, 4) -> phi: (n_gh^4, 4)
            z_flat = z_grid.reshape(-1, 4)  # (n_gh^4, 4)
            phi_flat = math_sqrt(2) * (z_flat @ L.T)  # (n_gh^4, 4)

            # Compute sigma for all points
            sigma_vals = self.sigma(phi_flat)  # (n_gh^4, 4)

            # Product of sigmas
            sigma_prod = sigma_vals[:, 0] * sigma_vals[:, 1] * sigma_vals[:, 2] * sigma_vals[:, 3]

            # Reshape and weight
            sigma_prod = sigma_prod.reshape(n_gh, n_gh, n_gh, n_gh)

            # Integrate: sum over all GH points with weights
            val = (w_prod * sigma_prod).sum() / (pi**2)

            result[i, j, k, m] = val

        return result

    def E4_pairwise_mc(
        self,
        K0: Tensor,
        n_samples: int = 10000,
        seed: int | None = None,
        batch_size: int = 500,
    ) -> Tensor:
        """Compute E[σ(φᵢ)σ(φⱼ)σ(φₖ)σ(φₗ)] using Monte Carlo sampling.

        This is much faster than GH quadrature for large N (N > 10).
        Recommended when GH is too slow or N exceeds the GH limit.

        Complexity: O(N^4 × n_samples / batch_size) time, O(N^4) memory
        For N=50, n_samples=10000: ~3 seconds

        Args:
            K0: Kernel matrix, shape (N, N)
            n_samples: Number of MC samples (more = higher precision)
            seed: Random seed for reproducibility
            batch_size: Batch size for vectorized computation

        Returns:
            4-point expectation tensor, shape (N, N, N, N)

        Note:
            Precision scales as 1/√n_samples. Typical max error:
            - n_samples=1000:  ~0.02
            - n_samples=10000: ~0.006
            - n_samples=50000: ~0.002
        """
        N = K0.shape[0]
        dtype = K0.dtype
        device = K0.device

        if seed is not None:
            torch.manual_seed(seed)

        L = cholesky_safe(K0, jitter=1e-6)
        E4 = zeros((N, N, N, N), dtype=dtype)

        n_batches = (n_samples + batch_size - 1) // batch_size
        total_samples = 0

        for _b in range(n_batches):
            actual_batch = min(batch_size, n_samples - total_samples)
            if actual_batch <= 0:
                break

            # Batched sampling: Z is (N, B), Phi = L @ Z is (N, B)
            Z = torch.randn(N, actual_batch, dtype=dtype, device=device)
            Phi = L @ Z  # (N, B)
            S = self.sigma(Phi)  # (N, B)

            # Batched 4-point outer product: E4 += Σ_b σᵢ(b)σⱼ(b)σₖ(b)σₗ(b)
            # einsum over batch dimension
            E4 += torch.einsum("ib,jb,kb,lb->ijkl", S, S, S, S)

            total_samples += actual_batch

        return E4 / total_samples

    def compute_V4_slices_mc(
        self,
        K0: Tensor,
        Cw: float,
        n_samples: int = 10000,
        seed: int | None = None,
        batch_size: int = 1000,
    ) -> dict[str, Tensor]:
        """Compute only V4 slices needed for K1 calculation using MC.

        This avoids storing the full O(N^4) V4 tensor, requiring only O(N^2) memory.
        Essential for large N (N > 100).

        The slices computed are:
        - diag_diag:    V4[i,i,j,j]   = Cov(K[i,i], K[j,j])
        - cross_diag:   V4[i,j,i,j]   = Cov(K[i,j], K[i,j])
        - diag_cross_L: V4[i,i,i,j]   = Cov(K[i,i], K[i,j])
        - diag_cross_R: V4[j,j,i,j]   = Cov(K[j,j], K[i,j])

        Implementation notes:
        - diag_diag and cross_diag share the same raw 4th moment M4[i,j] = E[σᵢ²σⱼ²]
          They differ only in subtraction: V4[i,i,j,j] - E[σᵢ²]E[σⱼ²], V4[i,j,i,j] - E[σᵢσⱼ]²
        - Batched computation uses BLAS for efficiency (critical for N > 100)
        - Returns O(1) coefficient form (consistent with KernelState convention)

        Complexity: O(N² × n_samples / batch_size) matrix ops, O(N² + N×batch_size) memory
        For N=1000, n_samples=10000, batch_size=1000: ~2 seconds

        Args:
            K0: Kernel matrix, shape (N, N)
            Cw: Weight variance parameter
            n_samples: Number of MC samples
            seed: Random seed for reproducibility
            batch_size: Batch size for vectorized computation

        Returns:
            Dictionary with keys: 'diag_diag', 'cross_diag', 'diag_cross_L', 'diag_cross_R'
            Each value is an (N, N) tensor (O(1) coefficient form).
        """
        N = K0.shape[0]
        dtype = K0.dtype
        device = K0.device

        if seed is not None:
            torch.manual_seed(seed)

        L = cholesky_safe(K0, jitter=1e-6)

        # Accumulators (running sums, normalized at the end)
        E2 = zeros((N, N), dtype=dtype)  # E[σᵢσⱼ]
        M4 = zeros((N, N), dtype=dtype)  # E[σᵢ²σⱼ²] - shared by diag_diag and cross_diag
        M31 = zeros((N, N), dtype=dtype)  # E[σᵢ³σⱼ] - for diag_cross_L
        M13 = zeros((N, N), dtype=dtype)  # E[σᵢσⱼ³] - for diag_cross_R

        n_batches = (n_samples + batch_size - 1) // batch_size
        total_samples = 0

        for _b in range(n_batches):
            actual_batch = min(batch_size, n_samples - total_samples)
            if actual_batch <= 0:
                break

            # Batched sampling: Z is (N, B), Phi = L @ Z is (N, B)
            Z = torch.randn(N, actual_batch, dtype=dtype, device=device)
            Phi = L @ Z  # (N, B)
            S = self.sigma(Phi)  # (N, B)

            S2 = S**2  # σᵢ²
            S3 = S**3  # σᵢ³

            # Accumulate using matrix products (BLAS-efficient)
            # E2 += (1/B) × S @ S.T  →  E2[i,j] += Σ_b σᵢ(b)σⱼ(b)
            E2 += S @ S.T

            # M4 += S² @ (S²).T  →  M4[i,j] += Σ_b σᵢ²(b)σⱼ²(b)
            M4 += S2 @ S2.T

            # M31 += S³ @ S.T  →  M31[i,j] += Σ_b σᵢ³(b)σⱼ(b)
            M31 += S3 @ S.T

            # M13 += S @ (S³).T  →  M13[i,j] += Σ_b σᵢ(b)σⱼ³(b)
            M13 += S @ S3.T

            total_samples += actual_batch

        # Normalize by total samples
        E2 /= total_samples
        M4 /= total_samples
        M31 /= total_samples
        M13 /= total_samples

        # Convert to V4 = Cw² × (E4 - E2 ⊗ E2) [coefficient form, O(1)]
        Cw2 = Cw**2
        E2_diag = E2.diag()  # E[σᵢ²]

        # V4[i,i,j,j] = Cw² × (M4[i,j] - E[σᵢ²]E[σⱼ²])
        V4_diag_diag = Cw2 * (M4 - torch.outer(E2_diag, E2_diag))

        # V4[i,j,i,j] = Cw² × (M4[i,j] - E[σᵢσⱼ]²)
        # Note: Same M4, different subtraction term
        V4_cross_diag = Cw2 * (M4 - E2**2)

        # V4[i,i,i,j] = Cw² × (M31[i,j] - E[σᵢ²]E[σᵢσⱼ])
        V4_diag_cross_L = Cw2 * (M31 - E2_diag[:, None] * E2)

        # V4[j,j,i,j] = Cw² × (M13[i,j] - E[σⱼ²]E[σᵢσⱼ])
        V4_diag_cross_R = Cw2 * (M13 - E2_diag[None, :] * E2)

        return {
            "diag_diag": V4_diag_diag,
            "cross_diag": V4_cross_diag,
            "diag_cross_L": V4_diag_cross_L,
            "diag_cross_R": V4_diag_cross_R,
        }

    # =========================================================================
    # Derivative expectations (for χ construction)
    # =========================================================================

    def E_sigma_prime_prime(self, K0: Tensor) -> Tensor:
        """Compute ⟨σ'(φᵢ)σ'(φⱼ)⟩.

        Args:
            K0: Kernel matrix, shape (N, N)

        Returns:
            Expectation matrix, shape (N, N)
        """
        act_name = self.params.act_name

        if act_name == "relu":
            return self._Epp_relu(K0)
        elif act_name == "erf":
            return self._Epp_erf(K0)
        elif act_name in ("tanh", "softplus", "gelu"):
            return self._Epp_numerical(K0)
        else:
            return self._Epp_numerical(K0)

    def _Epp_relu(self, K0: Tensor) -> Tensor:
        """ReLU: ⟨σ'σ'⟩ = (π - θ) / (2π) where θ = arccos(ρ)."""
        q = diagonal(K0)
        q1, q2 = q[:, None], q[None, :]
        rho = self._get_rho(q1, q2, K0)
        theta = arccos(clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho))
        return (PI - theta) / (2 * PI)

    def _Epp_erf(self, K0: Tensor) -> Tensor:
        """erf: Analytic formula for ⟨σ'σ'⟩.

        d/dφ erf(φ/s) = (2/√π)/s × exp(-(φ/s)²)

        ⟨σ'(φ₁)σ'(φ₂)⟩ = (4/πs²) × ∫ exp(-φ₁²/s² - φ₂²/s²) dφ
                       = (4/πs²) × 1/√det(2(I + 2K/s²))
        """
        s = self.params.act_input_scale
        q = diagonal(K0)
        q1, q2 = q[:, None], q[None, :]
        c = K0

        # Determinant of 2×2 matrix [1+2q₁/s², 2c/s²; 2c/s², 1+2q₂/s²]
        det = (1 + 2 * q1 / s**2) * (1 + 2 * q2 / s**2) - (2 * c / s**2) ** 2

        return (4 / (pi * s**2)) / sqrt(clip(det, self.params.eps_rho, float("inf")))

    def _Epp_numerical(self, K0: Tensor) -> Tensor:
        """Numerical computation of ⟨σ'σ'⟩."""
        gh_x, gh_w = self._get_gh_points()

        q = diagonal(K0)
        q1, q2 = q[:, None], q[None, :]
        rho = self._get_rho(q1, q2, K0)

        result = torch.zeros_like(K0)
        for _i, (xi, wi) in enumerate(zip(gh_x, gh_w)):
            for _j, (xj, wj) in enumerate(zip(gh_x, gh_w)):
                phi1 = sqrt(2 * q1) * xi
                rho_clamped = clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho)
                phi2 = sqrt(2 * q2) * (rho_clamped * xi + sqrt(1 - rho_clamped**2) * xj)

                result = result + wi * wj * self.sigma_prime(phi1) * self.sigma_prime(phi2) / pi

        return result

    def E_sigma_dprime_sigma(self, K0: Tensor) -> Tensor:
        """Compute ⟨σ''(φᵢ)σ(φⱼ)⟩.

        For ReLU, σ'' = δ(φ), requiring special handling.

        Args:
            K0: Kernel matrix, shape (N, N)

        Returns:
            Expectation matrix, shape (N, N)
        """
        act_name = self.params.act_name

        if act_name == "relu":
            return self._E2s_relu(K0)
        elif act_name == "erf":
            return self._E2s_erf(K0)
        elif act_name in ("tanh", "softplus", "gelu"):
            return self._E2s_numerical(K0)
        else:
            return self._E2s_numerical(K0)

    def _E2s_relu(self, K0: Tensor) -> Tensor:
        """ReLU: ⟨σ''(φᵢ)σ(φⱼ)⟩ = ⟨δ(φᵢ)σ(φⱼ)⟩.

        For ReLU, σ'' = δ(φ) (Dirac delta).

        E[δ(φᵢ)σ(φⱼ)] = E[σ(φⱼ)|φᵢ=0] × p(φᵢ=0)

        where:
        - p(φᵢ=0) = 1/√(2πqᵢ)
        - φⱼ|φᵢ=0 ~ N(0, qⱼ(1-ρ²))
        - E[ReLU(φⱼ)|φᵢ=0] = √(qⱼ(1-ρ²)/(2π))
        """
        N = K0.shape[0]
        q = diagonal(K0)

        result = zeros((N, N), dtype=K0.dtype)

        for i in range(N):
            for j in range(N):
                qi, qj = q[i], q[j]
                rho = K0[i, j] / sqrt(qi * qj + self.params.eps_rho)
                rho = clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho)

                # Conditional variance of φⱼ given φᵢ=0
                var_cond = qj * (1 - rho**2)

                # E[ReLU(φⱼ)|φᵢ=0] = √(var_cond/(2π))
                E_sigma_cond = sqrt(var_cond / (2 * PI))

                # p(φᵢ=0) = 1/√(2πqᵢ)
                p_zero = 1 / sqrt(2 * PI * qi)

                result[i, j] = E_sigma_cond * p_zero

        return result

    def _E2s_erf(self, K0: Tensor) -> Tensor:
        """erf: Analytic formula for ⟨σ''σ⟩.

        σ''(φ) = -(4/√π)/s³ × φ/s × exp(-(φ/s)²)
        """
        # Use numerical integration for simplicity
        return self._E2s_numerical(K0)

    def _E2s_numerical(self, K0: Tensor) -> Tensor:
        """Numerical computation of ⟨σ''σ⟩ using finite differences.

        For smooth activations, we use numerical differentiation.
        """
        eps_fd = 1e-5  # Finite difference epsilon

        gh_x, gh_w = self._get_gh_points()
        q = diagonal(K0)
        q1, q2 = q[:, None], q[None, :]
        rho = self._get_rho(q1, q2, K0)

        result = torch.zeros_like(K0)
        for _i, (xi, wi) in enumerate(zip(gh_x, gh_w)):
            for _j, (xj, wj) in enumerate(zip(gh_x, gh_w)):
                phi1 = sqrt(2 * q1) * xi
                rho_clamped = clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho)
                phi2 = sqrt(2 * q2) * (rho_clamped * xi + sqrt(1 - rho_clamped**2) * xj)

                # Finite difference for σ''
                sigma_pp = (
                    self.sigma(phi1 + eps_fd) - 2 * self.sigma(phi1) + self.sigma(phi1 - eps_fd)
                ) / (eps_fd**2)

                result = result + wi * wj * sigma_pp * self.sigma(phi2) / pi

        return result

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_rho(self, q1: Tensor, q2: Tensor, c: Tensor) -> Tensor:
        """Compute correlation coefficient ρ = c / √(q₁q₂).

        Args:
            q1: First variance (can be broadcasted)
            q2: Second variance (can be broadcasted)
            c: Covariance

        Returns:
            Correlation coefficient (clipped for stability)
        """
        rho = c / sqrt(q1 * q2 + self.params.eps_rho)
        return clip(rho, -1 + self.params.eps_rho, 1 - self.params.eps_rho)

    def build_chi_op(self, K0: Tensor) -> ChiOp:
        """Build ChiOp from this GaussianExpectation.

        Args:
            K0: Kernel matrix, shape (N, N)

        Returns:
            ChiOp instance
        """
        return ChiOp.from_gauss(self, K0)
