"""ChiOp: Susceptibility operator in pair-space.

χ is defined as the response of ⟨G⟩ to changes in the previous layer's kernel:
    χ(x₁,x₂;y₁,y₂) = δ⟨G(x₁,x₂)⟩ / δ⟨G(y₁,y₂)⟩

At leading order in 1/n, this becomes:
    χ = (Cw/2) × ⟨δ²[σσ]/δφ²⟩_{K₀}

The δ-structure allows efficient computation without storing the full (N,N,N,N) tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from resnet_eft.backend import allclose, diag_embed, diagonal

if TYPE_CHECKING:
    from resnet_eft.gaussian_expectation import GaussianExpectation


@dataclass
class ChiOp:
    """Susceptibility operator in pair-space.

    χ(x₁,x₂;y₁,y₂) = (Cw/2) × {
        [δ(x₁-y₁)δ(x₂-y₂) + δ(x₁-y₂)δ(x₂-y₁)] × ⟨σ'σ'⟩
      + δ(x₁-y₁)δ(x₁-y₂) × ⟨σ''σ⟩
      + δ(x₂-y₁)δ(x₂-y₂) × ⟨σσ''⟩
    }

    Due to the δ-structure, we don't need to store the full (N,N,N,N) tensor.

    Attributes:
        Epp: ⟨σ'(φᵢ)σ'(φⱼ)⟩, shape (N, N)
        E2s: ⟨σ''(φᵢ)σ(φⱼ)⟩, shape (N, N)
        Es2: ⟨σ(φᵢ)σ''(φⱼ)⟩, shape (N, N) (usually E2s.T)
        Cw: Weight variance scale
    """

    Epp: Tensor
    E2s: Tensor
    Es2: Tensor
    Cw: float

    @property
    def coeff(self) -> float:
        """Return Cw/2 coefficient."""
        return self.Cw / 2.0

    def apply_pair(self, A: Tensor) -> Tensor:
        """Apply χ in pair-space: B = χ(A).

        B[x₁,x₂] = Σ_{y₁,y₂} χ[x₁,x₂;y₁,y₂] A[y₁,y₂]

        Args:
            A: Input matrix of shape (N, N)

        Returns:
            Output matrix of shape (N, N)
        """
        # Term 1+2: from δ(x₁-y₁)δ(x₂-y₂) + δ(x₁-y₂)δ(x₂-y₁)
        # These select A[x₁,x₂] + A[x₂,x₁] = (A + A.T)
        term_pp = (A + A.T) * self.Epp

        # Term 3: from δ(x₁-y₁)δ(x₁-y₂)
        # This selects A[x₁,x₁] (diagonal of A) and broadcasts to x₂
        diag_A = diagonal(A)
        term_2s = diag_A[:, None] * self.E2s

        # Term 4: from δ(x₂-y₁)δ(x₂-y₂)
        # This selects A[x₂,x₂] (diagonal of A) and broadcasts to x₁
        term_s2 = diag_A[None, :] * self.Es2

        return self.coeff * (term_pp + term_2s + term_s2)

    def apply_pair_T(self, A: Tensor) -> Tensor:
        """Apply χᵀ in pair-space: B = χᵀ(A).

        B[y₁,y₂] = Σ_{x₁,x₂} χ[x₁,x₂;y₁,y₂] A[x₁,x₂]

        This is needed for transport: M_V_next = M_χ @ M_V_prev @ M_χᵀ

        Args:
            A: Input matrix of shape (N, N)

        Returns:
            Output matrix of shape (N, N)
        """
        # Term 1+2 (symmetric part)
        term_pp = (A + A.T) * self.Epp

        # Terms 3+4 contribute to diagonal
        # From term 3: when y₁=y₂=x₁, we get E2s[x₁,x₂] × A[x₁,x₂]
        # Summing over x₂ gives row sum weighted by E2s
        col_weighted = (self.E2s * A).sum(dim=1)  # sum over x₂

        # From term 4: when y₁=y₂=x₂, we get Es2[x₁,x₂] × A[x₁,x₂]
        # Summing over x₁ gives column sum weighted by Es2
        row_weighted = (self.Es2 * A).sum(dim=0)  # sum over x₁

        diag_contrib = col_weighted + row_weighted
        result = term_pp + diag_embed(diag_contrib)

        return self.coeff * result

    def is_symmetric(self) -> bool:
        """Check if χ is self-adjoint in pair-space.

        χ is symmetric if Epp is symmetric and E2s = Es2.T.

        Returns:
            True if χ is self-adjoint
        """
        return allclose(self.E2s, self.Es2.T) and allclose(self.Epp, self.Epp.T)

    @classmethod
    def from_gauss(cls, gauss: GaussianExpectation, K0: Tensor) -> ChiOp:
        """Build ChiOp from GaussianExpectation.

        Args:
            gauss: GaussianExpectation instance
            K0: Kernel matrix, shape (N, N)

        Returns:
            ChiOp instance
        """
        Epp = gauss.E_sigma_prime_prime(K0)
        E2s = gauss.E_sigma_dprime_sigma(K0)
        Es2 = E2s.T
        return cls(Epp=Epp, E2s=E2s, Es2=Es2, Cw=gauss.params.Cw)
