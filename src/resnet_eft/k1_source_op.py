"""K1SourceOp: Efficient operator for K1 ← V4 contribution using Hessian sparsity.

This operator computes the K1 source term by exploiting the fact that
E2[i,j] = f(q_i, q_j, c_ij) depends only on 3 variables: K[i,i], K[j,j], K[i,j].

The key insight is that the Hessian d²E2[i,j]/dK[a,b]dK[c,d] is sparse:
- Non-zero only when (a,b) and (c,d) are in {(i,i), (j,j), (i,j)}
- This reduces the contraction from O(N^6) to O(N^2)

The K1 source term formula:
    K1_source[i,j] = (Cw/2) × Σ H[α,β] × V4_αβ

where H is the 3×3 Hessian of f(q_i, q_j, c_ij) and V4_αβ are the relevant
V4 components (diag_diag, cross_diag, diag_offdiag).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from resnet_eft.backend import diagonal, zeros

if TYPE_CHECKING:
    from resnet_eft.gaussian_expectation import GaussianExpectation
    from resnet_eft.v4_repr import V4Repr


class K1SourceOp:
    """K1 source term operator using sparse Hessian structure.

    Exploits the fact that E2[i,j] = f(K[i,i], K[j,j], K[i,j]) to achieve
    O(N²) complexity instead of O(N^6).

    For uniform K0 (all diagonals equal, all off-diagonals equal), only 2
    Hessian computations are needed regardless of N.

    The 3×3 Hessian H for each (i,j) pair is:
        H = [[f_q1q1, f_q1q2, f_q1c ],
             [f_q2q1, f_q2q2, f_q2c ],
             [f_cq1,  f_cq2,  f_cc  ]]

    K1[i,j] = (Cw/2) × [
        H[0,0] × V4[i,i,i,i] + H[1,1] × V4[j,j,j,j] + H[2,2] × V4[i,j,i,j]
        + 2×H[0,1] × V4[i,i,j,j]
        + 2×H[0,2] × V4[i,i,i,j]
        + 2×H[1,2] × V4[j,j,i,j]
    ]

    Attributes:
        K0: Kernel matrix at which E2'' is evaluated
        gauss: GaussianExpectation instance for computing E2
        N: Dimension of the kernel
        eps: Finite difference step size for numerical differentiation
        mode: Computation mode ("auto", "uniform", "general")
    """

    def __init__(
        self,
        K0: Tensor,
        gauss: GaussianExpectation,
        eps: float = 0.0005,
        mode: str = "auto",
    ) -> None:
        """Initialize K1SourceOp.

        Args:
            K0: Kernel matrix, shape (N, N)
            gauss: GaussianExpectation instance
            eps: Finite difference step size (default: 0.0005)
            mode: Computation mode
                - "auto": Auto-detect uniform K0 and use optimized path
                - "uniform": Force uniform K0 optimization
                - "general": Force general O(N²) path
        """
        self.K0 = K0
        self.gauss = gauss
        self.N = K0.shape[0]
        self.eps = eps
        self.mode = mode

        # Cache diagonal for repeated use
        self._q = diagonal(K0)

        # Check if K0 is uniform (αI + β1 structure)
        is_uniform_detected, self._uniform_params = self._check_uniform()

        # Determine effective mode
        if mode == "auto":
            self._is_uniform = is_uniform_detected
        elif mode == "uniform":
            if not is_uniform_detected:
                raise ValueError(
                    "mode='uniform' requires K0 to have aI + b*1 structure, but K0 is not uniform"
                )
            self._is_uniform = True
        elif mode == "general":
            self._is_uniform = False
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'uniform', or 'general'")

        # Cache for uniform K0 Hessians
        self._cached_H_diag: float | None = None
        self._cached_H_offdiag: Tensor | None = None

    def _check_uniform(self, tol: float = 1e-10) -> tuple[bool, tuple[float, float] | None]:
        """Check if K0 has uniform structure (all diags equal, all off-diags equal).

        Returns:
            (is_uniform, (diag_val, offdiag_val) or None)
        """
        N = self.N
        if N == 1:
            return True, (self.K0[0, 0].item(), 0.0)

        diag_vals = self._q
        diag_val = diag_vals[0].item()

        # Check all diagonals are equal
        if not torch.allclose(diag_vals, torch.full_like(diag_vals, diag_val), atol=tol):
            return False, None

        # Check all off-diagonals are equal
        mask = ~torch.eye(N, dtype=torch.bool, device=self.K0.device)
        offdiag_vals = self.K0[mask]
        offdiag_val = offdiag_vals[0].item()

        if not torch.allclose(offdiag_vals, torch.full_like(offdiag_vals, offdiag_val), atol=tol):
            return False, None

        return True, (diag_val, offdiag_val)

    def _E2_offdiag(self, q1: float, q2: float, c: float) -> float:
        """Compute E2[0,1] for a 2×2 kernel with given parameters.

        Creates a 2×2 kernel matrix [[q1, c], [c, q2]] and computes E2[0,1].
        Used for off-diagonal elements E2[i,j] where i ≠ j.

        Args:
            q1: First diagonal element K[i,i]
            q2: Second diagonal element K[j,j]
            c: Off-diagonal element K[i,j]

        Returns:
            E2[0,1] value (scalar)
        """
        K_2x2 = torch.tensor([[q1, c], [c, q2]], dtype=self.K0.dtype, device=self.K0.device)
        E2 = self.gauss.E2_pairwise(K_2x2)
        return E2[0, 1].item()

    def _E2_diag(self, q: float) -> float:
        """Compute E2[0,0] = ⟨σ(φ)²⟩ for variance q.

        Used for diagonal elements E2[i,i].

        Args:
            q: Variance K[i,i]

        Returns:
            E2[0,0] value (scalar)
        """
        K_1x1 = torch.tensor([[q]], dtype=self.K0.dtype, device=self.K0.device)
        E2 = self.gauss.E2_pairwise(K_1x1)
        return E2[0, 0].item()

    def _compute_hessian_diag(self, i: int) -> float:
        """Compute d²E2[i,i]/dK[i,i]² for diagonal element.

        E2[i,i] = ⟨σ(φ_i)²⟩ depends only on K[i,i], so it's a 1-variable function.

        Args:
            i: Index

        Returns:
            Second derivative d²E2[i,i]/dq²
        """
        eps = self.eps
        q = self._q[i].item()

        f0 = self._E2_diag(q)
        f_p = self._E2_diag(q + eps)
        f_m = self._E2_diag(q - eps)

        return (f_p - 2 * f0 + f_m) / (eps * eps)

    def _compute_hessian_3x3(self, i: int, j: int) -> Tensor:
        """Compute 3×3 Hessian of f(q_i, q_j, c_ij) for off-diagonal (i,j) pair.

        Used only when i ≠ j. For diagonal elements, use _compute_hessian_diag.

        Uses second-order finite differences with symmetry-preserving perturbations.

        The Hessian structure:
            [d²f/dq1², d²f/dq1dq2, d²f/dq1dc]
            [d²f/dq2dq1, d²f/dq2², d²f/dq2dc]
            [d²f/dcdq1, d²f/dcdq2, d²f/dc²]

        Args:
            i, j: Indices for the E2 output (must have i ≠ j)

        Returns:
            3×3 Hessian matrix
        """
        assert i != j, "Use _compute_hessian_diag for diagonal elements"

        eps = self.eps
        q1 = self._q[i].item()
        q2 = self._q[j].item()
        c = self.K0[i, j].item()

        H = torch.zeros(3, 3, dtype=self.K0.dtype, device=self.K0.device)

        # Compute f at base point
        f0 = self._E2_offdiag(q1, q2, c)

        # --- Diagonal elements (second derivatives) ---
        # d²f/dq1² using f(q1+ε) - 2f(q1) + f(q1-ε)
        f_q1p = self._E2_offdiag(q1 + eps, q2, c)
        f_q1m = self._E2_offdiag(q1 - eps, q2, c)
        H[0, 0] = (f_q1p - 2 * f0 + f_q1m) / (eps * eps)

        # d²f/dq2²
        f_q2p = self._E2_offdiag(q1, q2 + eps, c)
        f_q2m = self._E2_offdiag(q1, q2 - eps, c)
        H[1, 1] = (f_q2p - 2 * f0 + f_q2m) / (eps * eps)

        # d²f/dc²
        f_cp = self._E2_offdiag(q1, q2, c + eps)
        f_cm = self._E2_offdiag(q1, q2, c - eps)
        H[2, 2] = (f_cp - 2 * f0 + f_cm) / (eps * eps)

        # --- Off-diagonal elements (mixed derivatives) ---
        # d²f/dq1dq2 using (f(+,+) - f(+,-) - f(-,+) + f(-,-)) / (4ε²)
        f_pp = self._E2_offdiag(q1 + eps, q2 + eps, c)
        f_pm = self._E2_offdiag(q1 + eps, q2 - eps, c)
        f_mp = self._E2_offdiag(q1 - eps, q2 + eps, c)
        f_mm = self._E2_offdiag(q1 - eps, q2 - eps, c)
        H[0, 1] = H[1, 0] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

        # d²f/dq1dc
        f_pp = self._E2_offdiag(q1 + eps, q2, c + eps)
        f_pm = self._E2_offdiag(q1 + eps, q2, c - eps)
        f_mp = self._E2_offdiag(q1 - eps, q2, c + eps)
        f_mm = self._E2_offdiag(q1 - eps, q2, c - eps)
        H[0, 2] = H[2, 0] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

        # d²f/dq2dc
        f_pp = self._E2_offdiag(q1, q2 + eps, c + eps)
        f_pm = self._E2_offdiag(q1, q2 + eps, c - eps)
        f_mp = self._E2_offdiag(q1, q2 - eps, c + eps)
        f_mm = self._E2_offdiag(q1, q2 - eps, c - eps)
        H[1, 2] = H[2, 1] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

        return H

    def contract(self, V4: V4Repr, Cw: float) -> Tensor:
        """Contract with V4 to get K1 source term using sparse Hessian.

        Complexity: O(N²) for V4 combination.
        For uniform K0, Hessian computation is O(1).
        For general K0, Hessian computation is O(N²).

        For diagonal (i=i):
            K1[i,i] = (Cw/2) × d²E2[i,i]/dq² × V4[i,i,i,i]

        For off-diagonal (i≠j):
            K1[i,j] = (Cw/2) × [
                H[0,0] × V4[i,i,i,i] + H[1,1] × V4[j,j,j,j] + H[2,2] × V4[i,j,i,j]
                + 2×H[0,1] × V4[i,i,j,j]
                + 2×H[0,2] × V4[i,i,i,j]
                + 2×H[1,2] × V4[j,j,i,j]
            ]

        Args:
            V4: V4 representation (coefficient, O(1))
            Cw: Weight variance scale

        Returns:
            K1 source contribution, shape (N, N)
        """
        if self._is_uniform:
            return self._contract_uniform(V4, Cw)
        else:
            return self._contract_general(V4, Cw)

    def _contract_uniform(self, V4: V4Repr, Cw: float) -> Tensor:
        """Fast contraction for uniform K0 (αI + β1 structure).

        Only 2 Hessian computations needed regardless of N.
        Fully vectorized - no Python loops over (i,j) pairs.
        """
        N = self.N
        assert self._uniform_params is not None
        q, c = self._uniform_params

        # Compute Hessians once (cached)
        if self._cached_H_diag is None:
            self._cached_H_diag = self._compute_hessian_diag_at(q)
        if self._cached_H_offdiag is None:
            self._cached_H_offdiag = self._compute_hessian_3x3_at(q, q, c)

        H_diag = self._cached_H_diag
        H = self._cached_H_offdiag

        # Extract V4 slices (N×N matrices)
        diag_diag = V4.get_diag_diag()  # V4[i,i,j,j]
        cross_diag = V4.get_cross_diag()  # V4[i,j,i,j]
        diag_cross_L = V4.get_diag_cross_left()  # V4[i,i,i,j]
        diag_cross_R = V4.get_diag_cross_right()  # V4[j,j,i,j]

        # Diagonal elements: K1[i,i] = (Cw/2) * H_diag * V4[i,i,i,i]
        diag_V4 = diagonal(diag_diag)  # V4[i,i,i,i] for all i

        # Off-diagonal elements (fully vectorized):
        # K1[i,j] = (Cw/2) * [H[0,0]*V4[i,i,i,i] + H[1,1]*V4[j,j,j,j] + H[2,2]*V4[i,j,i,j]
        #                    + 2*H[0,1]*V4[i,i,j,j] + 2*H[0,2]*V4[i,i,i,j] + 2*H[1,2]*V4[j,j,i,j]]

        # Build vectorized terms
        # V4[i,i,i,i] for row i → broadcast to (N, N)
        V4_iiii = diag_V4.unsqueeze(1).expand(N, N)  # (N,) → (N, N)
        # V4[j,j,j,j] for col j → broadcast to (N, N)
        V4_jjjj = diag_V4.unsqueeze(0).expand(N, N)  # (N,) → (N, N)

        # Compute K1 for all pairs
        result = (Cw / 2) * (
            H[0, 0] * V4_iiii
            + H[1, 1] * V4_jjjj
            + H[2, 2] * cross_diag
            + 2 * H[0, 1] * diag_diag
            + 2 * H[0, 2] * diag_cross_L
            + 2 * H[1, 2] * diag_cross_R
        )

        # Fix diagonal elements (different formula)
        for i in range(N):
            result[i, i] = (Cw / 2) * H_diag * diag_V4[i]

        return result

    def _contract_general(self, V4: V4Repr, Cw: float) -> Tensor:
        """General contraction for non-uniform K0."""
        N = self.N

        # Extract V4 slices (O(N²) or faster)
        diag_diag = V4.get_diag_diag()
        cross_diag = V4.get_cross_diag()
        diag_cross_left = V4.get_diag_cross_left()
        diag_cross_right = V4.get_diag_cross_right()

        result = zeros((N, N), dtype=self.K0.dtype)

        for i in range(N):
            # --- Diagonal element K1[i,i] ---
            d2E2_dq2 = self._compute_hessian_diag(i)
            V4_iiii = diag_diag[i, i]
            result[i, i] = (Cw / 2) * d2E2_dq2 * V4_iiii

            # --- Off-diagonal elements K1[i,j] for j > i ---
            for j in range(i + 1, N):
                H = self._compute_hessian_3x3(i, j)

                V4_iiii = diag_diag[i, i]
                V4_jjjj = diag_diag[j, j]
                V4_ijij = cross_diag[i, j]
                V4_iijj = diag_diag[i, j]
                V4_iiij = diag_cross_left[i, j]
                V4_jjij = diag_cross_right[i, j]

                k1_val = (
                    H[0, 0] * V4_iiii
                    + H[1, 1] * V4_jjjj
                    + H[2, 2] * V4_ijij
                    + 2 * H[0, 1] * V4_iijj
                    + 2 * H[0, 2] * V4_iiij
                    + 2 * H[1, 2] * V4_jjij
                )

                result[i, j] = (Cw / 2) * k1_val
                result[j, i] = result[i, j]

        return result

    def _compute_hessian_diag_at(self, q: float) -> float:
        """Compute d²E2/dq² at a specific q value."""
        eps = self.eps
        f0 = self._E2_diag(q)
        f_p = self._E2_diag(q + eps)
        f_m = self._E2_diag(q - eps)
        return (f_p - 2 * f0 + f_m) / (eps * eps)

    def _compute_hessian_3x3_at(self, q1: float, q2: float, c: float) -> Tensor:
        """Compute 3×3 Hessian at specific (q1, q2, c) values."""
        eps = self.eps
        H = torch.zeros(3, 3, dtype=self.K0.dtype, device=self.K0.device)

        f0 = self._E2_offdiag(q1, q2, c)

        # Diagonal elements
        f_q1p = self._E2_offdiag(q1 + eps, q2, c)
        f_q1m = self._E2_offdiag(q1 - eps, q2, c)
        H[0, 0] = (f_q1p - 2 * f0 + f_q1m) / (eps * eps)

        f_q2p = self._E2_offdiag(q1, q2 + eps, c)
        f_q2m = self._E2_offdiag(q1, q2 - eps, c)
        H[1, 1] = (f_q2p - 2 * f0 + f_q2m) / (eps * eps)

        f_cp = self._E2_offdiag(q1, q2, c + eps)
        f_cm = self._E2_offdiag(q1, q2, c - eps)
        H[2, 2] = (f_cp - 2 * f0 + f_cm) / (eps * eps)

        # Off-diagonal elements
        f_pp = self._E2_offdiag(q1 + eps, q2 + eps, c)
        f_pm = self._E2_offdiag(q1 + eps, q2 - eps, c)
        f_mp = self._E2_offdiag(q1 - eps, q2 + eps, c)
        f_mm = self._E2_offdiag(q1 - eps, q2 - eps, c)
        H[0, 1] = H[1, 0] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

        f_pp = self._E2_offdiag(q1 + eps, q2, c + eps)
        f_pm = self._E2_offdiag(q1 + eps, q2, c - eps)
        f_mp = self._E2_offdiag(q1 - eps, q2, c + eps)
        f_mm = self._E2_offdiag(q1 - eps, q2, c - eps)
        H[0, 2] = H[2, 0] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

        f_pp = self._E2_offdiag(q1, q2 + eps, c + eps)
        f_pm = self._E2_offdiag(q1, q2 + eps, c - eps)
        f_mp = self._E2_offdiag(q1, q2 - eps, c + eps)
        f_mm = self._E2_offdiag(q1, q2 - eps, c - eps)
        H[1, 2] = H[2, 1] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

        return H

    def contract_full(self, V4: V4Repr, Cw: float) -> Tensor:
        """Full O(N^6) contraction for verification (DEPRECATED).

        WARNING: This implementation has known numerical issues with
        off-diagonal elements due to how symmetric matrix perturbations
        are handled. When (c,d) == (e,f) for off-diagonal indices,
        the perturbation is applied twice, leading to incorrect Hessian.

        Use contract() (the sparse O(N²) method) for accurate results.
        The sparse method is validated against Monte Carlo simulations.

        This method is kept only for testing diagonal element agreement.

        Args:
            V4: V4 representation
            Cw: Weight variance scale

        Returns:
            K1 source contribution, shape (N, N)
        """
        N = self.N
        V4_tensor = V4.as_tensor()
        eps = self.eps
        K0 = self.K0

        result = zeros((N, N), dtype=K0.dtype)

        for i in range(N):
            for j in range(N):
                total = torch.tensor(0.0, dtype=K0.dtype, device=K0.device)

                for c in range(N):
                    for d in range(N):
                        for e in range(N):
                            for f in range(N):
                                # Compute d²E2[i,j]/dK[c,d]dK[e,f]
                                # K + ε_{cd} + ε_{ef}
                                K_pp = K0.clone()
                                K_pp[c, d] = K_pp[c, d] + eps
                                if c != d:
                                    K_pp[d, c] = K_pp[d, c] + eps
                                K_pp[e, f] = K_pp[e, f] + eps
                                if e != f:
                                    K_pp[f, e] = K_pp[f, e] + eps

                                # K + ε_{cd}
                                K_cd = K0.clone()
                                K_cd[c, d] = K_cd[c, d] + eps
                                if c != d:
                                    K_cd[d, c] = K_cd[d, c] + eps

                                # K + ε_{ef}
                                K_ef = K0.clone()
                                K_ef[e, f] = K_ef[e, f] + eps
                                if e != f:
                                    K_ef[f, e] = K_ef[f, e] + eps

                                E2_pp = self.gauss.E2_pairwise(K_pp)[i, j]
                                E2_cd = self.gauss.E2_pairwise(K_cd)[i, j]
                                E2_ef = self.gauss.E2_pairwise(K_ef)[i, j]
                                E2_0 = self.gauss.E2_pairwise(K0)[i, j]

                                d2E2 = (E2_pp - E2_cd - E2_ef + E2_0) / (eps * eps)
                                total = total + d2E2 * V4_tensor[c, d, e, f]

                result[i, j] = (Cw / 2) * total

        return result


def compute_k1_source_term(
    K0: Tensor,
    V4: V4Repr,
    gauss: GaussianExpectation,
    Cw: float,
    eps: float = 0.0005,
    mode: str = "auto",
) -> Tensor:
    """Convenience function to compute K1 source term.

    Uses the efficient O(N²) sparse Hessian method.

    Args:
        K0: Kernel matrix, shape (N, N)
        V4: V4 representation (coefficient, O(1))
        gauss: GaussianExpectation instance
        Cw: Weight variance scale
        eps: Finite difference step size
        mode: Computation mode ("auto", "uniform", "general")

    Returns:
        K1 source contribution, shape (N, N)
    """
    op = K1SourceOp(K0, gauss, eps, mode=mode)
    return op.contract(V4, Cw)
