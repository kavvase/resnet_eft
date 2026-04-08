"""V4 representation interfaces and implementations.

This module defines:
- V4Repr: Protocol for V4 representations
- V4Tensor: Full tensor representation for small N
- V4Operator: Composite operator representation for large N
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor

from resnet_eft.backend import einsum, zeros, zeros_like

if TYPE_CHECKING:
    from resnet_eft.chi_op import ChiOp


@runtime_checkable
class V4Repr(Protocol):
    """Protocol for V4 representations.

    V4 can be represented as:
    - V4Tensor: Full (N,N,N,N) tensor for small N
    - V4Operator: Composite operator for large N (transport-only mode)
    """

    def apply_pair(self, A: Tensor) -> Tensor:
        """Apply V4 in pair-space.

        B[x₁,x₂] = Σ_{x₃,x₄} V4[x₁,x₂;x₃,x₄] A[x₃,x₄]

        Args:
            A: Input matrix of shape (N, N)

        Returns:
            Output matrix of shape (N, N)
        """
        ...

    def apply_pair_T(self, A: Tensor) -> Tensor:
        """Apply V4^T (pair-space transpose) in pair-space.

        This is the "left-right pair swap" operation:
        V4^T[a,b;c,d] = V4[c,d;a,b]

        (M_V^T @ vec(A))[(a,b)] = Σ_{c,d} V4[c,d;a,b] A[c,d]

        This is NOT the same as A.T (within-pair swap).

        Args:
            A: Input matrix of shape (N, N)

        Returns:
            Output matrix of shape (N, N)
        """
        ...

    def as_tensor(self) -> Tensor:
        """Materialize as full (N,N,N,N) tensor.

        Warning: This is expensive for large N. Use only for testing.

        Returns:
            Full tensor of shape (N, N, N, N)
        """
        ...

    def get_diag_diag(self) -> Tensor:
        """Get V4[i,i,j,j] slice.

        Returns:
            Matrix of shape (N, N) where result[i,j] = V4[i,i,j,j]
        """
        ...

    def get_cross_diag(self) -> Tensor:
        """Get V4[i,j,i,j] slice.

        Returns:
            Matrix of shape (N, N) where result[i,j] = V4[i,j,i,j]
        """
        ...

    def get_diag_cross_left(self) -> Tensor:
        """Get V4[i,i,i,j] slice = Cov(K[i,i], K[i,j]).

        Used for K1 source term calculation (efficient O(N²) contraction).

        Returns:
            Matrix of shape (N, N) where result[i,j] = V4[i,i,i,j]
        """
        ...

    def get_diag_cross_right(self) -> Tensor:
        """Get V4[j,j,i,j] slice = Cov(K[j,j], K[i,j]).

        Used for K1 source term calculation (efficient O(N²) contraction).
        Note: This is NOT the same as get_diag_cross_left()[j,i] in general!

        Returns:
            Matrix of shape (N, N) where result[i,j] = V4[j,j,i,j]
        """
        ...

    def scale(self, c: float) -> V4Repr:
        """Return a scaled copy.

        Args:
            c: Scale factor

        Returns:
            New V4Repr with all values multiplied by c
        """
        ...


@dataclass
class V4Tensor:
    """Full tensor representation of V4 for small N.

    Attributes:
        data: Full tensor of shape (N, N, N, N)
    """

    data: Tensor

    def apply_pair(self, A: Tensor) -> Tensor:
        """Apply V4 in pair-space using einsum."""
        return einsum("ijkl,kl->ij", self.data, A)

    def apply_pair_T(self, A: Tensor) -> Tensor:
        """Apply V4^T (pair-space transpose) using einsum.

        V4^T[a,b;c,d] = V4[c,d;a,b]
        (M_V^T @ vec(A))[(a,b)] = Σ_{c,d} V4[c,d;a,b] A[c,d]
        """
        return einsum("klab,kl->ab", self.data, A)

    def as_tensor(self) -> Tensor:
        """Return the underlying tensor."""
        return self.data

    def get_diag_diag(self) -> Tensor:
        """Get V4[i,i,j,j] slice."""
        N = self.data.shape[0]
        result = zeros((N, N), dtype=self.data.dtype)
        for i in range(N):
            for j in range(N):
                result[i, j] = self.data[i, i, j, j]
        return result

    def get_cross_diag(self) -> Tensor:
        """Get V4[i,j,i,j] slice."""
        N = self.data.shape[0]
        result = zeros((N, N), dtype=self.data.dtype)
        for i in range(N):
            for j in range(N):
                result[i, j] = self.data[i, j, i, j]
        return result

    def get_diag_cross_left(self) -> Tensor:
        """Get V4[i,i,i,j] slice = Cov(K[i,i], K[i,j])."""
        N = self.data.shape[0]
        result = zeros((N, N), dtype=self.data.dtype)
        for i in range(N):
            for j in range(N):
                result[i, j] = self.data[i, i, i, j]
        return result

    def get_diag_cross_right(self) -> Tensor:
        """Get V4[j,j,i,j] slice = Cov(K[j,j], K[i,j])."""
        N = self.data.shape[0]
        result = zeros((N, N), dtype=self.data.dtype)
        for i in range(N):
            for j in range(N):
                result[i, j] = self.data[j, j, i, j]
        return result

    def scale(self, c: float) -> V4Tensor:
        """Return scaled copy."""
        return V4Tensor(data=c * self.data)


@dataclass
class V4SliceRepr:
    """Memory-efficient V4 representation storing only K1-relevant slices.

    This representation stores only 4 N×N matrices instead of the full N^4 tensor,
    reducing memory from O(N^4) to O(N^2). Essential for large N (N > 100).

    Limitations:
    - apply_pair() and apply_pair_T() are NOT supported (raises NotImplementedError)
    - as_tensor() is NOT supported (would defeat the purpose)
    - Only suitable for K1 source term computation via K1SourceOp.contract()

    The stored slices are:
    - diag_diag:    V4[i,i,j,j]   = Cov(K[i,i], K[j,j])
    - cross_diag:   V4[i,j,i,j]   = Cov(K[i,j], K[i,j])
    - diag_cross_L: V4[i,i,i,j]   = Cov(K[i,i], K[i,j])
    - diag_cross_R: V4[j,j,i,j]   = Cov(K[j,j], K[i,j])

    Attributes:
        _diag_diag: V4[i,i,j,j] slice, shape (N, N)
        _cross_diag: V4[i,j,i,j] slice, shape (N, N)
        _diag_cross_L: V4[i,i,i,j] slice, shape (N, N)
        _diag_cross_R: V4[j,j,i,j] slice, shape (N, N)
    """

    _diag_diag: Tensor
    _cross_diag: Tensor
    _diag_cross_L: Tensor
    _diag_cross_R: Tensor

    @classmethod
    def from_slices(cls, slices: dict[str, Tensor]) -> V4SliceRepr:
        """Create from dictionary of slices.

        Args:
            slices: Dictionary with keys 'diag_diag', 'cross_diag',
                   'diag_cross_L', 'diag_cross_R'

        Returns:
            V4SliceRepr instance
        """
        return cls(
            _diag_diag=slices["diag_diag"],
            _cross_diag=slices["cross_diag"],
            _diag_cross_L=slices["diag_cross_L"],
            _diag_cross_R=slices["diag_cross_R"],
        )

    def apply_pair(self, A: Tensor) -> Tensor:
        """Not supported for slice-only representation."""
        raise NotImplementedError(
            "V4SliceRepr does not support apply_pair(). "
            "Use V4Tensor or V4Operator for transport operations."
        )

    def apply_pair_T(self, A: Tensor) -> Tensor:
        """Not supported for slice-only representation."""
        raise NotImplementedError(
            "V4SliceRepr does not support apply_pair_T(). "
            "Use V4Tensor or V4Operator for transport operations."
        )

    def as_tensor(self) -> Tensor:
        """Not supported for slice-only representation."""
        raise NotImplementedError(
            "V4SliceRepr does not support as_tensor() to preserve memory efficiency. "
            "Use V4Tensor if you need the full tensor."
        )

    def get_diag_diag(self) -> Tensor:
        """Get V4[i,i,j,j] slice."""
        return self._diag_diag

    def get_cross_diag(self) -> Tensor:
        """Get V4[i,j,i,j] slice."""
        return self._cross_diag

    def get_diag_cross_left(self) -> Tensor:
        """Get V4[i,i,i,j] slice."""
        return self._diag_cross_L

    def get_diag_cross_right(self) -> Tensor:
        """Get V4[j,j,i,j] slice."""
        return self._diag_cross_R

    def scale(self, c: float) -> V4SliceRepr:
        """Return scaled copy."""
        return V4SliceRepr(
            _diag_diag=c * self._diag_diag,
            _cross_diag=c * self._cross_diag,
            _diag_cross_L=c * self._diag_cross_L,
            _diag_cross_R=c * self._diag_cross_R,
        )

    def transport_update(
        self,
        chi_op: ChiOp,
        Cw: float,
        width_ratio: float = 1.0,
    ) -> V4SliceRepr:
        """Transport slices through one layer using O(N²) closed-form formulas.

        The 4 slices are closed under transport due to χ's δ-structure:
        - K'_ii = a_i × K_ii  (diagonal)
        - K'_ij = α_ij × K_ij + β^L_ij × K_ii + β^R_ij × K_jj  (off-diagonal)

        The covariance transport is then:
        - diag_diag'[i,j] = wr × a_i × a_j × diag_diag[i,j]
        - cross_diag'[i,j] = wr × (α²×cd + β^L²×dd[i,i] + β^R²×dd[j,j]
                                  + 2αβ^L×dcL + 2αβ^R×dcR + 2β^Lβ^R×dd)
        - diag_cross_L'[i,j] = wr × a_i × (α×dcL + β^L×dd[i,i] + β^R×dd[i,j])
        - diag_cross_R'[i,j] = wr × a_j × (α×dcR + β^L×dd[i,j] + β^R×dd[j,j])

        Args:
            chi_op: ChiOp instance for this layer
            Cw: Weight variance parameter
            width_ratio: fan_in / prev.fan_in for coefficient propagation

        Returns:
            New V4SliceRepr with transported slices
        """
        import torch

        # Extract χ coefficients
        # a_i = (Cw/2) × (2×Epp[i,i] + E2s[i,i] + Es2[i,i])
        a = (Cw / 2) * (2 * chi_op.Epp.diag() + chi_op.E2s.diag() + chi_op.Es2.diag())
        # α_ij = Cw × Epp[i,j]
        alpha = Cw * chi_op.Epp
        # β^L_ij = (Cw/2) × E2s[i,j]
        beta_L = (Cw / 2) * chi_op.E2s
        # β^R_ij = (Cw/2) × Es2[i,j]
        beta_R = (Cw / 2) * chi_op.Es2

        wr = width_ratio
        N = self._diag_diag.shape[0]

        # Current slices
        dd = self._diag_diag
        cd = self._cross_diag
        dcL = self._diag_cross_L
        dcR = self._diag_cross_R

        # Transport formulas (vectorized for efficiency)

        # diag_diag'[i,j] = wr × a_i × a_j × diag_diag[i,j]
        dd_next = wr * torch.outer(a, a) * dd

        # For diagonal elements (i=j), all slices equal diag_diag[i,i]
        # For off-diagonal, use the full formulas

        # diag_cross_L'[i,j] = wr × a_i × (α_ij×dcL + β^L_ij×dd[i,i] + β^R_ij×dd[i,j])
        dcL_next = wr * a[:, None] * (alpha * dcL + beta_L * dd.diag()[:, None] + beta_R * dd)

        # diag_cross_R'[i,j] = wr × a_j × (α_ij×dcR + β^L_ij×dd[i,j] + β^R_ij×dd[j,j])
        dcR_next = wr * a[None, :] * (alpha * dcR + beta_L * dd + beta_R * dd.diag()[None, :])

        # cross_diag'[i,j] = wr × (α²×cd + β^L²×dd[i,i] + β^R²×dd[j,j]
        #                        + 2αβ^L×dcL + 2αβ^R×dcR + 2β^Lβ^R×dd)
        cd_next = wr * (
            alpha**2 * cd
            + beta_L**2 * dd.diag()[:, None]
            + beta_R**2 * dd.diag()[None, :]
            + 2 * alpha * beta_L * dcL
            + 2 * alpha * beta_R * dcR
            + 2 * beta_L * beta_R * dd
        )

        # Fix diagonal elements: for i=j, all slices should equal dd_next[i,i]
        for i in range(N):
            cd_next[i, i] = dd_next[i, i]
            dcL_next[i, i] = dd_next[i, i]
            dcR_next[i, i] = dd_next[i, i]

        return V4SliceRepr(
            _diag_diag=dd_next,
            _cross_diag=cd_next,
            _diag_cross_L=dcL_next,
            _diag_cross_R=dcR_next,
        )

    def add_local(self, local_slices: V4SliceRepr) -> V4SliceRepr:
        """Add local V4 contribution (from current layer's E4).

        Args:
            local_slices: Local term slices from GaussianExpectation.compute_V4_slices_mc()

        Returns:
            New V4SliceRepr with local term added
        """
        return V4SliceRepr(
            _diag_diag=self._diag_diag + local_slices._diag_diag,
            _cross_diag=self._cross_diag + local_slices._cross_diag,
            _diag_cross_L=self._diag_cross_L + local_slices._diag_cross_L,
            _diag_cross_R=self._diag_cross_R + local_slices._diag_cross_R,
        )


@dataclass
class LocalV4Op:
    """Local V4 term: Cw² × (E4 - E2⊗E2).

    Important: Wick decomposition is NOT used here because σ(φ) is non-Gaussian.

    Note: The local term is symmetric under left-right pair swap:
    E4[a,b,c,d] = E4[c,d,a,b] (by symmetry of Gaussian expectations)

    Attributes:
        local_tensor: Pre-computed local term tensor (N, N, N, N)
    """

    local_tensor: Tensor

    def apply_pair(self, A: Tensor) -> Tensor:
        """Apply local V4 term in pair-space."""
        return einsum("ijkl,kl->ij", self.local_tensor, A)

    def apply_pair_T(self, A: Tensor) -> Tensor:
        """Apply local V4^T (pair-space transpose).

        For the local term, V4^T = V4 due to symmetry of Gaussian expectations.
        """
        return einsum("klab,kl->ab", self.local_tensor, A)

    def as_tensor(self) -> Tensor:
        """Return the local tensor."""
        return self.local_tensor


@dataclass
class V4Operator:
    """Composite operator representation of V4 for large N.

    V4_next = local + transport

    where:
    - local: LocalV4Op (can be None for transport-only mode)
    - transport: χ⊗χ applied to prev_V4

    Transport formula (pair-space matrix form, from paper eq:dV4):
        term1 = M_χ @ M_V_prev @ M_χᵀ
        term2 = M_χ @ M_V_prev^T @ M_χᵀ

    where M_V_prev^T is the pair-space transpose (left-right pair swap):
        V4^T[a,b;c,d] = V4[c,d;a,b]

    This is NOT the same as within-pair swap (A.T).

    Paper eq:dV4:
        δV₄/δV₄ = (1/2)[χ(x₁,x₂;y₁,y₂)χ(x₃,x₄;y₃,y₄)    ... term1
                      + χ(x₁,x₂;y₃,y₄)χ(x₃,x₄;y₁,y₂)]   ... term2

    In term2, (y₁,y₂) ↔ (y₃,y₄) are swapped, corresponding to V^T.

    Attributes:
        local_op: Local V4 operator (None for transport-only)
        chi_op: χ operator for transport
        prev_V4: Previous layer's V4 (None if no transport)
        width_ratio: fan_in / prev.fan_in for coefficient propagation
    """

    local_op: LocalV4Op | None
    chi_op: ChiOp
    prev_V4: V4Repr | None
    width_ratio: float = 1.0

    def apply_pair(self, A: Tensor) -> Tensor:
        """Apply V4 in pair-space."""
        result = zeros_like(A)

        # Local term
        if self.local_op is not None:
            result = result + self.local_op.apply_pair(A)

        # Transport term (if previous V4 exists)
        if self.prev_V4 is not None:
            result = result + self._transport_term(A)

        return result

    def apply_pair_T(self, A: Tensor) -> Tensor:
        """Apply V4^T (pair-space transpose).

        For V4_next = local + transport, we have:
        V4_next^T = local^T + transport^T

        Since transport = width_ratio × 0.5 × (χ V χᵀ + χ V^T χᵀ),
        transport^T = width_ratio × 0.5 × (χ V^T χᵀ + χ V χᵀ) = transport.
        (The two terms simply swap, regardless of χ's symmetry.)

        For local term, local^T = local by Gaussian symmetry.
        """
        result = zeros_like(A)

        # Local term (symmetric under pair swap)
        if self.local_op is not None:
            result = result + self.local_op.apply_pair_T(A)

        # Transport term (symmetric under pair swap)
        if self.prev_V4 is not None:
            result = result + self._transport_term_T(A)

        return result

    def _transport_term(self, A: Tensor) -> Tensor:
        """Compute transport term: width_ratio × 0.5 × (χ V χᵀ + χ V^T χᵀ).

        Pair-space matrix form (from paper eq:dV4):
            term1 = M_χ @ M_V_prev @ M_χᵀ
            term2 = M_χ @ M_V_prev^T @ M_χᵀ  (V^T = left-right pair swap)

        In operator form:
            term1: χ(V(χᵀ(A)))
            term2: χ(V^T(χᵀ(A)))

        Args:
            A: Input matrix

        Returns:
            Transport contribution
        """
        chi = self.chi_op
        V_prev = self.prev_V4

        assert V_prev is not None

        # Apply χᵀ first
        B = chi.apply_pair_T(A)

        # Term 1: M_χ @ M_V_prev @ M_χᵀ
        term1 = chi.apply_pair(V_prev.apply_pair(B))

        # Term 2: M_χ @ M_V_prev^T @ M_χᵀ (V^T = left-right pair swap)
        term2 = chi.apply_pair(V_prev.apply_pair_T(B))

        return self.width_ratio * 0.5 * (term1 + term2)

    def _transport_term_T(self, A: Tensor) -> Tensor:
        """Compute transport^T (same as transport due to symmetry)."""
        # transport^T = χ(V^T(χᵀ(A))) + χ(V(χᵀ(A)))
        # which equals transport since we sum both terms
        return self._transport_term(A)

    def as_tensor(self) -> Tensor:
        """Materialize as full tensor (expensive!).

        For testing only. Computes apply_pair for each basis element.
        """
        N = self._get_N()
        result = zeros((N, N, N, N), dtype=self.chi_op.Epp.dtype)

        for k in range(N):
            for m in range(N):
                # Create basis element e_{km}
                e_km = zeros((N, N), dtype=self.chi_op.Epp.dtype)
                e_km[k, m] = 1.0

                # Compute column of the pair-space matrix
                col = self.apply_pair(e_km)
                result[:, :, k, m] = col

        return result

    def _get_N(self) -> int:
        """Get the dimension N."""
        return self.chi_op.Epp.shape[0]

    def get_diag_diag(self) -> Tensor:
        """Get V4[i,i,j,j] slice."""
        return self.as_tensor().diagonal(dim1=0, dim2=1).diagonal(dim1=0, dim2=1).T

    def get_cross_diag(self) -> Tensor:
        """Get V4[i,j,i,j] slice."""
        full = self.as_tensor()
        N = self._get_N()
        result = zeros((N, N), dtype=self.chi_op.Epp.dtype)
        for i in range(N):
            for j in range(N):
                result[i, j] = full[i, j, i, j]
        return result

    def get_diag_cross_left(self) -> Tensor:
        """Get V4[i,i,i,j] slice = Cov(K[i,i], K[i,j]).

        Computes V4[i,i,i,j] = (apply_pair(e_{ij}))[i,i] for each (i,j).
        """
        N = self._get_N()
        result = zeros((N, N), dtype=self.chi_op.Epp.dtype)

        for i in range(N):
            for j in range(N):
                # V4[i,i,i,j] = Σ_{k,l} V4[i,i,k,l] * δ_{ki}δ_{lj}
                e_ij = zeros((N, N), dtype=self.chi_op.Epp.dtype)
                e_ij[i, j] = 1.0
                col = self.apply_pair(e_ij)
                result[i, j] = col[i, i]

        return result

    def get_diag_cross_right(self) -> Tensor:
        """Get V4[j,j,i,j] slice = Cov(K[j,j], K[i,j]).

        Computes V4[j,j,i,j] = (apply_pair(e_{ij}))[j,j] for each (i,j).
        """
        N = self._get_N()
        result = zeros((N, N), dtype=self.chi_op.Epp.dtype)

        for i in range(N):
            for j in range(N):
                # V4[j,j,i,j] = Σ_{k,l} V4[j,j,k,l] * δ_{ki}δ_{lj}
                e_ij = zeros((N, N), dtype=self.chi_op.Epp.dtype)
                e_ij[i, j] = 1.0
                col = self.apply_pair(e_ij)
                result[i, j] = col[j, j]

        return result

    def scale(self, c: float) -> V4Operator:
        """Return scaled copy."""
        scaled_local = None
        if self.local_op is not None:
            scaled_local = LocalV4Op(local_tensor=c * self.local_op.local_tensor)

        return V4Operator(
            local_op=scaled_local,
            chi_op=self.chi_op,
            prev_V4=self.prev_V4.scale(c) if self.prev_V4 is not None else None,
            width_ratio=self.width_ratio,
        )
