"""Layer update API for finite-width correction calculation.

The main API is `step()` which advances the kernel state by one layer.
"""

from __future__ import annotations

from torch import Tensor

from resnet_eft.backend import einsum, ensure_psd, symmetrize, zeros_like
from resnet_eft.chi_op import ChiOp
from resnet_eft.core_types import KernelState, Params
from resnet_eft.gaussian_expectation import GaussianExpectation
from resnet_eft.k1_source_op import K1SourceOp
from resnet_eft.v4_repr import LocalV4Op, V4Operator, V4Repr, V4Tensor


def compute_V4_wishart(K0: Tensor) -> V4Tensor:
    """Compute initial V4 from Wishart distribution.

    For φ ~ N(0, K0), the empirical kernel G = φᵀφ/n has covariance:
        Cov(G[a,b], G[c,d]) = (1/n) × (K0[a,c] K0[b,d] + K0[a,d] K0[b,c])

    The V4 coefficient (= n × Cov) is:
        V4[a,b,c,d] = K0[a,c] K0[b,d] + K0[a,d] K0[b,c]

    This is derived from Isserlis' theorem (Wick's theorem) for Gaussian variables.

    Args:
        K0: Kernel matrix, shape (N, N)

    Returns:
        V4Tensor with the Wishart variance
    """
    # V4[a,b,c,d] = K0[a,c] K0[b,d] + K0[a,d] K0[b,c]
    # Using einsum for efficiency
    term1 = einsum("ac,bd->abcd", K0, K0)  # K0[a,c] K0[b,d]
    term2 = einsum("ad,bc->abcd", K0, K0)  # K0[a,d] K0[b,c]
    V4_data = term1 + term2
    return V4Tensor(data=V4_data)


def create_resnet_initial_state(
    K0: Tensor,
    fan_in: int,
    params: Params,
    include_wishart_v4: bool = True,
) -> KernelState:
    """Create initial KernelState for ResNet with Wishart V4.

    For real network simulation comparison, the initial V4 should include
    the Wishart variance from sampling φ ~ N(0, K0).

    Args:
        K0: Input kernel matrix, shape (N, N)
        fan_in: Width of the network
        params: Computation parameters
        include_wishart_v4: If True, initialize V4 with Wishart variance.
                           If False, V4 starts as None (theory-only mode).

    Returns:
        KernelState with properly initialized K0 and V4
    """
    V4_init = compute_V4_wishart(K0) if include_wishart_v4 else None

    return KernelState(
        N=K0.shape[0],
        depth=0,
        label="input",
        fan_in=fan_in,  # Used for width_ratio in subsequent layers
        fan_out=fan_in,
        params=params,
        K0=K0,
        K1=None,
        V4=V4_init,
        cache=None,
        K0_version=0,
        K1_version=0,
        V4_version=0 if V4_init is None else 1,
        meta=None,
    )


def step(
    prev: KernelState,
    params: Params,
    fan_out: int,
    label: str | None = None,
    compute_K1: bool = True,
    compute_V4: bool = True,
) -> KernelState:
    """Advance kernel state by one layer.

    This is the main API for layer-wise computation.

    Args:
        prev: Previous layer's KernelState
        params: Computation parameters
        fan_out: Number of neurons in this layer (n_ℓ)
        label: Label for this layer (auto-generated if None)
        compute_K1: Whether to compute K1 correction
        compute_V4: Whether to compute V4 correlation

    Returns:
        Next layer's KernelState

    Width relationships:
        prev.fan_out = this step's fan_in (summation dimension, 1/n denominator)
        fan_out = this step's fan_out (output dimension)

    Note:
        χ and δ4 expectations are evaluated at prev.K0 (layer ℓ-1 statistics)
    """
    fan_in = prev.fan_out

    gauss = GaussianExpectation(params)

    # === K0 update (infinite width) ===
    M = gauss.E2_pairwise(prev.K0)
    K0_next = params.Cb + params.Cw * M
    K0_next = symmetrize(K0_next)
    K0_next = ensure_psd(K0_next, params.psd_check)

    # === K1 update (coefficient, O(1)) ===
    K1_next = None
    if compute_K1:
        K1_next = _compute_K1_next(prev, params, gauss, fan_in)

    # === V4 update (coefficient, O(1)) ===
    V4_next = None
    if compute_V4:
        V4_next = _compute_V4_next(prev, params, gauss, fan_in)

    # === Create next state ===
    next_depth = prev.depth + 1
    next_label = label if label is not None else f"L{next_depth}"

    return KernelState(
        N=prev.N,
        depth=next_depth,
        label=next_label,
        fan_in=fan_in,
        fan_out=fan_out,
        params=params,
        K0=K0_next,
        K1=K1_next,
        V4=V4_next,
        cache=prev.cache,
        K0_version=prev.K0_version + 1,
        K1_version=prev.K1_version + 1 if K1_next is not None else prev.K1_version,
        V4_version=prev.V4_version + 1 if V4_next is not None else prev.V4_version,
        meta=prev.meta,
    )


def _compute_K1_next(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
) -> Tensor | None:
    """Compute K1 update (coefficient-based, MLP discrete step).

    K1 coefficient update formula:
        K1←K1: K1_next += width_ratio × χ(K1_prev)
        K1←V4: K1_next += width_ratio × K1SourceOp.contract(V4, Cw)

    The K1 source is (Cw/2) × Hess[K0] : V4, computed via K1SourceOp with
    the path controlled by Params.k1_mode:
        - "auto":    auto-detect uniform K0 and use optimized path
        - "uniform": force uniform K0 optimization
        - "general": force general O(N²) sparse Hessian path
    """
    if prev.K1 is None and prev.V4 is None:
        return None

    chi_op = gauss.build_chi_op(prev.K0)
    K1_next = zeros_like(prev.K0)

    # Width ratio for coefficient propagation
    width_ratio = fan_in / prev.fan_in if prev.fan_in is not None and prev.fan_in > 0 else 1.0

    # K1_prev contribution (width_ratio required)
    if prev.K1 is not None:
        K1_next = K1_next + width_ratio * chi_op.apply_pair(prev.K1)

    # V4_prev contribution: (Cw/2) × E2''(K0) × V4
    # Matches MC measurements of n×(E[G]-K0)
    if prev.V4 is not None and prev.fan_in is not None:
        k1_source = K1SourceOp(prev.K0, gauss, mode=params.k1_mode)
        K1_next = K1_next + width_ratio * k1_source.contract(prev.V4, params.Cw)

    return symmetrize(K1_next)


def _compute_V4_next(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
) -> V4Repr:
    """Compute V4 update (coefficient-based).

    V4 coefficient update formula:
        V4^{(ℓ)} = Cw² × ⟨ΔΔ⟩ + width_ratio × χ⊗χ(V4^{(ℓ-1)})

    Important: Local term has NO width_ratio. Transport term has width_ratio.

    For small N (≤6): Use full tensor computation
    For large N (>6): Use operator representation (not yet implemented)
    """
    mode = "tensor" if prev.N <= 6 else "operator"

    if mode == "tensor":
        return _update_V4_tensor(prev, params, gauss, fan_in)
    else:
        return _update_V4_operator(prev, params, gauss, fan_in)


def _update_V4_tensor(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
) -> V4Tensor:
    """Small-N: Full tensor computation of V4."""
    E2 = gauss.E2_pairwise(prev.K0)
    E4 = gauss.E4_pairwise(prev.K0)

    # Local term: ⟨ΔΔ⟩ = E4 - E2⊗E2 (NOT Wick decomposition!)
    disconnected = einsum("ij,kl->ijkl", E2, E2)
    local_term = (params.Cw**2) * (E4 - disconnected)

    # Transport term (if previous V4 exists)
    if prev.V4 is not None and prev.fan_in is not None:
        chi_op = gauss.build_chi_op(prev.K0)
        V4_prev = prev.V4.as_tensor()

        width_ratio = fan_in / prev.fan_in if prev.fan_in > 0 else 1.0
        transport_term = width_ratio * _compute_transport_tensor(chi_op, V4_prev)

        V4_data = local_term + transport_term
    else:
        V4_data = local_term

    return V4Tensor(data=V4_data)


def _update_V4_operator(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
) -> V4Operator:
    """Large-N: Operator representation of V4.

    Note: For large N, E4 computation is expensive, so we use LocalV4Op
    only when necessary.
    """
    chi_op = gauss.build_chi_op(prev.K0)

    # Local term (still requires E4 computation - expensive!)
    E2 = gauss.E2_pairwise(prev.K0)
    E4 = gauss.E4_pairwise(prev.K0)
    disconnected = einsum("ij,kl->ijkl", E2, E2)
    local_tensor = (params.Cw**2) * (E4 - disconnected)
    local_op = LocalV4Op(local_tensor=local_tensor)

    # Width ratio for transport
    width_ratio = fan_in / prev.fan_in if prev.fan_in is not None and prev.fan_in > 0 else 1.0

    return V4Operator(
        local_op=local_op,
        chi_op=chi_op,
        prev_V4=prev.V4,
        width_ratio=width_ratio,
    )


def _compute_transport_tensor(chi_op: ChiOp, V4_prev: Tensor) -> Tensor:
    """Compute V4 transport using naive einsum (for small N).

    Transport formula (pair-space):
        V4_next = 0.5 × (χ @ V4_prev @ χᵀ + χ @ S @ V4_prev @ S @ χᵀ)

    where S is the swap operator (transpose in pair indices).
    """
    # Build full χ tensor for naive einsum
    chi_tensor = _build_chi_tensor_naive(chi_op)

    # Term 1: χ[ab,mn] × V4[mn,pq] × χ[cd,pq] → V4[ab,cd]
    term1 = einsum("abmn,mnpq,cdpq->abcd", chi_tensor, V4_prev, chi_tensor)

    # Term 2: χ[ab,mn] × V4[nm,pq] × χ[cd,pq] → V4[ab,cd] (swap in V4)
    term2 = einsum("abmn,nmpq,cdpq->abcd", chi_tensor, V4_prev, chi_tensor)

    return 0.5 * (term1 + term2)


def resnet_step(
    prev: KernelState,
    params: Params,
    eps: float = 1.0,
    compute_K1: bool = False,
    compute_V4: bool = True,
) -> KernelState:
    """Single step of pre-activation ResNet (O(ε²) theory).

    Update rule: phi' = phi + eps * W * sigma(phi)

    Computes kernel updates accurate to O(ε²). Higher order terms (O(ε⁴))
    are neglected, so the theory is most accurate for small eps.

    K0 update:
        K0' = K0 + ε² × Cw × E2(K0)
        No Cb term in pre-activation ResNet

    V4 update:
        V4' = V4 + ε² × (width_ratio × transport + source)

        Transport: χ @ V4 + V4 @ χᵀ (linearized)
        Source: Σ(K)_{ab,cd} = Cw × [K0(a,c)E2(b,d) + K0(a,d)E2(b,c)
                                    + K0(b,c)E2(a,d) + K0(b,d)E2(a,c)]

        This source term is n × Cov(G^(1), G^(1)) where
        G^(1) = (φᵀWᵀσ + σᵀWφ) / n is the first-order increment.

    Validation:
        - For small ε (e.g., 0.1), theory matches MC/real network well (<1%)
        - For ε=1, O(ε⁴) terms contribute ~20% additional variance

    Args:
        prev: Previous layer's KernelState
        params: Computation parameters
        eps: Residual coefficient (use small eps for accurate theory)
        compute_K1: K1 computation (disabled, theory not finalized)
        compute_V4: Whether to compute V4 correlation

    Returns:
        Updated KernelState
    """
    fan_in = prev.fan_out
    gauss = GaussianExpectation(params)

    # === K0 update (incremental form) ===
    # K₀' = K₀ + ε² × Cw × E2(K₀)
    E2 = gauss.E2_pairwise(prev.K0)
    K0_increment = params.Cw * E2  # No Cb term in pre-activation ResNet
    K0_next = prev.K0 + eps**2 * K0_increment
    K0_next = symmetrize(K0_next)
    K0_next = ensure_psd(K0_next, params.psd_check)

    # === V4 update (incremental form) ===
    V4_next = None
    if compute_V4:
        V4_next = _compute_V4_resnet(prev, params, gauss, fan_in, eps)

    # === K1 update (incremental form) ===
    K1_next = None
    if compute_K1:
        K1_next = _compute_K1_resnet(prev, params, gauss, fan_in, eps)

    # === Create next state ===
    next_depth = prev.depth + 1
    next_label = f"R{next_depth}"

    return KernelState(
        N=prev.N,
        depth=next_depth,
        label=next_label,
        fan_in=fan_in,
        fan_out=prev.fan_out,  # ResNet preserves width
        params=params,
        K0=K0_next,
        K1=K1_next,
        V4=V4_next,
        cache=prev.cache,
        K0_version=prev.K0_version + 1,
        K1_version=prev.K1_version + 1 if K1_next is not None else prev.K1_version,
        V4_version=prev.V4_version + 1 if V4_next is not None else prev.V4_version,
        meta=prev.meta,
    )


def _compute_K1_resnet(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
    eps: float,
) -> Tensor | None:
    """Compute K1 update for ResNet (incremental form).

    Implements eq. (K1-resnet-diff) from Appendix A:
        K1' = K1 + eps^2 * (chi[K1] + (1/2) Hess[K0] : V4) + O(eps^4)
    """
    if prev.K1 is None and prev.V4 is None:
        return None

    chi_op = gauss.build_chi_op(prev.K0)
    K1_increment = zeros_like(prev.K0)

    # Width ratio for coefficient propagation
    width_ratio = fan_in / prev.fan_in if prev.fan_in is not None and prev.fan_in > 0 else 1.0

    # K1 transport term
    if prev.K1 is not None:
        K1_increment = K1_increment + width_ratio * chi_op.apply_pair(prev.K1)

    # K1 source from V4: (Cw/2) × E2''(K0) × V4
    if prev.V4 is not None and prev.fan_in is not None:
        k1_source = K1SourceOp(prev.K0, gauss, mode=params.k1_mode)
        K1_increment = K1_increment + width_ratio * k1_source.contract(prev.V4, params.Cw)

    # Incremental form: K1' = K1 + ε² × K1_increment
    if prev.K1 is not None:
        K1_next = prev.K1 + eps**2 * K1_increment
    else:
        K1_next = eps**2 * K1_increment

    return symmetrize(K1_next)


def _compute_V4_resnet(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
    eps: float,
) -> V4Tensor:
    """Compute V4 update for ResNet (O(ε²) theory).

    V4 update formula:
        V4' = width_ratio × V4 + ε² × width_ratio × transport + ε² × source

    where:
    - width_ratio = n_{ℓ-1}/n_{ℓ-2} = fan_in / prev.fan_in
    - Transport: χ @ V4 + V4 @ χᵀ (linearized)
    - Source: Σ(K)_{ab,cd} = Cw × [K0(a,c)E2(b,d) + K0(a,d)E2(b,c)
                                  + K0(b,c)E2(a,d) + K0(b,d)E2(a,c)]
              (4 Wick pairings from G^(1) = φᵀWᵀσ/n + σᵀWφ/n)

    Note:
    - Source is n × Cov(G^(1), G^(1)), NOT Cw²(E4-E2⊗E2) which is for MLP
    - Only O(ε²) terms are included; O(ε⁴) terms are neglected
    """
    # Always use tensor mode for ResNet (see docstring for rationale)
    return _update_V4_tensor_resnet(prev, params, gauss, fan_in, eps)


def _update_V4_tensor_resnet(
    prev: KernelState,
    params: Params,
    gauss: GaussianExpectation,
    fan_in: int,
    eps: float,
) -> V4Tensor:
    """Full tensor computation of V4 for ResNet (O(ε²) theory).

    V4 update formula:
        V4' = width_ratio × V4_prev + ε² × width_ratio × transport + ε² × source

    where:
    - transport = χ @ V4 + V4 @ χᵀ (linearized)
    - source = Cw × [K0⊗E2 + ...] (4 Wick pairings)
    - width_ratio = n_{ℓ-1}/n_{ℓ-2} = fan_in / prev.fan_in
    """
    E2 = gauss.E2_pairwise(prev.K0)

    # Source term for ResNet: Sigma(K) = n * Cov(G^(1), G^(1))
    # where G^(1) = (φ^T W^T σ + σ^T W φ) / n is the first-order increment.
    #
    # Σ(K)_{ab,cd} = C_W × [K0(a,c)E2(b,d) + K0(a,d)E2(b,c) + K0(b,c)E2(a,d) + K0(b,d)E2(a,c)]
    #
    # This is the 4 Wick pairings from the product of two G^(φδ) + G^(δφ) terms.
    source_term = params.Cw * (
        einsum("ac,bd->abcd", prev.K0, E2)
        + einsum("ad,bc->abcd", prev.K0, E2)
        + einsum("bc,ad->abcd", prev.K0, E2)
        + einsum("bd,ac->abcd", prev.K0, E2)
    )

    # Transport term (if previous V4 exists)
    if prev.V4 is not None and prev.fan_in is not None:
        chi_op = gauss.build_chi_op(prev.K0)
        V4_prev = prev.V4.as_tensor()

        # width_ratio for coefficient propagation
        width_ratio = fan_in / prev.fan_in if prev.fan_in > 0 else 1.0

        # Linearized transport: chi @ V4 + V4 @ chi^T
        transport = _compute_transport_linear(chi_op, V4_prev)

        # Full formula: V4' = width_ratio * V4_prev + eps^2 * width_ratio * transport + eps^2 * source
        V4_data = width_ratio * V4_prev + eps**2 * width_ratio * transport + eps**2 * source_term
    else:
        # First layer: only source term (no V4 to transport)
        V4_data = eps**2 * source_term

    return V4Tensor(data=V4_data)


def _compute_transport_linear(chi_op: ChiOp, V4_prev: Tensor) -> Tensor:
    """Compute linearized V4 transport: χ @ V4 + V4 @ χᵀ.

    This is the linearized form used in ResNet continuous limit,
    different from the quadratic form 0.5(χ @ V4 @ χᵀ + χ @ V4ᵀ @ χᵀ) used in MLP.

    For ResNet:
        dV4/dt = χ @ V4 + V4 @ χᵀ + source
    """
    chi_tensor = _build_chi_tensor_naive(chi_op)

    # Term 1: χ[ab,mn] × V4[mn,cd] → result[ab,cd]
    term1 = einsum("abmn,mncd->abcd", chi_tensor, V4_prev)

    # Term 2: V4[ab,mn] × χᵀ[mn,cd] = V4[ab,mn] × χ[cd,mn] → result[ab,cd]
    term2 = einsum("abmn,cdmn->abcd", V4_prev, chi_tensor)

    return term1 + term2


def _build_chi_tensor_naive(chi_op: ChiOp) -> Tensor:
    """Build full χ tensor for testing/naive computation.

    χ[x₁,x₂,y₁,y₂] = (Cw/2) × {
        [δ(x₁-y₁)δ(x₂-y₂) + δ(x₁-y₂)δ(x₂-y₁)] × Epp[x₁,x₂]
      + δ(x₁-y₁)δ(x₁-y₂) × E2s[x₁,x₂]
      + δ(x₂-y₁)δ(x₂-y₂) × Es2[x₁,x₂]
    }
    """
    from resnet_eft.backend import zeros

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
