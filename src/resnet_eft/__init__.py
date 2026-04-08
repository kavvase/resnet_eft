"""resnet_eft: Collective-kernel EFT for finite-width pre-activation ResNets.

This package implements the collective bilocal stochastic EFT for pre-activation ResNets.
It computes the leading-order mean kernel K0, the kernel fluctuation covariance V4,
and the NLO mean-kernel correction Keff via the exact block-law + Gaussian closure hierarchy.

Main API:
    step(): Advance kernel state by one layer
    KernelState: State carried through layers
    Params: Computation parameters

Example:
    >>> from resnet_eft import KernelState, Params, step
    >>> import torch
    >>>
    >>> # Input points
    >>> X = torch.randn(N, d_in)
    >>> K0_init = X @ X.T / d_in
    >>>
    >>> # Initial state
    >>> state = KernelState.from_input(K0_init, fan_out=n_hidden)
    >>>
    >>> # Forward through layers
    >>> params = Params(act='relu', Cw=2.0, Cb=0.0)
    >>> for _ in range(L):
    ...     state = step(state, params, fan_out=n_hidden)
    >>>
    >>> # Get physical quantities
    >>> K0 = state.K0                          # Infinite-width kernel
    >>> K1_phys = state.get_physical_K1()       # NLO 2-point correction
    >>> V4_phys = state.get_physical_V4()       # 4-point connected correlation
"""

__version__ = "0.1.0"

# Validation submodule (import explicitly: from resnet_eft.validation import ...)
from resnet_eft import validation
from resnet_eft.chi_op import ChiOp
from resnet_eft.core_types import ActivationSpec, Cache, KernelState, Params
from resnet_eft.gaussian_expectation import GaussianExpectation
from resnet_eft.k1_source_op import K1SourceOp, compute_k1_source_term
from resnet_eft.layer_update import (
    compute_V4_wishart,
    create_resnet_initial_state,
    resnet_step,
    step,
)
from resnet_eft.v4_repr import (
    LocalV4Op,
    V4Operator,
    V4Repr,
    V4SliceRepr,
    V4Tensor,
)

__all__ = [
    # Main API
    "step",
    "resnet_step",
    "create_resnet_initial_state",
    "compute_V4_wishart",
    "KernelState",
    "Params",
    # Supporting types
    "ActivationSpec",
    "Cache",
    "GaussianExpectation",
    "ChiOp",
    "K1SourceOp",
    "compute_k1_source_term",
    # V4 representations
    "V4Repr",
    "V4Tensor",
    "V4SliceRepr",
    "V4Operator",
    "LocalV4Op",
]
