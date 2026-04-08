"""Validation utilities for NNGP perturbation theory.

This module provides Monte Carlo and real network simulations
for validating theoretical predictions.

Usage:
    from resnet_eft.validation import mc_kernel_statistics, real_network_kernel_statistics

    # MC simulation (Gaussian pre-activations)
    result = mc_kernel_statistics(K0_input, n_layers=3, n_hidden=256, ...)
    K1_mc = n_hidden * (result['G_mean'] - K0_theory)

    # Real network simulation
    result = real_network_kernel_statistics(K0_input, n_layers=3, n_hidden=256, ...)
"""

from resnet_eft.validation.mc_simulation import (
    get_activation_fn,
    mc_kernel_estimate_batched,
    mc_kernel_statistics,
    mc_resnet_kernel_statistics,
)
from resnet_eft.validation.real_network import (
    real_network_kernel_statistics,
    real_network_resnet_statistics,
)

__all__ = [
    "mc_kernel_statistics",
    "mc_resnet_kernel_statistics",
    "mc_kernel_estimate_batched",
    "real_network_kernel_statistics",
    "real_network_resnet_statistics",
    "get_activation_fn",
]
