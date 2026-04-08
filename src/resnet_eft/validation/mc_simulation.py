"""Monte Carlo simulation for kernel statistics.

This module implements MC simulation with Gaussian pre-activations,
which matches the assumptions of the NNGP perturbation theory.

The key assumption is that pre-activations are Gaussian:
    φ_ℓ ~ N(0, K_ℓ)
where K_ℓ is the kernel at layer ℓ.

This is the "annealed" or "Gaussian" approximation that becomes
exact in the infinite-width limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from resnet_eft.core_types import ActivationSpec, Params


def get_activation_fn(act_name: str, beta: float | None = None) -> Callable[[Tensor], Tensor]:
    """Get activation function by name.

    Args:
        act_name: Activation name ('relu', 'tanh', 'erf', 'softplus', 'gelu')
        beta: For softplus, the sharpness parameter

    Returns:
        Activation function that takes and returns Tensor
    """
    if act_name == "relu":
        return torch.relu
    elif act_name == "softplus":
        beta_val = beta if beta is not None else 10.0
        return lambda x: torch.nn.functional.softplus(x * beta_val) / beta_val
    elif act_name == "gelu":
        return torch.nn.functional.gelu
    elif act_name == "erf":
        return lambda x: x.erf()
    elif act_name == "tanh":
        return torch.tanh
    else:
        raise ValueError(f"Unknown activation: {act_name}")


def mc_kernel_statistics(
    K0_input: Tensor,
    n_layers: int,
    n_hidden: int,
    activation: str | ActivationSpec,
    Cw: float,
    Cb: float,
    n_samples: int,
    n_seeds: int = 1,
) -> dict[str, Tensor]:
    """Run MC simulation with Gaussian pre-activations and compute kernel statistics.

    This implements the "annealed" MC simulation where pre-activations
    are sampled from a Gaussian distribution with covariance equal to
    the current kernel. This matches the theory's assumptions.

    Args:
        K0_input: Input kernel, shape (N, N)
        n_layers: Number of layers (each applies activation then computes kernel)
        n_hidden: Width of hidden layers
        activation: Activation function name or ActivationSpec
        Cw: Weight variance scale
        Cb: Bias variance
        n_samples: Number of samples per seed
        n_seeds: Number of independent runs (for SE estimation)

    Returns:
        dict with keys:
            - G_mean: Mean of G across samples, shape (N, N)
            - G_var: Variance of G (element-wise), shape (N, N)
            - G_mean_se: Standard error of G_mean (if n_seeds > 1)

    Example:
        >>> K0 = torch.eye(2)
        >>> result = mc_kernel_statistics(K0, n_layers=3, n_hidden=256,
        ...                               activation='tanh', Cw=1.0, Cb=0.1,
        ...                               n_samples=10000, n_seeds=5)
        >>> K1_mc = n_hidden * (result['G_mean'] - K0_theory)
    """
    N = K0_input.shape[0]
    dtype = K0_input.dtype
    device = K0_input.device

    # Get activation function
    if isinstance(activation, str):
        act_fn = get_activation_fn(activation)
    else:
        # ActivationSpec
        act_fn = get_activation_fn(activation.name, getattr(activation, "smoothing_beta", None))

    seed_means: list[Tensor] = []
    seed_vars: list[Tensor] = []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 10000)

        # Welford's algorithm for online mean/variance
        G_mean = torch.zeros(N, N, dtype=dtype, device=device)
        G_M2 = torch.zeros(N, N, dtype=dtype, device=device)

        for i in range(n_samples):
            # Forward pass through network with Gaussian pre-activations
            G = K0_input.clone()
            for _layer in range(n_layers):
                # Sample pre-activations: φ ~ N(0, G)
                L = torch.linalg.cholesky(G + 1e-8 * torch.eye(N, dtype=dtype, device=device))
                phi = torch.randn(n_hidden, N, dtype=dtype, device=device) @ L.T

                # Apply activation
                sigma = act_fn(phi)

                # Compute kernel: G = Cb + Cw * σᵀσ / n
                G = Cb + Cw * (sigma.T @ sigma) / n_hidden

            # Welford update
            delta = G - G_mean
            G_mean = G_mean + delta / (i + 1)
            delta2 = G - G_mean
            G_M2 = G_M2 + delta * delta2

        G_var = G_M2 / (n_samples - 1) if n_samples > 1 else torch.zeros_like(G_M2)
        seed_means.append(G_mean)
        seed_vars.append(G_var)

    # Aggregate across seeds
    G_mean_avg = torch.stack(seed_means).mean(dim=0)
    G_var_avg = torch.stack(seed_vars).mean(dim=0)

    result: dict[str, Tensor] = {
        "G_mean": G_mean_avg,
        "G_var": G_var_avg,
    }

    if n_seeds > 1:
        # Standard error of the mean
        G_mean_var = torch.zeros(N, N, dtype=dtype, device=device)
        for gm in seed_means:
            G_mean_var = G_mean_var + (gm - G_mean_avg) ** 2
        result["G_mean_se"] = (G_mean_var / (n_seeds * (n_seeds - 1))) ** 0.5

    return result


def mc_kernel_estimate_batched(
    X: Tensor,
    n_hidden: int,
    n_layers: int,
    params: Params,
    n_samples: int,
    batch_size: int = 50,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Estimate kernel mean and variance using batched MC (memory-efficient).

    This is an alternative implementation that:
    1. Takes input data X (not kernel)
    2. Uses batched computation for memory efficiency
    3. Builds actual weight matrices (like a real network)

    Computes:
    - E[K] where K[a,b] = (1/n) Σ_i φ_i(a)φ_i(b)
    - Var(K) = E[K²] - E[K]² (element-wise variance)

    Uses Welford's online algorithm for numerical stability.

    Args:
        X: Input data, shape (N, d_in)
        n_hidden: Width of hidden layers
        n_layers: Number of layers (first is linear, rest are nonlinear)
        params: Network parameters (Cw, Cb, activation)
        n_samples: Total number of network samples
        batch_size: Samples per batch (controls memory)
        seed: Random seed

    Returns:
        K_mean: Mean kernel, shape (N, N)
        K_var: Variance of kernel elements, shape (N, N)
    """
    torch.manual_seed(seed)
    N, d_in = X.shape
    dtype = X.dtype
    device = X.device
    Cw, Cb = params.Cw, params.Cb

    # Get activation function
    act_spec = params.act if not isinstance(params.act, str) else None
    if act_spec is None:
        raise ValueError("params.act should be ActivationSpec")
    act_fn = get_activation_fn(act_spec.name, getattr(act_spec, "smoothing_beta", None))

    # Welford's online algorithm for mean and variance
    K_mean = torch.zeros(N, N, dtype=dtype, device=device)
    K_M2 = torch.zeros(N, N, dtype=dtype, device=device)
    count = 0

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        current_batch = min(batch_size, n_samples - batch_idx * batch_size)
        if current_batch <= 0:
            break

        # Forward pass for this batch
        h = X.unsqueeze(0).expand(current_batch, -1, -1)  # (batch, N, d_in)

        for layer in range(n_layers):
            fan_in = d_in if layer == 0 else n_hidden
            fan_out = n_hidden

            # Sample weights: W ~ N(0, Cw/fan_in)
            W = torch.randn(current_batch, fan_out, fan_in, dtype=dtype, device=device)
            W = W * (Cw / fan_in) ** 0.5
            b = None
            if Cb > 0:
                b = torch.randn(current_batch, fan_out, dtype=dtype, device=device) * Cb**0.5

            # Apply layer
            if layer > 0:
                h = act_fn(h)  # Activation on previous preactivation

            # Linear: h_new[s,a,i] = Σ_j W[s,i,j] * h[s,a,j] + b[s,i]
            h = torch.einsum("sij,saj->sai", W, h)
            if b is not None:
                h = h + b.unsqueeze(1)

        # Compute kernel for each sample: K[a,b] = (1/n) Σ_i h[a,i] * h[b,i]
        K_batch = torch.einsum("sai,sbi->sab", h, h) / n_hidden  # (batch, N, N)

        # Update statistics using Welford's algorithm
        for s in range(current_batch):
            count += 1
            K_sample = K_batch[s]
            delta = K_sample - K_mean
            K_mean = K_mean + delta / count
            delta2 = K_sample - K_mean
            K_M2 = K_M2 + delta * delta2

    # Compute unbiased variance (divide by count-1)
    K_var = K_M2 / (count - 1) if count > 1 else torch.zeros_like(K_mean)

    return K_mean, K_var


def mc_resnet_kernel_statistics(
    K0_input: Tensor,
    n_layers: int,
    n_hidden: int,
    activation: str | ActivationSpec,
    Cw: float,
    eps: float,
    n_samples: int,
    n_seeds: int = 1,
) -> dict[str, Tensor]:
    """Run MC simulation for pre-activation ResNet with Gaussian pre-activations.

    Pre-activation ResNet update rule:
        phi' = phi + eps * W * sigma(phi)

    Kernel update (incremental form):
        G' = G + eps^2 * Cw * sigma.T @ sigma / n

    Note: No Cb (bias variance) term in pre-activation ResNet.

    Args:
        K0_input: Input kernel, shape (N, N)
        n_layers: Number of ResNet layers
        n_hidden: Width of hidden layers
        activation: Activation function name or ActivationSpec
        Cw: Weight variance scale
        eps: Residual coefficient (use eps << 1 for continuous limit)
        n_samples: Number of samples per seed
        n_seeds: Number of independent runs (for SE estimation)

    Returns:
        dict with keys:
            - G_mean: Mean of G across samples, shape (N, N)
            - G_var: Variance of G (element-wise), shape (N, N)
            - G_mean_se: Standard error of G_mean (if n_seeds > 1)

    Example:
        >>> K0 = torch.eye(2)
        >>> result = mc_resnet_kernel_statistics(
        ...     K0, n_layers=10, n_hidden=256,
        ...     activation='tanh', Cw=1.0, eps=0.1,
        ...     n_samples=10000, n_seeds=5
        ... )
        >>> K1_mc = n_hidden * (result['G_mean'] - K0_theory)
    """
    N = K0_input.shape[0]
    dtype = K0_input.dtype
    device = K0_input.device

    # Get activation function
    if isinstance(activation, str):
        act_fn = get_activation_fn(activation)
    else:
        # ActivationSpec
        act_fn = get_activation_fn(activation.name, getattr(activation, "smoothing_beta", None))

    eps2 = eps**2  # Pre-compute for efficiency
    seed_means: list[Tensor] = []
    seed_vars: list[Tensor] = []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 10000)

        # Welford's algorithm for online mean/variance
        G_mean = torch.zeros(N, N, dtype=dtype, device=device)
        G_M2 = torch.zeros(N, N, dtype=dtype, device=device)

        for i in range(n_samples):
            # Forward pass through ResNet with Gaussian pre-activations
            G = K0_input.clone()
            for _layer in range(n_layers):
                # Sample pre-activations: phi ~ N(0, G)
                L = torch.linalg.cholesky(G + 1e-8 * torch.eye(N, dtype=dtype, device=device))
                phi = torch.randn(n_hidden, N, dtype=dtype, device=device) @ L.T

                # Apply activation
                sigma = act_fn(phi)

                # ResNet kernel update (incremental form):
                # G' = G + eps^2 * Cw * sigma.T @ sigma / n
                # Note: No Cb term in pre-activation ResNet
                G = G + eps2 * Cw * (sigma.T @ sigma) / n_hidden

            # Welford update
            delta = G - G_mean
            G_mean = G_mean + delta / (i + 1)
            delta2 = G - G_mean
            G_M2 = G_M2 + delta * delta2

        G_var = G_M2 / (n_samples - 1) if n_samples > 1 else torch.zeros_like(G_M2)
        seed_means.append(G_mean)
        seed_vars.append(G_var)

    # Aggregate across seeds
    G_mean_avg = torch.stack(seed_means).mean(dim=0)
    G_var_avg = torch.stack(seed_vars).mean(dim=0)

    result: dict[str, Tensor] = {
        "G_mean": G_mean_avg,
        "G_var": G_var_avg,
    }

    if n_seeds > 1:
        # Standard error of the mean
        G_mean_var = torch.zeros(N, N, dtype=dtype, device=device)
        for gm in seed_means:
            G_mean_var = G_mean_var + (gm - G_mean_avg) ** 2
        result["G_mean_se"] = (G_mean_var / (n_seeds * (n_seeds - 1))) ** 0.5

    return result
