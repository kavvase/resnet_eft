"""Real network simulation for kernel statistics.

This module implements simulations using actual PyTorch networks
with random weights, which can be compared against the theoretical
predictions and Gaussian MC simulations.

The difference from mc_simulation.py is that this builds real
weight matrices and performs actual linear transformations,
rather than assuming Gaussian pre-activations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from resnet_eft.validation.mc_simulation import get_activation_fn

if TYPE_CHECKING:
    from resnet_eft.core_types import ActivationSpec


def real_network_kernel_statistics(
    K0_input: Tensor,
    n_layers: int,
    n_hidden: int,
    activation: str | ActivationSpec,
    Cw: float,
    Cb: float,
    n_samples: int,
    n_seeds: int = 1,
) -> dict[str, Tensor]:
    """Run real network simulation and compute kernel statistics.

    This creates actual neural network computations with random weights.
    To match the MC/Theory convention, we:
    1. Sample input φ₀ ~ N(0, K0_input)
    2. Apply n_layers of transformations
    3. Compute G = Cb + Cw * σ @ σ.T / n (post-activation kernel)

    Note: The layer count matches MC exactly:
    - n_layers=3 applies 3 activations (not 4!)
    - φ₀ → σ₁ → φ₁ → σ₂ → φ₂ → σ₃ → G

    Args:
        K0_input: Input kernel, shape (N, N)
        n_layers: Number of layers (each applies activation then linear)
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
        >>> result = real_network_kernel_statistics(K0, n_layers=3, n_hidden=256,
        ...                                         activation='tanh', Cw=1.0, Cb=0.1,
        ...                                         n_samples=10000, n_seeds=5)
        >>> K1_real = n_hidden * (result['G_mean'] - K0_theory)
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
            # Sample initial pre-activations from N(0, K0_input)
            L0 = torch.linalg.cholesky(K0_input + 1e-8 * torch.eye(N, dtype=dtype, device=device))
            phi = torch.randn(n_hidden, N, dtype=dtype, device=device) @ L0.T

            # Apply n_layers of transformation
            # Each layer: activation -> linear transform (except last layer)
            for layer_idx in range(n_layers):
                # Activation
                sigma = act_fn(phi)  # (n_hidden, N)

                # Linear transform to next layer (only for intermediate layers)
                if layer_idx < n_layers - 1:
                    # W ~ N(0, Cw/n_hidden), b ~ N(0, Cb)
                    W = torch.randn(n_hidden, n_hidden, dtype=dtype, device=device)
                    W = W * (Cw / n_hidden) ** 0.5
                    b = torch.randn(n_hidden, 1, dtype=dtype, device=device) * Cb**0.5
                    phi = W @ sigma + b  # (n_hidden, N)

            # sigma is now the n_layers-th activation (no extra activation!)
            # Compute kernel G = Cb + Cw * σ.T @ σ / n
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


def real_network_resnet_statistics(
    K0_input: Tensor,
    n_layers: int,
    n_hidden: int,
    activation: str | ActivationSpec,
    Cw: float,
    eps: float,
    n_samples: int,
    n_seeds: int = 1,
    batch_size: int = 100,
    use_gpu: bool = False,
) -> dict[str, Tensor]:
    """Run real network simulation for pre-activation ResNet.

    Pre-activation ResNet update rule:
        phi' = phi + eps * W @ sigma(phi)

    This uses actual weight matrices (not Gaussian assumption).

    Args:
        K0_input: Input kernel, shape (N, N)
        n_layers: Number of ResNet layers
        n_hidden: Width of hidden layers
        activation: Activation function name or ActivationSpec
        Cw: Weight variance scale
        eps: Residual coefficient (use eps << 1 for continuous limit)
        n_samples: Number of samples per seed
        n_seeds: Number of independent runs (for SE estimation)
        batch_size: Number of samples to process in parallel
        use_gpu: If True, use MPS (Apple Silicon) or CUDA for acceleration

    Returns:
        dict with keys:
            - G_mean: Mean of G across samples, shape (N, N)
            - G_var: Variance of G (element-wise), shape (N, N)
            - G_mean_se: Standard error of G_mean (if n_seeds > 1)
    """
    N = K0_input.shape[0]
    original_dtype = K0_input.dtype
    original_device = K0_input.device

    # Select device for computation
    if use_gpu:
        if torch.backends.mps.is_available():
            compute_device = torch.device("mps")
            # MPS works better with float32
            compute_dtype = torch.float32
        elif torch.cuda.is_available():
            compute_device = torch.device("cuda")
            compute_dtype = torch.float32
        else:
            compute_device = original_device
            compute_dtype = original_dtype
    else:
        compute_device = original_device
        compute_dtype = original_dtype

    # Move input to compute device
    K0_compute = K0_input.to(device=compute_device, dtype=compute_dtype)

    # Get activation function
    if isinstance(activation, str):
        act_fn = get_activation_fn(activation)
    else:
        act_fn = get_activation_fn(activation.name, getattr(activation, "smoothing_beta", None))

    # Pre-compute Cholesky decomposition (moved outside loop)
    L0 = torch.linalg.cholesky(K0_compute + 1e-6 * torch.eye(N, dtype=compute_dtype, device=compute_device))
    W_scale = (Cw / n_hidden) ** 0.5

    seed_means: list[Tensor] = []
    seed_vars: list[Tensor] = []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 10000)

        # Welford's algorithm for online mean/variance
        G_mean = torch.zeros(N, N, dtype=compute_dtype, device=compute_device)
        G_M2 = torch.zeros(N, N, dtype=compute_dtype, device=compute_device)
        count = 0

        # Process in batches
        for batch_start in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - batch_start)

            # Sample initial pre-activations: phi ~ N(0, K0_input)
            # Shape: (current_batch, n_hidden, N)
            z = torch.randn(current_batch, n_hidden, N, dtype=compute_dtype, device=compute_device)
            phi = z @ L0.T  # Broadcast over batch dimension

            # Apply n_layers of ResNet transformation
            for _layer in range(n_layers):
                # Apply activation
                sigma = act_fn(phi)  # (current_batch, n_hidden, N)

                # Sample weights for each sample in batch
                # W: (current_batch, n_hidden, n_hidden)
                W = torch.randn(current_batch, n_hidden, n_hidden, dtype=compute_dtype, device=compute_device)
                W = W * W_scale

                # ResNet update: phi' = phi + eps * W @ sigma
                # bmm: (B, n, n) @ (B, n, N) -> (B, n, N)
                phi = phi + eps * torch.bmm(W, sigma)

            # Compute kernel for each sample: G = (1/n) * phi.T @ phi
            # phi: (B, n_hidden, N) -> G: (B, N, N)
            G_batch = torch.bmm(phi.transpose(1, 2), phi) / n_hidden

            # Welford update for each sample in batch
            for i in range(current_batch):
                G = G_batch[i]
                count += 1
                delta = G - G_mean
                G_mean = G_mean + delta / count
                delta2 = G - G_mean
                G_M2 = G_M2 + delta * delta2

        G_var = G_M2 / (n_samples - 1) if n_samples > 1 else torch.zeros_like(G_M2)
        seed_means.append(G_mean)
        seed_vars.append(G_var)

    # Aggregate across seeds
    G_mean_avg = torch.stack(seed_means).mean(dim=0)
    G_var_avg = torch.stack(seed_vars).mean(dim=0)

    # Move results back to original device and dtype
    # Note: MPS doesn't support float64, so we move to CPU first then convert
    result: dict[str, Tensor] = {
        "G_mean": G_mean_avg.cpu().to(dtype=original_dtype).to(device=original_device),
        "G_var": G_var_avg.cpu().to(dtype=original_dtype).to(device=original_device),
    }

    if n_seeds > 1:
        G_mean_var = torch.zeros(N, N, dtype=compute_dtype, device=compute_device)
        for gm in seed_means:
            G_mean_var = G_mean_var + (gm - G_mean_avg) ** 2
        se = (G_mean_var / (n_seeds * (n_seeds - 1))) ** 0.5
        result["G_mean_se"] = se.cpu().to(dtype=original_dtype).to(device=original_device)

    return result
