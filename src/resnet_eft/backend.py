"""Backend utilities for tensor operations.

This module provides a thin abstraction layer over PyTorch tensor operations.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def diag_embed(v: Tensor) -> Tensor:
    """Create diagonal matrix from vector.

    Args:
        v: Vector of shape (N,)

    Returns:
        Diagonal matrix of shape (N, N)
    """
    return torch.diag_embed(v)


def diagonal(A: Tensor) -> Tensor:
    """Extract diagonal from matrix.

    Args:
        A: Matrix of shape (N, N)

    Returns:
        Diagonal elements of shape (N,)
    """
    return torch.diagonal(A)


def symmetrize(K: Tensor) -> Tensor:
    """Symmetrize a matrix.

    Args:
        K: Matrix of shape (N, N)

    Returns:
        Symmetrized matrix (K + K.T) / 2
    """
    return (K + K.T) / 2


def ensure_psd(
    K: Tensor,
    psd_check: Literal["none", "cheap", "eigh"] = "cheap",
    eps: float = 1e-6,
) -> Tensor:
    """Ensure matrix is positive semi-definite.

    Args:
        K: Matrix to check
        psd_check: Checking strategy
            - "none": No checking
            - "cheap": Check diagonal positivity only
            - "eigh": Full eigenvalue check
        eps: Tolerance for eigenvalue check

    Returns:
        The input matrix (raises if not PSD)

    Raises:
        ValueError: If matrix is not PSD
    """
    if psd_check == "none":
        return K
    elif psd_check == "cheap":
        diag = diagonal(K)
        if (diag < 0).any():
            raise ValueError("Matrix has negative diagonal elements")
        return K
    else:  # 'eigh'
        eigvals = torch.linalg.eigvalsh(K)
        if eigvals.min() < -eps:
            raise ValueError(f"Matrix is not PSD: min eigenvalue = {eigvals.min().item()}")
        return K


def cholesky_safe(K: Tensor, jitter: float = 1e-6) -> Tensor:
    """Cholesky decomposition with jitter for numerical stability.

    Args:
        K: Positive definite matrix
        jitter: Small value added to diagonal for stability

    Returns:
        Lower triangular Cholesky factor L such that K = L @ L.T
    """
    N = K.shape[0]
    K_jittered = K + jitter * torch.eye(N, dtype=K.dtype, device=K.device)
    result: Tensor = torch.linalg.cholesky(K_jittered)
    return result


def allclose(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two tensors are element-wise equal within tolerance.

    Args:
        a: First tensor
        b: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if all elements are close
    """
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def zeros(shape: tuple[int, ...], dtype: torch.dtype | None = None) -> Tensor:
    """Create zero tensor.

    Args:
        shape: Shape of the tensor
        dtype: Data type (default: torch.float64)

    Returns:
        Zero tensor of specified shape
    """
    if dtype is None:
        dtype = torch.float64
    return torch.zeros(shape, dtype=dtype)


def zeros_like(x: Tensor) -> Tensor:
    """Create zero tensor with same shape and dtype.

    Args:
        x: Reference tensor

    Returns:
        Zero tensor with same properties
    """
    return torch.zeros_like(x)


def einsum(subscripts: str, *operands: Tensor) -> Tensor:
    """Einstein summation.

    Args:
        subscripts: Subscript string
        *operands: Input tensors

    Returns:
        Result of einsum operation
    """
    return torch.einsum(subscripts, *operands)


def eye(N: int, dtype: torch.dtype | None = None) -> Tensor:
    """Create identity matrix.

    Args:
        N: Size of identity matrix
        dtype: Data type (default: torch.float64)

    Returns:
        Identity matrix of shape (N, N)
    """
    if dtype is None:
        dtype = torch.float64
    return torch.eye(N, dtype=dtype)


def sqrt(x: Tensor) -> Tensor:
    """Element-wise square root."""
    return torch.sqrt(x)


def arccos(x: Tensor) -> Tensor:
    """Element-wise arccos."""
    return torch.acos(x)


def arcsin(x: Tensor) -> Tensor:
    """Element-wise arcsin."""
    return torch.asin(x)


def sin(x: Tensor) -> Tensor:
    """Element-wise sin."""
    return torch.sin(x)


def cos(x: Tensor) -> Tensor:
    """Element-wise cos."""
    return torch.cos(x)


def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """Clip tensor values to range.

    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped tensor
    """
    return torch.clamp(x, min=min_val, max=max_val)


def relu(x: Tensor) -> Tensor:
    """ReLU activation."""
    return torch.relu(x)


def erf(x: Tensor) -> Tensor:
    """Error function."""
    return torch.erf(x)


PI = torch.pi
