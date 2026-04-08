"""Core data types for finite-width correction calculations.

This module defines:
- ActivationSpec: Activation function specification
- Params: Computation parameters
- Cache: Layer-wise caching
- KernelState: State carried through layers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from torch import Tensor

if TYPE_CHECKING:
    from resnet_eft.v4_repr import V4Repr


@dataclass
class ActivationSpec:
    """Activation function specification.

    Attributes:
        name: Name of the activation ('relu', 'erf', 'tanh', 'softplus', 'gelu')
        input_scale: Scale factor for input (σ(u) = f(u / input_scale))
        critical_Cw: Critical C_W value for edge-of-chaos
        smoothing_beta: For softplus, β parameter (higher = closer to ReLU)
    """

    name: str
    input_scale: float = 1.0
    critical_Cw: float | None = None
    smoothing_beta: float | None = None

    @classmethod
    def relu(
        cls,
        mode: Literal["exact", "smooth"] = "smooth",
        beta: float = 10.0,
    ) -> ActivationSpec:
        """Create ReLU activation spec.

        Args:
            mode: "smooth" uses softplus(βx)/β approximation (recommended),
                  "exact" uses true ReLU (non-smooth, K1 has limitations)
            beta: Smoothing parameter for mode="smooth" (higher = closer to ReLU)
        """
        if mode == "smooth":
            return cls(name="softplus", smoothing_beta=beta, critical_Cw=2.0)
        return cls(name="relu", input_scale=1.0, critical_Cw=2.0)

    @classmethod
    def softplus(cls, beta: float = 10.0) -> ActivationSpec:
        """Create softplus activation spec (smooth ReLU approximation).

        σ(x) = softplus(βx)/β = log(1 + exp(βx))/β

        Args:
            beta: Sharpness parameter (higher = closer to ReLU)
        """
        return cls(name="softplus", smoothing_beta=beta, critical_Cw=2.0)

    @classmethod
    def gelu(cls) -> ActivationSpec:
        """Create GELU activation spec (another smooth ReLU-like function)."""
        return cls(name="gelu", input_scale=1.0, critical_Cw=2.0)

    @classmethod
    def erf(cls, scale: float = 1.0) -> ActivationSpec:
        """Create erf activation spec.

        Args:
            scale: Input scale factor
                   scale=1: standard erf
                   scale=√2: close to Gaussian CDF normalization
        """
        return cls(name="erf", input_scale=scale, critical_Cw=1.0)

    @classmethod
    def tanh(cls) -> ActivationSpec:
        """Create tanh activation spec."""
        return cls(name="tanh", input_scale=1.0, critical_Cw=1.0)


@dataclass
class Params:
    """Computation parameters.

    Attributes:
        act: Activation function specification
        Cw: Weight variance scale (C_W = n × Var(W))
        Cb: Bias variance (C_b = Var(b))
        eps_rho: Epsilon for correlation coefficient clipping
        gh_order: Gauss-Hermite quadrature order
        psd_check: PSD checking strategy
        nan_mode: NaN handling strategy
        k1_mode: K1 computation optimization mode
            - "auto": Auto-detect uniform K0 and use optimized path
            - "uniform": Force uniform K0 optimization (requires αI + β1 structure)
            - "general": Force general O(N²) path (no optimization)
        e4_mode: E4 computation mode
            - "gh": Gauss-Hermite quadrature (accurate, slow for large N)
            - "mc": Monte Carlo sampling (fast for large N, approximate)
            - "auto": Use GH for N≤10, MC for N>10
        e4_mc_samples: Number of MC samples when e4_mode is "mc" or "auto"
        v4_mode: V4 storage mode
            - "full": Store full N^4 tensor (accurate, memory-intensive)
            - "slice": Store only K1-relevant slices (memory-efficient)
            - "auto": Use full for N≤50, slice for N>50
    """

    act: ActivationSpec | str
    Cw: float
    Cb: float = 0.0
    eps_rho: float = 1e-6
    gh_order: int = 32
    psd_check: Literal["none", "cheap", "eigh"] = "cheap"
    nan_mode: Literal["raise", "warn", "replace_zero"] = "raise"
    k1_mode: Literal["auto", "uniform", "general"] = "auto"
    e4_mode: Literal["gh", "mc", "auto"] = "auto"
    e4_mc_samples: int = 10000
    v4_mode: Literal["full", "slice", "auto"] = "auto"

    def __post_init__(self) -> None:
        """Convert string activation to ActivationSpec."""
        if isinstance(self.act, str):
            if self.act == "relu":
                self.act = ActivationSpec.relu()
            elif self.act == "erf":
                self.act = ActivationSpec.erf()
            elif self.act == "tanh":
                self.act = ActivationSpec.tanh()
            elif self.act == "softplus":
                self.act = ActivationSpec.softplus()
            elif self.act == "gelu":
                self.act = ActivationSpec.gelu()
            else:
                raise ValueError(f"Unknown activation: {self.act}")

    @property
    def act_name(self) -> str:
        """Get activation name for string comparison."""
        if isinstance(self.act, ActivationSpec):
            return self.act.name
        return self.act

    @property
    def act_input_scale(self) -> float:
        """Get activation input scale."""
        if isinstance(self.act, ActivationSpec):
            return self.act.input_scale
        return 1.0

    @property
    def act_smoothing_beta(self) -> float | None:
        """Get smoothing beta for softplus activation."""
        if isinstance(self.act, ActivationSpec):
            return self.act.smoothing_beta
        return None


def _params_hash(params: Params) -> int:
    """Compute hash of Params for cache key."""
    return hash(
        (
            params.Cw,
            params.Cb,
            params.act_name,
            params.act_input_scale,
            params.act_smoothing_beta,
        )
    )


@dataclass
class CacheEntry:
    """Cache entry with value and dependency signature.

    Attributes:
        value: Cached value
        deps_signature: Tuple identifying dependencies
    """

    value: Any
    deps_signature: tuple[Any, ...]


CacheKey = tuple[int, str, int]  # (depth, name, params_hash)


class Cache:
    """Layer-wise, key-wise cache for intermediate values."""

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._store: dict[CacheKey, CacheEntry] = {}

    def get(
        self, depth: int, name: str, params: Params, deps_signature: tuple[Any, ...]
    ) -> Any | None:
        """Get cached value if dependencies match.

        Args:
            depth: Layer depth
            name: Cache key name (e.g., 'E2', 'chi')
            params: Computation parameters
            deps_signature: Dependency signature to check

        Returns:
            Cached value if found and dependencies match, None otherwise
        """
        key: CacheKey = (depth, name, _params_hash(params))
        entry = self._store.get(key)
        if entry and entry.deps_signature == deps_signature:
            return entry.value
        return None

    def set(
        self,
        depth: int,
        name: str,
        params: Params,
        value: Any,
        deps_signature: tuple[Any, ...],
    ) -> None:
        """Set cached value with dependency signature.

        Args:
            depth: Layer depth
            name: Cache key name
            params: Computation parameters
            value: Value to cache
            deps_signature: Dependency signature
        """
        key: CacheKey = (depth, name, _params_hash(params))
        self._store[key] = CacheEntry(value, deps_signature)

    def invalidate(self, depth: int | None = None, name: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            depth: If specified, only invalidate entries at this depth
            name: If specified, only invalidate entries with this name
        """
        if depth is None and name is None:
            self._store.clear()
        else:
            keys_to_remove = [
                k
                for k in self._store
                if (depth is None or k[0] == depth) and (name is None or k[1] == name)
            ]
            for k in keys_to_remove:
                del self._store[k]


@dataclass
class KernelState:
    """State carried through layers during computation.

    Attributes:
        N: Number of input points
        depth: Layer depth (0, 1, 2, ...)
        label: Label for display/logging ("L1", "res_3+7", etc.)
        fan_in: n_{ℓ-1}, dimension for 1/n denominator (None for input layer)
        fan_out: n_ℓ, output dimension of this layer
        params: Computation parameters (None for input layer)
        K0: Infinite-width kernel, shape (N, N)
        K1: NLO 2-point correction coefficient (O(1)), shape (N, N) or None
        V4: 4-point connected correlation coefficient (O(1)) or None
        cache: Cache for intermediate values
        K0_version: Version counter for K0 changes
        K1_version: Version counter for K1 changes
        V4_version: Version counter for V4 changes
        meta: Arbitrary metadata

    Normalization convention:
        K1, V4 are stored as O(1) coefficients.
        Physical quantities: K1_phys = K1 / fan_in, V4_phys = V4 / fan_in
    """

    N: int
    depth: int
    label: str
    fan_in: int | None
    fan_out: int
    params: Params | None
    K0: Tensor
    K1: Tensor | None
    V4: V4Repr | None
    cache: Cache
    K0_version: int = 0
    K1_version: int = 0
    V4_version: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(
        cls,
        K0_input: Tensor,
        fan_out: int,
        label: str = "L0",
    ) -> KernelState:
        """Create initial state from input kernel.

        Args:
            K0_input: Input kernel matrix, shape (N, N)
            fan_out: Number of neurons in the first hidden layer
            label: Label for this state

        Returns:
            Initial KernelState for the input layer
        """
        N = K0_input.shape[0]
        return cls(
            N=N,
            depth=0,
            label=label,
            fan_in=None,  # Input layer has no fan_in
            fan_out=fan_out,
            params=None,  # Input layer has no params
            K0=K0_input,
            K1=None,  # No finite-width correction for input
            V4=None,  # No 4-point correlation for input
            cache=Cache(),
            K0_version=0,
            K1_version=0,
            V4_version=0,
            meta={},
        )

    def get_physical_K1(self) -> Tensor | None:
        """Get K1 as physical quantity (divided by fan_in).

        Returns:
            K1 / fan_in if K1 is not None and fan_in is set, else None
        """
        if self.K1 is None or self.fan_in is None:
            return None
        return self.K1 / self.fan_in

    def get_physical_V4(self) -> V4Repr | None:
        """Get V4 as physical quantity (divided by fan_in).

        Returns:
            V4.scale(1/fan_in) if V4 is not None and fan_in is set, else None
        """
        if self.V4 is None or self.fan_in is None:
            return None
        return self.V4.scale(1.0 / self.fan_in)
