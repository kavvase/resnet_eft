"""Tests for the validation module.

This module contains:
1. API Tests: Unit tests for validation module functions
2. Smoke Tests: Theory vs MC/Real Network validation

For rigorous validation, see experiments/mc_validation.ipynb and experiments/k1_validation.ipynb.
"""

import pytest
import torch

from resnet_eft import KernelState, Params, step
from resnet_eft.backend import symmetrize
from resnet_eft.core_types import ActivationSpec
from resnet_eft.validation import (
    get_activation_fn,
    mc_kernel_estimate_batched,
    mc_kernel_statistics,
    mc_resnet_kernel_statistics,
    real_network_kernel_statistics,
    real_network_resnet_statistics,
)


def compute_theory_k0(
    K0_input: torch.Tensor,
    n_layers: int,
    n_hidden: int,
    activation: str,
    Cw: float,
    Cb: float,
) -> torch.Tensor:
    """Compute theoretical K0 for comparison."""
    if activation == "relu":
        act_spec = ActivationSpec.relu(mode="exact")
    elif activation == "tanh":
        act_spec = ActivationSpec.tanh()
    elif activation == "erf":
        act_spec = ActivationSpec.erf()
    else:
        raise ValueError(f"Unknown activation: {activation}")

    params = Params(act=act_spec, Cw=Cw, Cb=Cb)
    state = KernelState.from_input(K0_input, fan_out=n_hidden)
    for _ in range(n_layers):
        state = step(state, params, fan_out=n_hidden, compute_K1=False, compute_V4=False)
    return state.K0


# =============================================================================
# Part 1: API Tests (Unit tests for validation module functions)
# =============================================================================


class TestGetActivationFn:
    """Unit tests for get_activation_fn."""

    def test_relu(self):
        """Test ReLU activation."""
        fn = get_activation_fn("relu")
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = fn(x)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected)

    def test_tanh(self):
        """Test tanh activation."""
        fn = get_activation_fn("tanh")
        x = torch.tensor([0.0, 1.0])
        expected = torch.tanh(x)
        assert torch.allclose(fn(x), expected)

    def test_erf(self):
        """Test erf activation."""
        fn = get_activation_fn("erf")
        x = torch.tensor([0.0, 1.0])
        expected = torch.erf(x)
        assert torch.allclose(fn(x), expected)

    def test_softplus(self):
        """Test softplus activation."""
        fn = get_activation_fn("softplus", beta=5.0)
        x = torch.tensor([0.0, 1.0])
        expected = torch.nn.functional.softplus(x * 5.0) / 5.0
        assert torch.allclose(fn(x), expected)

    def test_gelu(self):
        """Test GELU activation."""
        fn = get_activation_fn("gelu")
        x = torch.tensor([0.0, 1.0])
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(fn(x), expected)

    def test_unknown_activation_raises(self):
        """Test unknown activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation_fn("unknown")


class TestMCKernelStatisticsAPI:
    """Unit tests for mc_kernel_statistics API."""

    def test_output_shape(self):
        """Test output shape is correct."""
        N = 3
        K0 = torch.eye(N, dtype=torch.float64)
        result = mc_kernel_statistics(
            K0, n_layers=2, n_hidden=50, activation="tanh", Cw=1.0, Cb=0.1, n_samples=100
        )
        assert result["G_mean"].shape == (N, N)
        assert result["G_var"].shape == (N, N)

    def test_returns_se_with_multiple_seeds(self):
        """Test SE is returned when n_seeds > 1."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        result = mc_kernel_statistics(
            K0, n_layers=1, n_hidden=50, activation="tanh", Cw=1.0, Cb=0.1, n_samples=50, n_seeds=3
        )
        assert "G_mean_se" in result
        assert result["G_mean_se"].shape == (N, N)
        assert (result["G_mean_se"] >= 0).all()

    def test_no_se_with_single_seed(self):
        """Test SE is not returned when n_seeds == 1."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        result = mc_kernel_statistics(
            K0, n_layers=1, n_hidden=50, activation="tanh", Cw=1.0, Cb=0.1, n_samples=50, n_seeds=1
        )
        assert "G_mean_se" not in result


class TestRealNetworkKernelStatisticsAPI:
    """Unit tests for real_network_kernel_statistics API."""

    def test_output_shape(self):
        """Test output shape is correct."""
        N = 3
        K0 = torch.eye(N, dtype=torch.float64)
        result = real_network_kernel_statistics(
            K0, n_layers=2, n_hidden=50, activation="tanh", Cw=1.0, Cb=0.1, n_samples=100
        )
        assert result["G_mean"].shape == (N, N)
        assert result["G_var"].shape == (N, N)

    def test_returns_se_with_multiple_seeds(self):
        """Test SE is returned when n_seeds > 1."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        result = real_network_kernel_statistics(
            K0, n_layers=1, n_hidden=50, activation="tanh", Cw=1.0, Cb=0.1, n_samples=50, n_seeds=3
        )
        assert "G_mean_se" in result
        assert result["G_mean_se"].shape == (N, N)


class TestMCKernelEstimateBatchedAPI:
    """Unit tests for mc_kernel_estimate_batched API."""

    def test_output_shape(self):
        """Test output shape is correct."""
        N = 3
        d_in = 4
        X = torch.randn(N, d_in, dtype=torch.float32)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.1)

        K_mean, K_var = mc_kernel_estimate_batched(
            X, n_hidden=50, n_layers=2, params=params, n_samples=100, batch_size=20
        )

        assert K_mean.shape == (N, N)
        assert K_var.shape == (N, N)

    def test_symmetry(self):
        """Test output is symmetric."""
        N = 3
        d_in = 4
        X = torch.randn(N, d_in, dtype=torch.float32)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1)

        K_mean, _ = mc_kernel_estimate_batched(
            X, n_hidden=100, n_layers=2, params=params, n_samples=200, batch_size=50
        )

        assert torch.allclose(K_mean, K_mean.T, atol=1e-4)


# =============================================================================
# Part 2: Smoke Tests (Theory vs MC/Real Network Validation)
# =============================================================================


class TestTheoryVsSimulationK0:
    """Smoke tests: Theory K0 should match MC and Real Network K0."""

    @pytest.mark.parametrize("activation,Cw,Cb", [("tanh", 1.0, 0.1), ("relu", 2.0, 0.0)])
    def test_k0_theory_vs_mc(self, activation: str, Cw: float, Cb: float):
        """Theory K0 should match MC K0 within sampling error."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_layers = 2
        n_samples = 2000

        K0_th = compute_theory_k0(K0_input, n_layers, n_hidden, activation, Cw, Cb)
        result = mc_kernel_statistics(
            K0_input, n_layers, n_hidden, activation, Cw, Cb, n_samples
        )
        K0_mc = result["G_mean"]

        rel_err = (K0_th - K0_mc).abs() / K0_th.abs().clamp(min=0.1)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 0.05, f"K0 rel error {max_rel_err:.3f} > 5% for {activation}"

    @pytest.mark.parametrize("activation,Cw,Cb", [("tanh", 1.0, 0.1), ("relu", 2.0, 0.0)])
    def test_k0_theory_vs_real_network(self, activation: str, Cw: float, Cb: float):
        """Theory K0 should match Real Network K0 within sampling error."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_layers = 2
        n_samples = 2000

        K0_th = compute_theory_k0(K0_input, n_layers, n_hidden, activation, Cw, Cb)
        result = real_network_kernel_statistics(
            K0_input, n_layers, n_hidden, activation, Cw, Cb, n_samples
        )
        K0_real = result["G_mean"]

        rel_err = (K0_th - K0_real).abs() / K0_th.abs().clamp(min=0.1)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 0.05, f"K0 rel error {max_rel_err:.3f} > 5% for {activation}"

    def test_mc_vs_real_network_consistency(self):
        """MC and Real Network should give consistent results."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_hidden = 128
        n_layers = 2
        n_samples = 2000

        mc_result = mc_kernel_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, Cb=0.1, n_samples=n_samples
        )
        real_result = real_network_kernel_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, Cb=0.1, n_samples=n_samples
        )

        rel_err = (mc_result["G_mean"] - real_result["G_mean"]).abs()
        rel_err = rel_err / mc_result["G_mean"].abs().clamp(min=0.1)
        assert rel_err.max() < 0.05


class TestWidthScaling:
    """Smoke tests: Finite-width corrections should scale as 1/n."""

    def test_mc_correction_scales_with_width(self):
        """||G_mc - K0_th|| should decrease with width."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_layers = 2
        n_samples = 1000
        widths = [64, 256]

        corrections = []
        for n_hidden in widths:
            K0_th = compute_theory_k0(K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1)
            result = mc_kernel_statistics(
                K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1, n_samples
            )
            correction = (result["G_mean"] - K0_th).abs().mean().item()
            corrections.append(correction)

        ratio = corrections[0] / (corrections[1] + 1e-10)
        assert ratio > 1.5, f"Correction should decrease with width, ratio={ratio:.2f}"

    def test_real_network_correction_scales_with_width(self):
        """||G_real - K0_th|| should decrease with width."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_layers = 2
        n_samples = 2000
        widths = [50, 400]  # 8x ratio for clear signal

        corrections = []
        for n_hidden in widths:
            K0_th = compute_theory_k0(K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1)
            result = real_network_kernel_statistics(
                K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1, n_samples
            )
            correction = (result["G_mean"] - K0_th).abs().mean().item()
            corrections.append(correction)

        ratio = corrections[0] / (corrections[1] + 1e-10)
        assert ratio > 1.2, f"Correction should decrease with width, ratio={ratio:.2f}"


class TestVarianceScaling:
    """Smoke tests: Kernel variance (V4) should scale as 1/n."""

    def test_mc_variance_scales_with_width(self):
        """Var(G)_mc should scale as 1/n."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_layers = 2
        n_samples = 800
        widths = [64, 256]

        variances = []
        for n_hidden in widths:
            result = mc_kernel_statistics(
                K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1, n_samples
            )
            mean_var = result["G_var"].mean().item()
            variances.append(mean_var)

        ratio = variances[0] / (variances[1] + 1e-15)
        assert ratio > 2.0, f"Variance should scale as 1/n, ratio={ratio:.2f}"

    def test_real_network_variance_scales_with_width(self):
        """Var(G)_real should scale as 1/n."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_layers = 2
        n_samples = 800
        widths = [64, 256]

        variances = []
        for n_hidden in widths:
            result = real_network_kernel_statistics(
                K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1, n_samples
            )
            mean_var = result["G_var"].mean().item()
            variances.append(mean_var)

        ratio = variances[0] / (variances[1] + 1e-15)
        assert ratio > 2.0, f"Variance should scale as 1/n, ratio={ratio:.2f}"


class TestOutputSymmetry:
    """Smoke tests: Kernel outputs should be symmetric."""

    def test_mc_output_symmetric(self):
        """MC kernel should be symmetric."""
        N = 3
        K0_input = symmetrize(torch.randn(N, N, dtype=torch.float64) * 0.3 + torch.eye(N))
        result = mc_kernel_statistics(
            K0_input, n_layers=2, n_hidden=100, activation="tanh", Cw=1.0, Cb=0.1, n_samples=500
        )
        G = result["G_mean"]
        assert torch.allclose(G, G.T, atol=1e-6), "MC kernel should be symmetric"

    def test_real_network_output_symmetric(self):
        """Real network kernel should be symmetric."""
        N = 3
        K0_input = symmetrize(torch.randn(N, N, dtype=torch.float64) * 0.3 + torch.eye(N))
        result = real_network_kernel_statistics(
            K0_input, n_layers=2, n_hidden=100, activation="tanh", Cw=1.0, Cb=0.1, n_samples=500
        )
        G = result["G_mean"]
        assert torch.allclose(G, G.T, atol=1e-6), "Real network kernel should be symmetric"


class TestIntegration:
    """Integration and sanity tests."""

    def test_theory_predictions_finite(self):
        """Sanity check: theory predictions should be finite."""
        N = 3
        n_hidden = 50
        n_layers = 4

        torch.manual_seed(192021)
        K0_input = symmetrize(torch.randn(N, N, dtype=torch.float64) * 0.2 + torch.eye(N))

        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1)
        state = KernelState.from_input(K0_input, fan_out=n_hidden)

        for _ in range(n_layers):
            state = step(state, params, fan_out=n_hidden, compute_K1=True, compute_V4=True)
            assert torch.isfinite(state.K0).all(), "K0 should be finite"
            if state.K1 is not None:
                assert torch.isfinite(state.K1).all(), "K1 should be finite"

    def test_deeper_network(self):
        """Test with deeper network (5 layers)."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_hidden = 100
        n_layers = 5

        K0_th = compute_theory_k0(K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1)

        mc_result = mc_kernel_statistics(
            K0_input, n_layers, n_hidden, "tanh", 1.0, 0.1, n_samples=1000
        )

        rel_err = (K0_th - mc_result["G_mean"]).abs() / K0_th.abs().clamp(min=0.1)
        assert rel_err.max() < 0.1, "K0 should match for deeper networks"


# =============================================================================
# Part 3: ResNet MC Simulation Tests
# =============================================================================


class TestMCResnetKernelStatisticsAPI:
    """Unit tests for mc_resnet_kernel_statistics API."""

    def test_output_shape(self):
        """Test output shape is correct."""
        N = 3
        K0 = torch.eye(N, dtype=torch.float64)
        result = mc_resnet_kernel_statistics(
            K0, n_layers=2, n_hidden=50, activation="tanh", Cw=1.0, eps=1.0, n_samples=100
        )
        assert result["G_mean"].shape == (N, N)
        assert result["G_var"].shape == (N, N)

    def test_returns_se_with_multiple_seeds(self):
        """Test SE is returned when n_seeds > 1."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        result = mc_resnet_kernel_statistics(
            K0, n_layers=1, n_hidden=50, activation="tanh", Cw=1.0, eps=1.0, n_samples=50, n_seeds=3
        )
        assert "G_mean_se" in result
        assert result["G_mean_se"].shape == (N, N)
        assert (result["G_mean_se"] >= 0).all()

    def test_incremental_update(self):
        """Test that G increases incrementally (not replacement)."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        n_layers = 3
        n_samples = 500

        # With eps=1, each layer adds Cw * E[sigma.T @ sigma] / n
        result = mc_resnet_kernel_statistics(
            K0, n_layers=n_layers, n_hidden=100, activation="tanh", Cw=1.0, eps=1.0, n_samples=n_samples
        )

        # G_mean should be larger than K0 (incremental addition)
        assert (result["G_mean"].diag() > K0.diag()).all(), "ResNet should add to kernel"


class TestResnetTheoryVsMC:
    """Smoke tests: ResNet theory vs MC simulation."""

    def test_resnet_k0_theory_vs_mc(self):
        """ResNet theory K0 should match MC K0."""
        from resnet_eft import resnet_step

        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_layers = 3
        n_samples = 3000
        eps = 1.0

        # Theory
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)
        state = KernelState.from_input(K0_input, fan_out=n_hidden)
        for _ in range(n_layers):
            state = resnet_step(state, params, eps=eps, compute_K1=False, compute_V4=False)
        K0_th = state.K0

        # MC
        result = mc_resnet_kernel_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
        )
        K0_mc = result["G_mean"]

        rel_err = (K0_th - K0_mc).abs() / K0_th.abs().clamp(min=0.1)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 0.05, f"ResNet K0 rel error {max_rel_err:.3f} > 5%"

    def test_resnet_eps_scaling_vs_mc(self):
        """Test that smaller eps gives ODE-like behavior in MC."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_samples = 2000

        # Compare: eps=1.0 with 4 steps vs eps=0.5 with 16 steps (same total "time" = 1)
        # Both should give similar final K0 as eps -> 0
        result_coarse = mc_resnet_kernel_statistics(
            K0_input, n_layers=4, n_hidden=n_hidden, activation="tanh", Cw=1.0, eps=0.5, n_samples=n_samples
        )
        result_fine = mc_resnet_kernel_statistics(
            K0_input, n_layers=16, n_hidden=n_hidden, activation="tanh", Cw=1.0, eps=0.25, n_samples=n_samples
        )

        # Fine and coarse should be relatively close
        diff = (result_coarse["G_mean"] - result_fine["G_mean"]).abs().max().item()
        assert diff < 0.1, f"Coarse vs fine difference {diff:.3f} > 0.1"


# =============================================================================
# Part 4: ResNet Real Network Simulation Tests
# =============================================================================


class TestRealNetworkResnetStatisticsAPI:
    """Unit tests for real_network_resnet_statistics API."""

    def test_output_shape(self):
        """Test output shape is correct."""
        N = 3
        K0 = torch.eye(N, dtype=torch.float64)
        result = real_network_resnet_statistics(
            K0, n_layers=2, n_hidden=50, activation="tanh", Cw=1.0, eps=1.0, n_samples=100
        )
        assert result["G_mean"].shape == (N, N)
        assert result["G_var"].shape == (N, N)

    def test_returns_se_with_multiple_seeds(self):
        """Test SE is returned when n_seeds > 1."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        result = real_network_resnet_statistics(
            K0, n_layers=1, n_hidden=50, activation="tanh", Cw=1.0, eps=1.0, n_samples=50, n_seeds=3
        )
        assert "G_mean_se" in result
        assert result["G_mean_se"].shape == (N, N)
        assert (result["G_mean_se"] >= 0).all()

    def test_incremental_update(self):
        """Test that G increases incrementally (not replacement)."""
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        n_layers = 3
        n_samples = 500

        result = real_network_resnet_statistics(
            K0, n_layers=n_layers, n_hidden=100, activation="tanh", Cw=1.0, eps=1.0, n_samples=n_samples
        )

        # G_mean should be larger than K0 (incremental addition)
        assert (result["G_mean"].diag() > K0.diag()).all(), "ResNet should add to kernel"

    def test_output_symmetric(self):
        """Test output is symmetric."""
        N = 3
        K0 = torch.eye(N, dtype=torch.float64)
        result = real_network_resnet_statistics(
            K0, n_layers=2, n_hidden=100, activation="tanh", Cw=1.0, eps=1.0, n_samples=500
        )
        G = result["G_mean"]
        assert torch.allclose(G, G.T, atol=1e-5), "Real network kernel should be symmetric"


class TestResnetTheoryVsRealNetwork:
    """Tests: ResNet theory vs real network simulation."""

    def test_resnet_k0_theory_vs_real_network(self):
        """ResNet theory K0 should match real network K0."""
        from resnet_eft import resnet_step

        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_layers = 3
        n_samples = 3000
        eps = 1.0

        # Theory
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)
        state = KernelState.from_input(K0_input, fan_out=n_hidden)
        for _ in range(n_layers):
            state = resnet_step(state, params, eps=eps, compute_K1=False, compute_V4=False)
        K0_th = state.K0

        # Real network
        result = real_network_resnet_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
        )
        K0_real = result["G_mean"]

        rel_err = (K0_th - K0_real).abs() / K0_th.abs().clamp(min=0.1)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 0.05, f"ResNet K0 rel error {max_rel_err:.3f} > 5%"

    def test_resnet_mc_vs_real_network_consistency(self):
        """MC and real network should give consistent ResNet results."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_layers = 2
        n_samples = 3000
        eps = 1.0

        mc_result = mc_resnet_kernel_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
        )
        real_result = real_network_resnet_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
        )

        rel_err = (mc_result["G_mean"] - real_result["G_mean"]).abs()
        rel_err = rel_err / mc_result["G_mean"].abs().clamp(min=0.1)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 0.05, f"MC vs real network difference {max_rel_err:.3f} > 5%"


class TestResnetVarianceScaling:
    """Tests: ResNet variance (V4) should scale as 1/n."""

    def test_resnet_mc_variance_scales_with_width(self):
        """Var(G)_mc for ResNet should scale as 1/n."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_layers = 2
        n_samples = 1000
        widths = [64, 256]
        eps = 1.0

        variances = []
        for n_hidden in widths:
            result = mc_resnet_kernel_statistics(
                K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
            )
            mean_var = result["G_var"].mean().item()
            variances.append(mean_var)

        ratio = variances[0] / (variances[1] + 1e-15)
        assert ratio > 2.0, f"Variance should scale as 1/n, ratio={ratio:.2f}"

    def test_resnet_real_network_variance_scales_with_width(self):
        """Var(G)_real for ResNet should scale as 1/n."""
        N = 2
        K0_input = torch.eye(N, dtype=torch.float64)
        n_layers = 2
        n_samples = 1000
        widths = [64, 256]
        eps = 1.0

        variances = []
        for n_hidden in widths:
            result = real_network_resnet_statistics(
                K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
            )
            mean_var = result["G_var"].mean().item()
            variances.append(mean_var)

        ratio = variances[0] / (variances[1] + 1e-15)
        assert ratio > 2.0, f"Variance should scale as 1/n, ratio={ratio:.2f}"


class TestResnetV4TheoryVsRealNetwork:
    """Tests: ResNet V4 theory vs real network."""

    def test_resnet_v4_theory_vs_real_network_small_eps(self):
        """V4 theory should match real network at small eps (O(eps^2) accuracy)."""
        from resnet_eft import resnet_step, create_resnet_initial_state

        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 256
        n_samples = 3000
        eps = 0.2  # Small eps for O(eps^2) accuracy
        T_final = 0.2
        n_layers = max(1, int(T_final / eps**2))
        Cw = 2.0

        # Theory: with Wishart initial V4
        params = Params(act=ActivationSpec.tanh(), Cw=Cw, Cb=0.0)
        state = create_resnet_initial_state(K0_input, fan_in=n_hidden, params=params)
        for _ in range(n_layers):
            state = resnet_step(state, params, eps=eps, compute_V4=True)
        V4_th = state.V4.as_tensor()

        # Real network simulation
        result = real_network_resnet_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=Cw, eps=eps, n_samples=n_samples
        )
        V4_real = n_hidden * result["G_var"]

        # Compare at small eps: should be within 15%
        for a in range(N):
            for b in range(N):
                v4_th_ab = V4_th[a, b, a, b].item()
                v4_real_ab = V4_real[a, b].item()
                if abs(v4_real_ab) > 0.01:
                    rel_err = abs(v4_th_ab - v4_real_ab) / abs(v4_real_ab)
                    assert rel_err < 0.15, f"V4[{a},{b},{a},{b}] rel error {rel_err:.2f} > 15%"

    def test_resnet_v4_real_network_includes_wishart_variance(self):
        """Real network V4 includes Wishart variance when theory starts with V4=None.

        This test verifies that when theory is initialized WITHOUT Wishart V4,
        the real network V4 ≈ V4_theory + V4_Wishart.
        """
        from resnet_eft import resnet_step

        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 256
        n_layers = 1
        n_samples = 5000
        eps = 1.0

        # Theory: only source term
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)
        state = KernelState.from_input(K0_input, fan_out=n_hidden)
        for _ in range(n_layers):
            state = resnet_step(state, params, eps=eps, compute_V4=True)
        V4_th = state.V4.as_tensor()

        # Real network
        result = real_network_resnet_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
        )
        V4_real = n_hidden * result["G_var"]

        # Wishart base variance: V4_Wishart[a,b,a,b] ≈ K0[a,a]*K0[b,b] + K0[a,b]^2
        # For diagonal (a=b): V4_Wishart[a,a,a,a] = 2 * K0[a,a]^2
        K0_final = state.K0
        for a in range(N):
            for b in range(N):
                v4_theory = V4_th[a, b, a, b].item()
                v4_real = V4_real[a, b].item()

                # Wishart variance: V4[a,b,a,b] = K0[a,a]*K0[b,b] + K0[a,b]^2
                v4_wishart = K0_final[a, a].item() * K0_final[b, b].item() + K0_final[a, b].item() ** 2

                # Loose bounds (eps=1.0 has higher-order effects)
                v4_expected_lower = v4_theory + v4_wishart * 0.5
                v4_expected_upper = v4_theory + v4_wishart * 2.0

                assert v4_expected_lower < v4_real < v4_expected_upper, (
                    f"V4[{a},{b},{a},{b}]: real={v4_real:.2f} not in "
                    f"[{v4_expected_lower:.2f}, {v4_expected_upper:.2f}]"
                )


class TestResnetContinuousLimitConvergence:
    """Tests: ResNet continuous limit convergence."""

    def test_euler_convergence_to_continuous_limit(self):
        """Test that finer discretization converges."""
        from resnet_eft import resnet_step

        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 100
        T = 1.0  # Total time

        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # Compute K0 at different discretization levels
        K0_results = []
        for n_steps in [4, 16, 64]:
            eps = (T / n_steps) ** 0.5  # dt = eps^2, so eps = sqrt(T/n_steps)
            state = KernelState.from_input(K0_input, fan_out=n_hidden)
            for _ in range(n_steps):
                state = resnet_step(state, params, eps=eps, compute_K1=False, compute_V4=False)
            K0_results.append(state.K0)

        # Check convergence: finer should be closer to each other
        diff_coarse = (K0_results[0] - K0_results[1]).abs().max().item()
        diff_fine = (K0_results[1] - K0_results[2]).abs().max().item()

        # Error should decrease (roughly by factor of 4 for O(eps^2) = O(1/n_steps))
        assert diff_fine < diff_coarse, f"Finer discretization should reduce error"

    def test_mc_agrees_with_theory_at_small_eps(self):
        """MC should agree with theory for small eps (many layers)."""
        from resnet_eft import resnet_step

        N = 2
        K0_input = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        n_hidden = 128
        n_samples = 2000
        n_layers = 16
        eps = 0.25  # Small eps for continuous limit

        # Theory
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)
        state = KernelState.from_input(K0_input, fan_out=n_hidden)
        for _ in range(n_layers):
            state = resnet_step(state, params, eps=eps, compute_K1=False, compute_V4=False)
        K0_th = state.K0

        # MC
        result = mc_resnet_kernel_statistics(
            K0_input, n_layers, n_hidden, "tanh", Cw=1.0, eps=eps, n_samples=n_samples
        )
        K0_mc = result["G_mean"]

        rel_err = (K0_th - K0_mc).abs() / K0_th.abs().clamp(min=0.1)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 0.05, f"Small eps K0 rel error {max_rel_err:.3f} > 5%"
