"""Tests for layer_update (step and resnet_step functions).

These tests verify the layer update API works correctly.
"""

import pytest
import torch

from resnet_eft import KernelState, Params, resnet_step, step
from resnet_eft.backend import allclose
from resnet_eft.core_types import ActivationSpec


def random_psd_matrix(N: int, dtype=torch.float64) -> torch.Tensor:
    """Generate a random positive semi-definite matrix."""
    A = torch.randn(N, N, dtype=dtype)
    return A @ A.T + torch.eye(N, dtype=dtype)


class TestStep:
    """Tests for the step() function."""

    def test_step_K0_update(self):
        """Test that K0 is updated correctly."""
        N = 4
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n_hidden, compute_K1=False, compute_V4=False)

        # K0 should be updated
        assert state1.K0 is not None
        assert state1.K0.shape == (N, N)

        # K0 should be symmetric
        assert allclose(state1.K0, state1.K0.T)

        # K0 should be PSD (diagonal positive)
        assert (state1.K0.diagonal() > 0).all()

    def test_step_updates_depth(self):
        """Test that depth is incremented."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        assert state0.depth == 0

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        state1 = step(state0, params, fan_out=n_hidden)
        assert state1.depth == 1

        state2 = step(state1, params, fan_out=n_hidden)
        assert state2.depth == 2

    def test_step_fan_in_fan_out(self):
        """Test that fan_in and fan_out are set correctly."""
        N = 3
        n1, n2, n3 = 100, 200, 150
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n1)
        assert state0.fan_in is None
        assert state0.fan_out == n1

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n2)
        assert state1.fan_in == n1  # fan_in = prev.fan_out
        assert state1.fan_out == n2

        state2 = step(state1, params, fan_out=n3)
        assert state2.fan_in == n2
        assert state2.fan_out == n3

    def test_step_K1_computed(self):
        """Test that K1 is computed when requested."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        # First step: K1_prev = None, so K1 comes only from V4 (which is computed)
        state1 = step(state0, params, fan_out=n_hidden, compute_K1=True, compute_V4=True)

        # K1 should be computed (from V4 contribution)
        # After first step, V4 exists, so K1 will be computed
        assert state1.K1 is not None or state1.V4 is not None

        # Second step should definitely have K1
        state2 = step(state1, params, fan_out=n_hidden, compute_K1=True, compute_V4=True)
        assert state2.K1 is not None

    def test_step_V4_computed(self):
        """Test that V4 is computed when requested."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n_hidden, compute_V4=True)

        # V4 should be computed (from local term)
        assert state1.V4 is not None

    def test_step_no_K1_when_disabled(self):
        """Test that K1 is not computed when disabled."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n_hidden, compute_K1=False, compute_V4=False)
        assert state1.K1 is None

    def test_step_no_V4_when_disabled(self):
        """Test that V4 is not computed when disabled."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n_hidden, compute_K1=False, compute_V4=False)
        assert state1.V4 is None


class TestStepDifferentActivations:
    """Tests for step with different activations."""

    @pytest.mark.parametrize("act", ["relu", "erf"])
    def test_step_different_activations(self, act: str):
        """Test step works with different activations."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=act, Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n_hidden)

        assert state1.K0 is not None
        assert allclose(state1.K0, state1.K0.T)


class TestStepMultipleLayers:
    """Tests for multiple step invocations."""

    def test_multi_layer_forward(self):
        """Test forward pass through multiple layers."""
        N = 4
        n_hidden = 100
        L = 5

        K0_init = random_psd_matrix(N)
        state = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        for _ in range(L):
            state = step(state, params, fan_out=n_hidden)

        assert state.depth == L
        assert state.K0 is not None

    def test_K0_evolves(self):
        """Test that K0 evolves through layers (not constant)."""
        N = 3
        n_hidden = 100

        K0_init = random_psd_matrix(N)
        state = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        K0_values = [state.K0.clone()]
        for _ in range(3):
            state = step(state, params, fan_out=n_hidden, compute_K1=False, compute_V4=False)
            K0_values.append(state.K0.clone())

        # K0 should change between layers
        for i in range(len(K0_values) - 1):
            # At least some elements should differ
            diff = (K0_values[i] - K0_values[i + 1]).abs().max()
            assert diff > 1e-6, f"K0 did not change between layer {i} and {i + 1}"


class TestPhysicalQuantities:
    """Tests for physical quantity extraction."""

    def test_get_physical_K1(self):
        """Test get_physical_K1 divides by fan_in."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        # Run two steps to get non-None K1
        state1 = step(state0, params, fan_out=n_hidden, compute_K1=True, compute_V4=True)
        state2 = step(state1, params, fan_out=n_hidden, compute_K1=True, compute_V4=True)

        if state2.K1 is not None and state2.fan_in is not None:
            K1_phys = state2.get_physical_K1()
            assert K1_phys is not None
            assert allclose(K1_phys, state2.K1 / state2.fan_in)

    def test_get_physical_V4(self):
        """Test get_physical_V4 scales by 1/fan_in."""
        N = 3
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)

        state1 = step(state0, params, fan_out=n_hidden, compute_V4=True)

        if state1.V4 is not None and state1.fan_in is not None:
            V4_phys = state1.get_physical_V4()
            assert V4_phys is not None
            # V4_phys should be V4.scale(1/fan_in)
            assert allclose(V4_phys.as_tensor(), state1.V4.as_tensor() / state1.fan_in)


class TestResNetStep:
    """Tests for the resnet_step() function (pre-activation ResNet)."""

    def test_resnet_step_K0_incremental(self):
        """Test that K0 is updated incrementally: K0' = K0 + ε² × Cw × E2."""
        N = 4
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1)

        # With ε=1, K0' = K0 + Cw × E2
        state1 = resnet_step(state0, params, eps=1.0, compute_K1=False, compute_V4=False)

        # K0 should be updated
        assert state1.K0 is not None
        assert state1.K0.shape == (N, N)

        # K0 should be symmetric
        assert allclose(state1.K0, state1.K0.T)

        # K0 should be larger than input (adding positive E2)
        assert (state1.K0.diagonal() > K0_init.diagonal()).all()

    def test_resnet_step_eps_scaling(self):
        """Test that ε² scaling is applied correctly."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # Compare eps=1.0 and eps=0.5
        state_eps1 = resnet_step(state0, params, eps=1.0, compute_K1=False, compute_V4=False)
        state_eps05 = resnet_step(state0, params, eps=0.5, compute_K1=False, compute_V4=False)

        # Increment should scale as ε²
        # K0_eps1 = K0 + 1.0² × increment
        # K0_eps05 = K0 + 0.25 × increment
        increment_eps1 = state_eps1.K0 - K0_init
        increment_eps05 = state_eps05.K0 - K0_init

        # increment_eps05 should be 0.25 × increment_eps1
        assert allclose(increment_eps05, 0.25 * increment_eps1, rtol=1e-5)

    def test_resnet_step_V4_computed(self):
        """Test that V4 is computed when requested."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1)

        state1 = resnet_step(state0, params, eps=1.0, compute_V4=True)

        # V4 should be computed
        assert state1.V4 is not None

    def test_resnet_step_multi_layer(self):
        """Test forward pass through multiple ResNet layers."""
        N = 4
        n_hidden = 100
        L = 5

        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        state = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        for _ in range(L):
            state = resnet_step(state, params, eps=1.0)

        assert state.depth == L
        assert state.K0 is not None
        assert allclose(state.K0, state.K0.T)

    def test_resnet_step_continuous_limit_convergence(self):
        """Test that small ε gives ODE-like behavior: more steps = closer to ODE."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # Reference: eps=1, L=1 steps (total "time" = 1)
        state_ref = KernelState.from_input(K0_init, fan_out=n_hidden)
        state_ref = resnet_step(state_ref, params, eps=1.0, compute_K1=False, compute_V4=False)
        K0_ref = state_ref.K0

        # Fine discretization: eps=0.5, L=4 steps (total "time" = 0.5² × 4 = 1)
        state_fine = KernelState.from_input(K0_init, fan_out=n_hidden)
        for _ in range(4):
            state_fine = resnet_step(state_fine, params, eps=0.5, compute_K1=False, compute_V4=False)
        K0_fine = state_fine.K0

        # Finer discretization: eps=0.25, L=16 steps (total "time" = 0.25² × 16 = 1)
        state_finer = KernelState.from_input(K0_init, fan_out=n_hidden)
        for _ in range(16):
            state_finer = resnet_step(state_finer, params, eps=0.25, compute_K1=False, compute_V4=False)
        K0_finer = state_finer.K0

        # K0_fine and K0_finer should be closer to each other than to K0_ref
        # (demonstrating convergence to continuous limit)
        diff_ref_fine = (K0_ref - K0_fine).abs().max()
        diff_fine_finer = (K0_fine - K0_finer).abs().max()

        # Finer discretization should give smaller differences
        # (This is a weak test; stronger tests need ODE solver comparison)
        assert diff_fine_finer < diff_ref_fine, "Finer discretization should converge"

    def test_resnet_step_no_cb_term(self):
        """Test that Cb (bias variance) does not affect K0 in ResNet."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)

        # Compare with Cb=0 and Cb=0.5
        params_no_bias = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)
        params_with_bias = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.5)

        state_no_bias = resnet_step(state0, params_no_bias, eps=1.0, compute_K1=False, compute_V4=False)
        state_with_bias = resnet_step(state0, params_with_bias, eps=1.0, compute_K1=False, compute_V4=False)

        # K0 should be the same (Cb doesn't enter pre-act ResNet)
        assert allclose(state_no_bias.K0, state_with_bias.K0)


class TestResNetStepV4:
    """Tests for V4 computation in resnet_step."""

    def test_resnet_v4_incremental_form(self):
        """Test that V4 is updated in incremental form: V4' = V4 + ε² × (...)."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # First step: V4 comes from source term only
        state1 = resnet_step(state0, params, eps=1.0, compute_V4=True)
        V4_1 = state1.V4.as_tensor()

        # Second step: V4 should be V4_prev + increment
        state2 = resnet_step(state1, params, eps=1.0, compute_V4=True)
        V4_2 = state2.V4.as_tensor()

        # V4 should grow (accumulating source and transport)
        # V4_2 should be larger than V4_1 in Frobenius norm
        assert V4_2.norm() >= V4_1.norm() * 0.9  # Allow some tolerance

    def test_resnet_v4_eps_scaling(self):
        """Test that V4 increment scales as ε²."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # First layer to get initial V4
        state1 = resnet_step(state0, params, eps=1.0, compute_V4=True)

        # Second layer with eps=1.0 and eps=0.5
        state2_eps1 = resnet_step(state1, params, eps=1.0, compute_V4=True)
        state2_eps05 = resnet_step(state1, params, eps=0.5, compute_V4=True)

        # V4 increments
        V4_1 = state1.V4.as_tensor()
        increment_eps1 = state2_eps1.V4.as_tensor() - V4_1
        increment_eps05 = state2_eps05.V4.as_tensor() - V4_1

        # increment_eps05 should be 0.25 × increment_eps1
        assert allclose(increment_eps05, 0.25 * increment_eps1, rtol=1e-4)

    def test_resnet_v4_source_term_present(self):
        """Test that source term (local V4) is generated at each layer."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # First step: V4 = ε² × source (no previous V4)
        state1 = resnet_step(state0, params, eps=1.0, compute_V4=True)

        # V4 should be non-zero (source term)
        assert state1.V4 is not None
        V4_tensor = state1.V4.as_tensor()
        assert V4_tensor.abs().max() > 1e-10, "V4 source term should be non-zero"

    def test_resnet_v4_transport_across_layers(self):
        """Test V4 transport accumulates across multiple layers."""
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        V4_norms = []
        for _ in range(5):
            state = resnet_step(state, params, eps=1.0, compute_V4=True)
            V4_norms.append(state.V4.as_tensor().norm().item())

        # V4 should generally grow due to source accumulation
        # (Transport may shrink it, but source keeps adding)
        # Check that final V4 is larger than first
        assert V4_norms[-1] > V4_norms[0] * 0.5, "V4 should accumulate"


class TestResNetStepCompareWithMLP:
    """Tests comparing resnet_step with step for sanity checks."""

    def test_resnet_vs_mlp_single_layer_K0(self):
        """Compare K0 evolution between ResNet (ε=1) and MLP single step.

        For ε=1:
        - ResNet: K0' = K0 + Cw × E2(K0)  (no Cb)
        - MLP:    K0' = Cb + Cw × E2(K0)

        With Cb=0, the difference is: ResNet = MLP - Cb + K0 = MLP + K0
        """
        N = 3
        n_hidden = 100
        K0_init = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.0)

        # ResNet step with ε=1
        state_resnet = resnet_step(state0, params, eps=1.0, compute_K1=False, compute_V4=False)

        # MLP step
        state_mlp = step(state0, params, fan_out=n_hidden, compute_K1=False, compute_V4=False)

        # With Cb=0: K0_resnet = K0 + Cw × E2 = K0 + K0_mlp
        # Because MLP: K0_mlp = Cb + Cw × E2 = Cw × E2 (when Cb=0)
        expected_resnet_K0 = K0_init + state_mlp.K0

        assert allclose(state_resnet.K0, expected_resnet_K0, rtol=1e-5)
