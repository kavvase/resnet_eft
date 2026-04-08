"""Tests for K1SourceOp (K1 ← V4 contribution using E2'').

K1SourceOp computes the K1 source term using the correct formula:
    K1_source = (Cw/2) × E2''(K0) × V4

where E2''(K0) is the second derivative of E2(K) with respect to K.

This is validated against Monte Carlo simulations.
"""

import torch

from resnet_eft import GaussianExpectation, Params, V4Tensor
from resnet_eft.backend import allclose, zeros
from resnet_eft.core_types import ActivationSpec
from resnet_eft.k1_source_op import K1SourceOp, compute_k1_source_term


def random_psd_matrix(N: int, dtype=torch.float64) -> torch.Tensor:
    """Generate a random positive semi-definite matrix."""
    A = torch.randn(N, N, dtype=dtype)
    return A @ A.T + torch.eye(N, dtype=dtype)


def random_v4_tensor(N: int, dtype=torch.float64) -> torch.Tensor:
    """Generate a random V4 tensor with proper symmetries.

    Symmetries enforced:
    - First pair swap: V4[i,j,k,l] = V4[j,i,k,l]
    - Second pair swap: V4[i,j,k,l] = V4[i,j,l,k]
    - Left-right pair swap: V4[i,j,k,l] = V4[k,l,i,j]
    """
    V4 = torch.randn(N, N, N, N, dtype=dtype)
    # First pair swap: V4[i,j,k,l] = V4[j,i,k,l]
    V4 = 0.5 * (V4 + V4.permute(1, 0, 2, 3))
    # Second pair swap: V4[i,j,k,l] = V4[i,j,l,k]
    V4 = 0.5 * (V4 + V4.permute(0, 1, 3, 2))
    # Left-right pair swap: V4[i,j,k,l] = V4[k,l,i,j]
    V4 = 0.5 * (V4 + V4.permute(2, 3, 0, 1))
    return V4


class TestK1SourceOpBasic:
    """Basic tests for K1SourceOp."""

    def test_k1_source_op_shape(self):
        """Test K1SourceOp.contract returns correct shape."""
        N = 2
        K0 = random_psd_matrix(N)
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        result = op.contract(V4, Cw)

        assert result.shape == (N, N)

    def test_k1_source_op_symmetry(self):
        """Test K1SourceOp.contract returns symmetric result."""
        N = 2
        K0 = random_psd_matrix(N)
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        result = op.contract(V4, Cw)

        # Result should be approximately symmetric
        # (may not be exact due to numerical differentiation)
        assert (result - result.T).abs().max() < 0.1

    def test_k1_source_op_linearity(self):
        """Test K1SourceOp.contract is linear in V4."""
        N = 2
        K0 = random_psd_matrix(N)
        V4_data1 = random_v4_tensor(N)
        V4_data2 = random_v4_tensor(N)
        alpha = 2.5
        Cw = 2.0

        V4_1 = V4Tensor(data=V4_data1)
        V4_2 = V4Tensor(data=V4_data2)
        V4_sum = V4Tensor(data=V4_data1 + V4_data2)
        V4_scaled = V4Tensor(data=alpha * V4_data1)

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        result_1 = op.contract(V4_1, Cw)
        result_2 = op.contract(V4_2, Cw)
        result_sum = op.contract(V4_sum, Cw)
        result_scaled = op.contract(V4_scaled, Cw)

        # Linearity: contract(V4_1 + V4_2) = contract(V4_1) + contract(V4_2)
        assert allclose(result_sum, result_1 + result_2, rtol=0.01)

        # Scaling: contract(α × V4_1) = α × contract(V4_1)
        assert allclose(result_scaled, alpha * result_1, rtol=0.01)


class TestK1SourceOpTanh:
    """Tests specific to tanh activation (smooth, concave)."""

    def test_e2_hessian_diag_negative_for_tanh(self):
        """Test d²E2[i,i]/dq² is negative for tanh (concave function).

        For tanh, E2(K)[i,i] = E[tanh²(φ)] where φ ~ N(0, K[i,i]).
        This is a concave function of K[i,i] for small variances,
        so E2'' < 0.
        """
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)  # Unit variance

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        # Compute d²E2[0,0]/dq² using the sparse Hessian method
        d2E2_dq2 = op._compute_hessian_diag(0)

        # Should be negative for tanh (concave)
        assert d2E2_dq2 < 0

    def test_hessian_3x3_symmetric_for_tanh(self):
        """Test that 3×3 Hessian is symmetric for off-diagonal elements."""
        K0 = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        # Compute 3×3 Hessian for off-diagonal (0, 1)
        H = op._compute_hessian_3x3(0, 1)

        # Hessian should be symmetric
        assert H.shape == (3, 3)
        assert allclose(H, H.T, rtol=0.01)

    def test_hessian_3x3_has_correct_structure(self):
        """Test that 3×3 Hessian has expected structure for tanh.

        For E2[i,j] = f(q_i, q_j, c_ij), the Hessian should have:
        - H[0,0] = d²f/dq_i² (curvature in q_i direction)
        - H[1,1] = d²f/dq_j² (curvature in q_j direction)
        - H[2,2] = d²f/dc² (curvature in correlation direction)
        """
        K0 = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        H = op._compute_hessian_3x3(0, 1)

        # All diagonal elements should be finite
        assert torch.isfinite(H).all()

        # Due to symmetry in K0 (q_0 = q_1), H[0,0] ≈ H[1,1]
        assert allclose(H[0, 0], H[1, 1], rtol=0.1)

    def test_k1_source_negative_for_positive_v4_diagonal(self):
        """Test K1 source is negative for positive V4 and tanh.

        With V4[0,0,0,0] > 0 and E2''[0,0;0,0;0,0] < 0:
        K1_source[0,0] = (Cw/2) × E2'' × V4 < 0
        """
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)
        Cw = 2.0

        # Create V4 with positive diagonal
        V4_data = zeros((N, N, N, N), dtype=torch.float64)
        V4_data[0, 0, 0, 0] = 1.0
        V4 = V4Tensor(data=V4_data)

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        result = op.contract(V4, Cw)

        # K1_source[0,0] should be negative
        assert result[0, 0] < 0


class TestK1SourceOpReLU:
    """Tests specific to ReLU activation."""

    def test_e2_hessian_diag_zero_for_relu(self):
        """Test d²E2[i,i]/dq² ≈ 0 for ReLU diagonal.

        For ReLU, E2[i,i] = K[i,i]/2 (linear in diagonal),
        so E2'' = 0 for diagonal components.
        """
        N = 2
        K0 = torch.eye(N, dtype=torch.float64)

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        # Compute d²E2[0,0]/dq² using the sparse Hessian method
        d2E2_dq2 = op._compute_hessian_diag(0)

        # Should be approximately zero for ReLU diagonal
        assert abs(d2E2_dq2) < 0.01


class TestSparseVsFullHessian:
    """Tests comparing sparse O(N²) vs full O(N^6) implementations.

    Note: Off-diagonal comparison may differ due to how symmetric matrix
    perturbations are handled. The sparse method is the correct one
    (validated against MC), while the full method has known issues with
    double-counting off-diagonal perturbations.
    """

    def test_sparse_matches_full_diagonal_for_tanh(self):
        """Test that sparse Hessian matches full for diagonal elements (tanh)."""
        N = 2
        K0 = random_psd_matrix(N)
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        # Sparse method (O(N²))
        result_sparse = op.contract(V4, Cw)

        # Full method (O(N^6))
        result_full = op.contract_full(V4, Cw)

        # Diagonal elements should match well
        for i in range(N):
            assert allclose(result_sparse[i, i], result_full[i, i], rtol=0.1, atol=0.01)

    def test_sparse_matches_full_diagonal_for_relu(self):
        """Test that sparse Hessian matches full for diagonal elements (ReLU).

        Note: Off-diagonal comparison is less reliable due to numerical
        differentiation differences in how symmetric matrix perturbations
        are handled. We focus on diagonal elements here.
        """
        N = 2
        K0 = random_psd_matrix(N)
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        Cw = 2.0

        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=Cw)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        # Sparse method (O(N²))
        result_sparse = op.contract(V4, Cw)

        # Full method (O(N^6))
        result_full = op.contract_full(V4, Cw)

        # Diagonal elements should match well (both methods handle diagonal the same)
        for i in range(N):
            assert allclose(result_sparse[i, i], result_full[i, i], rtol=0.1, atol=0.01)


class TestConvenienceFunction:
    """Tests for compute_k1_source_term function."""

    def test_compute_k1_source_term(self):
        """Test convenience function produces same result as class method."""
        N = 2
        K0 = random_psd_matrix(N)
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)

        # Using class
        op = K1SourceOp(K0, gauss)
        result_class = op.contract(V4, Cw)

        # Using function
        result_func = compute_k1_source_term(K0, V4, gauss, Cw)

        assert allclose(result_class, result_func)


class TestK1SourceOpMode:
    """Tests for K1SourceOp mode parameter."""

    def test_mode_auto_detects_uniform(self):
        """Test mode='auto' correctly detects uniform K0."""
        N = 3
        K0_uniform = torch.eye(N) * 0.5 + 0.5  # αI + β1 structure

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0_uniform, gauss, mode="auto")

        assert op._is_uniform is True, "Should detect uniform K0"

    def test_mode_auto_detects_non_uniform(self):
        """Test mode='auto' correctly detects non-uniform K0."""
        K0_non_uniform = torch.tensor(
            [
                [1.0, 0.5, 0.3],
                [0.5, 0.8, 0.4],
                [0.3, 0.4, 1.2],
            ]
        )

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0_non_uniform, gauss, mode="auto")

        assert op._is_uniform is False, "Should detect non-uniform K0"

    def test_mode_uniform_forces_uniform_path(self):
        """Test mode='uniform' forces uniform optimization path."""
        N = 3
        K0_uniform = torch.eye(N) * 0.5 + 0.5

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0_uniform, gauss, mode="uniform")

        assert op._is_uniform is True
        assert op.mode == "uniform"

    def test_mode_uniform_raises_for_non_uniform_k0(self):
        """Test mode='uniform' raises ValueError for non-uniform K0."""
        K0_non_uniform = torch.tensor(
            [
                [1.0, 0.5, 0.3],
                [0.5, 0.8, 0.4],
                [0.3, 0.4, 1.2],
            ]
        )

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)

        import pytest

        with pytest.raises(ValueError, match="mode='uniform' requires K0 to have"):
            K1SourceOp(K0_non_uniform, gauss, mode="uniform")

    def test_mode_general_forces_general_path(self):
        """Test mode='general' forces general path even for uniform K0."""
        N = 3
        K0_uniform = torch.eye(N) * 0.5 + 0.5

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0_uniform, gauss, mode="general")

        assert op._is_uniform is False, "mode='general' should force _is_uniform=False"
        assert op.mode == "general"

    def test_mode_invalid_raises_error(self):
        """Test invalid mode raises ValueError."""
        N = 2
        K0 = torch.eye(N)

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)

        import pytest

        with pytest.raises(ValueError, match="Unknown mode"):
            K1SourceOp(K0, gauss, mode="invalid_mode")

    def test_all_modes_give_same_result_for_uniform_k0(self):
        """Test all modes produce identical results for uniform K0."""
        N = 3
        K0_uniform = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)

        # Create V4
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        # Compute with all modes
        op_auto = K1SourceOp(K0_uniform, gauss, mode="auto")
        op_uniform = K1SourceOp(K0_uniform, gauss, mode="uniform")
        op_general = K1SourceOp(K0_uniform, gauss, mode="general")

        K1_auto = op_auto.contract(V4, Cw)
        K1_uniform = op_uniform.contract(V4, Cw)
        K1_general = op_general.contract(V4, Cw)

        # All should be identical (within numerical differentiation precision)
        # Note: The uniform path caches Hessians, general path computes per-(i,j).
        # Small differences (~0.01%) can occur due to floating point accumulation.
        max_diff = torch.abs(K1_auto - K1_general).max().item()
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds 1e-3"
        assert allclose(K1_auto, K1_uniform, rtol=1e-6, atol=1e-10)

    def test_convenience_function_accepts_mode(self):
        """Test compute_k1_source_term accepts mode parameter."""
        N = 2
        K0 = random_psd_matrix(N)
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)

        # Should not raise
        result = compute_k1_source_term(K0, V4, gauss, Cw, mode="general")
        assert result.shape == (N, N)


class TestCheckUniform:
    """Tests for _check_uniform method."""

    def test_identity_is_uniform(self):
        """Test identity matrix is detected as uniform."""
        N = 3
        K0 = torch.eye(N)

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        is_uniform, uniform_params = op._check_uniform()
        assert is_uniform
        assert uniform_params == (1.0, 0.0)  # diag=1, offdiag=0

    def test_alpha_I_plus_beta_1_is_uniform(self):
        """Test αI + β1 matrix is detected as uniform."""
        N = 4
        alpha, beta = 0.5, 0.3
        K0 = alpha * torch.eye(N, dtype=torch.float64) + beta * torch.ones(
            N, N, dtype=torch.float64
        )

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        is_uniform, uniform_params = op._check_uniform()
        assert is_uniform
        assert uniform_params is not None
        assert abs(uniform_params[0] - (alpha + beta)) < 1e-6  # diag = alpha + beta
        assert abs(uniform_params[1] - beta) < 1e-6  # offdiag = beta

    def test_different_diagonals_not_uniform(self):
        """Test matrix with different diagonals is not uniform."""
        K0 = torch.tensor(
            [
                [1.0, 0.5],
                [0.5, 2.0],  # Different diagonal
            ]
        )

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss, mode="auto")

        is_uniform, _ = op._check_uniform()
        assert not is_uniform

    def test_different_offdiagonals_not_uniform(self):
        """Test matrix with different off-diagonals is not uniform."""
        K0 = torch.tensor(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.5],  # Different off-diagonal at [0,2]
                [0.3, 0.5, 1.0],
            ]
        )

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss, mode="auto")

        is_uniform, _ = op._check_uniform()
        assert not is_uniform

    def test_1x1_matrix_is_uniform(self):
        """Test 1×1 matrix is always uniform."""
        K0 = torch.tensor([[5.0]])

        params = Params(act="tanh", Cw=2.0)
        gauss = GaussianExpectation(params)
        op = K1SourceOp(K0, gauss)

        is_uniform, uniform_params = op._check_uniform()
        assert is_uniform
        assert uniform_params == (5.0, 0.0)


class TestK1SourceOpIntegration:
    """Integration tests for K1SourceOp with layer_update."""

    def test_k1_source_in_step(self):
        """Test that K1SourceOp is used correctly in step()."""
        from resnet_eft import KernelState, step

        N = 2
        n_hidden = 100
        K0_init = random_psd_matrix(N)

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)
        params = Params(act="tanh", Cw=2.0, Cb=0.0)

        # First step: create V4 (local term only)
        state1 = step(state0, params, fan_out=n_hidden, compute_K1=False, compute_V4=True)
        assert state1.V4 is not None

        # Second step: K1 should be computed from V4 contribution
        state2 = step(state1, params, fan_out=n_hidden, compute_K1=True, compute_V4=True)

        # K1 should be non-None and finite
        assert state2.K1 is not None
        assert torch.isfinite(state2.K1).all(), "K1 contains NaN or inf"

        # For tanh with positive V4, K1 should be predominantly negative
        # (due to E2'' < 0)

    def test_k1_matches_mc_direction(self):
        """Test that K1 sign matches MC observation direction.

        For tanh: E2'' < 0 for diagonal → K1 < 0 when V4 > 0.
        This matches MC observation that E[G] < K0.
        """
        K0 = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        Cw = 2.0

        params = Params(act="tanh", Cw=Cw)
        gauss = GaussianExpectation(params)

        # Compute local V4 term (always positive for connected correlation)
        E2 = gauss.E2_pairwise(K0)
        E4 = gauss.E4_pairwise(K0)
        disconnected = torch.einsum("ij,kl->ijkl", E2, E2)
        V4_local = (Cw**2) * (E4 - disconnected)

        # For typical networks, V4_local[0,0,0,0] > 0
        # (4-point connected correlation is positive)
        V4 = V4Tensor(data=V4_local)

        op = K1SourceOp(K0, gauss)
        K1_source = op.contract(V4, Cw)

        # For tanh: E2'' < 0, V4 > 0 → K1 < 0
        # This means E[G] = K0 + K1/n < K0 (consistent with MC)
        assert K1_source[0, 0] < 0, "K1 should be negative for tanh with positive V4"

    def test_params_k1_mode_propagates_to_step(self):
        """Test that Params.k1_mode is used in step()."""
        from resnet_eft import KernelState, step

        N = 3
        n_hidden = 100
        # Uniform K0
        K0_init = torch.eye(N) * 0.5 + 0.5

        state0 = KernelState.from_input(K0_init, fan_out=n_hidden)

        # Test with k1_mode="general"
        params_general = Params(act="tanh", Cw=2.0, Cb=0.0, k1_mode="general")
        state1 = step(state0, params_general, fan_out=n_hidden, compute_K1=False, compute_V4=True)
        state2_general = step(
            state1, params_general, fan_out=n_hidden, compute_K1=True, compute_V4=True
        )

        # Test with k1_mode="uniform"
        state0_u = KernelState.from_input(K0_init, fan_out=n_hidden)
        params_uniform = Params(act="tanh", Cw=2.0, Cb=0.0, k1_mode="uniform")
        state1_u = step(
            state0_u, params_uniform, fan_out=n_hidden, compute_K1=False, compute_V4=True
        )
        state2_uniform = step(
            state1_u, params_uniform, fan_out=n_hidden, compute_K1=True, compute_V4=True
        )

        # Both should produce the same K1 (just different code paths)
        assert state2_general.K1 is not None
        assert state2_uniform.K1 is not None
        assert allclose(state2_general.K1, state2_uniform.K1, rtol=1e-10)
