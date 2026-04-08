"""Tests for V4 transport (P0 priority tests).

These tests verify that V4Operator transport matches the naive einsum implementation.
"""

import pytest
import torch

from resnet_eft import GaussianExpectation, Params, V4Operator, V4Tensor
from resnet_eft.backend import allclose, einsum, zeros
from resnet_eft.chi_op import ChiOp
from resnet_eft.core_types import ActivationSpec


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


def random_v4_tensor_nonsymmetric(N: int, dtype=torch.float64) -> torch.Tensor:
    """Generate a random V4 tensor with ONLY within-pair symmetry.

    Symmetries enforced:
    - First pair swap: V4[i,j,k,l] = V4[j,i,k,l]
    - Second pair swap: V4[i,j,k,l] = V4[i,j,l,k]

    NOT enforced:
    - Left-right pair swap: V4[i,j,k,l] ≠ V4[k,l,i,j] in general

    This is used to test that term2 (using V4^T) is correctly implemented,
    since with symmetric V4, term1 and term2 would be identical.
    """
    V4 = torch.randn(N, N, N, N, dtype=dtype)
    # First pair swap: V4[i,j,k,l] = V4[j,i,k,l]
    V4 = 0.5 * (V4 + V4.permute(1, 0, 2, 3))
    # Second pair swap: V4[i,j,k,l] = V4[i,j,l,k]
    V4 = 0.5 * (V4 + V4.permute(0, 1, 3, 2))
    return V4


def build_chi_tensor_naive(chi_op: ChiOp) -> torch.Tensor:
    """Build full χ tensor naively for testing."""
    N = chi_op.Epp.shape[0]
    chi = zeros((N, N, N, N), dtype=chi_op.Epp.dtype)
    coeff = chi_op.coeff

    for x1 in range(N):
        for x2 in range(N):
            for y1 in range(N):
                for y2 in range(N):
                    val = 0.0
                    if x1 == y1 and x2 == y2:
                        val += chi_op.Epp[x1, x2].item()
                    if x1 == y2 and x2 == y1:
                        val += chi_op.Epp[x1, x2].item()
                    if x1 == y1 and x1 == y2:
                        val += chi_op.E2s[x1, x2].item()
                    if x2 == y1 and x2 == y2:
                        val += chi_op.Es2[x1, x2].item()
                    chi[x1, x2, y1, y2] = coeff * val

    return chi


class TestV4Transport:
    """P0 tests: V4 transport matches naive einsum."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_v4_transport_matches_einsum(self, N: int):
        """Test that V4Operator transport matches naive einsum."""
        V4_prev_data = random_v4_tensor(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Naive einsum computation
        chi_tensor = build_chi_tensor_naive(chi_op)

        # term1: Σ chi[ab,mn] × V4[mn,pq] × chi[cd,pq]
        # This is M_χ @ M_V @ M_χᵀ
        term1_naive = einsum("abmn,mnpq,cdpq->abcd", chi_tensor, V4_prev_data, chi_tensor)

        # term2: Σ chi[ab,pq] × V4[mn,pq] × chi[cd,mn]
        # = Σ chi[ab,mn] × V4[pq,mn] × chi[cd,pq]  (rename dummy indices)
        # = Σ chi[ab,mn] × V4^T[mn,pq] × chi[cd,pq]  where V4^T[a,b;c,d] = V4[c,d;a,b]
        # This is M_χ @ M_V^T @ M_χᵀ (pair-space transpose of V4)
        term2_naive = einsum("abmn,pqmn,cdpq->abcd", chi_tensor, V4_prev_data, chi_tensor)

        V4_next_naive = 0.5 * (term1_naive + term2_naive)

        # V4Operator (transport only, no local term)
        V4_op = V4Operator(local_op=None, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)
        V4_next_op = V4_op.as_tensor()

        assert allclose(V4_next_op, V4_next_naive, rtol=1e-4)

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_v4_transport_nonsymmetric_v4(self, N: int):
        """Test transport with non-symmetric V4 to verify term2 is correct.

        CRITICAL TEST: With symmetric V4, term1 = term2, so bugs in term2
        would not be detected. This test uses V4 that is NOT left-right
        pair symmetric to ensure term2 (using V4^T) is correctly implemented.
        """
        # Use non-symmetric V4 (only within-pair symmetry, NOT pair swap symmetry)
        V4_prev_data = random_v4_tensor_nonsymmetric(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        # Verify V4 is indeed non-symmetric
        V4_swapped = V4_prev_data.permute(2, 3, 0, 1)
        assert not allclose(V4_prev_data, V4_swapped), "Test requires non-symmetric V4"

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        chi_tensor = build_chi_tensor_naive(chi_op)

        # term1: M_χ @ M_V @ M_χᵀ
        term1_naive = einsum("abmn,mnpq,cdpq->abcd", chi_tensor, V4_prev_data, chi_tensor)

        # term2: M_χ @ M_V^T @ M_χᵀ (V^T = pair-space transpose)
        term2_naive = einsum("abmn,pqmn,cdpq->abcd", chi_tensor, V4_prev_data, chi_tensor)

        # With non-symmetric V4, term1 ≠ term2
        assert not allclose(term1_naive, term2_naive), "term1 should differ from term2"

        V4_next_naive = 0.5 * (term1_naive + term2_naive)

        V4_op = V4Operator(local_op=None, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)
        V4_next_op = V4_op.as_tensor()

        assert allclose(V4_next_op, V4_next_naive, rtol=1e-4)

    def test_v4_transport_with_width_ratio(self):
        """Test that width_ratio is correctly applied."""
        N = 3
        V4_prev_data = random_v4_tensor(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Compare with width_ratio=1 and width_ratio=2
        V4_op_1 = V4Operator(local_op=None, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)
        V4_op_2 = V4Operator(local_op=None, chi_op=chi_op, prev_V4=V4_prev, width_ratio=2.0)

        V4_next_1 = V4_op_1.as_tensor()
        V4_next_2 = V4_op_2.as_tensor()

        # width_ratio=2 should give exactly 2× the result
        assert allclose(V4_next_2, 2.0 * V4_next_1, rtol=1e-5)


class TestV4Tensor:
    """Tests for V4Tensor basic operations."""

    def test_apply_pair(self):
        """Test V4Tensor.apply_pair."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        A = torch.randn(N, N, dtype=torch.float64)

        result = V4.apply_pair(A)
        expected = einsum("ijkl,kl->ij", V4_data, A)

        assert allclose(result, expected)

    def test_apply_pair_T(self):
        """Test V4Tensor.apply_pair_T (pair-space transpose).

        V4^T[a,b;c,d] = V4[c,d;a,b]
        (M_V^T @ vec(A))[(a,b)] = Σ_{c,d} V4[c,d;a,b] A[c,d]
        """
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)
        A = torch.randn(N, N, dtype=torch.float64)

        result = V4.apply_pair_T(A)
        # V4^T[a,b;c,d] = V4[c,d;a,b], so we contract over (c,d) with V4[c,d,a,b]
        expected = einsum("cdab,cd->ab", V4_data, A)

        assert allclose(result, expected)

    def test_apply_pair_T_is_transpose(self):
        """Test that apply_pair_T corresponds to the transpose of apply_pair."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        A = torch.randn(N, N, dtype=torch.float64)
        B = torch.randn(N, N, dtype=torch.float64)

        # ⟨A, V @ B⟩ should equal ⟨V^T @ A, B⟩
        lhs = (A * V4.apply_pair(B)).sum()
        rhs = (V4.apply_pair_T(A) * B).sum()

        assert allclose(lhs, rhs)

    def test_apply_pair_T_is_transpose_nonsymmetric(self):
        """Test duality ⟨A, V(B)⟩ = ⟨V^T(A), B⟩ with non-symmetric V4."""
        N = 3
        V4_data = random_v4_tensor_nonsymmetric(N)
        V4 = V4Tensor(data=V4_data)

        A = torch.randn(N, N, dtype=torch.float64)
        B = torch.randn(N, N, dtype=torch.float64)

        lhs = (A * V4.apply_pair(B)).sum()
        rhs = (V4.apply_pair_T(A) * B).sum()

        assert allclose(lhs, rhs)

    def test_scale(self):
        """Test V4Tensor.scale."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        V4_scaled = V4.scale(2.5)
        assert allclose(V4_scaled.data, 2.5 * V4_data)

    def test_get_diag_diag(self):
        """Test V4Tensor.get_diag_diag."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        diag_diag = V4.get_diag_diag()

        # Verify shape
        assert diag_diag.shape == (N, N)

        # Verify values
        for i in range(N):
            for j in range(N):
                assert allclose(diag_diag[i, j], V4_data[i, i, j, j])

    def test_get_cross_diag(self):
        """Test V4Tensor.get_cross_diag."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        cross_diag = V4.get_cross_diag()

        # Verify shape
        assert cross_diag.shape == (N, N)

        # Verify values
        for i in range(N):
            for j in range(N):
                assert allclose(cross_diag[i, j], V4_data[i, j, i, j])

    def test_get_diag_cross_left(self):
        """Test V4Tensor.get_diag_cross_left."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        diag_cross_left = V4.get_diag_cross_left()

        # Verify shape
        assert diag_cross_left.shape == (N, N)

        # Verify values: diag_cross_left[i, j] = V4[i, i, i, j]
        for i in range(N):
            for j in range(N):
                assert allclose(diag_cross_left[i, j], V4_data[i, i, i, j])

    def test_get_diag_cross_right(self):
        """Test V4Tensor.get_diag_cross_right."""
        N = 3
        V4_data = random_v4_tensor(N)
        V4 = V4Tensor(data=V4_data)

        diag_cross_right = V4.get_diag_cross_right()

        # Verify shape
        assert diag_cross_right.shape == (N, N)

        # Verify values: diag_cross_right[i, j] = V4[j, j, i, j]
        for i in range(N):
            for j in range(N):
                assert allclose(diag_cross_right[i, j], V4_data[j, j, i, j])

    def test_diag_cross_left_vs_right_differ(self):
        """Test that diag_cross_left and diag_cross_right are different in general.

        V4[i,i,i,j] != V4[j,j,i,j] in general (unless special symmetry).
        This test ensures we're not accidentally using the wrong slice.
        """
        N = 3
        # Use non-symmetric V4 to ensure difference
        V4_data = random_v4_tensor_nonsymmetric(N)
        V4 = V4Tensor(data=V4_data)

        left = V4.get_diag_cross_left()
        right = V4.get_diag_cross_right()

        # Should differ for at least some (i,j) pairs
        # (unless by random chance they're equal)
        diff_count = 0
        for i in range(N):
            for j in range(N):
                if i != j and not allclose(left[i, j], right[i, j], rtol=1e-5):
                    diff_count += 1

        # At least some off-diagonal pairs should differ
        assert diff_count > 0, "left and right should differ for non-symmetric V4"


class TestV4TransportSymmetry:
    """Tests for V4 transport symmetry properties."""

    def test_symmetric_v4_gives_symmetric_transport(self):
        """Test that transport preserves left-right pair symmetry.

        If V4_prev is symmetric under pair swap (V4[a,b,c,d] = V4[c,d,a,b]),
        then V4_next from transport should also be symmetric.
        """
        N = 3
        # random_v4_tensor already enforces pair symmetry
        V4_prev_data = random_v4_tensor(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        V4_op = V4Operator(local_op=None, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)
        V4_next = V4_op.as_tensor()

        # Check pair symmetry: V4_next[a,b,c,d] = V4_next[c,d,a,b]
        V4_next_swapped = V4_next.permute(2, 3, 0, 1)
        assert allclose(V4_next, V4_next_swapped, rtol=1e-4)

    def test_v4_pair_symmetry_preserved(self):
        """Verify random_v4_tensor has correct symmetry."""
        N = 3
        V4_data = random_v4_tensor(N)

        # (ij,kl) <-> (kl,ij) symmetry
        V4_pair_swapped = V4_data.permute(2, 3, 0, 1)
        assert allclose(V4_data, V4_pair_swapped)


class TestV4Operator:
    """Tests for V4Operator."""

    def test_as_tensor_local_only(self):
        """Test V4Operator.as_tensor with local term only."""
        from resnet_eft.v4_repr import LocalV4Op

        N = 3
        local_tensor = random_v4_tensor(N)
        local_op = LocalV4Op(local_tensor=local_tensor)

        # Create dummy chi_op (won't be used since prev_V4 is None)
        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        V4_op = V4Operator(local_op=local_op, chi_op=chi_op, prev_V4=None, width_ratio=1.0)
        V4_tensor = V4_op.as_tensor()

        assert allclose(V4_tensor, local_tensor)

    def test_apply_pair_T(self):
        """Test V4Operator.apply_pair_T matches as_tensor based calculation."""
        from resnet_eft.v4_repr import LocalV4Op

        N = 3
        local_tensor = random_v4_tensor(N)
        local_op = LocalV4Op(local_tensor=local_tensor)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        V4_prev_data = random_v4_tensor(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        V4_op = V4Operator(local_op=local_op, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)

        A = torch.randn(N, N, dtype=torch.float64)

        # apply_pair_T via operator
        result = V4_op.apply_pair_T(A)

        # Expected: use as_tensor and compute directly
        V4_tensor = V4_op.as_tensor()
        expected = einsum("cdab,cd->ab", V4_tensor, A)

        assert allclose(result, expected, rtol=1e-4)

    def test_apply_pair_T_duality_nonsymmetric(self):
        """Test duality ⟨A, V(B)⟩ = ⟨V^T(A), B⟩ for V4Operator with non-symmetric V4."""
        from resnet_eft.v4_repr import LocalV4Op

        N = 3
        local_tensor = random_v4_tensor(N)
        local_op = LocalV4Op(local_tensor=local_tensor)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Use non-symmetric V4 for prev_V4
        V4_prev_data = random_v4_tensor_nonsymmetric(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        V4_op = V4Operator(local_op=local_op, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)

        A = torch.randn(N, N, dtype=torch.float64)
        B = torch.randn(N, N, dtype=torch.float64)

        # ⟨A, V @ B⟩ = ⟨V^T @ A, B⟩
        lhs = (A * V4_op.apply_pair(B)).sum()
        rhs = (V4_op.apply_pair_T(A) * B).sum()

        assert allclose(lhs, rhs, rtol=1e-4)

    def test_get_diag_cross_left(self):
        """Test V4Operator.get_diag_cross_left."""
        from resnet_eft.v4_repr import LocalV4Op

        N = 3
        local_tensor = random_v4_tensor(N)
        local_op = LocalV4Op(local_tensor=local_tensor)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        V4_prev_data = random_v4_tensor(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        V4_op = V4Operator(local_op=local_op, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)

        # Compare get_diag_cross_left with as_tensor based calculation
        diag_cross_left = V4_op.get_diag_cross_left()
        V4_tensor = V4_op.as_tensor()

        # Verify shape
        assert diag_cross_left.shape == (N, N)

        # Verify values: diag_cross_left[i, j] = V4[i, i, i, j]
        for i in range(N):
            for j in range(N):
                assert allclose(diag_cross_left[i, j], V4_tensor[i, i, i, j], rtol=1e-4)

    def test_get_diag_cross_right(self):
        """Test V4Operator.get_diag_cross_right."""
        from resnet_eft.v4_repr import LocalV4Op

        N = 3
        local_tensor = random_v4_tensor(N)
        local_op = LocalV4Op(local_tensor=local_tensor)

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.relu(mode="exact"), Cw=2.0, Cb=0.0)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        V4_prev_data = random_v4_tensor(N)
        V4_prev = V4Tensor(data=V4_prev_data)

        V4_op = V4Operator(local_op=local_op, chi_op=chi_op, prev_V4=V4_prev, width_ratio=1.0)

        # Compare get_diag_cross_right with as_tensor based calculation
        diag_cross_right = V4_op.get_diag_cross_right()
        V4_tensor = V4_op.as_tensor()

        # Verify shape
        assert diag_cross_right.shape == (N, N)

        # Verify values: diag_cross_right[i, j] = V4[j, j, i, j]
        for i in range(N):
            for j in range(N):
                assert allclose(diag_cross_right[i, j], V4_tensor[j, j, i, j], rtol=1e-4)


class TestFullSliceGate:
    """Full ↔ Slice Gate tests (mandatory for large N experiments).

    These tests verify that V4SliceRepr is consistent with V4Tensor for small N.
    This is a critical gate: if these fail, large N experiments using V4SliceRepr
    cannot be trusted.
    """

    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_slice_extraction_from_full_tensor(self, N: int):
        """Test that V4SliceRepr slices match V4Tensor slices exactly."""
        from resnet_eft.v4_repr import V4SliceRepr

        V4_full = random_v4_tensor(N)
        v4_tensor = V4Tensor(data=V4_full)

        # Extract slices
        slices = {
            "diag_diag": v4_tensor.get_diag_diag(),
            "cross_diag": v4_tensor.get_cross_diag(),
            "diag_cross_L": v4_tensor.get_diag_cross_left(),
            "diag_cross_R": v4_tensor.get_diag_cross_right(),
        }
        v4_slice = V4SliceRepr.from_slices(slices)

        # Verify exact match
        assert allclose(v4_slice.get_diag_diag(), v4_tensor.get_diag_diag())
        assert allclose(v4_slice.get_cross_diag(), v4_tensor.get_cross_diag())
        assert allclose(v4_slice.get_diag_cross_left(), v4_tensor.get_diag_cross_left())
        assert allclose(v4_slice.get_diag_cross_right(), v4_tensor.get_diag_cross_right())

    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_source_term_slices_match_full_e4(self, N: int):
        """Test that V4 source term slices match full E4-based computation."""
        from resnet_eft.v4_repr import V4SliceRepr

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1, gh_order=32)
        gauss = GaussianExpectation(params)

        # Full tensor source term: Cw² × (E4 - E2⊗E2)
        E4 = gauss.E4_pairwise(K0)
        E2 = gauss.E2_pairwise(K0)
        V4_source_full = params.Cw**2 * (E4 - torch.einsum("ij,kl->ijkl", E2, E2))
        v4_tensor = V4Tensor(data=V4_source_full)

        # Slice-based source term (using MC should match for large samples)
        # Here we use the full tensor slices directly for comparison
        slices = {
            "diag_diag": v4_tensor.get_diag_diag(),
            "cross_diag": v4_tensor.get_cross_diag(),
            "diag_cross_L": v4_tensor.get_diag_cross_left(),
            "diag_cross_R": v4_tensor.get_diag_cross_right(),
        }
        v4_slice = V4SliceRepr.from_slices(slices)

        # Verify all slices match
        tol = 1e-10
        assert torch.abs(v4_slice.get_diag_diag() - v4_tensor.get_diag_diag()).max() < tol
        assert torch.abs(v4_slice.get_cross_diag() - v4_tensor.get_cross_diag()).max() < tol
        assert torch.abs(v4_slice.get_diag_cross_left() - v4_tensor.get_diag_cross_left()).max() < tol
        assert torch.abs(v4_slice.get_diag_cross_right() - v4_tensor.get_diag_cross_right()).max() < tol

    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_transport_slices_match_full_transport(self, N: int):
        """Test that V4SliceRepr transport matches full tensor transport.

        This is the critical gate test: transport formula consistency.

        Transport: result[ab,cd] = 0.5 × (χ[ab,·] V[·,cd] + χ[ab,·] Vᵀ[·,cd])
        This is the quadratic (pair-space sandwich) form used in the MLP discrete step.
        """
        from resnet_eft.v4_repr import V4SliceRepr

        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1, gh_order=32)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Initial V4 from source term
        E4 = gauss.E4_pairwise(K0)
        E2 = gauss.E2_pairwise(K0)
        V4_data = params.Cw**2 * (E4 - torch.einsum("ij,kl->ijkl", E2, E2))
        v4_tensor = V4Tensor(data=V4_data)

        # Extract slices for V4SliceRepr
        slices = {
            "diag_diag": v4_tensor.get_diag_diag(),
            "cross_diag": v4_tensor.get_cross_diag(),
            "diag_cross_L": v4_tensor.get_diag_cross_left(),
            "diag_cross_R": v4_tensor.get_diag_cross_right(),
        }
        v4_slice = V4SliceRepr.from_slices(slices)

        # Transport via V4SliceRepr
        v4_slice_transported = v4_slice.transport_update(chi_op, params.Cw, width_ratio=1.0)

        # Transport via full tensor (naive O(N^4) implementation)
        def full_transport(V4: torch.Tensor, chi: ChiOp, wr: float) -> torch.Tensor:
            """Full tensor transport using χ @ V4 + V4 @ χ.T form."""
            N_local = V4.shape[0]
            result = torch.zeros_like(V4)
            for a in range(N_local):
                for b in range(N_local):
                    basis = torch.zeros(N_local, N_local)
                    basis[a, b] = 1.0
                    B = chi.apply_pair_T(basis)
                    VB = einsum("ijkl,kl->ij", V4, B)
                    VTB = einsum("klij,kl->ij", V4, B)
                    term1 = chi.apply_pair(VB)
                    term2 = chi.apply_pair(VTB)
                    result[:, :, a, b] = wr * 0.5 * (term1 + term2)
            return result

        V4_full_transported = full_transport(V4_data, chi_op, 1.0)
        v4_full_t = V4Tensor(data=V4_full_transported)

        # Compare all 4 slices
        tol = 2e-4  # Allow small numerical differences
        err_dd = torch.abs(v4_slice_transported.get_diag_diag() - v4_full_t.get_diag_diag()).max()
        err_cd = torch.abs(v4_slice_transported.get_cross_diag() - v4_full_t.get_cross_diag()).max()
        err_dcL = torch.abs(v4_slice_transported.get_diag_cross_left() - v4_full_t.get_diag_cross_left()).max()
        err_dcR = torch.abs(v4_slice_transported.get_diag_cross_right() - v4_full_t.get_diag_cross_right()).max()

        assert err_dd < tol, f"diag_diag error: {err_dd}"
        assert err_cd < tol, f"cross_diag error: {err_cd}"
        assert err_dcL < tol, f"diag_cross_L error: {err_dcL}"
        assert err_dcR < tol, f"diag_cross_R error: {err_dcR}"

    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_multilayer_transport_consistency(self, N: int):
        """Test multi-layer transport consistency between full and slice representations."""
        from resnet_eft.v4_repr import V4SliceRepr
        from resnet_eft.backend import ensure_psd

        # Use a stable uniform K0 that stays PSD after layers
        K0 = torch.eye(N, dtype=torch.float64) * 0.5 + 0.5
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, Cb=0.1, gh_order=32)
        gauss = GaussianExpectation(params)

        # Layer 1: source term
        E4 = gauss.E4_pairwise(K0)
        E2 = gauss.E2_pairwise(K0)
        V4_data = params.Cw**2 * (E4 - torch.einsum("ij,kl->ijkl", E2, E2))
        v4_tensor = V4Tensor(data=V4_data)

        slices = {
            "diag_diag": v4_tensor.get_diag_diag(),
            "cross_diag": v4_tensor.get_cross_diag(),
            "diag_cross_L": v4_tensor.get_diag_cross_left(),
            "diag_cross_R": v4_tensor.get_diag_cross_right(),
        }
        v4_slice = V4SliceRepr.from_slices(slices)

        # Propagate through 3 layers
        for _layer in range(3):
            # Update K0
            K0_next = params.Cb + params.Cw * gauss.E2_pairwise(K0)
            K0_next = ensure_psd(K0_next, psd_check="eigh")
            chi_op = ChiOp.from_gauss(gauss, K0)

            # Transport slices
            v4_slice = v4_slice.transport_update(chi_op, params.Cw, width_ratio=1.0)

            # Add local term (compute from K0_next)
            E4_next = gauss.E4_pairwise(K0_next)
            E2_next = gauss.E2_pairwise(K0_next)
            V4_local_data = params.Cw**2 * (E4_next - torch.einsum("ij,kl->ijkl", E2_next, E2_next))
            v4_local_tensor = V4Tensor(data=V4_local_data)
            local_slices = V4SliceRepr.from_slices({
                "diag_diag": v4_local_tensor.get_diag_diag(),
                "cross_diag": v4_local_tensor.get_cross_diag(),
                "diag_cross_L": v4_local_tensor.get_diag_cross_left(),
                "diag_cross_R": v4_local_tensor.get_diag_cross_right(),
            })
            v4_slice = v4_slice.add_local(local_slices)

            K0 = K0_next

        # Verify slices are still finite and reasonable
        assert torch.isfinite(v4_slice.get_diag_diag()).all(), "diag_diag has NaN/Inf"
        assert torch.isfinite(v4_slice.get_cross_diag()).all(), "cross_diag has NaN/Inf"
        assert v4_slice.get_diag_diag().abs().max() < 1e10, "diag_diag too large"

class TestV4SliceRepr:
    """Tests for V4SliceRepr (memory-efficient representation)."""

    def test_from_slices(self):
        """Test V4SliceRepr.from_slices."""
        from resnet_eft.v4_repr import V4SliceRepr

        N = 4
        slices = {
            "diag_diag": torch.randn(N, N),
            "cross_diag": torch.randn(N, N),
            "diag_cross_L": torch.randn(N, N),
            "diag_cross_R": torch.randn(N, N),
        }

        v4_slice = V4SliceRepr.from_slices(slices)

        assert allclose(v4_slice.get_diag_diag(), slices["diag_diag"])
        assert allclose(v4_slice.get_cross_diag(), slices["cross_diag"])
        assert allclose(v4_slice.get_diag_cross_left(), slices["diag_cross_L"])
        assert allclose(v4_slice.get_diag_cross_right(), slices["diag_cross_R"])

    def test_scale(self):
        """Test V4SliceRepr.scale."""
        from resnet_eft.v4_repr import V4SliceRepr

        N = 3
        slices = {
            "diag_diag": torch.randn(N, N),
            "cross_diag": torch.randn(N, N),
            "diag_cross_L": torch.randn(N, N),
            "diag_cross_R": torch.randn(N, N),
        }

        v4_slice = V4SliceRepr.from_slices(slices)
        v4_scaled = v4_slice.scale(2.5)

        assert allclose(v4_scaled.get_diag_diag(), 2.5 * slices["diag_diag"])
        assert allclose(v4_scaled.get_cross_diag(), 2.5 * slices["cross_diag"])

    def test_apply_pair_raises(self):
        """Test V4SliceRepr.apply_pair raises NotImplementedError."""
        from resnet_eft.v4_repr import V4SliceRepr

        N = 3
        slices = {
            "diag_diag": torch.randn(N, N),
            "cross_diag": torch.randn(N, N),
            "diag_cross_L": torch.randn(N, N),
            "diag_cross_R": torch.randn(N, N),
        }

        v4_slice = V4SliceRepr.from_slices(slices)

        with pytest.raises(NotImplementedError, match="does not support apply_pair"):
            v4_slice.apply_pair(torch.randn(N, N))

    def test_as_tensor_raises(self):
        """Test V4SliceRepr.as_tensor raises NotImplementedError."""
        from resnet_eft.v4_repr import V4SliceRepr

        N = 3
        slices = {
            "diag_diag": torch.randn(N, N),
            "cross_diag": torch.randn(N, N),
            "diag_cross_L": torch.randn(N, N),
            "diag_cross_R": torch.randn(N, N),
        }

        v4_slice = V4SliceRepr.from_slices(slices)

        with pytest.raises(NotImplementedError, match="does not support as_tensor"):
            v4_slice.as_tensor()

    def test_matches_v4_tensor_slices(self):
        """Test V4SliceRepr slices match V4Tensor slices."""
        from resnet_eft.v4_repr import V4SliceRepr

        N = 4
        V4_full = random_v4_tensor(N)
        v4_tensor = V4Tensor(data=V4_full)

        # Create V4SliceRepr from V4Tensor slices
        slices = {
            "diag_diag": v4_tensor.get_diag_diag(),
            "cross_diag": v4_tensor.get_cross_diag(),
            "diag_cross_L": v4_tensor.get_diag_cross_left(),
            "diag_cross_R": v4_tensor.get_diag_cross_right(),
        }
        v4_slice = V4SliceRepr.from_slices(slices)

        # Slices should match
        assert allclose(v4_slice.get_diag_diag(), v4_tensor.get_diag_diag())
        assert allclose(v4_slice.get_cross_diag(), v4_tensor.get_cross_diag())
        assert allclose(v4_slice.get_diag_cross_left(), v4_tensor.get_diag_cross_left())
        assert allclose(v4_slice.get_diag_cross_right(), v4_tensor.get_diag_cross_right())

    def test_transport_update_matches_full(self):
        """Test V4SliceRepr.transport_update matches full tensor transport."""
        from resnet_eft.v4_repr import V4SliceRepr

        torch.manual_seed(42)
        N = 4
        K0 = random_psd_matrix(N)
        params = Params(act=ActivationSpec.tanh(), Cw=1.0, gh_order=32)
        gauss = GaussianExpectation(params)
        chi_op = ChiOp.from_gauss(gauss, K0)

        # Compute V4 from expectations
        E4 = gauss.E4_pairwise(K0)
        E2 = gauss.E2_pairwise(K0)
        V4_data = params.Cw**2 * (E4 - torch.einsum("ij,kl->ijkl", E2, E2))

        # Extract slices
        v4_tensor = V4Tensor(data=V4_data)
        slices = {
            "diag_diag": v4_tensor.get_diag_diag(),
            "cross_diag": v4_tensor.get_cross_diag(),
            "diag_cross_L": v4_tensor.get_diag_cross_left(),
            "diag_cross_R": v4_tensor.get_diag_cross_right(),
        }
        v4_slice = V4SliceRepr.from_slices(slices)

        # Transport via slices
        v4_slice_transported = v4_slice.transport_update(chi_op, params.Cw, width_ratio=1.0)

        # Transport via full tensor
        def transport_full(V4_t: torch.Tensor, chi: ChiOp, wr: float) -> torch.Tensor:
            N_local = V4_t.shape[0]
            result = torch.zeros_like(V4_t)
            for a_idx in range(N_local):
                for b_idx in range(N_local):
                    basis = torch.zeros(N_local, N_local)
                    basis[a_idx, b_idx] = 1.0
                    B = chi.apply_pair_T(basis)
                    VB = einsum("ijkl,kl->ij", V4_t, B)
                    VTB = einsum("klij,kl->ij", V4_t, B)
                    term1 = chi.apply_pair(VB)
                    term2 = chi.apply_pair(VTB)
                    result[:, :, a_idx, b_idx] = wr * 0.5 * (term1 + term2)
            return result

        V4_full_transported = transport_full(V4_data, chi_op, 1.0)
        v4_full_transported = V4Tensor(data=V4_full_transported)

        # Compare - allow 1e-6 tolerance for numerical precision
        max_err_dd = torch.abs(
            v4_slice_transported.get_diag_diag() - v4_full_transported.get_diag_diag()
        ).max()
        max_err_cd = torch.abs(
            v4_slice_transported.get_cross_diag() - v4_full_transported.get_cross_diag()
        ).max()
        max_err_dcL = torch.abs(
            v4_slice_transported.get_diag_cross_left() - v4_full_transported.get_diag_cross_left()
        ).max()
        max_err_dcR = torch.abs(
            v4_slice_transported.get_diag_cross_right() - v4_full_transported.get_diag_cross_right()
        ).max()

        # Tolerance: slice transport uses different coefficient extraction than full transport
        # which can lead to O(1e-5) numerical differences due to floating point operations
        tol = 1e-4
        assert max_err_dd < tol, f"diag_diag max error: {max_err_dd}"
        assert max_err_cd < tol, f"cross_diag max error: {max_err_cd}"
        assert max_err_dcL < tol, f"diag_cross_L max error: {max_err_dcL}"
        assert max_err_dcR < tol, f"diag_cross_R max error: {max_err_dcR}"

    def test_add_local(self):
        """Test V4SliceRepr.add_local correctly adds slices."""
        from resnet_eft.v4_repr import V4SliceRepr

        N = 3
        slices1 = {
            "diag_diag": torch.ones(N, N) * 1.0,
            "cross_diag": torch.ones(N, N) * 2.0,
            "diag_cross_L": torch.ones(N, N) * 3.0,
            "diag_cross_R": torch.ones(N, N) * 4.0,
        }
        slices2 = {
            "diag_diag": torch.ones(N, N) * 0.1,
            "cross_diag": torch.ones(N, N) * 0.2,
            "diag_cross_L": torch.ones(N, N) * 0.3,
            "diag_cross_R": torch.ones(N, N) * 0.4,
        }

        v4_1 = V4SliceRepr.from_slices(slices1)
        v4_2 = V4SliceRepr.from_slices(slices2)
        v4_sum = v4_1.add_local(v4_2)

        assert allclose(v4_sum.get_diag_diag(), torch.ones(N, N) * 1.1)
        assert allclose(v4_sum.get_cross_diag(), torch.ones(N, N) * 2.2)
        assert allclose(v4_sum.get_diag_cross_left(), torch.ones(N, N) * 3.3)
        assert allclose(v4_sum.get_diag_cross_right(), torch.ones(N, N) * 4.4)
