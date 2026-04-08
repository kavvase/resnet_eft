"""Tests for backend utilities."""

import pytest
import torch

from resnet_eft.backend import (
    allclose,
    diag_embed,
    diagonal,
    einsum,
    ensure_psd,
    symmetrize,
    zeros,
)


class TestDiagEmbed:
    """Tests for diag_embed."""

    def test_basic(self):
        """Test basic diag_embed."""
        v = torch.tensor([1.0, 2.0, 3.0])
        result = diag_embed(v)
        expected = torch.diag(v)
        assert allclose(result, expected)

    def test_shape(self):
        """Test output shape."""
        v = torch.randn(5)
        result = diag_embed(v)
        assert result.shape == (5, 5)


class TestDiagonal:
    """Tests for diagonal."""

    def test_basic(self):
        """Test basic diagonal extraction."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = diagonal(A)
        expected = torch.tensor([1.0, 4.0])
        assert allclose(result, expected)

    def test_shape(self):
        """Test output shape."""
        A = torch.randn(5, 5)
        result = diagonal(A)
        assert result.shape == (5,)


class TestSymmetrize:
    """Tests for symmetrize."""

    def test_basic(self):
        """Test basic symmetrization."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = symmetrize(A)
        expected = torch.tensor([[1.0, 2.5], [2.5, 4.0]])
        assert allclose(result, expected)

    def test_already_symmetric(self):
        """Test symmetrization of already symmetric matrix."""
        A = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        result = symmetrize(A)
        assert allclose(result, A)

    def test_result_is_symmetric(self):
        """Test that result is always symmetric."""
        A = torch.randn(5, 5)
        result = symmetrize(A)
        assert allclose(result, result.T)


class TestEnsurePsd:
    """Tests for ensure_psd."""

    def test_psd_matrix_passes(self):
        """Test that PSD matrix passes."""
        A = torch.eye(3)
        result = ensure_psd(A, psd_check="cheap")
        assert allclose(result, A)

    def test_negative_diagonal_fails(self):
        """Test that negative diagonal fails."""
        A = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        with pytest.raises(ValueError, match="negative diagonal"):
            ensure_psd(A, psd_check="cheap")

    def test_none_mode_passes(self):
        """Test that 'none' mode always passes."""
        A = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        result = ensure_psd(A, psd_check="none")
        assert allclose(result, A)


class TestZeros:
    """Tests for zeros."""

    def test_shape(self):
        """Test zeros shape."""
        result = zeros((3, 4))
        assert result.shape == (3, 4)

    def test_values(self):
        """Test all values are zero."""
        result = zeros((3, 4))
        assert (result == 0).all()


class TestEinsum:
    """Tests for einsum."""

    def test_matrix_multiply(self):
        """Test matrix multiplication."""
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        result = einsum("ij,jk->ik", A, B)
        expected = A @ B
        assert allclose(result, expected)

    def test_trace(self):
        """Test trace computation."""
        A = torch.randn(4, 4)
        result = einsum("ii->", A)
        expected = A.trace()
        assert allclose(result, expected)
