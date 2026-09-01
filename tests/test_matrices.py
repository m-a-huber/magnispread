import pytest
import torch

from magnispread.matrices import get_distance_matrix, get_similarity_matrix
from magnispread.metrics import pairwise_cosine_distance


def test_get_distance_matrix_euclidean_matches_cdist():
    X = torch.randn(6, 4)
    expected = torch.cdist(X, X, p=2)
    result = get_distance_matrix(
        X, metric="euclidean", use_double_precision=False
    )
    assert torch.allclose(result, expected)


def test_get_distance_matrix_cosine_matches_pairwise_cosine_distance():
    X = torch.randn(6, 4)
    expected = pairwise_cosine_distance(X, X)
    result = get_distance_matrix(
        X, metric="cosine", use_double_precision=False
    )
    assert torch.allclose(result, expected)


def test_get_distance_matrix_precomputed_returns_input_unchanged():
    # Includes a negative entry, which a real distance never has, to confirm
    # the matrix is used exactly as supplied (no clamping).
    D = torch.tensor([[0.0, -1e-4], [-1e-4, 0.0]])
    result = get_distance_matrix(
        D, metric="precomputed", use_double_precision=False
    )
    assert torch.equal(result, D)


def test_get_distance_matrix_rejects_invalid_metric():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        get_distance_matrix(X, metric="manhattan", use_double_precision=False)


def test_get_distance_matrix_casts_non_floating_point_input():
    X = torch.randint(0, 5, (4, 3))
    result = get_distance_matrix(
        X, metric="euclidean", use_double_precision=False
    )
    assert result.dtype == torch.float32


def test_get_distance_matrix_use_double_precision():
    X = torch.randn(4, 3, dtype=torch.float32)
    result = get_distance_matrix(
        X, metric="euclidean", use_double_precision=True
    )
    assert result.dtype == torch.float64


def test_get_similarity_matrix_formula():
    D = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = get_similarity_matrix(
        D, scale=2.0, symmetrize=False, force_diagonal=False
    )
    expected = torch.exp(-2.0 * D)
    assert torch.allclose(result, expected)


def test_get_similarity_matrix_symmetrize():
    D = torch.tensor([[0.0, 1.0], [2.0, 0.0]])  # asymmetric
    result = get_similarity_matrix(
        D, scale=1.0, symmetrize=True, force_diagonal=False
    )
    assert torch.allclose(result, result.mT)


def test_get_similarity_matrix_without_symmetrize_stays_asymmetric():
    D = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    result = get_similarity_matrix(
        D, scale=1.0, symmetrize=False, force_diagonal=False
    )
    assert not torch.allclose(result, result.mT)


def test_get_similarity_matrix_force_diagonal():
    D = torch.tensor([[0.5, 1.0], [1.0, 0.5]])  # nonzero diagonal
    result = get_similarity_matrix(
        D, scale=1.0, symmetrize=False, force_diagonal=True
    )
    assert torch.allclose(torch.diagonal(result), torch.ones(2))


def test_get_similarity_matrix_without_force_diag_leaves_diag_as_supplied():
    D = torch.tensor([[0.5, 1.0], [1.0, 0.5]])
    result = get_similarity_matrix(
        D, scale=1.0, symmetrize=False, force_diagonal=False
    )
    expected_diag = torch.exp(-1.0 * torch.tensor([0.5, 0.5]))
    assert torch.allclose(torch.diagonal(result), expected_diag)
    assert not torch.allclose(torch.diagonal(result), torch.ones(2))
