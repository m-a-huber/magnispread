import pytest
import torch

from magnispread.validation import validate_inputs


def test_requires_2d_input():
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        validate_inputs(X, metric="euclidean", scale=1.0)


def test_rejects_empty_point_cloud():
    X = torch.randn(0, 4)
    with pytest.raises(ValueError, match="at least one point"):
        validate_inputs(X, metric="euclidean", scale=1.0)


def test_rejects_invalid_metric():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        validate_inputs(X, metric="manhattan", scale=1.0)


def test_precomputed_requires_square_matrix():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="square"):
        validate_inputs(X, metric="precomputed", scale=1.0)


def test_precomputed_accepts_square_matrix():
    X = torch.randn(4, 4)
    validate_inputs(X, metric="precomputed", scale=1.0)


def test_rejects_non_positive_scale():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`scale`"):
        validate_inputs(X, metric="euclidean", scale=0.0)
    with pytest.raises(ValueError, match="`scale`"):
        validate_inputs(X, metric="euclidean", scale=-1.0)


def test_rejects_negative_jitter():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`jitter`"):
        validate_inputs(X, metric="euclidean", scale=1.0, jitter=-1e-6)


def test_jitter_not_validated_when_none():
    X = torch.randn(4, 3)
    validate_inputs(X, metric="euclidean", scale=1.0, jitter=None)


def test_rejects_invalid_solver():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`solver`"):
        validate_inputs(X, metric="euclidean", scale=1.0, solver="svd")


def test_solver_not_validated_when_none():
    X = torch.randn(4, 3)
    validate_inputs(X, metric="euclidean", scale=1.0, solver=None)
