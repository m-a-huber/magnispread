import pytest
import torch

from magnispread.functional import magnitude, spread
from magnispread.metrics import pairwise_cosine_distance


def test_magnitude_returns_scalar_and_backward():
    X = torch.randn(8, 4, requires_grad=True)
    loss = magnitude(X, metric="euclidean")

    assert loss.ndim == 0
    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_magnitude_supports_cosine_metric():
    X = torch.randn(10, 6, requires_grad=True)
    loss = magnitude(X, metric="cosine", scale=0.5)

    assert torch.isfinite(loss)
    loss.backward()
    assert X.grad is not None


def test_magnitude_supports_inverse_solver():
    X = torch.randn(6, 3, requires_grad=True)
    loss = magnitude(X, metric="euclidean", solver="inverse", jitter=1e-4)

    assert torch.isfinite(loss)
    loss.backward()
    assert X.grad is not None


def test_magnitude_auto_solver_matches_cholesky_when_well_conditioned(recwarn):
    X = torch.randn(8, 4)

    result_auto = magnitude(X, metric="euclidean", solver="auto")
    result_cholesky = magnitude(X, metric="euclidean", solver="cholesky")

    assert torch.allclose(result_auto, result_cholesky)
    assert len(recwarn) == 0


def test_magnitude_auto_solver_falls_back_to_inverse_on_cholesky_failure(
    monkeypatch,
):
    X = torch.randn(6, 3)
    original_cholesky_ex = torch.linalg.cholesky_ex

    def _failing_cholesky_ex(*args, **kwargs):
        L, info = original_cholesky_ex(*args, **kwargs)
        return L, torch.ones_like(info)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", _failing_cholesky_ex)

    with pytest.warns(UserWarning, match="falling back"):
        result_auto = magnitude(X, metric="euclidean", solver="auto")

    result_inverse = magnitude(X, metric="euclidean", solver="inverse")

    assert torch.allclose(result_auto, result_inverse)


def test_magnitude_auto_solver_backward_after_fallback(monkeypatch):
    X = torch.randn(6, 3, requires_grad=True)
    original_cholesky_ex = torch.linalg.cholesky_ex

    def _failing_cholesky_ex(*args, **kwargs):
        L, info = original_cholesky_ex(*args, **kwargs)
        return L, torch.ones_like(info)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", _failing_cholesky_ex)

    with pytest.warns(UserWarning):
        loss = magnitude(X, metric="euclidean", solver="auto")

    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_magnitude_supports_precomputed_distances():
    X = torch.randn(7, 5)
    distances = torch.cdist(X, X, p=2)

    loss_from_points = magnitude(X, metric="euclidean", scale=0.75)
    loss_from_distances = magnitude(
        distances, metric="precomputed", scale=0.75
    )

    assert torch.allclose(loss_from_points, loss_from_distances)


def test_magnitude_supports_precomputed_cosine_distances():
    X = torch.randn(7, 5)
    distances = pairwise_cosine_distance(X, X)

    loss_from_points = magnitude(X, metric="cosine", scale=0.75)
    loss_from_distances = magnitude(
        distances, metric="precomputed", scale=0.75
    )

    assert torch.allclose(loss_from_points, loss_from_distances)


def test_magnitude_backward_with_precomputed_distances():
    distances = torch.zeros(6, 6, requires_grad=True)
    loss = magnitude(distances, metric="precomputed", jitter=1e-4)

    loss.backward()
    assert distances.grad is not None
    assert torch.isfinite(distances.grad).all()


def test_magnitude_rejects_invalid_metric():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        magnitude(X, metric="manhattan")


def test_magnitude_requires_2d_input():
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        magnitude(X)


def test_magnitude_requires_square_precomputed_input():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="square"):
        magnitude(X, metric="precomputed")


def test_magnitude_rejects_invalid_solver():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="solver"):
        magnitude(X, solver="svd")


def test_magnitude_validates_t_and_jitter():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`scale`"):
        magnitude(X, scale=0)

    with pytest.raises(ValueError, match="`jitter`"):
        magnitude(X, jitter=-1e-6)


def test_spread_returns_scalar_and_backward():
    X = torch.randn(8, 4, requires_grad=True)
    loss = spread(X, metric="euclidean")

    assert loss.ndim == 0
    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_spread_supports_cosine_metric():
    X = torch.randn(10, 6, requires_grad=True)
    loss = spread(X, metric="cosine", scale=0.5)

    assert torch.isfinite(loss)
    loss.backward()
    assert X.grad is not None


def test_spread_supports_precomputed_distances():
    X = torch.randn(7, 5)
    distances = torch.cdist(X, X, p=2)

    loss_from_points = spread(X, metric="euclidean", scale=0.75)
    loss_from_distances = spread(
        distances,
        metric="precomputed",
        scale=0.75,
    )

    assert torch.allclose(loss_from_points, loss_from_distances)


def test_spread_supports_precomputed_cosine_distances():
    X = torch.randn(7, 5)
    distances = pairwise_cosine_distance(X, X)

    loss_from_points = spread(X, metric="cosine", scale=0.75)
    loss_from_distances = spread(
        distances,
        metric="precomputed",
        scale=0.75,
    )

    assert torch.allclose(loss_from_points, loss_from_distances)


def test_spread_backward_with_precomputed_distances():
    distances = torch.zeros(6, 6, requires_grad=True)
    loss = spread(distances, metric="precomputed")

    loss.backward()
    assert distances.grad is not None
    assert torch.isfinite(distances.grad).all()


def test_spread_rejects_invalid_metric():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        spread(X, metric="manhattan")


def test_spread_requires_2d_input():
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        spread(X)


def test_spread_requires_square_precomputed_input():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="square"):
        spread(X, metric="precomputed")


def test_spread_validates_t():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`scale`"):
        spread(X, scale=0)


def test_magnitude_and_spread_of_empty_point_cloud_are_zero():
    X = torch.randn(0, 4)

    assert magnitude(X) == 0.0  # ruff: ignore[float-equality-comparison]
    assert spread(X) == 0.0  # ruff: ignore[float-equality-comparison]
