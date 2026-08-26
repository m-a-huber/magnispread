import pytest
import torch

from magnispread.functional import magnitude, spread, spread_dim
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
    distances = distances.fill_diagonal_(0.0)

    loss_from_points = magnitude(X, metric="euclidean", scale=0.75)
    loss_from_distances = magnitude(
        distances, metric="precomputed", scale=0.75
    )

    assert torch.allclose(loss_from_points, loss_from_distances)


def test_magnitude_supports_precomputed_cosine_distances():
    X = torch.randn(7, 5)
    distances = pairwise_cosine_distance(X, X)
    distances = distances.fill_diagonal_(0.0)

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
    distances = distances.fill_diagonal_(0.0)

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
    distances = distances.fill_diagonal_(0.0)

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


@pytest.mark.parametrize("function", [magnitude, spread, spread_dim])
def test_functionals_reject_empty_point_clouds(function):
    X = torch.randn(0, 4)

    with pytest.raises(ValueError, match="at least one point"):
        function(X)


def test_spread_dim_returns_scalar_and_backward():
    X = torch.randn(8, 4, requires_grad=True)
    loss = spread_dim(X, metric="euclidean")

    assert loss.ndim == 0
    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_spread_dim_supports_cosine_metric():
    X = torch.randn(10, 6, requires_grad=True)
    loss = spread_dim(X, metric="cosine", scale=0.5)

    assert torch.isfinite(loss)
    loss.backward()
    assert X.grad is not None


def test_spread_dim_matches_spread_scale_derivative():
    X = torch.randn(6, 3, dtype=torch.float64)
    scale = 0.75
    distances = torch.cdist(X, X, p=2)
    similarity = torch.exp(-scale * distances)
    row_sums = similarity.sum(dim=1)
    spread_value = (1 / row_sums).sum()
    spread_derivative = (
        (distances * similarity).sum(dim=1) / row_sums.square()
    ).sum()
    expected = scale / spread_value * spread_derivative

    assert torch.allclose(
        spread_dim(X, scale=scale, use_double_precision=True),
        expected.to(dtype=torch.float32),
        rtol=1e-4,
        atol=1e-5,
    )


def test_spread_dim_supports_precomputed_distances():
    X = torch.randn(7, 5)
    distances = torch.cdist(X, X, p=2)
    distances = distances.fill_diagonal_(0.0)

    result_from_points = spread_dim(X, metric="euclidean", scale=0.75)
    result_from_distances = spread_dim(
        distances,
        metric="precomputed",
        scale=0.75,
    )

    assert torch.allclose(result_from_points, result_from_distances)


@pytest.mark.parametrize("use_double_precision", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_spread_dim_always_returns_float32(use_double_precision, input_dtype):
    X = torch.randn(6, 3).to(dtype=input_dtype)

    assert (
        spread_dim(
            X,
            use_double_precision=use_double_precision,
        ).dtype
        == torch.float32
    )


def test_spread_dim_casts_int_input_and_returns_float32():
    X = torch.randint(0, 5, (6, 3))

    assert spread_dim(X).dtype == torch.float32


def test_spread_dim_rejects_invalid_metric():
    X = torch.randn(4, 3)

    with pytest.raises(ValueError, match="metric"):
        spread_dim(X, metric="manhattan")


def test_spread_dim_requires_2d_input():
    X = torch.randn(4, 3, 2)

    with pytest.raises(ValueError, match="2D"):
        spread_dim(X)


def test_spread_dim_requires_square_precomputed_input():
    X = torch.randn(4, 3)

    with pytest.raises(ValueError, match="square"):
        spread_dim(X, metric="precomputed")


def test_spread_dim_validates_scale():
    X = torch.randn(4, 3)

    with pytest.raises(ValueError, match="`scale`"):
        spread_dim(X, scale=0)


def test_spread_dim_uses_row_sums_without_symmetrization():
    distances = torch.tensor(
        [
            [0.1, 0.5, 1.0],
            [0.2, 0.3, 0.7],
            [0.4, 0.8, 0.6],
        ]
    )
    similarity = torch.exp(-distances)
    row_sums = similarity.sum(dim=1)
    expected_spread = (1 / row_sums).sum()
    expected = (
        1.0
        / expected_spread
        * (distances * similarity).sum(dim=1).div(row_sums**2).sum()
    )

    result = spread_dim(
        distances,
        metric="precomputed",
        symmetrize=False,
    )

    assert torch.allclose(result, expected.to(dtype=torch.float32))


def test_magnitude_and_spread_stay_finite_at_large_scale_float32():
    rng = torch.Generator().manual_seed(42)
    X = torch.randn(50, 2, generator=rng)

    loss_magnitude = magnitude(X, metric="euclidean", scale=1e5)
    loss_spread = spread(X, metric="euclidean", scale=1e5)

    assert torch.isfinite(loss_magnitude)
    assert torch.isfinite(loss_spread)
    assert 0.0 < loss_magnitude <= X.shape[0] + 1e-3


def test_magnitude_no_cholesky_fallback_at_large_scale(recwarn):
    X = torch.randn(20, 3)

    magnitude(X, metric="euclidean", scale=1e4, solver="auto")

    assert len(recwarn) == 0


@pytest.mark.parametrize("use_double_precision", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_magnitude_and_spread_always_return_float32(
    use_double_precision, input_dtype
):
    X = torch.randn(6, 3).to(dtype=input_dtype)

    assert magnitude(X, use_double_precision=use_double_precision).dtype == (
        torch.float32
    )
    assert spread(X, use_double_precision=use_double_precision).dtype == (
        torch.float32
    )


def test_magnitude_and_spread_cast_int_input_and_return_float32():
    X = torch.randint(0, 5, (6, 3))

    assert magnitude(X).dtype == torch.float32
    assert spread(X).dtype == torch.float32


def test_magnitude_and_spread_symmetrize_defaults_to_true():
    X = torch.randn(8, 4)

    assert torch.allclose(
        magnitude(X, metric="euclidean"),
        magnitude(X, metric="euclidean", symmetrize=True),
    )
    assert torch.allclose(
        spread(X, metric="euclidean"),
        spread(X, metric="euclidean", symmetrize=True),
    )


def test_magnitude_and_spread_support_symmetrize_false():
    X = torch.randn(8, 4)

    loss_magnitude = magnitude(X, metric="euclidean", symmetrize=False)
    loss_spread = spread(X, metric="euclidean", symmetrize=False)

    assert torch.isfinite(loss_magnitude)
    assert torch.isfinite(loss_spread)


def test_magnitude_and_spread_do_not_alter_precomputed_matrix():
    distances = torch.tensor(
        [
            [1e-3, 0.5, 1.0],
            [0.5, 1e-3, -1e-4],
            [1.0, -1e-4, 1e-3],
        ]
    )

    expected_similarity = torch.exp(-1.0 * distances)
    expected_magnitude = torch.linalg.inv(expected_similarity).sum()
    expected_spread = (1 / expected_similarity.sum(dim=1)).sum()

    loss_magnitude = magnitude(
        distances,
        metric="precomputed",
        scale=1.0,
        solver="inverse",
        jitter=0.0,
    )
    loss_spread = spread(distances, metric="precomputed", scale=1.0)

    assert torch.allclose(loss_magnitude, expected_magnitude.to(torch.float32))
    assert torch.allclose(loss_spread, expected_spread.to(torch.float32))
