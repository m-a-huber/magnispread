import pytest
import torch

from magnispread.functional import magnitude, spread, spread_dim
from magnispread.metrics import pairwise_cosine_distance

FUNCTIONS = [magnitude, spread, spread_dim]


# ---------------------------------------------------------------------------
# Shared input validation, as documented in each function's docstring.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
def test_rejects_non_2d_input(function):
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        function(X)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_rejects_empty_point_cloud(function):
    X = torch.randn(0, 4)
    with pytest.raises(ValueError, match="at least one point"):
        function(X)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_rejects_invalid_metric(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        function(X, metric="manhattan")


@pytest.mark.parametrize("function", FUNCTIONS)
def test_precomputed_requires_square_matrix(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="square"):
        function(X, metric="precomputed")


@pytest.mark.parametrize("function", FUNCTIONS)
def test_rejects_non_positive_scale(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`scale`"):
        function(X, scale=0.0)


def test_magnitude_rejects_negative_jitter():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`jitter`"):
        magnitude(X, jitter=-1e-6)


def test_magnitude_rejects_invalid_solver():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`solver`"):
        magnitude(X, solver="svd")


# ---------------------------------------------------------------------------
# General shape/gradient behavior.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_returns_scalar_and_supports_backward(function, metric):
    X = torch.randn(8, 4, requires_grad=True)
    result = function(X, metric=metric)

    assert result.ndim == 0
    assert torch.isfinite(result)

    result.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


# ---------------------------------------------------------------------------
# `metric="precomputed"` equivalence with the on-the-fly computation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
def test_precomputed_euclidean_matches_on_the_fly(function):
    X = torch.randn(7, 5)
    distances = torch.cdist(X, X, p=2).fill_diagonal_(0.0)

    from_points = function(X, metric="euclidean", scale=0.75)
    from_distances = function(distances, metric="precomputed", scale=0.75)

    assert torch.allclose(from_points, from_distances)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_precomputed_cosine_matches_on_the_fly(function):
    X = torch.randn(7, 5)
    distances = pairwise_cosine_distance(X, X).fill_diagonal_(0.0)

    from_points = function(X, metric="cosine", scale=0.75)
    from_distances = function(distances, metric="precomputed", scale=0.75)

    assert torch.allclose(from_points, from_distances)


# ---------------------------------------------------------------------------
# dtype handling. README: "The returned tensor's dtype matches `X`'s
# floating-point dtype (or `float32` if `X` is not floating-point),
# regardless of `use_double_precision`."
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("use_double_precision", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_output_dtype_matches_input_dtype(
    function, use_double_precision, input_dtype
):
    X = torch.randn(6, 3).to(dtype=input_dtype)

    result = function(X, use_double_precision=use_double_precision)

    assert result.dtype == input_dtype


@pytest.mark.parametrize("function", FUNCTIONS)
def test_non_floating_point_input_returns_float32(function):
    X = torch.randint(0, 5, (6, 3))

    assert function(X).dtype == torch.float32


# ---------------------------------------------------------------------------
# `symmetrize`. README: defaults to `True`; controls whether the similarity
# matrix is symmetrized before use.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
def test_symmetrize_defaults_to_true(function):
    X = torch.randn(8, 4)

    assert torch.allclose(function(X), function(X, symmetrize=True))


@pytest.mark.parametrize("function", FUNCTIONS)
def test_symmetrize_changes_result_for_asymmetric_precomputed_input(function):
    D = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [0.2, 0.0, 0.7],
            [0.4, 0.8, 0.0],
        ]
    )

    symmetrized = function(D, metric="precomputed", symmetrize=True)
    not_symmetrized = function(D, metric="precomputed", symmetrize=False)

    assert not torch.allclose(symmetrized, not_symmetrized)


# ---------------------------------------------------------------------------
# `force_diagonal`. README / docstrings: defaults to `True` for
# `metric="euclidean"`/`"cosine"`, and to `False` for `metric="precomputed"`
# (the matrix is used as supplied); either default is overridable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_force_diagonal_defaults_to_true_for_computed_metrics(
    function, metric
):
    X = torch.randn(8, 4)

    assert torch.allclose(
        function(X, metric=metric),
        function(X, metric=metric, force_diagonal=True),
    )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_force_diagonal_defaults_to_false_for_precomputed(function):
    # A "wrong" diagonal that a real self-distance would never have.
    D = torch.tensor([[0.1, 2.0], [2.0, 0.3]])

    default = function(D, metric="precomputed")
    explicit_false = function(D, metric="precomputed", force_diagonal=False)
    explicit_true = function(D, metric="precomputed", force_diagonal=True)

    assert torch.allclose(default, explicit_false)
    assert not torch.allclose(default, explicit_true)


# ---------------------------------------------------------------------------
# `jitter` (magnitude only). README: defaults to `1e-6`, guards against a
# singular/ill-conditioned similarity matrix.
# ---------------------------------------------------------------------------


def test_magnitude_jitter_defaults_to_1e_minus_6():
    X = torch.randn(8, 4)

    assert torch.equal(magnitude(X), magnitude(X, jitter=1e-6))


def test_magnitude_jitter_guards_against_singular_similarity_matrix():
    # An exact duplicate point makes the similarity matrix exactly singular
    # (two identical rows/columns).
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    X_with_duplicate = torch.cat([X, X[:1]])

    result = magnitude(X_with_duplicate)  # default jitter=1e-6
    assert torch.isfinite(result)

    with (
        pytest.warns(UserWarning, match="linsolve"),
        pytest.raises(torch.linalg.LinAlgError),
    ):
        magnitude(X_with_duplicate, jitter=0.0)


# ---------------------------------------------------------------------------
# `solver` (magnitude only).
# ---------------------------------------------------------------------------


def test_magnitude_auto_matches_cholesky_when_well_conditioned(recwarn):
    X = torch.randn(8, 4)

    result_auto = magnitude(X, solver="auto")
    result_cholesky = magnitude(X, solver="cholesky")

    assert torch.allclose(result_auto, result_cholesky)
    assert len(recwarn) == 0


def test_magnitude_auto_falls_back_to_linsolve_on_cholesky_failure(
    monkeypatch,
):
    X = torch.randn(6, 3)
    original_cholesky_ex = torch.linalg.cholesky_ex

    def _failing_cholesky_ex(*args, **kwargs):
        L, info = original_cholesky_ex(*args, **kwargs)
        return L, torch.ones_like(info)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", _failing_cholesky_ex)

    with pytest.warns(UserWarning, match="linsolve"):
        result_auto = magnitude(X, solver="auto")

    result_linsolve = magnitude(X, solver="linsolve")
    assert torch.allclose(result_auto, result_linsolve)


def test_magnitude_solver_variants_agree():
    X = torch.randn(6, 3, dtype=torch.float64)

    results = {
        solver: magnitude(X, solver=solver, jitter=0.0)
        for solver in ["cholesky", "linsolve", "inverse"]
    }

    assert torch.allclose(results["cholesky"], results["linsolve"])
    assert torch.allclose(results["cholesky"], results["inverse"])


# ---------------------------------------------------------------------------
# `spread_dim` math. Docstring: "the logarithmic derivative of spread with
# respect to scale, namely scale / spread * d(spread) / d(scale)".
# ---------------------------------------------------------------------------


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
        expected,
        rtol=1e-4,
        atol=1e-6,
    )
