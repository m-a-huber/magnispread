import pytest
import torch

from magnispread.functional import (
    DimMaxResult,
    magnitude,
    magnitude_dim,
    magnitude_dim_max,
    spread,
    spread_dim,
    spread_dim_max,
)
from magnispread.metrics import pairwise_cosine_distance

FUNCTIONS = [magnitude, magnitude_dim, spread, spread_dim]
JITTER_SOLVER_FUNCTIONS = [magnitude, magnitude_dim]
MAX_FUNCTIONS = [magnitude_dim_max, spread_dim_max]
JITTER_SOLVER_MAX_FUNCTIONS = [magnitude_dim_max]


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


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_rejects_negative_jitter(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`jitter`"):
        function(X, jitter=-1e-6)


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_rejects_invalid_solver(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`solver`"):
        function(X, solver="svd")


# ---------------------------------------------------------------------------
# Shared input validation for the scale-agnostic ("_max") functions. These
# have no `scale` parameter, so the validation tests above that pass
# `scale=...` don't apply, but the rest of `validate_inputs`'s checks do.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
def test_max_functions_reject_non_2d_input(function):
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        function(X)


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
def test_max_functions_reject_empty_point_cloud(function):
    X = torch.randn(0, 4)
    with pytest.raises(ValueError, match="at least one point"):
        function(X)


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
def test_max_functions_reject_invalid_metric(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        function(X, metric="manhattan")


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
def test_max_functions_precomputed_requires_square_matrix(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="square"):
        function(X, metric="precomputed")


@pytest.mark.parametrize("function", JITTER_SOLVER_MAX_FUNCTIONS)
def test_max_functions_reject_negative_jitter(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`jitter`"):
        function(X, jitter=-1e-6)


@pytest.mark.parametrize("function", JITTER_SOLVER_MAX_FUNCTIONS)
def test_max_functions_reject_invalid_solver(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`solver`"):
        function(X, solver="svd")


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
def test_max_functions_reject_invalid_search_params(function):
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`t_min`"):
        function(X, t_min=-1.0)
    with pytest.raises(ValueError, match="`t_max`"):
        function(X, t_min=1.0, t_max=0.5)
    with pytest.raises(ValueError, match="`num_coarse_steps`"):
        function(X, num_coarse_steps=2)
    with pytest.raises(ValueError, match="`max_refine_iter`"):
        function(X, max_refine_iter=-1)
    with pytest.raises(ValueError, match="`log_tol`"):
        function(X, log_tol=0.0)


# ---------------------------------------------------------------------------
# Scale-agnostic ("_max") functions: return shape and gradient behavior.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_max_functions_return_dim_max_result(function, metric):
    X = torch.randn(8, 4)

    result = function(X, metric=metric, t_min=1e-3, t_max=1e3)

    assert isinstance(result, DimMaxResult)
    assert result.dim.ndim == 0
    assert torch.isfinite(result.dim)
    assert isinstance(result.scale, float)
    assert 1e-3 <= result.scale <= 1e3


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_max_functions_support_backward(function, metric):
    X = torch.randn(8, 4, requires_grad=True)

    result = function(X, metric=metric)
    result.dim.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


@pytest.mark.parametrize("function", MAX_FUNCTIONS)
def test_max_functions_warn_on_boundary_search_range(function):
    # The 2-point space's magnitude/spread dimension peaks around t ~= 1.28
    # (see test_math_correctness.py), far below this deliberately narrow,
    # high search range.
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    with pytest.warns(UserWarning, match="boundary"):
        function(X, t_min=50.0, t_max=100.0)


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
# `jitter` (magnitude and magnitude_dim only). README: defaults to `1e-6`,
# guards against a singular/ill-conditioned similarity matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_jitter_defaults_to_1e_minus_6(function):
    X = torch.randn(8, 4)

    assert torch.equal(function(X), function(X, jitter=1e-6))


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_jitter_guards_against_singular_similarity_matrix(function):
    # An exact duplicate point makes the similarity matrix exactly singular
    # (two identical rows/columns).
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    X_with_duplicate = torch.cat([X, X[:1]])

    result = function(X_with_duplicate)  # default jitter=1e-6
    assert torch.isfinite(result)

    with (
        pytest.warns(UserWarning, match="linsolve"),
        pytest.raises(torch.linalg.LinAlgError),
    ):
        function(X_with_duplicate, jitter=0.0)


# ---------------------------------------------------------------------------
# `solver` (magnitude and magnitude_dim only).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_auto_matches_cholesky_when_well_conditioned(function, recwarn):
    X = torch.randn(8, 4)

    result_auto = function(X, solver="auto")
    result_cholesky = function(X, solver="cholesky")

    assert torch.allclose(result_auto, result_cholesky)
    assert len(recwarn) == 0


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_auto_falls_back_to_linsolve_on_cholesky_failure(
    function, monkeypatch
):
    X = torch.randn(6, 3)
    original_cholesky_ex = torch.linalg.cholesky_ex

    def _failing_cholesky_ex(*args, **kwargs):
        L, info = original_cholesky_ex(*args, **kwargs)
        return L, torch.ones_like(info)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", _failing_cholesky_ex)

    with pytest.warns(UserWarning, match="linsolve"):
        result_auto = function(X, solver="auto")

    result_linsolve = function(X, solver="linsolve")
    assert torch.allclose(result_auto, result_linsolve)


@pytest.mark.parametrize("function", JITTER_SOLVER_FUNCTIONS)
def test_solver_variants_agree(function):
    X = torch.randn(6, 3, dtype=torch.float64)

    results = {
        solver: function(X, solver=solver, jitter=0.0)
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


# ---------------------------------------------------------------------------
# `magnitude_dim` math. Docstring: "the logarithmic derivative of magnitude
# with respect to scale, namely scale / magnitude * d(magnitude) / d(scale)".
# ---------------------------------------------------------------------------


def test_magnitude_dim_matches_magnitude_scale_derivative():
    X = torch.randn(6, 3, dtype=torch.float64)
    scale = 0.75

    distances = torch.cdist(X, X, p=2)
    similarity = torch.exp(-scale * distances)
    ones = torch.ones(len(X), 1, dtype=torch.float64)
    w = torch.linalg.solve(similarity, ones).squeeze(-1)
    magnitude_value = w.sum()
    magnitude_derivative = w @ ((distances * similarity) @ w)
    expected = scale / magnitude_value * magnitude_derivative

    assert torch.allclose(
        magnitude_dim(
            X, scale=scale, use_double_precision=True, jitter=0.0
        ),
        expected,
        rtol=1e-4,
        atol=1e-6,
    )
