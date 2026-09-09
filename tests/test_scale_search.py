import math

import pytest

from magnispread.scale_search import ScaleSearchResult, maximize_over_log_scale


def test_finds_maximum_of_simple_unimodal_function():
    # A parabola in log(t), with its peak at t = e.
    objective = lambda t: -((math.log(t) - 1.0) ** 2)

    result = maximize_over_log_scale(objective, t_min=0.1, t_max=100.0)

    assert result.t == pytest.approx(math.e, rel=1e-4)


def test_handles_non_unimodal_function_via_coarse_grid():
    # Two bumps in log(t)-space of different heights, centered at t=1 and
    # t=100. A local search started near the smaller bump (without the
    # coarse-grid bracketing phase) would get stuck there.
    def objective(t):
        log_t = math.log(t)
        small_bump = 0.5 * math.exp(-((log_t - 0.0) ** 2) / 0.01)
        tall_bump = 1.0 * math.exp(-((log_t - math.log(100.0)) ** 2) / 0.01)
        return small_bump + tall_bump

    result = maximize_over_log_scale(
        objective, t_min=0.01, t_max=10000.0, num_coarse_steps=100
    )

    assert result.t == pytest.approx(100.0, rel=1e-2)


def test_warns_and_returns_boundary_when_argmax_at_t_min():
    objective = lambda t: -t  # strictly decreasing -> argmax at t_min

    with pytest.warns(UserWarning, match="boundary"):
        result = maximize_over_log_scale(objective, t_min=1.0, t_max=10.0)

    assert result.t == pytest.approx(1.0)


def test_warns_and_returns_boundary_when_argmax_at_t_max():
    objective = lambda t: t  # strictly increasing -> argmax at t_max

    with pytest.warns(UserWarning, match="boundary"):
        result = maximize_over_log_scale(objective, t_min=1.0, t_max=10.0)

    assert result.t == pytest.approx(10.0)


def test_rejects_non_positive_t_min():
    with pytest.raises(ValueError, match="`t_min`"):
        maximize_over_log_scale(lambda t: t, t_min=0.0, t_max=10.0)


def test_rejects_t_max_not_greater_than_t_min():
    with pytest.raises(ValueError, match="`t_max`"):
        maximize_over_log_scale(lambda t: t, t_min=10.0, t_max=10.0)


def test_rejects_too_few_coarse_steps():
    with pytest.raises(ValueError, match="`num_coarse_steps`"):
        maximize_over_log_scale(lambda t: t, num_coarse_steps=2)


def test_rejects_negative_max_refine_iter():
    with pytest.raises(ValueError, match="`max_refine_iter`"):
        maximize_over_log_scale(lambda t: t, max_refine_iter=-1)


def test_rejects_non_positive_log_tol():
    with pytest.raises(ValueError, match="`log_tol`"):
        maximize_over_log_scale(lambda t: t, log_tol=0.0)


def test_return_type_is_namedtuple_with_float_fields():
    result = maximize_over_log_scale(lambda t: -((math.log(t)) ** 2))

    assert isinstance(result, ScaleSearchResult)
    assert isinstance(result.t, float)
    assert isinstance(result.value, float)
