"""Mathematical correctness check against a closed-form two-point example.

For the two-point space `{[0, 0], [1, 0]}` (Euclidean distance 1 apart), the
2x2 similarity matrix at scale `t` is `[[1, s], [s, 1]]` with `s = exp(-t)`.
Both magnitude and spread work out to the same closed form:

    magnitude(t) = spread(t) = 2 / (1 + exp(-t))

Spread dimension (`t / spread * d(spread)/dt`) simplifies to:

    spread_dim(t) = t * (1 - sigma(t)) = t * exp(-t) / (1 + exp(-t))

Since magnitude and spread coincide on this space, their scale-derivatives
coincide too, so magnitude dimension works out to the exact same closed form
as spread dimension:

    magnitude_dim(t) = t * exp(-t) / (1 + exp(-t))
"""

import math

import pytest
import torch

from magnispread.functional import (
    magnitude,
    magnitude_dim,
    magnitude_dim_max,
    spread,
    spread_dim,
    spread_dim_max,
)

X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
SCALES = torch.logspace(math.log10(0.01), math.log10(100.0), steps=25).tolist()


@pytest.mark.parametrize("t", SCALES)
def test_magnitude_matches_closed_form(t):
    expected = 2.0 / (1.0 + math.exp(-t))
    result = magnitude(
        X, scale=t, use_double_precision=True, jitter=0.0
    ).item()
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("t", SCALES)
def test_spread_matches_closed_form(t):
    expected = 2.0 / (1.0 + math.exp(-t))
    result = spread(X, scale=t, use_double_precision=True).item()
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("t", SCALES)
def test_spread_dim_matches_closed_form(t):
    expected = t * math.exp(-t) / (1.0 + math.exp(-t))
    result = spread_dim(X, scale=t, use_double_precision=True).item()
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("t", SCALES)
def test_magnitude_dim_matches_closed_form(t):
    expected = t * math.exp(-t) / (1.0 + math.exp(-t))
    result = magnitude_dim(
        X, scale=t, use_double_precision=True, jitter=0.0
    ).item()
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_magnitude_equals_spread_for_two_point_space():
    # A special property of two-point spaces: magnitude and spread coincide.
    for t in SCALES:
        m = magnitude(X, scale=t, use_double_precision=True, jitter=0.0)
        s = spread(X, scale=t, use_double_precision=True)
        assert torch.allclose(m, s)


def test_magnitude_dim_equals_spread_dim_for_two_point_space():
    # Since magnitude and spread coincide on two-point spaces, so do their
    # scale-derivatives, and hence magnitude dimension and spread dimension.
    for t in SCALES:
        md = magnitude_dim(X, scale=t, use_double_precision=True, jitter=0.0)
        sd = spread_dim(X, scale=t, use_double_precision=True)
        assert torch.allclose(md, sd)


# ---------------------------------------------------------------------------
# Scale-agnostic dimension (`sup_{t>0} dim(t)`) against an independent
# ground truth for the two-point space.
#
# Setting d(dim)/dt = 0 for dim(t) = t * exp(-t) / (1 + exp(-t)) gives the
# critical-point equation `t - 1 = exp(-t)`. The function
# `g(t) = t - 1 - exp(-t)` is strictly increasing (`g'(t) = 1 + exp(-t) > 0`),
# so this equation has a unique root, found below by plain bisection --
# independent of the production golden-section search, so this is a real
# ground truth rather than a self-consistency check.
#
# Substituting `exp(-t*) = t* - 1` (from the critical-point equation) into
# `dim(t*) = t* * exp(-t*) / (1 + exp(-t*))` gives
# `dim(t*) = t*(t*-1) / ((t*-1)+1) = t*-1`. So `dim(t*) = t*-1` is an exact
# algebraic identity for this example, not a numerical coincidence.
# ---------------------------------------------------------------------------


def _bisect_critical_point(lo=0.01, hi=100.0, iterations=200):
    g = lambda t: t - 1.0 - math.exp(-t)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


T_STAR = _bisect_critical_point()
DIM_MAX = T_STAR - 1.0


def test_magnitude_dim_max_matches_independent_ground_truth():
    result = magnitude_dim_max(
        X, use_double_precision=True, jitter=0.0, t_min=0.01, t_max=100.0
    )

    assert result.scale == pytest.approx(T_STAR, rel=1e-3)
    assert result.dim.item() == pytest.approx(DIM_MAX, rel=1e-4)


def test_spread_dim_max_matches_independent_ground_truth():
    result = spread_dim_max(
        X, use_double_precision=True, t_min=0.01, t_max=100.0
    )

    assert result.scale == pytest.approx(T_STAR, rel=1e-3)
    assert result.dim.item() == pytest.approx(DIM_MAX, rel=1e-4)


def test_magnitude_dim_max_equals_spread_dim_max_for_two_point_space():
    # Since magnitude_dim and spread_dim coincide on two-point spaces at
    # every scale, so do their maxima over scale.
    md_max = magnitude_dim_max(
        X, use_double_precision=True, jitter=0.0, t_min=0.01, t_max=100.0
    )
    sd_max = spread_dim_max(X, use_double_precision=True, t_min=0.01, t_max=100.0)

    assert torch.allclose(md_max.dim, sd_max.dim)
    assert md_max.scale == pytest.approx(sd_max.scale, rel=1e-6)
