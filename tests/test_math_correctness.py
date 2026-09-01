"""Mathematical correctness check against a closed-form two-point example.

For the two-point space `{[0, 0], [1, 0]}` (Euclidean distance 1 apart), the
2x2 similarity matrix at scale `t` is `[[1, s], [s, 1]]` with `s = exp(-t)`.
Both magnitude and spread work out to the same closed form:

    magnitude(t) = spread(t) = 2 / (1 + exp(-t))

Spread dimension (`t / spread * d(spread)/dt`) simplifies to:

    spread_dim(t) = t * (1 - sigma(t)) = t * exp(-t) / (1 + exp(-t))
"""

import math

import pytest
import torch

from magnispread.functional import magnitude, spread, spread_dim

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


def test_magnitude_equals_spread_for_two_point_space():
    # A special property of two-point spaces: magnitude and spread coincide.
    for t in SCALES:
        m = magnitude(X, scale=t, use_double_precision=True, jitter=0.0)
        s = spread(X, scale=t, use_double_precision=True)
        assert torch.allclose(m, s)
