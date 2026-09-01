import inspect

import pytest
import torch
import torch.nn as nn

from magnispread import MagLoss, SpreadDimLoss, SpreadLoss
from magnispread.functional import magnitude, spread, spread_dim


@pytest.mark.parametrize("cls", [MagLoss, SpreadLoss, SpreadDimLoss])
def test_is_nn_module(cls):
    assert isinstance(cls(), nn.Module)


@pytest.mark.parametrize(
    "cls, function",
    [(MagLoss, magnitude), (SpreadLoss, spread), (SpreadDimLoss, spread_dim)],
)
def test_signature_mirrors_functional_counterpart(cls, function):
    # Parameter lists (aside from `X`/`self`) must match exactly, since the
    # loss modules are documented to mirror the functional API.
    loss_params = list(inspect.signature(cls.__init__).parameters)[1:]
    func_params = list(inspect.signature(function).parameters)[1:]
    assert loss_params == func_params


@pytest.mark.parametrize(
    "cls, function",
    [(MagLoss, magnitude), (SpreadLoss, spread), (SpreadDimLoss, spread_dim)],
)
def test_defaults_match_functional_counterpart(cls, function):
    X = torch.randn(8, 4)
    assert torch.allclose(cls()(X), function(X))


@pytest.mark.parametrize(
    "cls, function",
    [(MagLoss, magnitude), (SpreadLoss, spread), (SpreadDimLoss, spread_dim)],
)
@pytest.mark.parametrize("metric", ["euclidean", "cosine", "precomputed"])
@pytest.mark.parametrize("force_diagonal", [None, True, False])
def test_matches_functional_counterpart_across_settings(
    cls, function, metric, force_diagonal
):
    X = torch.randn(6, 4)
    D = torch.cdist(X, X, p=2) if metric == "precomputed" else X

    kwargs = {"metric": metric, "scale": 0.5, "force_diagonal": force_diagonal}
    if cls is MagLoss:
        kwargs.update(jitter=1e-5, solver="linsolve")

    assert torch.allclose(cls(**kwargs)(D), function(D, **kwargs))


@pytest.mark.parametrize("cls", [MagLoss, SpreadLoss, SpreadDimLoss])
def test_supports_backward(cls):
    X = torch.randn(8, 4, requires_grad=True)
    loss = cls()(X)

    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


# ---------------------------------------------------------------------------
# dtype handling. README: "The returned tensor's dtype matches `X`'s
# floating-point dtype (or `float32` if `X` is not floating-point),
# regardless of `use_double_precision`." Applies equally to the module API.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [MagLoss, SpreadLoss, SpreadDimLoss])
@pytest.mark.parametrize("use_double_precision", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_output_dtype_matches_input_dtype(
    cls, use_double_precision, input_dtype
):
    X = torch.randn(6, 3).to(dtype=input_dtype)

    result = cls(use_double_precision=use_double_precision)(X)

    assert result.dtype == input_dtype


@pytest.mark.parametrize("cls", [MagLoss, SpreadLoss, SpreadDimLoss])
def test_non_floating_point_input_returns_float32(cls):
    X = torch.randint(0, 5, (6, 3))

    assert cls()(X).dtype == torch.float32
