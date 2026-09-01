import pytest
import torch
import torch.nn.functional as F

from magnispread.metrics import pairwise_cosine_distance


def test_requires_2d_input_for_x():
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        pairwise_cosine_distance(X)


def test_requires_2d_input_for_y():
    X = torch.randn(4, 3)
    Y = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        pairwise_cosine_distance(X, Y)


def test_defaults_y_to_x():
    X = torch.randn(5, 4)
    assert torch.equal(
        pairwise_cosine_distance(X), pairwise_cosine_distance(X, X)
    )


def test_output_within_valid_range():
    torch.manual_seed(0)
    for _ in range(200):
        X = torch.randn(8, 6)
        D = pairwise_cosine_distance(X)
        assert (D >= 0.0).all()
        assert (D <= 2.0).all()


def test_self_distance_is_approximately_zero():
    X = torch.randn(5, 4)
    D = pairwise_cosine_distance(X)
    assert torch.allclose(torch.diagonal(D), torch.zeros(5), atol=1e-5)


def test_clamp_preserves_gradient_for_near_parallel_vectors():
    # This seed reliably produces a raw cosine similarity strictly greater
    # than 1.0 for a vector against itself (float32 rounding in
    # normalize + dot), which is exactly the case the clamp guards against.
    torch.manual_seed(2)
    base = torch.randn(1, 32)
    x = base.clone().requires_grad_(True)
    y = base.clone().requires_grad_(True)
    X = torch.cat([x, y])

    raw_similarity = (F.normalize(x, dim=1) @ F.normalize(y, dim=1).T).item()
    assert raw_similarity > 1.0  # sanity check: the clamp is actually hit

    D = pairwise_cosine_distance(X)
    assert (
        D[0, 1].item() == 0.0  # ruff: ignore[float-equality-comparison]
    )  # forward value is still clamped

    D[0, 1].backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0  # old hard clamp would have zeroed this
