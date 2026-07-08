import pytest
import torch

from mag_torch.functional import mag_loss


def test_mag_loss_returns_scalar_and_backward():
    X = torch.randn(8, 4, requires_grad=True)
    loss = mag_loss(X, metric="euclidean")

    assert loss.ndim == 0
    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_mag_loss_supports_cosine_metric():
    X = torch.randn(10, 6, requires_grad=True)
    loss = mag_loss(X, metric="cosine", t=0.5)

    assert torch.isfinite(loss)
    loss.backward()
    assert X.grad is not None


def test_mag_loss_rejects_invalid_metric():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="metric"):
        mag_loss(X, metric="manhattan")


def test_mag_loss_requires_2d_input():
    X = torch.randn(4, 3, 2)
    with pytest.raises(ValueError, match="2D"):
        mag_loss(X)


def test_mag_loss_validates_t_and_jitter():
    X = torch.randn(4, 3)
    with pytest.raises(ValueError, match="`t`"):
        mag_loss(X, t=0)

    with pytest.raises(ValueError, match="`jitter`"):
        mag_loss(X, jitter=-1e-6)
