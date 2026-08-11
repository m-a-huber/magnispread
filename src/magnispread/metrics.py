import torch
import torch.nn.functional as F


def pairwise_cosine_distance(
    X: torch.Tensor,
    Y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes cosine distance between rows of X and Y. Only intended for
    2D-tensors."""
    if X.ndim != 2:
        raise ValueError(
            f"`X` must be a 2D-tensor, got shape {tuple(X.shape)}"
        )
    if Y is not None and Y.ndim != 2:
        raise ValueError(
            f"If provided, `Y` must be a 2D-tensor, got shape {tuple(Y.shape)}"
        )
    if Y is None:
        Y = X
    x1 = F.normalize(X, dim=1)
    x2 = F.normalize(Y, dim=1)
    similarity = x1 @ x2.T
    return torch.clamp(1.0 - similarity, min=0.0, max=2.0)
