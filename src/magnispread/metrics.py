import torch
import torch.nn.functional as F


def pairwise_cosine_distance(
    X: torch.Tensor,
    Y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute cosine distance between rows of X and Y."""
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
    # Clamp distance while preserving gradients
    distance = 1.0 - similarity
    clamped_distance = distance.clamp(min=0.0, max=2.0)
    return distance + (clamped_distance - distance).detach()
