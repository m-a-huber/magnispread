import torch

from .metrics import pairwise_cosine_distance


def get_distance_matrix(
    X: torch.Tensor,
    metric: str,
    use_double_precision: bool,
) -> torch.Tensor:
    """Compute distance matrix from point cloud or precomputed distances."""

    if not X.is_floating_point():
        X = X.to(dtype=torch.float32)

    X_kernel = X if not use_double_precision else X.to(dtype=torch.float64)

    if metric == "euclidean":
        distance_matrix = torch.cdist(X_kernel, X_kernel, p=2)
    elif metric == "cosine":
        distance_matrix = pairwise_cosine_distance(X_kernel, X_kernel)
    elif metric == "precomputed":
        distance_matrix = X_kernel
    else:
        raise ValueError(
            "`metric` must be either 'euclidean', 'cosine', or "
            f"'precomputed', got {metric}"
        )

    return distance_matrix


def get_similarity_matrix(
    distance_matrix: torch.Tensor,
    scale: float,
    symmetrize: bool,
    force_diagonal: bool,
) -> torch.Tensor:
    """Compute similarity matrix from a distance matrix."""

    similarity_matrix = torch.exp(-scale * distance_matrix)

    if symmetrize:
        similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.mT)

    if force_diagonal:
        similarity_matrix = torch.diagonal_scatter(
            similarity_matrix,
            torch.ones(
                similarity_matrix.shape[0],
                dtype=similarity_matrix.dtype,
                device=similarity_matrix.device,
            ),
        )

    return similarity_matrix
