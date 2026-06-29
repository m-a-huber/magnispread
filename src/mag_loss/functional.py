import torch

from ._metrics import pairwise_cosine_distance


def magnitude_loss(
    X: torch.Tensor,
    metric: str = "euclidean",
    t: float = 1.0,
    use_double_precision: bool = True,
    symmetrize: bool = True,
    jitter: float = 1e-6,
) -> torch.Tensor:
    if use_double_precision:
        X_kernel = X.to(dtype=torch.float64)
    else:
        X_kernel = X

    if metric == "euclidean":
        similarity_matrix = torch.exp(
            -t * torch.cdist(X_kernel, X_kernel, p=2)
        )
    elif metric == "cosine":
        similarity_matrix = torch.exp(
            -t * pairwise_cosine_distance(X_kernel, X_kernel)
        )
    else:
        raise ValueError(
            f"`metric` must be either 'euclidean' or 'cosine', got {metric}"
        )

    if symmetrize:
        similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.mT)

    if jitter:
        similarity_matrix = similarity_matrix + jitter * torch.eye(
            len(X),
            dtype=similarity_matrix.dtype,
            device=similarity_matrix.device,
        )

    L = torch.linalg.cholesky(similarity_matrix, upper=False)
    ones = torch.ones(
        len(X),
        1,
        dtype=similarity_matrix.dtype,
        device=similarity_matrix.device,
    )
    x = torch.linalg.solve_triangular(L, ones, upper=False)
    return (x.mT @ x).squeeze()
