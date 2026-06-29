import torch

from ._metrics import pairwise_cosine_distance


def mag_loss(
    X: torch.Tensor,
    metric: str = "euclidean",
    t: float = 1.0,
    use_double_precision: bool = True,
    symmetrize: bool = True,
    jitter: float = 1e-6,
) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError(
            f"`X` must be a 2D tensor, got shape {tuple(X.shape)}"
        )

    if t <= 0:
        raise ValueError(f"`t` must be > 0, got {t}")

    if jitter < 0:
        raise ValueError(f"`jitter` must be >= 0, got {jitter}")

    if metric not in {"euclidean", "cosine"}:
        raise ValueError(
            f"`metric` must be either 'euclidean' or 'cosine', got {metric}"
        )

    if not X.is_floating_point():
        X = X.to(dtype=torch.float32)

    if use_double_precision:
        X_kernel = X.to(dtype=torch.float64)
    else:
        X_kernel = X

    if metric == "euclidean":
        similarity_matrix = torch.exp(
            -t * torch.cdist(X_kernel, X_kernel, p=2)
        )
    else:
        similarity_matrix = torch.exp(
            -t * pairwise_cosine_distance(X_kernel, X_kernel)
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
