import torch

from .metrics import pairwise_cosine_distance


def _validate_common_inputs(
    X: torch.Tensor,
    metric: str,
    t: float,
) -> None:
    """Validate input constraints shared by `magnitude` and `spread`."""
    if X.ndim != 2:
        raise ValueError(
            f"`X` must be a 2D-tensor, got shape {tuple(X.shape)}"
        )

    if metric not in {"euclidean", "cosine", "precomputed"}:
        raise ValueError(
            "`metric` must be either 'euclidean', 'cosine', or "
            f"'precomputed', got {metric}"
        )

    if metric == "precomputed" and X.shape[0] != X.shape[1]:
        raise ValueError(
            "`X` must be a square pairwise-distance matrix when "
            f"`metric='precomputed'`, got shape {tuple(X.shape)}"
        )

    if t <= 0:
        raise ValueError(f"`t` must be positive, got {t}")


def _similarity_matrix_from_input(
    X: torch.Tensor,
    metric: str,
    t: float,
    use_double_precision: bool,
) -> torch.Tensor:
    """Build a similarity matrix from validated input."""

    if not X.is_floating_point():
        X = X.to(dtype=torch.float32)

    X_kernel = X.to(dtype=torch.float64) if use_double_precision else X

    if metric == "euclidean":
        distances = torch.cdist(X_kernel, X_kernel, p=2)
    elif metric == "cosine":
        distances = pairwise_cosine_distance(X_kernel, X_kernel)
    else:
        distances = X_kernel

    return torch.exp(-t * distances)


def magnitude(
    X: torch.Tensor,
    metric: str = "euclidean",
    t: float = 1.0,
    use_double_precision: bool = True,
    symmetrize: bool = True,
    jitter: float = 1e-6,
    solver: str = "cholesky",
) -> torch.Tensor:
    """Computes metric space magnitude from a point cloud or from a matrix of
    pairwise distances.

    Parameters
    ----------
    X : torch.Tensor
        Tensor of shape `(n, d)` representing a `d`-dimensional point cloud
        with `n` elements, or a tensor of shape `(n,n)` representing a matrix
        of pairwise distances.
    metric : str, optional
        Metric to use for computing pairwise distances. If set to
        `"precomputed"`, `X` is assumed to be a square matrix containing
        pairwise distances. Defaults to `"euclidean"`.
    t : float, optional
        Scale at which to compute magnitude. Must be positive. Defaults to
        `1.0`.
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. Defaults to
        `True`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `True`.
    jitter : float, optional
        Small constant added to the diagonal of the similarity matrix for
        numerical stability. Defaults to `1e-6`.
    solver : str, optional
        Solver to use for computing the magnitude. Must be either `"cholesky"`
        or `"inverse"`. Defaults to `"cholesky"`.

    Returns
    -------
    torch.Tensor
        Tensor containing the magnitude of the point cloud specified by `X`.

    Raises
    ------
    ValueError
        If `X` is not a 2D-tensor.
    ValueError
        If `metric` is not `"euclidean"`, `"cosine"`, or `"precomputed"`.
    ValueError
        If `metric` is `"precomputed"` and `X` is not square.
    ValueError
        If `t` is not positive.
    ValueError
        If `jitter` is negative.
    ValueError
        If `solver` is not `"cholesky"` or `"inverse"`.
    """
    _validate_common_inputs(X=X, metric=metric, t=t)

    if jitter < 0:
        raise ValueError(f"`jitter` must be non-negative, got {jitter}")

    if solver not in {"cholesky", "inverse"}:
        raise ValueError(
            f"`solver` must be either 'cholesky' or 'inverse', got {solver}"
        )

    similarity_matrix = _similarity_matrix_from_input(
        X,
        metric=metric,
        t=t,
        use_double_precision=use_double_precision,
    )

    if symmetrize:
        similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.mT)

    if jitter:
        similarity_matrix = similarity_matrix + jitter * torch.eye(
            len(X),
            dtype=similarity_matrix.dtype,
            device=similarity_matrix.device,
        )

    if solver == "cholesky":
        L = torch.linalg.cholesky(similarity_matrix, upper=False)
        ones = torch.ones(
            len(X),
            1,
            dtype=similarity_matrix.dtype,
            device=similarity_matrix.device,
        )
        x = torch.linalg.solve_triangular(L, ones, upper=False)
        return (x.mT @ x).squeeze()
    if solver == "inverse":
        return torch.linalg.inv(similarity_matrix).sum().squeeze()


def spread(
    X: torch.Tensor,
    metric: str = "euclidean",
    t: float = 1.0,
    use_double_precision: bool = True,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Computes metric space spread from a point cloud or from a matrix of
    pairwise distances.

    Parameters
    ----------
    X : torch.Tensor
        Tensor of shape `(n, d)` representing a `d`-dimensional point cloud
        with `n` elements, or a tensor of shape `(n,n)` representing a matrix
        of pairwise distances.
    metric : str, optional
        Metric to use for computing pairwise distances. If set to
        `"precomputed"`, `X` is assumed to be a square matrix containing
        pairwise distances. Defaults to `"euclidean"`.
    t : float, optional
        Scale at which to compute spread. Must be positive. Defaults to
        `1.0`.
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. Defaults to
        `True`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `True`.

    Returns
    -------
    torch.Tensor
        Tensor containing the spread of the point cloud specified by `X`.

    Raises
    ------
    ValueError
        If `X` is not a 2D-tensor.
    ValueError
        If `metric` is not `"euclidean"`, `"cosine"`, or `"precomputed"`.
    ValueError
        If `metric` is `"precomputed"` and `X` is not square.
    ValueError
        If `t` is not positive.
    """
    _validate_common_inputs(X=X, metric=metric, t=t)

    similarity_matrix = _similarity_matrix_from_input(
        X,
        metric=metric,
        t=t,
        use_double_precision=use_double_precision,
    )

    if symmetrize:
        similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.mT)

    return (1 / similarity_matrix.sum(dim=1)).sum().squeeze()
