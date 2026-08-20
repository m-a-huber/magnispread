import warnings

import torch

from .metrics import pairwise_cosine_distance


def _validate_inputs(
    X: torch.Tensor,
    metric: str,
    scale: float,
    jitter: float | None = None,
    solver: str | None = None,
) -> None:
    """Validate input arguments."""

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

    if scale <= 0:
        raise ValueError(f"`scale` must be positive, got {scale}")

    if jitter is not None and jitter < 0:
        raise ValueError(f"`jitter` must be non-negative, got {jitter}")

    if solver is not None and solver not in {"auto", "cholesky", "inverse"}:
        raise ValueError(
            "`solver` must be either 'auto', 'cholesky', or 'inverse', got "
            f"{solver}"
        )


def _similarity_matrix_from_input(
    X: torch.Tensor,
    metric: str,
    scale: float,
    use_double_precision: bool,
) -> torch.Tensor:
    """Build a similarity matrix from validated input."""

    if not X.is_floating_point():
        X = X.to(dtype=torch.float32)

    X_kernel = X if not use_double_precision else X.to(dtype=torch.float64)

    if metric == "euclidean":
        distances = torch.cdist(X_kernel, X_kernel, p=2)
    elif metric == "cosine":
        distances = pairwise_cosine_distance(X_kernel, X_kernel)
    else:
        distances = X_kernel

    return torch.exp(-scale * distances)


def magnitude(
    X: torch.Tensor,
    metric: str = "euclidean",
    scale: float = 1.0,
    use_double_precision: bool = False,
    symmetrize: bool = False,
    jitter: float = 1e-6,
    solver: str = "auto",
) -> torch.Tensor:  # ty: ignore[invalid-return-type]
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
    scale : float, optional
        Scale at which to compute magnitude. Must be positive. Defaults to
        `1.0`.
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. Defaults to
        `False`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `False`.
    jitter : float, optional
        Small constant added to the diagonal of the similarity matrix for
        numerical stability. Defaults to `1e-6`.
    solver : str, optional
        Solver to use for computing the magnitude. Must be either
        `"auto"`, `"cholesky"`, or `"inverse"`. `"auto"` attempts Cholesky
        decomposition first and falls back to the direct matrix inverse,
        emitting a `UserWarning`, if the similarity matrix is not
        positive-definite. Defaults to `"auto"`.

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
        If `scale` is not positive.
    ValueError
        If `jitter` is negative.
    ValueError
        If `solver` is not `"auto"`, `"cholesky"`, or `"inverse"`.
    """

    _validate_inputs(
        X=X,
        metric=metric,
        scale=scale,
        jitter=jitter,
        solver=solver,
    )

    similarity_matrix = _similarity_matrix_from_input(
        X,
        metric=metric,
        scale=scale,
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

    if solver == "auto":
        L, info = torch.linalg.cholesky_ex(similarity_matrix, upper=False)
        if info.item() == 0:
            ones = torch.ones(
                len(X),
                1,
                dtype=similarity_matrix.dtype,
                device=similarity_matrix.device,
            )
            x = torch.linalg.solve_triangular(L, ones, upper=False)
            return (x.mT @ x).squeeze()
        warnings.warn(
            "Cholesky decomposition failed; falling back to solver='inverse'",
            stacklevel=2,
        )
        return torch.linalg.inv(similarity_matrix).sum().squeeze()
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
    scale: float = 1.0,
    use_double_precision: bool = False,
    symmetrize: bool = False,
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
    scale : float, optional
        Scale at which to compute spread. Must be positive. Defaults to
        `1.0`.
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. Defaults to
        `False`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `False`.

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
        If `scale` is not positive.
    """

    _validate_inputs(
        X=X,
        metric=metric,
        scale=scale,
    )

    similarity_matrix = _similarity_matrix_from_input(
        X,
        metric=metric,
        scale=scale,
        use_double_precision=use_double_precision,
    )

    if symmetrize:
        similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.mT)

    return (1 / similarity_matrix.sum(dim=1)).sum().squeeze()
