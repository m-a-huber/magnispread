import torch

from .magnitude import magnitude_from_similarity_matrix
from .matrices import get_distance_matrix, get_similarity_matrix
from .spread import (
    spread_dim_from_distance_matrix,
    spread_from_similarity_matrix,
)
from .validation import _validate_inputs


def _get_output_dtype(
    X: torch.Tensor,
) -> torch.dtype:
    """Determine the output dtype, preserving `X`'s floating-point dtype."""

    return X.dtype if X.is_floating_point() else torch.float32


def _resolve_force_diagonal(
    metric: str,
    force_diagonal: bool | None,
) -> bool:
    """Resolve `force_diagonal`'s default, which depends on `metric`."""
    return (
        force_diagonal
        if force_diagonal is not None
        else metric != "precomputed"
    )


def magnitude(
    X: torch.Tensor,
    metric: str = "euclidean",
    scale: float = 1.0,
    use_double_precision: bool = False,
    symmetrize: bool = True,
    force_diagonal: bool | None = None,
    jitter: float = 1e-6,
    solver: str = "auto",
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
    scale : float, optional
        Scale at which to compute magnitude. Must be positive. Defaults to
        `1.0`.
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. This is
        independent of the returned tensor's dtype, which matches `X`'s dtype
        if `X` is floating-point, or `float32` otherwise. Defaults to `False`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `True`.
    force_diagonal : bool or None, optional
        Whether to force the diagonal of the similarity matrix to be exactly
        `1`. If `None`, defaults to `True` unless `metric="precomputed"`, in
        which case it defaults to `False` (the precomputed matrix is used as
        supplied). Defaults to `None`.
    jitter : float, optional
        Small constant added to the diagonal of the similarity matrix for
        numerical stability. Defaults to `1e-6`.
    solver : str, optional
        Solver to use for computing the magnitude. Must be either
        `"auto"`, `"cholesky"`, `"linsolve"`, or `"inverse"`. `"auto"` attempts
        Cholesky decomposition first and falls back to `"linsolve"` which
        computes magnitude by solving the linear system
        `similarity_matrix @ weights = 1` and summing the weights (emitting a
        `UserWarning` if Cholesky decomposition fails). If solver is set to
        `"inverse"`, magnitude is computed by directly inverting the similarity
        matrix and summing its entries. Defaults to `"auto"`.

    Notes
    -----
    By default, the similarity matrix's diagonal is forced to exactly `1`
    (self-similarity) for `metric="euclidean"` and `metric="cosine"`, which
    guards against float32-precision noise in the underlying distance
    computation. This does not happen by default for `metric="precomputed"`;
    see `force_diagonal`.

    Returns
    -------
    torch.Tensor
        Tensor containing the magnitude of the point cloud specified by `X`,
        with the same floating-point dtype as `X` (or `float32` if `X` is not
        floating-point).

    Raises
    ------
    ValueError
        If `X` contains no points.
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
        If `solver` is not `"auto"`, `"cholesky"`, `"linsolve"`, or
        `"inverse"`.
    """

    _validate_inputs(
        X=X,
        metric=metric,
        scale=scale,
        jitter=jitter,
        solver=solver,
    )

    output_dtype = _get_output_dtype(X)
    force_diagonal = _resolve_force_diagonal(metric, force_diagonal)

    distance_matrix = get_distance_matrix(
        X,
        metric=metric,
        use_double_precision=use_double_precision,
    )
    similarity_matrix = get_similarity_matrix(
        distance_matrix,
        scale=scale,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
    )

    if jitter:
        similarity_matrix = similarity_matrix + jitter * torch.eye(
            len(X),
            dtype=similarity_matrix.dtype,
            device=similarity_matrix.device,
        )

    return magnitude_from_similarity_matrix(
        similarity_matrix=similarity_matrix,
        solver=solver,
        output_dtype=output_dtype,
    )


def spread(
    X: torch.Tensor,
    metric: str = "euclidean",
    scale: float = 1.0,
    use_double_precision: bool = False,
    symmetrize: bool = True,
    force_diagonal: bool | None = None,
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
        Whether to use double precision for internal computations. This is
        independent of the returned tensor's dtype, which matches `X`'s dtype
        if `X` is floating-point, or `float32` otherwise. Defaults to `False`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `True`.
    force_diagonal : bool or None, optional
        Whether to force the diagonal of the similarity matrix to be exactly
        `1`. If `None`, defaults to `True` unless `metric="precomputed"`, in
        which case it defaults to `False` (the precomputed matrix is used as
        supplied). Defaults to `None`.

    Notes
    -----
    By default, the similarity matrix's diagonal is forced to exactly `1`
    (self-similarity) for `metric="euclidean"` and `metric="cosine"`, which
    guards against float32-precision noise in the underlying distance
    computation. This does not happen by default for `metric="precomputed"`;
    see `force_diagonal`.

    Returns
    -------
    torch.Tensor
        Tensor containing the spread of the point cloud specified by `X`, with
        the same floating-point dtype as `X` (or `float32` if `X` is not
        floating-point).

    Raises
    ------
    ValueError
        If `X` contains no points.
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

    output_dtype = _get_output_dtype(X)
    force_diagonal = _resolve_force_diagonal(metric, force_diagonal)

    distance_matrix = get_distance_matrix(
        X,
        metric=metric,
        use_double_precision=use_double_precision,
    )
    similarity_matrix = get_similarity_matrix(
        distance_matrix,
        scale=scale,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
    )

    return spread_from_similarity_matrix(
        similarity_matrix=similarity_matrix,
        output_dtype=output_dtype,
    )


def spread_dim(
    X: torch.Tensor,
    metric: str = "euclidean",
    scale: float = 1.0,
    use_double_precision: bool = False,
    symmetrize: bool = True,
    force_diagonal: bool | None = None,
) -> torch.Tensor:
    """Computes spread dimension from a point cloud or matrix of pairwise
    distances.

    Spread dimension is the logarithmic derivative of spread with respect to
    scale, namely ``scale / spread * d(spread) / d(scale)``.

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
        Scale at which to compute spread dimension. Must be positive. Defaults
        to `1.0`.
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. This is
        independent of the returned tensor's dtype, which matches `X`'s dtype
        if `X` is floating-point, or `float32` otherwise. Defaults to `False`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `True`.
    force_diagonal : bool or None, optional
        Whether to force the diagonal of the similarity matrix to be exactly
        `1`. If `None`, defaults to `True` unless `metric="precomputed"`, in
        which case it defaults to `False` (the precomputed matrix is used as
        supplied). Defaults to `None`.

    Notes
    -----
    By default, the similarity matrix's diagonal is forced to exactly `1`
    (self-similarity) for `metric="euclidean"` and `metric="cosine"`, which
    guards against float32-precision noise in the underlying distance
    computation. This does not happen by default for `metric="precomputed"`;
    see `force_diagonal`.

    Returns
    -------
    torch.Tensor
        Tensor containing the spread dimension of the point cloud specified by
        `X`, with the same floating-point dtype as `X` (or `float32` if `X` is
        not floating-point).

    Raises
    ------
    ValueError
        If `X` contains no points.
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

    output_dtype = _get_output_dtype(X)
    force_diagonal = _resolve_force_diagonal(metric, force_diagonal)

    distance_matrix = get_distance_matrix(
        X,
        metric=metric,
        use_double_precision=use_double_precision,
    )

    return spread_dim_from_distance_matrix(
        distance_matrix=distance_matrix,
        scale=scale,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
        output_dtype=output_dtype,
    )
