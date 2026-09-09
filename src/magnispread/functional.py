import warnings
from typing import NamedTuple

import torch

from .magnitude import (
    magnitude_dim_from_distance_matrix,
    magnitude_from_similarity_matrix,
)
from .matrices import get_distance_matrix, get_similarity_matrix
from .scale_search import maximize_over_log_scale
from .spread import (
    spread_dim_from_distance_matrix,
    spread_from_similarity_matrix,
)
from .validation import validate_inputs


class DimMaxResult(NamedTuple):
    """Result of maximizing a per-scale dimension function over `t > 0`."""

    dim: torch.Tensor
    scale: float


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

    validate_inputs(
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


def magnitude_dim(
    X: torch.Tensor,
    metric: str = "euclidean",
    scale: float = 1.0,
    use_double_precision: bool = False,
    symmetrize: bool = True,
    force_diagonal: bool | None = None,
    jitter: float = 1e-6,
    solver: str = "auto",
) -> torch.Tensor:
    """Computes magnitude dimension from a point cloud or matrix of pairwise
    distances.

    Magnitude dimension is the logarithmic derivative of magnitude with
    respect to scale, namely ``scale / magnitude * d(magnitude) / d(scale)``.

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
        Scale at which to compute magnitude dimension. Must be positive.
        Defaults to `1.0`.
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
        Solver to use for computing the magnitude weight vector. Must be
        either `"auto"`, `"cholesky"`, `"linsolve"`, or `"inverse"`. `"auto"`
        attempts Cholesky decomposition first and falls back to
        `"linsolve"` which computes the weight vector by solving the linear
        system `similarity_matrix @ weights = 1` (emitting a `UserWarning` if
        Cholesky decomposition fails). If solver is set to `"inverse"`, the
        weight vector is computed by directly inverting the similarity
        matrix. Defaults to `"auto"`.

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
        Tensor containing the magnitude dimension of the point cloud
        specified by `X`, with the same floating-point dtype as `X` (or
        `float32` if `X` is not floating-point).

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

    validate_inputs(
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

    return magnitude_dim_from_distance_matrix(
        distance_matrix=distance_matrix,
        scale=scale,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
        jitter=jitter,
        solver=solver,
        output_dtype=output_dtype,
    )


def magnitude_dim_max(
    X: torch.Tensor,
    metric: str = "euclidean",
    use_double_precision: bool = False,
    symmetrize: bool = True,
    force_diagonal: bool | None = None,
    jitter: float = 1e-6,
    solver: str = "auto",
    t_min: float = 1e-3,
    t_max: float = 1e3,
    num_coarse_steps: int = 50,
    max_refine_iter: int = 60,
    log_tol: float = 1e-6,
) -> DimMaxResult:
    """Computes the scale-agnostic ("scale-free") magnitude dimension of a
    point cloud or matrix of pairwise distances, defined as `sup_{t>0}
    dim(t)`, where `dim(t)` is the magnitude dimension at scale `t` (see
    :func:`magnispread.functional.magnitude_dim`).

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
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. This is
        more often warranted here than for a single fixed `scale`, since
        searching a wide `[t_min, t_max]` range is more likely to probe
        similarity matrices that are nearly singular (small `t`). This is
        independent of the returned tensor's dtype, which matches `X`'s
        dtype if `X` is floating-point, or `float32` otherwise. Defaults to
        `False`.
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
        Solver to use for computing the magnitude weight vector at each
        candidate scale. Must be either `"auto"`, `"cholesky"`, `"linsolve"`,
        or `"inverse"`. Defaults to `"auto"`.
    t_min : float, optional
        Lower bound of the scale range searched for the maximizer. Must be
        positive. Defaults to `1e-3`.
    t_max : float, optional
        Upper bound of the scale range searched for the maximizer. Must be
        greater than `t_min`. Defaults to `1e3`.
    num_coarse_steps : int, optional
        Number of log-spaced grid points used to bracket the global maximum
        before refinement. Must be at least `3`. Defaults to `50`.
    max_refine_iter : int, optional
        Maximum number of golden-section refinement iterations. Defaults to
        `60`.
    log_tol : float, optional
        Convergence tolerance on the golden-section search bracket's width
        in `log(t)`-space. Defaults to `1e-6`.

    Notes
    -----
    The optimal scale `t*` is found via a derivative-free search performed
    entirely inside `torch.no_grad()`: a coarse log-spaced grid scan over
    `[t_min, t_max]` brackets the global maximum, followed by a golden-section
    search in `log(t)`-space for cheap, high-precision refinement (see
    :func:`magnispread.scale_search.maximize_over_log_scale`). Exactly one
    additional, fully differentiable call to the existing
    `magnitude_dim_from_distance_matrix` kernel is then made at `scale=t*`,
    outside `torch.no_grad()`. By the envelope theorem, this yields the
    exact gradient of `sup_t dim(t)` with respect to `X`: since `t*` is (by
    construction) a stationary point of `dim(t)` in `t`, differentiating
    through the fixed scale `t*` as if it were a constant gives the same
    gradient as differentiating through the full `argmax`, with no need to
    differentiate through the search procedure itself.

    If the coarse grid's maximum falls on the boundary of `[t_min, t_max]`,
    a `UserWarning` is emitted, since the true maximizer may lie outside the
    searched range. `UserWarning`s raised by the solver during the (many)
    internal search evaluations are suppressed; a genuine solver failure at
    the final scale `t*` still raises its usual `UserWarning`.

    By default, the similarity matrix's diagonal is forced to exactly `1`
    (self-similarity) for `metric="euclidean"` and `metric="cosine"`, which
    guards against float32-precision noise in the underlying distance
    computation. This does not happen by default for `metric="precomputed"`;
    see `force_diagonal`.

    Returns
    -------
    DimMaxResult
        A `NamedTuple` with fields `dim` (a scalar `torch.Tensor` containing
        `sup_{t>0} dim(t)`, with the same floating-point dtype as `X`, or
        `float32` if `X` is not floating-point, and supporting `.backward()`)
        and `scale` (the Python `float` scale `t*` at which that maximum is
        attained).

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
        If `jitter` is negative.
    ValueError
        If `solver` is not `"auto"`, `"cholesky"`, `"linsolve"`, or
        `"inverse"`.
    ValueError
        If `t_min` is not positive.
    ValueError
        If `t_max` is not greater than `t_min`.
    ValueError
        If `num_coarse_steps` is less than `3`.
    ValueError
        If `max_refine_iter` is negative.
    ValueError
        If `log_tol` is not positive.
    """

    validate_inputs(
        X=X,
        metric=metric,
        scale=None,
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

    with torch.no_grad():

        def _objective(t: float) -> float:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return magnitude_dim_from_distance_matrix(
                    distance_matrix,
                    scale=t,
                    symmetrize=symmetrize,
                    force_diagonal=force_diagonal,
                    jitter=jitter,
                    solver=solver,
                    output_dtype=output_dtype,
                ).item()

        t_star, _ = maximize_over_log_scale(
            _objective,
            t_min=t_min,
            t_max=t_max,
            num_coarse_steps=num_coarse_steps,
            max_refine_iter=max_refine_iter,
            log_tol=log_tol,
        )

    dim_value = magnitude_dim_from_distance_matrix(
        distance_matrix,
        scale=t_star,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
        jitter=jitter,
        solver=solver,
        output_dtype=output_dtype,
    )

    return DimMaxResult(dim=dim_value, scale=t_star)


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

    validate_inputs(
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

    validate_inputs(
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


def spread_dim_max(
    X: torch.Tensor,
    metric: str = "euclidean",
    use_double_precision: bool = False,
    symmetrize: bool = True,
    force_diagonal: bool | None = None,
    t_min: float = 1e-3,
    t_max: float = 1e3,
    num_coarse_steps: int = 50,
    max_refine_iter: int = 60,
    log_tol: float = 1e-6,
) -> DimMaxResult:
    """Computes the scale-agnostic ("scale-free") spread dimension of a point
    cloud or matrix of pairwise distances, defined as `sup_{t>0} dim(t)`,
    where `dim(t)` is the spread dimension at scale `t` (see
    :func:`magnispread.functional.spread_dim`).

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
    use_double_precision : bool, optional
        Whether to use double precision for internal computations. This is
        more often warranted here than for a single fixed `scale`, since
        searching a wide `[t_min, t_max]` range is more likely to probe
        similarity matrices that are nearly singular (small `t`). This is
        independent of the returned tensor's dtype, which matches `X`'s
        dtype if `X` is floating-point, or `float32` otherwise. Defaults to
        `False`.
    symmetrize : bool, optional
        Whether to symmetrize the similarity matrix. Defaults to `True`.
    force_diagonal : bool or None, optional
        Whether to force the diagonal of the similarity matrix to be exactly
        `1`. If `None`, defaults to `True` unless `metric="precomputed"`, in
        which case it defaults to `False` (the precomputed matrix is used as
        supplied). Defaults to `None`.
    t_min : float, optional
        Lower bound of the scale range searched for the maximizer. Must be
        positive. Defaults to `1e-3`.
    t_max : float, optional
        Upper bound of the scale range searched for the maximizer. Must be
        greater than `t_min`. Defaults to `1e3`.
    num_coarse_steps : int, optional
        Number of log-spaced grid points used to bracket the global maximum
        before refinement. Must be at least `3`. Defaults to `50`.
    max_refine_iter : int, optional
        Maximum number of golden-section refinement iterations. Defaults to
        `60`.
    log_tol : float, optional
        Convergence tolerance on the golden-section search bracket's width
        in `log(t)`-space. Defaults to `1e-6`.

    Notes
    -----
    The optimal scale `t*` is found via a derivative-free search performed
    entirely inside `torch.no_grad()`: a coarse log-spaced grid scan over
    `[t_min, t_max]` brackets the global maximum, followed by a golden-section
    search in `log(t)`-space for cheap, high-precision refinement (see
    :func:`magnispread.scale_search.maximize_over_log_scale`). Exactly one
    additional, fully differentiable call to the existing
    `spread_dim_from_distance_matrix` kernel is then made at `scale=t*`,
    outside `torch.no_grad()`. By the envelope theorem, this yields the
    exact gradient of `sup_t dim(t)` with respect to `X`: since `t*` is (by
    construction) a stationary point of `dim(t)` in `t`, differentiating
    through the fixed scale `t*` as if it were a constant gives the same
    gradient as differentiating through the full `argmax`, with no need to
    differentiate through the search procedure itself.

    If the coarse grid's maximum falls on the boundary of `[t_min, t_max]`,
    a `UserWarning` is emitted, since the true maximizer may lie outside the
    searched range.

    By default, the similarity matrix's diagonal is forced to exactly `1`
    (self-similarity) for `metric="euclidean"` and `metric="cosine"`, which
    guards against float32-precision noise in the underlying distance
    computation. This does not happen by default for `metric="precomputed"`;
    see `force_diagonal`.

    Returns
    -------
    DimMaxResult
        A `NamedTuple` with fields `dim` (a scalar `torch.Tensor` containing
        `sup_{t>0} dim(t)`, with the same floating-point dtype as `X`, or
        `float32` if `X` is not floating-point, and supporting `.backward()`)
        and `scale` (the Python `float` scale `t*` at which that maximum is
        attained).

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
        If `t_min` is not positive.
    ValueError
        If `t_max` is not greater than `t_min`.
    ValueError
        If `num_coarse_steps` is less than `3`.
    ValueError
        If `max_refine_iter` is negative.
    ValueError
        If `log_tol` is not positive.
    """

    validate_inputs(
        X=X,
        metric=metric,
        scale=None,
    )

    output_dtype = _get_output_dtype(X)
    force_diagonal = _resolve_force_diagonal(metric, force_diagonal)

    distance_matrix = get_distance_matrix(
        X,
        metric=metric,
        use_double_precision=use_double_precision,
    )

    with torch.no_grad():

        def _objective(t: float) -> float:
            return spread_dim_from_distance_matrix(
                distance_matrix,
                scale=t,
                symmetrize=symmetrize,
                force_diagonal=force_diagonal,
                output_dtype=output_dtype,
            ).item()

        t_star, _ = maximize_over_log_scale(
            _objective,
            t_min=t_min,
            t_max=t_max,
            num_coarse_steps=num_coarse_steps,
            max_refine_iter=max_refine_iter,
            log_tol=log_tol,
        )

    dim_value = spread_dim_from_distance_matrix(
        distance_matrix,
        scale=t_star,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
        output_dtype=output_dtype,
    )

    return DimMaxResult(dim=dim_value, scale=t_star)
