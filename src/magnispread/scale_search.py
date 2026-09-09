import math
import warnings
from collections.abc import Callable
from typing import NamedTuple


class ScaleSearchResult(NamedTuple):
    """Result of maximizing a scalar objective over a positive scale range."""

    t: float
    value: float


def maximize_over_log_scale(
    objective: Callable[[float], float],
    t_min: float = 1e-3,
    t_max: float = 1e3,
    num_coarse_steps: int = 50,
    max_refine_iter: int = 60,
    log_tol: float = 1e-6,
) -> ScaleSearchResult:
    """Maximizes a scalar objective over `t > 0` via a coarse log-spaced grid
    scan followed by golden-section refinement in `log(t)`-space.

    The coarse grid scan brackets the global maximum without assuming
    `objective` is unimodal in `log(t)`; the golden-section search then
    refines within the bracket around the grid's best point for cheap,
    high-precision convergence.

    Parameters
    ----------
    objective : Callable[[float], float]
        Function mapping a candidate scale `t > 0` to a scalar value to be
        maximized.
    t_min : float, optional
        Lower bound of the search range. Must be positive. Defaults to
        `1e-3`.
    t_max : float, optional
        Upper bound of the search range. Must be greater than `t_min`.
        Defaults to `1e3`.
    num_coarse_steps : int, optional
        Number of log-spaced grid points used to bracket the global maximum.
        Must be at least `3`. Defaults to `50`.
    max_refine_iter : int, optional
        Maximum number of golden-section refinement iterations. Defaults to
        `60`.
    log_tol : float, optional
        Convergence tolerance on the golden-section bracket's width in
        `log(t)`-space. Must be positive. Defaults to `1e-6`.

    Returns
    -------
    ScaleSearchResult
        A `NamedTuple` with fields `t` (the scale maximizing `objective`) and
        `value` (the corresponding objective value).

    Raises
    ------
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

    if t_min <= 0:
        raise ValueError(f"`t_min` must be positive, got {t_min}")
    if t_max <= t_min:
        raise ValueError(
            f"`t_max` must be greater than `t_min`, got t_min={t_min}, "
            f"t_max={t_max}"
        )
    if num_coarse_steps < 3:
        raise ValueError(
            f"`num_coarse_steps` must be at least 3, got {num_coarse_steps}"
        )
    if max_refine_iter < 0:
        raise ValueError(
            f"`max_refine_iter` must be non-negative, got {max_refine_iter}"
        )
    if log_tol <= 0:
        raise ValueError(f"`log_tol` must be positive, got {log_tol}")

    log_t_min = math.log(t_min)
    log_t_max = math.log(t_max)
    step = (log_t_max - log_t_min) / (num_coarse_steps - 1)
    log_t_grid = [log_t_min + i * step for i in range(num_coarse_steps)]
    t_grid = [math.exp(log_t) for log_t in log_t_grid]

    raw_values = [objective(t) for t in t_grid]
    key_values = [
        value if not math.isnan(value) else -math.inf for value in raw_values
    ]
    best_idx = max(range(num_coarse_steps), key=lambda i: key_values[i])

    if best_idx in (0, num_coarse_steps - 1):
        warnings.warn(
            "Coarse grid search for the scale maximizing the objective "
            f"landed on the boundary of the search range "
            f"[t_min={t_min:g}, t_max={t_max:g}] (t={t_grid[best_idx]:.4g}). "
            "The true maximizer may lie outside this range; consider "
            "widening `t_min`/`t_max`.",
            stacklevel=2,
        )
        return ScaleSearchResult(
            t=t_grid[best_idx], value=raw_values[best_idx]
        )

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a = log_t_grid[best_idx - 1]
    b = log_t_grid[best_idx + 1]

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = objective(math.exp(c))
    fd = objective(math.exp(d))

    for _ in range(max_refine_iter):
        if (b - a) < log_tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = objective(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = objective(math.exp(d))

    if fc >= fd:
        best_log_t, best_value = c, fc
    else:
        best_log_t, best_value = d, fd

    return ScaleSearchResult(t=math.exp(best_log_t), value=best_value)
