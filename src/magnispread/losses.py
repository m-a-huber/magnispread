import torch.nn as nn

from .functional import (
    magnitude,
    magnitude_dim,
    magnitude_dim_max,
    spread,
    spread_dim,
    spread_dim_max,
)


class MagLoss(nn.Module):
    """Module wrapper for metric space magnitude. Delegates computation to
    :func:`magnispread.functional.magnitude`."""

    def __init__(
        self,
        metric: str = "euclidean",
        scale: float = 1.0,
        use_double_precision: bool = False,
        symmetrize: bool = True,
        force_diagonal: bool | None = None,
        jitter: float = 1e-6,
        solver: str = "auto",
    ):
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.force_diagonal = force_diagonal
        self.jitter = jitter
        self.solver = solver

    def forward(self, pred):
        """Compute magnitude via `magnispread.functional.magnitude`."""
        return magnitude(
            pred,
            metric=self.metric,
            scale=self.scale,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            force_diagonal=self.force_diagonal,
            jitter=self.jitter,
            solver=self.solver,
        )


class MagDimLoss(nn.Module):
    """Module wrapper for magnitude dimension. Delegates computation to
    :func:`magnispread.functional.magnitude_dim`."""

    def __init__(
        self,
        metric: str = "euclidean",
        scale: float = 1.0,
        use_double_precision: bool = False,
        symmetrize: bool = True,
        force_diagonal: bool | None = None,
        jitter: float = 1e-6,
        solver: str = "auto",
    ):
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.force_diagonal = force_diagonal
        self.jitter = jitter
        self.solver = solver

    def forward(self, pred):
        """Compute magnitude dimension via
        `magnispread.functional.magnitude_dim`."""
        return magnitude_dim(
            pred,
            metric=self.metric,
            scale=self.scale,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            force_diagonal=self.force_diagonal,
            jitter=self.jitter,
            solver=self.solver,
        )


class MagDimMaxLoss(nn.Module):
    """Module wrapper for scale-agnostic magnitude dimension. Delegates
    computation to :func:`magnispread.functional.magnitude_dim_max`."""

    def __init__(
        self,
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
    ):
        super().__init__()
        self.metric = metric
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.force_diagonal = force_diagonal
        self.jitter = jitter
        self.solver = solver
        self.t_min = t_min
        self.t_max = t_max
        self.num_coarse_steps = num_coarse_steps
        self.max_refine_iter = max_refine_iter
        self.log_tol = log_tol

    def forward(self, pred):
        """Compute scale-agnostic magnitude dimension via
        `magnispread.functional.magnitude_dim_max`."""
        return magnitude_dim_max(
            pred,
            metric=self.metric,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            force_diagonal=self.force_diagonal,
            jitter=self.jitter,
            solver=self.solver,
            t_min=self.t_min,
            t_max=self.t_max,
            num_coarse_steps=self.num_coarse_steps,
            max_refine_iter=self.max_refine_iter,
            log_tol=self.log_tol,
        )


class SpreadLoss(nn.Module):
    """Module wrapper for metric space spread. Delegates computation to
    :func:`magnispread.functional.spread`."""

    def __init__(
        self,
        metric: str = "euclidean",
        scale: float = 1.0,
        use_double_precision: bool = False,
        symmetrize: bool = True,
        force_diagonal: bool | None = None,
    ):
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.force_diagonal = force_diagonal

    def forward(self, pred):
        """Compute spread via `magnispread.functional.spread`."""
        return spread(
            pred,
            metric=self.metric,
            scale=self.scale,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            force_diagonal=self.force_diagonal,
        )


class SpreadDimLoss(nn.Module):
    """Module wrapper for spread dimension. Delegates computation to
    :func:`magnispread.functional.spread_dim`."""

    def __init__(
        self,
        metric: str = "euclidean",
        scale: float = 1.0,
        use_double_precision: bool = False,
        symmetrize: bool = True,
        force_diagonal: bool | None = None,
    ):
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.force_diagonal = force_diagonal

    def forward(self, pred):
        """Compute spread dimension via `magnispread.functional.spread_dim`."""
        return spread_dim(
            pred,
            metric=self.metric,
            scale=self.scale,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            force_diagonal=self.force_diagonal,
        )


class SpreadDimMaxLoss(nn.Module):
    """Module wrapper for scale-agnostic spread dimension. Delegates
    computation to :func:`magnispread.functional.spread_dim_max`."""

    def __init__(
        self,
        metric: str = "euclidean",
        use_double_precision: bool = False,
        symmetrize: bool = True,
        force_diagonal: bool | None = None,
        t_min: float = 1e-3,
        t_max: float = 1e3,
        num_coarse_steps: int = 50,
        max_refine_iter: int = 60,
        log_tol: float = 1e-6,
    ):
        super().__init__()
        self.metric = metric
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.force_diagonal = force_diagonal
        self.t_min = t_min
        self.t_max = t_max
        self.num_coarse_steps = num_coarse_steps
        self.max_refine_iter = max_refine_iter
        self.log_tol = log_tol

    def forward(self, pred):
        """Compute scale-agnostic spread dimension via
        `magnispread.functional.spread_dim_max`."""
        return spread_dim_max(
            pred,
            metric=self.metric,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            force_diagonal=self.force_diagonal,
            t_min=self.t_min,
            t_max=self.t_max,
            num_coarse_steps=self.num_coarse_steps,
            max_refine_iter=self.max_refine_iter,
            log_tol=self.log_tol,
        )
