import torch.nn as nn

from .functional import magnitude, spread, spread_dim


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
