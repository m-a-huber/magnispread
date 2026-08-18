import torch.nn as nn

from .functional import magnitude, spread


class MagLoss(nn.Module):
    """Module wrapper for metric space magnitude. Delegates computation to
    :func:`magnispread.functional.magnitude`."""

    def __init__(
        self,
        metric: str = "euclidean",
        scale: float = 1.0,
        use_double_precision: bool = False,
        symmetrize: bool = False,
        jitter: float = 1e-6,
        solver: str = "cholesky",
    ):
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
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
        symmetrize: bool = False,
    ):
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize

    def forward(self, pred):
        """Compute spread via `magnispread.functional.spread`."""
        return spread(
            pred,
            metric=self.metric,
            scale=self.scale,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
        )
