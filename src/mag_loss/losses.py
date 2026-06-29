import torch.nn as nn

from .functional import mag_loss


class MagLoss(nn.Module):
    def __init__(
        self,
        metric: str = "euclidean",
        t: float = 1.0,
        use_double_precision: bool = True,
        symmetrize: bool = True,
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.metric = metric
        self.t = t
        self.use_double_precision = use_double_precision
        self.symmetrize = symmetrize
        self.jitter = jitter

    def forward(self, pred):
        return mag_loss(
            pred,
            metric=self.metric,
            t=self.t,
            use_double_precision=self.use_double_precision,
            symmetrize=self.symmetrize,
            jitter=self.jitter,
        )
