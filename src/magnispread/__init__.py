from .functional import (
    DimMaxResult,
    magnitude,
    magnitude_dim,
    magnitude_dim_max,
    spread,
    spread_dim,
    spread_dim_max,
)
from .losses import (
    MagDimLoss,
    MagDimMaxLoss,
    MagLoss,
    SpreadDimLoss,
    SpreadDimMaxLoss,
    SpreadLoss,
)

__all__ = [
    "DimMaxResult",
    "MagDimLoss",
    "MagDimMaxLoss",
    "MagLoss",
    "SpreadDimLoss",
    "SpreadDimMaxLoss",
    "SpreadLoss",
    "magnitude",
    "magnitude_dim",
    "magnitude_dim_max",
    "spread",
    "spread_dim",
    "spread_dim_max",
]
