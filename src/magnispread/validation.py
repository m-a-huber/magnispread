import torch


def validate_inputs(
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

    if X.shape[0] == 0:
        raise ValueError("`X` must contain at least one point")

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

    if solver is not None and solver not in {
        "auto",
        "cholesky",
        "linsolve",
        "inverse",
    }:
        raise ValueError(
            "`solver` must be either 'auto', 'cholesky', 'linsolve', or "
            f"'inverse', got {solver}"
        )
