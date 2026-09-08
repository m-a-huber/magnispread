import warnings

import torch

from .matrices import get_similarity_matrix


def _magnitude_from_cholesky(
    L: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    ones = torch.ones(
        len(L),
        1,
        dtype=L.dtype,
        device=L.device,
    )
    x = torch.linalg.solve_triangular(L, ones, upper=False)
    return (x.mT @ x).squeeze().to(dtype=output_dtype)


def magnitude_from_similarity_matrix(
    similarity_matrix: torch.Tensor,
    solver: str,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if solver == "auto":
        L, info = torch.linalg.cholesky_ex(similarity_matrix, upper=False)
        if info.item() == 0:
            return _magnitude_from_cholesky(
                L,
                output_dtype=output_dtype,
            )
        warnings.warn(
            "Cholesky decomposition failed; falling back to solver='linsolve'",
            stacklevel=2,
        )
        return magnitude_from_similarity_matrix(
            similarity_matrix,
            solver="linsolve",
            output_dtype=output_dtype,
        )
    elif solver == "cholesky":
        L = torch.linalg.cholesky(similarity_matrix, upper=False)
        return _magnitude_from_cholesky(
            L,
            output_dtype=output_dtype,
        )
    elif solver == "linsolve":
        ones = torch.ones(
            len(similarity_matrix),
            1,
            dtype=similarity_matrix.dtype,
            device=similarity_matrix.device,
        )
        w = torch.linalg.solve(similarity_matrix, ones)
        return w.sum().to(dtype=output_dtype)
    elif solver == "inverse":
        return torch.linalg.inv(similarity_matrix).sum().to(dtype=output_dtype)
    else:
        raise ValueError(
            "Solver must be one of 'auto', 'cholesky', 'linsolve', or "
            f"'inverse', got '{solver}'"
        )


def _magnitude_weights_from_similarity_matrix(
    similarity_matrix: torch.Tensor,
    solver: str,
) -> torch.Tensor:
    ones = torch.ones(
        len(similarity_matrix),
        1,
        dtype=similarity_matrix.dtype,
        device=similarity_matrix.device,
    )
    if solver == "auto":
        L, info = torch.linalg.cholesky_ex(similarity_matrix, upper=False)
        if info.item() == 0:
            x = torch.linalg.solve_triangular(L, ones, upper=False)
            return torch.linalg.solve_triangular(L.mT, x, upper=True).squeeze(
                -1
            )
        warnings.warn(
            "Cholesky decomposition failed; falling back to solver='linsolve'",
            stacklevel=2,
        )
        return _magnitude_weights_from_similarity_matrix(
            similarity_matrix,
            solver="linsolve",
        )
    elif solver == "cholesky":
        L = torch.linalg.cholesky(similarity_matrix, upper=False)
        x = torch.linalg.solve_triangular(L, ones, upper=False)
        return torch.linalg.solve_triangular(L.mT, x, upper=True).squeeze(-1)
    elif solver == "linsolve":
        return torch.linalg.solve(similarity_matrix, ones).squeeze(-1)
    elif solver == "inverse":
        return torch.linalg.inv(similarity_matrix).sum(dim=1)
    else:
        raise ValueError(
            "Solver must be one of 'auto', 'cholesky', 'linsolve', or "
            f"'inverse', got '{solver}'"
        )


def magnitude_dim_from_distance_matrix(
    distance_matrix: torch.Tensor,
    scale: float,
    symmetrize: bool = True,
    force_diagonal: bool = True,
    jitter: float = 1e-6,
    solver: str = "auto",
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    similarity_matrix = get_similarity_matrix(
        distance_matrix,
        scale=scale,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
    )

    if jitter:
        similarity_matrix = similarity_matrix + jitter * torch.eye(
            len(distance_matrix),
            dtype=similarity_matrix.dtype,
            device=similarity_matrix.device,
        )

    w = _magnitude_weights_from_similarity_matrix(similarity_matrix, solver)
    magnitude = w.sum()
    derivative_term = w @ ((distance_matrix * similarity_matrix) @ w)

    return ((scale / magnitude) * derivative_term).to(dtype=output_dtype)
