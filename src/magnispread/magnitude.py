import warnings

import torch


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
