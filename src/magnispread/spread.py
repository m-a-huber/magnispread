import torch

from .matrices import get_similarity_matrix


def spread_from_similarity_matrix(
    similarity_matrix: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (1 / similarity_matrix.sum(dim=1)).sum().to(dtype=output_dtype)


def spread_dim_from_distance_matrix(
    distance_matrix: torch.Tensor,
    scale: float,
    symmetrize: bool = True,
    force_diagonal: bool = True,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    similarity_matrix = get_similarity_matrix(
        distance_matrix,
        scale=scale,
        symmetrize=symmetrize,
        force_diagonal=force_diagonal,
    )
    row_sums = similarity_matrix.sum(dim=1)
    spread = (1 / row_sums).sum()

    factor_1 = scale / spread
    factor_2 = (
        (distance_matrix * similarity_matrix).sum(dim=1) / (row_sums**2)
    ).sum()
    return (factor_1 * factor_2).to(dtype=output_dtype)
