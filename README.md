# MagniSpread

`magnispread` provides PyTorch implementations of metric space magnitude, spread, and spread dimension. These are described, for instance, in [Limbeck et al. (2024)](#references), [Willerton (2025)](#references) and [Dunne (2023)](#references), respectively.

The package supports computation of magnitude, spread, and spread dimension using Euclidean and cosine distances, as well as from a matrix of pairwise distances directly.

## Example Usage

**Functional API:**

```python
import torch
from magnispread import magnitude, spread, spread_dim

X = torch.randn(16, 64, requires_grad=True)

# Usage for magnitude
loss_magnitude = magnitude(
    X,
    scale=1.0,
)
loss_magnitude.backward()

# Usage for spread
loss_spread = spread(
    X,
    scale=1.0,
)
loss_spread.backward()

# Usage for spread dimension
loss_spread_dim = spread_dim(
    X,
    scale=1.0,
)
loss_spread_dim.backward()
```

Precomputed pairwise distances are also supported:

```python
import torch
from magnispread import magnitude, spread, spread_dim

X = torch.randn(16, 64, requires_grad=True)
D = torch.cdist(X, X, p=2)

# Usage for magnitude
loss_magnitude = magnitude(
    D,
    metric="precomputed",
    scale=1.0,
)

# Usage for spread
loss_spread = spread(
    D,
    metric="precomputed",
    scale=1.0,
)

# Usage for spread dimension
loss_spread_dim = spread_dim(
    D,
    metric="precomputed",
    scale=1.0,
)
loss_spread_dim.backward()
```

**Module API:**

```python
import torch
from magnispread import MagLoss, SpreadLoss, SpreadDimLoss

X = torch.randn(32, 128, requires_grad=True)

# Usage for magnitude
criterion_magnitude = MagLoss(
    metric="cosine",
    scale=1.0,
)

loss_magnitude = criterion_magnitude(X)
loss_magnitude.backward()

# Usage for spread
criterion_spread = SpreadLoss(
    metric="cosine",
    scale=1.0,
)

loss_spread = criterion_spread(X)
loss_spread.backward()

# Usage for spread dimension
criterion_spread_dim = SpreadDimLoss(
    metric="cosine",
    scale=1.0,
)

loss_spread_dim = criterion_spread_dim(X)
loss_spread_dim.backward()
```

The same `metric="precomputed"` mode is available in `MagLoss`,
`SpreadLoss` and `SpreadDimLoss` when the module input is already a pairwise-distance matrix.

## Notes

- `metric="euclidean"` and `metric="cosine"` expect a point cloud with shape `(n_samples, n_features)`.
- `metric="precomputed"` expects a square 2D tensor containing pairwise distances.
- `scale` must be positive.
- The diagonal of the similarity matrix is forced to exactly `1.0` by default for `metric="euclidean"` and `metric="cosine"` (guarding against float32-precision noise in the underlying distance computation that would otherwise destabilize the computation at large `scale`), but not for `metric="precomputed"`. This is controlled by the `force_diagonal` argument (available on the functional API as well as `MagLoss`, `SpreadLoss`, and `SpreadDimLoss`), which can be passed explicitly to override either default.
- The similarity matrix is symmetrized by default for the same numerical-stability reasons. Set `symmetrize` to `False` to opt out.
- Numerical stability can be further improved by using double precision for internal computations, via `use_double_precision=True`.
- The returned tensor's dtype matches `X`'s floating-point dtype (or `float32` if `X` is not floating-point), regardless of `use_double_precision`.
- `magnitude` adds a small `jitter` (default `1e-6`) to the diagonal of the similarity matrix before solving, since that matrix can otherwise become singular or ill-conditioned whenever points coincide or cluster tightly.
- By default, `magnitude` uses `solver="auto"`: it first attempts Cholesky decomposition (see Appendix A.5 in [Limbeck et al. (2024)](#references) for details) and, if that fails, falls back to `solver="linsolve"`, which solves `similarity_matrix @ weights = 1` directly (emitting a `UserWarning` when this happens). `solver="inverse"` computes magnitude by directly inverting the similarity matrix instead; `solver="cholesky"` and `solver="linsolve"` can also be set explicitly to skip the fallback logic.
- Empty point clouds are rejected by all three functional APIs.

## References

1. Katharina Limbeck, Rayna Andreeva, Rik Sarkar, and Bastian Rieck. 2024. [Metric space Magnitude for Evaluating the Diversity of Latent Representations](https://doi.org/10.52202/079017-3937). In *Advances in Neural Information Processing Systems*, volume 37, pages 123911–123953. Curran Associates, Inc.

2. Kevin Dunne. 2023. [Metric Space Spread, Intrinsic Dimension and the Manifold Hypothesis](https://arxiv.org/abs/2308.01382). *Preprint*, arXiv:2308.01382. ArXiv:2308.01382 [math.MG], https://arxiv.org/abs/2308.01382.

3. Simon Willerton. 2025. [Spread: a measure of the size of metric spaces](https://arxiv.org/abs/2508.08025). *Preprint*, arXiv:2508.08025. ArXiv:2508.08025 [math.AT], https://arxiv.org/abs/2508.08025.