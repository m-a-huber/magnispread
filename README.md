# MagniSpread

`magnispread` provides PyTorch implementations of metric space magnitude, magnitude dimension, spread, and spread dimension. These are described, for instance, in [Limbeck et al. (2024)](#references), [Meckes (2015)](#references), [Willerton (2025)](#references), and [Dunne (2023)](#references), respectively.

**Note:** We treat magnitude dimension and spread dimension as functions depending on the scale parameter $t>0$, defined as

$$\mathrm{dim_{X}^{Mag}}(t)=\frac{d\log\mathrm{Mag_{X}}(t)}{d\log t}=\frac{t}{\mathrm{Mag_{X}}(t)}\frac{d\mathrm{Mag_{X}}(t)}{dt}$$

and

$$\mathrm{dim_{X}^{Spr}}(t)=\frac{d\log\mathrm{Spr_{X}}(t)}{d\log t}=\frac{t}{\mathrm{Spr_{X}}(t)}\frac{d\mathrm{Spr_{X}}(t)}{dt},$$

respectively. This is in contrast to many sources that define magnitude dimension and spread dimension as the limit as $t\to\infty$ in the respective expressions above.

In addition to the dimension curves above, the package also provides functions that compute the maximal magnitude dimension and spread dimension across all scales. This serves as a heuristic for the dimension of a finite metric space, as the limit as $t\to\infty$ of the dimension curves above is trivially zero for finite spaces.

The package supports computation of magnitude, spread, spread dimension, and magnitude dimension using Euclidean and cosine distances, as well as from a matrix of pairwise distances directly.

## Example Usage

**Functional API:**

```python
import torch
from magnispread import (
    magnitude,
    magnitude_dim,
    magnitude_dim_max,
    spread,
    spread_dim,
    spread_dim_max,
)

X = torch.randn(16, 64, requires_grad=True)

# Usage for magnitude
loss_magnitude = magnitude(
    X,
    scale=1.0,
)
loss_magnitude.backward()

# Usage for magnitude dimension
loss_magnitude_dim = magnitude_dim(
    X,
    scale=1.0,
)
loss_magnitude_dim.backward()

# Usage for maximal magnitude dimension
result_magnitude_dim_max = magnitude_dim_max(X)
result_magnitude_dim_max.dim.backward()

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

# Usage for maximal spread dimension
result_spread_dim_max = spread_dim_max(X)
result_spread_dim_max.dim.backward()
```

Precomputed pairwise distances are also supported:

```python
import torch
from magnispread import (
    magnitude,
    magnitude_dim,
    magnitude_dim_max,
    spread,
    spread_dim,
    spread_dim_max,
)

X = torch.randn(16, 64, requires_grad=True)
D = torch.cdist(X, X, p=2)

# Usage for magnitude
loss_magnitude = magnitude(
    D,
    metric="precomputed",
    scale=1.0,
)

# Usage for magnitude dimension
loss_magnitude_dim = magnitude_dim(
    D,
    metric="precomputed",
    scale=1.0,
)
loss_magnitude_dim.backward()

# Usage for maximal magnitude dimension
result_magnitude_dim_max = magnitude_dim_max(D, metric="precomputed")
result_magnitude_dim_max.dim.backward()

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

# Usage for maximal spread dimension
result_spread_dim_max = spread_dim_max(D, metric="precomputed")
result_spread_dim_max.dim.backward()
```

**Module API:**

```python
import torch
from magnispread import (
    MagDimLoss,
    MagDimMaxLoss,
    MagLoss,
    SpreadDimLoss,
    SpreadDimMaxLoss,
    SpreadLoss,
)

X = torch.randn(32, 128, requires_grad=True)

# Usage for magnitude
criterion_magnitude = MagLoss(
    metric="cosine",
    scale=1.0,
)

loss_magnitude = criterion_magnitude(X)
loss_magnitude.backward()

# Usage for magnitude dimension
criterion_magnitude_dim = MagDimLoss(
    metric="cosine",
    scale=1.0,
)

loss_magnitude_dim = criterion_magnitude_dim(X)
loss_magnitude_dim.backward()

# Usage for maximal magnitude dimension
criterion_magnitude_dim_max = MagDimMaxLoss(metric="cosine")

result_magnitude_dim_max = criterion_magnitude_dim_max(X)
result_magnitude_dim_max.dim.backward()

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

# Usage for maximal spread dimension
criterion_spread_dim_max = SpreadDimMaxLoss(metric="cosine")

result_spread_dim_max = criterion_spread_dim_max(X)
result_spread_dim_max.dim.backward()
```

The same `metric="precomputed"` mode is available in `MagLoss`,
`SpreadLoss`, `SpreadDimLoss`, `MagDimLoss`, `MagDimMaxLoss`, and
`SpreadDimMaxLoss` when the module input is already a pairwise-distance
matrix.

## Notes

- `metric="euclidean"` and `metric="cosine"` expect a point cloud with shape `(n_samples, n_features)`.
- `metric="precomputed"` expects a square 2D tensor containing pairwise distances.
- `scale` must be positive.
- The diagonal of the similarity matrix is forced to exactly `1.0` by default for `metric="euclidean"` and `metric="cosine"` (guarding against float32-precision noise in the underlying distance computation that would otherwise destabilize the computation at large `scale`), but not for `metric="precomputed"`. This is controlled by the `force_diagonal` argument (available on the functional API as well as `MagLoss`, `MagDimLoss`, `MagDimMaxLoss`, `SpreadLoss`, `SpreadDimLoss`, and `SpreadDimMaxLoss`), which can be passed explicitly to override either default.
- The similarity matrix is symmetrized by default for the same numerical-stability reasons. Set `symmetrize` to `False` to opt out.
- Numerical stability can be further improved by using double precision for internal computations, via `use_double_precision=True`.
- The returned tensor's dtype matches `X`'s floating-point dtype (or `float32` if `X` is not floating-point), regardless of `use_double_precision`.
- `magnitude` and `magnitude_dim` both add a small `jitter` (default `1e-6`) to the diagonal of the similarity matrix before solving, since that matrix can otherwise become singular or ill-conditioned whenever points coincide or cluster tightly. `magnitude_dim` needs the magnitude weight vector (the solution of `similarity_matrix @ weights = 1`) to compute the scale-derivative of magnitude, so it solves the same kind of linear system as `magnitude` itself and shares its `jitter`/`solver` arguments; `spread` and `spread_dim` need no such solve, since spread only depends on row sums of the similarity matrix. `magnitude_dim_max` solves this same linear system once per candidate scale visited during its search and shares the same `jitter`/`solver` arguments; `spread_dim_max`, like `spread`/`spread_dim`, needs no such solve.
- By default, `magnitude` and `magnitude_dim` use `solver="auto"`: they first attempt Cholesky decomposition (see Appendix A.5 in [Limbeck et al. (2024)](#references) for details) and, if that fails, fall back to `solver="linsolve"`, which solves `similarity_matrix @ weights = 1` directly (emitting a `UserWarning` when this happens). `solver="inverse"` computes the weight vector by directly inverting the similarity matrix instead; `solver="cholesky"` and `solver="linsolve"` can also be set explicitly to skip the fallback logic. `magnitude_dim_max` uses the same `solver` logic at each candidate scale, but suppresses the many `UserWarning` during its internal search; a genuine solver failure at the final, reported scale still raises it as usual.
- `magnitude_dim_max` and `spread_dim_max` (and their module counterparts, `MagDimMaxLoss` and `SpreadDimMaxLoss`) return a `DimMaxResult` namedtuple with two fields: `dim`, the maximal dimension value `sup_{t>0} dim(t)` as a differentiable scalar tensor, and `scale`, the Python `float` value of the scale `t*` at which that maximum is attained.
- The maximizing scale `t*` is found by a derivative-free search: a coarse, log-spaced grid scan over `[t_min, t_max]` (defaults `1e-3` and `1e3`) brackets the global maximum, followed by golden-section refinement in `log(t)`-space for cheap, high-precision convergence. The search's cost and precision can be tuned via `t_min`, `t_max`, `num_coarse_steps` (default `50`), `max_refine_iter` (default `60`), and `log_tol` (default `1e-6`). If the coarse grid's maximum lands on the boundary of `[t_min, t_max]`, a `UserWarning` is emitted, since the true maximizer may lie outside the searched range — widen `t_min`/`t_max` in that case.
- Gradients of `dim` with respect to the input are exact, not approximate: the search itself runs entirely inside `torch.no_grad()`, and exactly one further, fully differentiable evaluation of the underlying `dim(t)` kernel is made at the fixed scale `t*`. By the envelope theorem, since `t*` is a stationary point of `dim(t)`, differentiating this single fixed-scale evaluation gives the same gradient as differentiating through the full maximization.
- As with `magnitude`/`magnitude_dim`, `use_double_precision=True` is recommended for `magnitude_dim_max`/`spread_dim_max` more often than for a single fixed `scale`, since the search sweeps into small-`t` regimes where the similarity matrix is more likely to become nearly singular.
- Empty point clouds are rejected by all six functional APIs.

## References

1. Katharina Limbeck, Rayna Andreeva, Rik Sarkar, and Bastian Rieck. 2024. [Metric space Magnitude for Evaluating the Diversity of Latent Representations](https://doi.org/10.52202/079017-3937). In *Advances in Neural Information Processing Systems*, volume 37, pages 123911–123953. Curran Associates, Inc.

1. Mark W. Meckes. 2025. [Magnitude, Diversity, Capacities, and Dimensions of Metric Spaces](https://doi.org/10.1007/s11118-014-9444-3). In *Potential Anal*, volume 42, pages 549–572.

3. Kevin Dunne. 2023. [Metric Space Spread, Intrinsic Dimension and the Manifold Hypothesis](https://arxiv.org/abs/2308.01382). *Preprint*, arXiv:2308.01382. ArXiv:2308.01382 [math.MG], https://arxiv.org/abs/2308.01382.

4. Simon Willerton. 2025. [Spread: a measure of the size of metric spaces](https://arxiv.org/abs/2508.08025). *Preprint*, arXiv:2508.08025. ArXiv:2508.08025 [math.AT], https://arxiv.org/abs/2508.08025.