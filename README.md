# magnispread

`magnispread` provides PyTorch-implementations of metric space magnitude and spread. These are described, for instance, in [Limbeck et al. (2024)](#references) and [Willerton (2015)](#references), respectively.

The package supports computation of magnitude and spread using Euclidean and cosine distances, as well as from a matrix of pairwise distances directly.

## Example Usage

**Functional API:**

```python
import torch
from magnispread import magnitude, spread

X = torch.randn(16, 64, requires_grad=True)

loss_magnitude = magnitude(
    X,
    t=1.0,
)
loss_magnitude.backward()

loss_spread = spread(
    X,
    t=1.0,
)
loss_spread.backward()
```

Precomputed pairwise distances are also supported:

```python
import torch
from magnispread import magnitude, spread

X = torch.randn(16, 64)
D = torch.cdist(X, X, p=2)

# Usage for magnitude
loss_magnitude = magnitude(
    D,
    metric="precomputed",
    t=1.0,
)

# Usage for spread
loss_spread = spread(
    D,
    metric="precomputed",
    t=1.0,
)
```

By default, magnitude is computed via Cholesky decomposition (see Appendix A.5 in [Limbeck et al. (2024)](#references) for details).
This can be changed by setting `solver="inverse"`, in which case magnitude is computed as the sum of the entries of the inverse of the similarity matrix.

**Module API:**

```python
import torch
from magnispread import MagLoss, SpreadLoss

X = torch.randn(32, 128, requires_grad=True)

# Usage for magnitude
criterion_magnitude = MagLoss(
    metric="cosine",
    t=1.0,
)

loss_magnitude = criterion_magnitude(X)
loss_magnitude.backward()

# Usage for spread
criterion_spread = SpreadLoss(
    metric="cosine",
    t=1.0,
)

loss_spread = criterion_spread(X)
loss_spread.backward()
```

The same `metric="precomputed"` mode is available in `MagLoss` and
`SpreadLoss` when the module input is already a pairwise-distance matrix.

## Notes

- `metric="euclidean"` and `metric="cosine"` expect a point cloud with shape `(n_samples, n_features)`.
- `metric="precomputed"` expects a square 2D tensor containing pairwise distances.
- `t` must be positive.
- The implementation uses double precision and symmetrization for the computation of the similarity matrix $\zeta$ by default for numerical stability. This behavior can be turned off by setting ``use_double_precision=False`` and ``symmetrize=False``, respectively.
- If computation of magnitude fails because of `torch.linalg.cholesky`, increase `jitter` (default value if `1e-6`), or set `solver="inverse"` (default value is `"cholesky"`).

## References

1. Katharina Limbeck, Rayna Andreeva, Rik Sarkar, and Bastian Rieck. 2024. [Metric space Magnitude for Evaluating the Diversity of Latent Representations](https://doi.org/10.52202/079017-3937). In *Advances in Neural Information Processing Systems*, volume 37, pages 123911–123953. Curran Associates, Inc.

2. Simon Willerton. 2025. [Spread: a measure of the size of metric spaces](https://arxiv.org/abs/2508.08025). *Preprint*, arXiv:2508.08025. ArXiv:2508.08025 [math.AT], https://arxiv.org/abs/2508.08025.
