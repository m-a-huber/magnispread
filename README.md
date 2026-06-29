# mag-loss

`mag-loss` provides a magnitude-based PyTorch loss that builds a similarity
matrix $\zeta$ from pairwise distances and returns:

$$
\mathbf{1}^\top \zeta^{-1} \mathbf{1}.
$$

The package supports Euclidean as well as cosine distance.

## Usage

Functional API:

```python
import torch
from mag_loss import mag_loss

X = torch.randn(16, 64, requires_grad=True)
loss = mag_loss(
    X,
    metric="euclidean",
    t=1.0,
)
loss.backward()
```

Module API:

```python
import torch
from mag_loss import MagLoss

criterion = MagLoss(
    metric="cosine",
    t=0.5,
    jitter=1e-6,
)
X = torch.randn(32, 128, requires_grad=True)
loss = criterion(X)
loss.backward()
```

## Notes

- Input must be a point cloud, given as a 2D-tensor with shape `(n_samples, n_features)`.
- `t` must be positive.
- `jitter` must be non-negative.
- The implementation uses double precision and symmetrization for the computation of the similarity matrix $\zeta$ by default for numerical stability. This behavior can be turned off by setting ``use_double_precision=False`` and ``symmetrize=False``, respectively.
- If `torch.linalg.cholesky` fails for difficult batches, increase `jitter` (for example, from `1e-6` to `1e-4`).
