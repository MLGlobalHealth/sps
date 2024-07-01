import jax.numpy as jnp
from jax import config, jit
from jax.typing import ArrayLike

# improves numerical stability for small lengthscales
config.update("jax_enable_x64", True)


@jit
def _prepare_dims(x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Prepares dims for use in kernel functions.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        `[N_x, D]` and `[N_y, D]` arrays.
    """
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y


@jit
def l2_dist_sq(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    r"""L2 distance between two [..., D] arrays.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Uses $\mathbf{D}=\mathbf{\hat{x}}\mathbf{1}^\intercal_{N_y}-2\mathbf{XY}^\intercal-1_{N_x}\mathbf{\hat{y}}^\intercal$
    from Probabilistic Machine Learning by Kevin Murphy, p.245.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = _prepare_dims(x, y)
    _1_N_x, _1_N_y = jnp.ones(x.shape[0]), jnp.ones(y.shape[0])
    x_hat, y_hat = (x**2).sum(-1), (y**2).sum(-1)
    return jnp.outer(x_hat, _1_N_y) - 2 * x @ y.T + jnp.outer(_1_N_x, y_hat)


@jit
def rbf(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Radial Basis kernel, aka Squared Exponential kernel.

    $K(x, y) = \text{var}\cdot\exp\left(-\frac{\lVert x-y\rVert^2}{2\text{ls}^2}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    return var * jnp.exp(-l2_dist_sq(x, y) / (2 * ls**2))


@jit
def periodic(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
    period: float = 2 * jnp.pi,
) -> ArrayLike:
    r"""Periodic kernel.

    $K(x, y) = \text{var}\cdot\exp\left(-\frac{2\sin^2\frac{\left(\lVert x-y\rVert\right)}{\text{period}}}{2\text{ls}^2}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    x, y = _prepare_dims(x, y)
    return var * jnp.exp(-2 / ls**2 * jnp.sin(jnp.pi * jnp.abs(x - y.T) / period) ** 2)


@jit
def matern_3_2(
    x: ArrayLike,
    y: ArrayLike,
    variance: float,
    lengthscale: float,
) -> ArrayLike:
    r"""Matern 3/2 kernel.

    $K(x, y) = \text{var}\cdot\left(1 + \frac{\sqrt{3}\lVert x-y\rVert}{\text{ls}}\right)\cdot\exp\left(-\frac{\sqrt{3}\lVert x-y\rVert}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    d = jnp.sqrt(l2_dist_sq(x, y))
    sqrt3 = 3.0 ** (1 / 2)
    return variance * (1 + sqrt3 * d / lengthscale) * jnp.exp(-sqrt3 * d / lengthscale)


@jit
def matern_5_2(
    x: ArrayLike,
    y: ArrayLike,
    variance: float,
    lengthscale: float,
) -> ArrayLike:
    r"""Matern 5/2 kernel.

    $K(x, y) = \text{var}\cdot\left(1 + \frac{\sqrt{5}\lVert x-y\rVert}{\text{ls}} + \frac{5}{3}\cdot\frac{\lVert x-y\rVert^2}{\text{ls}^2}\right)\cdot\exp\left(-\frac{\sqrt{5}\lVert x-y\rVert}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    dsq = l2_dist_sq(x, y)
    d = jnp.sqrt(dsq)
    sqrt5 = jnp.sqrt(5.0)
    return (
        variance
        * (1 + sqrt5 * d / lengthscale + 5 / 3 * dsq / lengthscale**2)
        * jnp.exp(-sqrt5 * d / lengthscale)
    )
