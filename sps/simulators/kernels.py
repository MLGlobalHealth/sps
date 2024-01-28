import jax.numpy as jnp
from jax import jit
from . import shared_types as T


@jit
def _prepare_dims(x: T.Locations, y: T.Locations) -> tuple[T.Locations, T.Locations]:
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y


@jit
def l2_dist_sq(x: T.Locations, y: T.Locations):
    """L2 distance between two Locations arrays."""
    x, y = _prepare_dims(x, y)
    dsq = (x**2).sum(-1)[:, None] + (y**2).sum(-1).T - 2 * x @ y.T
    # can produce small (negative) values on the diagonal,
    # e.g. distance d(x_i, x_i) = -1.2384e-8, so fill with 0s
    return jnp.fill_diagonal(dsq, 0, inplace=False)


@jit
def rbf(
    x: T.Locations,
    y: T.Locations,
    variance: T.Variance,
    lengthscale: T.Lengthscale,
) -> T.Covariance:
    """K(x, y) = variance * exp{-||x-y||^2 / (2 * lengthscale^2)}"""
    return variance * jnp.exp(-l2_dist_sq(x, y) / (2 * lengthscale**2))


@jit
def matern_3_2(
    x: T.Locations,
    y: T.Locations,
    variance: T.Variance,
    lengthscale: T.Lengthscale,
) -> T.Covariance:
    """K(x, y) = variance * (1 + √3 * ||x-y|| / lengthscale) * exp{-√3 * ||x-y|| / lengthscale}"""
    d = l2_dist_sq(x, y) ** (1 / 2)
    sqrt3 = 3.0 ** (1 / 2)
    return variance * (1 + sqrt3 * d / lengthscale) * jnp.exp(-sqrt3 * d / lengthscale)


@jit
def matern_5_2(
    x: T.Locations,
    y: T.Locations,
    variance: T.Variance,
    lengthscale: T.Lengthscale,
) -> T.Covariance:
    """K(x, y) = variance * (1 + √5 * ||x-y|| / lengthscale + 5/3 * ||x-y||^2 / lengthscale^2) * exp{-√5 * ||x-y|| / lengthscale}"""
    dsq = l2_dist_sq(x, y)
    d = jnp.sqrt(dsq)
    sqrt5 = jnp.sqrt(5.0)
    return (
        variance
        * (1 + sqrt5 * d / lengthscale + 5 / 3 * dsq / lengthscale**2)
        * jnp.exp(-sqrt5 * d / lengthscale)
    )
