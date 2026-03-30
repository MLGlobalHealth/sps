from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random
from jax.tree_util import Partial


@dataclass
class Prior:
    """Parameterized random prior backed by `jax.random`.

    Args:
        dist: Distribution name from this module or `jax.random`.
        kwargs: Keyword arguments bound to the distribution function.
    """

    dist: str
    kwargs: dict[str, float]

    def __post_init__(self):
        """Resolve and partially apply the configured distribution function."""
        dist_func = globals().get(self.dist, getattr(random, self.dist, None))
        self.dist_func = Partial(dist_func, **self.kwargs)

    def __hash__(self):
        """Return a stable hash based on the distribution configuration."""
        return hash((self.dist, repr(self.kwargs)))

    def __eq__(self, other):
        """Compare priors by their hashed distribution configuration.

        Args:
            other: Object to compare against.

        Returns:
            Whether the two prior configurations are equal.
        """
        return hash(self) == hash(other)

    def sample(self, rng: Array, shape: Sequence[int] = (1,)) -> Array:
        """Sample from the configured prior.

        Args:
            rng: Pseudo-random key from `jax.random`.
            shape: Output shape of the samples.

        Returns:
            Sample array with shape `shape`.
        """
        return self.dist_func(rng, shape=shape)


# JAX doesn't have a parameterized normal
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html
def normal(
    rng: Array,
    mu: float,
    sigma: float,
    shape: Sequence[int],
) -> Array:
    r"""Sample from a normal distribution parameterized by mean and stddev.

    $$
    X \sim \mathcal{N}(\mu, \sigma^2)
    $$

    Args:
        rng: Pseudo-random key from `jax.random`.
        mu: Mean of the distribution.
        sigma: Standard deviation.
        shape: Output shape of the samples.

    Returns:
        Sample array with shape `shape`.
    """
    return mu + sigma * random.normal(rng, shape)


# JAX doesn't have a lambda parameterized exponential
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
def exponential(
    rng: Array,
    lam: float,
    shape: Sequence[int],
) -> Array:
    r"""Sample from an exponential distribution parameterized by rate.

    $$
    X \sim \operatorname{Exp}(\lambda), \quad
    f(x) = \lambda e^{-\lambda x}
    $$

    Args:
        rng: Pseudo-random key from `jax.random`.
        lam: Rate parameter of the exponential distribution.
        shape: Output shape of the samples.

    Returns:
        Sample array with shape `shape`.
    """
    return 1 / lam * random.exponential(rng, shape)


# JAX doesn't have a rate parameterized gamma
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html
def gamma(
    rng: Array,
    alpha: float,  # shape
    beta: float,  # rate
    shape: Sequence[int],
) -> Array:
    r"""Sample from a gamma distribution parameterized by shape and rate.

    $$
    X \sim \operatorname{Gamma}(\alpha, \beta), \quad
    f(x) \propto x^{\alpha - 1} e^{-\beta x}
    $$

    Args:
        rng: Pseudo-random key from `jax.random`.
        alpha: Shape parameter.
        beta: The rate parameter.
        shape: Output shape of the samples.

    Returns:
        Sample array with shape `shape`.
    """
    return random.gamma(rng, alpha, shape) / beta


# JAX doesn't have a lambda parameterized inverse-gamma
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html
def inverse_gamma(
    rng: Array,
    alpha: float,
    beta: float,
    shape: Sequence[int],
) -> Array:
    r"""Sample from an inverse-gamma distribution.

    This implementation samples `Y ~ Gamma(alpha, beta)` and returns `1 / Y`.

    Args:
        rng: Pseudo-random key from `jax.random`.
        alpha: Shape parameter.
        beta: Rate parameter.
        shape: Output shape of the samples.

    Returns:
        Sample array with shape `shape`.
    """
    return 1 / gamma(rng, alpha, beta, shape)


def fixed(
    rng: Array,
    value: float,
    shape: Sequence[int],
) -> Array:
    """Return a constant-valued sample array.

    This behaves like a degenerate distribution concentrated at `value`.

    Args:
        rng: Unused pseudo-random key kept for API compatibility.
        value: Scalar value to broadcast across the sample.
        shape: Output shape of the samples.

    Returns:
        Constant array with shape `shape`.
    """
    return jnp.full(shape, value)
