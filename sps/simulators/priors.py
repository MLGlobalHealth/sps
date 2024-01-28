from dataclasses import dataclass
import jax.numpy as jnp
from jax import random, jit
from jax.tree_util import Partial
from jaxtyping import Float, Array, PRNGKeyArray, Num
from collections.abc import Sequence


@dataclass
class Prior:
    dist: str
    kwargs: dict[str, Num]

    def __post_init__(self):
        dist_func = globals().get(self.dist, getattr(random, self.dist, None))
        self.dist_func = Partial(dist_func, **self.kwargs)

    def __hash__(self):
        return hash((self.dist, repr(self.kwargs)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def sample(self, key: PRNGKeyArray, shape: Sequence[int]) -> Float[Array, "..."]:
        return self.dist_func(key, shape=shape)


# JAX doesn't have a lambda parameterized exponential (28-01-2024)
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
def exponential(
    key: PRNGKeyArray,
    lam: Float,
    shape: Sequence[int],
) -> Float[Array, "..."]:
    """Exponential parameterized by lambda `lam`."""
    return 1 / lam * random.exponential(key, shape)


def fixed(
    key: PRNGKeyArray,
    value: Float,
    shape: Sequence[int],
) -> Float[Array, "..."]:
    """Fixed distribution."""
    return jnp.full(shape, value)
