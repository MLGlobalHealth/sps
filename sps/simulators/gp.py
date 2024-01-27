import jax.numpy as jnp
from jax import config, jit, random, vmap
from dataclasses import dataclass
from jax.random import PRNGKey
from jax.tree_util import Partial
from . import kernels
from .priors import Prior
from jaxtyped import Float, Array

# improves numerical stability for small lengthscales
config.update("jax_enable_x64", True)


@dataclass
class GP:
    kernel: str = "rbf"
    variance: Prior = Prior("fixed", {"value": 1})
    lengthscale: Prior = Prior("beta", {"a": 2.5, "b": 6.0})
    noise: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        self.kernel_func = getattr(kernels, self.kernel)
        self.key = PRNGKey(self.seed)

    def simulate(
        self,
        locations: Float[Array, "... D"],
        batch_size: int = 1,
        approx: bool = False,
    ):
        self.key, rng_var, rng_ls, rng_z = random.split(self.key, 4)
        var = self.variance.sample(rng_var, batch_size)
        ls = self.lengthscale.sample(rng_ls, batch_size)
        factorize = vmap(kronecker if approx else cholesky, in_axes=(None, None, 0, 0))
        L = factorize(self.kernel_func, locations, var, ls)
        z = random.normal(rng_z, shape=(batch_size, *locations.shape[:-1]))
        return L @ z


@jit
def kronecker(kernel, locations, var, ls):
    pass


@jit
def cholesky(kernel, locations, var, ls):
    K = kernel(locations, var, ls)
    return jnp.linalg.cholesky(K)
