import math
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.lax import conv_general_dilated

from .priors import Prior


@dataclass
class LatticeSI:
    """A Susceptible Infected (SI) model simulated on a lattice.

    Args:
        beta: A prior over the infection rate.
        num_init: A prior over the initial number of infected (nearest integer is used).

    Returns:
        An instance of the `LatticeSI` dataclass.
    """

    beta: Prior = Prior("beta", {"a": 2, "b": 18})
    num_init: Prior = Prior("uniform", {"minval": 1, "maxval": 5})

    def simulate(
        self,
        rng: Array,
        dims: Tuple[int, int] = (64, 64),
        num_steps: int = 100,
    ):
        """Simulate `num_steps` of SI model on a lattice of `dims`."""
        rng_beta, rng_num_init, rng_init, rng = random.split(rng, 4)
        beta = self.beta.sample(rng_beta)
        num_init = int(jnp.round(self.num_init.sample(rng_num_init, (1,)))[0])
        init_locs = random.choice(rng_init, math.prod(dims), (num_init,), replace=False)
        state = jnp.zeros(dims).at[jnp.unravel_index(init_locs, dims)].set(1.0)
        kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.float32)
        kernel = kernel[None, None, :, :]  # add dummy batch and channel dims

        @jit
        def step(rng, state):
            neighbor_sum = conv_general_dilated(
                state[None, None, :, :],  # add dummy batch and channel dims
                kernel,
                window_strides=(1, 1),
                padding="SAME",
            )[0, 0]  # remove batch and channel dimensions
            infection_prob = beta * neighbor_sum
            u = random.uniform(rng, state.shape)
            new_infections = (state == 0.0) & (u < infection_prob)
            return jnp.where(new_infections, 1.0, state)

        def step_scanner(state, rng):
            state = step(rng, state)
            return state, state

        rngs = random.split(rng, num_steps - 1)
        _, steps = jax.lax.scan(step_scanner, state, rngs)
        return jnp.vstack([state[None, ...], steps]), beta, num_init
