import math
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.lax import conv_general_dilated

from .priors import Prior


@dataclass
class LatticeSIR:
    """A Susceptible-Infected-Recovered (SIR) model simulated on a lattice.

    Args:
        beta: A prior over the infection rate.
        gamma: A prior over the recovery rate.
        num_init: A prior over the initial number of infected (nearest integer is used).

    Returns:
        An instance of the `LatticeSIR` dataclass.
    """

    beta: Prior = Prior("beta", {"a": 5, "b": 10})
    gamma: Prior = Prior("inverse_gamma", {"alpha": 5, "beta": 0.4})
    num_init: Prior = Prior("uniform", {"minval": 1, "maxval": 5})

    def simulate(
        self,
        rng: Array,
        dims: Tuple[int, int] = (64, 64),
        num_steps: int = 100,
    ):
        """Simulate `num_steps` of SIR model on a lattice of `dims`."""
        rng_beta, rng_gamma, rng_num_init, rng_init, rng = random.split(rng, 5)
        beta = self.beta.sample(rng_beta)
        gamma = self.gamma.sample(rng_gamma)
        num_init = int(jnp.round(self.num_init.sample(rng_num_init, (1,)))[0])
        init_locs = random.choice(rng_init, math.prod(dims), (num_init,), replace=False)

        # initialize state array: 0 = susceptible, 1 = infected, -1 = recovered
        state = jnp.zeros(dims).at[jnp.unravel_index(init_locs, dims)].set(1.0)

        kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.float32)
        kernel = kernel[None, None, :, :]  # add dummy batch and channel dims

        @jit
        def step(rng, state):
            neighbor_sum = conv_general_dilated(
                jnp.float32(state == 1.0)[None, None, :, :],  # infected only
                kernel,
                window_strides=(1, 1),
                padding="SAME",
            )[0, 0]  # remove batch and channel dimensions
            rng_infect, rng_recover = random.split(rng)
            u_recover = random.uniform(rng, state.shape)
            u_infect = random.uniform(rng_infect, state.shape)
            new_infections = (state == 0.0) & (u_infect < beta * neighbor_sum)
            new_recoveries = (state == 1.0) & (u_recover < gamma)
            state = jnp.where(new_infections, 1.0, state)  # susceptible -> infected
            state = jnp.where(new_recoveries, -1.0, state)  # infected -> recovered
            return state

        def step_scanner(state, rng):
            state = step(rng, state)
            return state, state

        rngs = random.split(rng, num_steps - 1)
        _, steps = jax.lax.scan(step_scanner, state, rngs)

        # Return states over time
        return jnp.vstack([state[None, ...], steps]), beta, gamma, num_init
