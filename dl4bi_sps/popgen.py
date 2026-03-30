from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.lax import conv_general_dilated

from .priors import Prior


@dataclass(frozen=True)
class PopGenState:
    """Immutable state container for the population genetics simulator.

    Attributes:
        migration: Migration rate scalar.
        mutation: Mutation rate scalar.
        population: Population size per deme.
        prevalence: Current prevalence field with shape `[B, C, H, W]`.
    """

    migration: Array  # [1]
    mutation: Array  # [1]
    population: Array  # [1]
    prevalence: Array  # [B, C, H, W]


jax.tree_util.register_pytree_node(
    PopGenState,
    lambda d: ((d.migration, d.mutation, d.population, d.prevalence), None),
    lambda _aux, children: PopGenState(*children),
)


@dataclass
class PopGen:
    """Population genetics simulator on a lattice.

    Args:
        migration: A prior over the migration rate.
        mutation: A prior over the mutation rate.
        population: A prior over the population size per deme.
    """

    migration: Prior = Prior("uniform", {"minval": 10**-3.3, "maxval": 10**-1.3})
    mutation: Prior = Prior("uniform", {"minval": 10e-6, "maxval": 1e-3})
    population: Prior = Prior("fixed", {"value": 1000})

    def simulate(
        self,
        rng: Array,
        num_warmup: int = 2000,
        num_steps: int = 16,
        step_interval: int = 16,
        batch_size: int = 32,
        dims: Tuple[int, int] = (32, 32),
        wrap_edges: bool = True,
        state: Optional[PopGenState] = None,
    ):
        r"""Simulate allele prevalence trajectories on a lattice.

        Each step applies spatial migration, mutation toward the symmetric
        equilibrium `0.5`, and then binomial genetic drift.

        Args:
            rng: Pseudo-random key.
            num_warmup: Number of warmup steps (thrown away).
            num_steps: Total number of steps kept at the end.
            step_interval: Number of steps to skip between kept steps.
            batch_size: Number of sequences of steps to keep.
            dims: Surface deme dimensions as `(height, width)`.
            wrap_edges: Whether the lattice uses wraparound boundaries.
            state: Optional state to continue simulating from.

        Returns:
            Tuple of prevalence trajectories with shape `[B, C, T, H, W]`
            and the final simulation state.
        """
        if state is None:
            rng_mi, rng_mu, rng_po, rng = random.split(rng, 4)
            migration = self.migration.sample(rng_mi, (1,))
            mutation = self.mutation.sample(rng_mu, (1,))
            population = self.population.sample(rng_po, (1,))
            prevalence = jnp.zeros((batch_size, 1, *dims))  # [B, C=1, H, W]
            state = PopGenState(migration, mutation, population, prevalence)
        return _simulate(
            rng,
            state,
            num_warmup,
            num_steps,
            step_interval,
            wrap_edges,
        )


@partial(
    jit,
    static_argnames=(
        "num_warmup",
        "num_steps",
        "step_interval",
        "wrap_edges",
    ),
)
def _simulate(
    rng: Array,
    state: PopGenState,
    num_warmup: int,
    num_steps: int,
    step_interval: int,
    wrap_edges: bool = True,
):
    """Run the jitted population genetics simulation loop.

    Args:
        rng: Pseudo-random key.
        state: Initial simulation state.
        num_warmup: Number of steps discarded before collecting outputs.
        num_steps: Number of states to record.
        step_interval: Number of simulation steps between recorded states.
        wrap_edges: Whether the lattice uses wraparound boundaries.

    Returns:
        Tuple of recorded prevalences and the final simulation state.
    """
    migration = state.migration
    mutation = state.mutation
    population = state.population
    prevalence = state.prevalence
    T, (B, C, H, W) = num_steps, prevalence.shape
    buffer = jnp.zeros((T, B, C, H, W))

    def step(carry, i):
        """Advance the prevalence state by one simulation step.

        Args:
            carry: Tuple containing the RNG key, output buffer, and prevalence.
            i: Current step index.

        Returns:
            Updated carry tuple and a dummy scan output.
        """
        rng, buffer, prevalence = carry
        rng, rng_step = random.split(rng)
        prevalence = _migrate_and_mutate(migration, mutation, prevalence, wrap_edges)
        prevalence = random.binomial(rng_step, population, prevalence) / population
        idx = (i - num_warmup) // step_interval
        update = lambda b: b.at[idx].set(prevalence)
        keep = jnp.logical_and(i >= num_warmup, (i - num_warmup) % step_interval == 0)
        buffer = lax.cond(keep, update, lambda b: b, buffer)
        return (rng, buffer, prevalence), None

    total_steps = num_warmup + num_steps * step_interval
    (rng, buffer, last_prev), _ = lax.scan(
        step, (rng, buffer, prevalence), jnp.arange(total_steps)
    )
    prevalences = jnp.moveaxis(buffer, 0, 2)  # [B, C, T, H, W]
    last_state = replace(state, prevalence=last_prev)
    return prevalences, last_state


@partial(jit, static_argnames=("wrap_edges"))
def _migrate_and_mutate(
    migration: Array,  # [1]
    mutation: Array,  # [1]
    prevalence: Array,  # [B, C, H, W]
    wrap_edges: bool = True,
):
    r"""Apply one migration-plus-mutation update to the lattice.

    The update is

    $$
    p' = (K_m * p)(1 - \mu) + 0.5\mu,
    $$

    where `K_m` is the local migration kernel, `*` denotes convolution, and
    `\mu` is the mutation rate.

    Args:
        migration: Migration rate scalar.
        mutation: Mutation rate scalar.
        prevalence: Current prevalence field with shape `[B, C, H, W]`.
        wrap_edges: Whether to use wraparound instead of edge padding.

    Returns:
        Updated prevalence field with the same shape as `prevalence`.
    """
    k_neighbor = jnp.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
    )
    k_center = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    kernel = k_neighbor * migration + k_center * (1 - 2 * migration)
    prevalence_padded = jnp.pad(
        prevalence,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="wrap" if wrap_edges else "edge",
    )
    prevalence_migrated = conv_general_dilated(
        lhs=prevalence_padded,
        rhs=kernel[None, None, :, :],
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return prevalence_migrated * (1 - mutation) + 0.5 * mutation
