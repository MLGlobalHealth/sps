from jax import random

from dl4bi_sps.priors import Prior
from dl4bi_sps.sir import LatticeSIR


def test_lattice_sir():
    """Verify the SIR simulator returns the expected number of time steps."""
    rng = random.key(42)
    dims = (8, 8)
    num_steps = 10
    num_init = Prior("fixed", {"value": 1.0})
    steps, beta, gamma, num_init = LatticeSIR(kernel_width=3).simulate(
        rng, dims, num_steps
    )
    assert len(steps) == num_steps
