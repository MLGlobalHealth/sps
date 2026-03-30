from jax import random

from dl4bi_sps.priors import Prior
from dl4bi_sps.si import LatticeSI


def test_lattice_si():
    """Verify the SI simulator returns the expected number of time steps."""
    rng = random.key(42)
    dims = (8, 8)
    num_steps = 10
    num_init = Prior("fixed", {"value": 1.0})
    steps, beta, num_init = LatticeSI(num_init=num_init, kernel_width=3).simulate(
        rng, dims, num_steps
    )
    assert len(steps) == num_steps
