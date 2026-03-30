from jax import random

from dl4bi_sps.si import LatticeSI


def test_lattice_si():
    """Verify the SI simulator returns the expected number of time steps."""
    rng = random.key(42)
    dims = (64, 64)
    num_steps = 100
    steps, beta, num_init = LatticeSI().simulate(rng, dims, num_steps)
    assert len(steps) == num_steps
