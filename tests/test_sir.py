from jax import random

from dl4bi_sps.sir import LatticeSIR


def test_lattice_sir():
    """Verify the SIR simulator returns the expected number of time steps."""
    rng = random.key(42)
    dims = (64, 64)
    num_steps = 100
    steps, beta, gamma, num_init = LatticeSIR().simulate(rng, dims, num_steps)
    assert len(steps) == num_steps
