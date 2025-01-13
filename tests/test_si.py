from jax import random

from sps.si import LatticeSI


def test_lattice_si():
    rng = random.key(42)
    dims = (64, 64)
    num_steps = 100
    steps, beta, num_init = LatticeSI().simulate(rng, dims, num_steps)
    assert len(steps) == num_steps
