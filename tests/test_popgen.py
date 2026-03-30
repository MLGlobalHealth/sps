from jax import random

from dl4bi_sps.popgen import PopGen


def test_popgen():
    """Verify the population genetics simulator returns expected tensor shapes."""
    rng = random.key(42)
    num_warmup = 100
    num_steps = 16
    step_interval = 16
    batch_size = 32
    dims = (32, 32)
    prevalences, state = PopGen().simulate(
        rng,
        num_warmup,
        num_steps,
        step_interval,
        batch_size,
        dims,
    )
    assert prevalences.shape == (batch_size, 1, num_steps, *dims)
    assert state.prevalence.shape == (batch_size, 1, *dims)
    assert (state.prevalence.sum(axis=(1, 2, 3)) > 0).all()


def test_popgen_speed(benchmark):
    """Benchmark the population genetics simulator and validate the final state."""
    rng = random.key(42)
    num_warmup = 200000
    num_steps = 1
    step_interval = 1
    batch_size = 32
    dims = (32, 32)
    _, state = benchmark.pedantic(
        PopGen().simulate,
        args=(rng, num_warmup, num_steps, step_interval, batch_size, dims),
        iterations=1,
        rounds=1,
    )
    assert state.prevalence.shape == (batch_size, 1, *dims)
