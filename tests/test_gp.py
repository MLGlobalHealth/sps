from sps.simulators.gp import GP
from sps.metrics import maximum_mean_discrepancy
from sps.kernels import rbf


def test_gp_simulator():
    gp = GP()
    num_batches, batch_size = 3, 1024
    _test_batching(gp, num_batches, batch_size, approx=False)
    _test_batching(gp, num_batches, batch_size, approx=True)
    samples_true = gp.simulate(num=batch_size, approx=False)
    samples_approx = gp.simulate(num=batch_size, approx=True)
    assert maximum_mean_discrepancy(samples_true, samples_approx) < 0.1, "MMD too high!"


def _test_batching(simulator, num_batches, batch_size, approx):
    batches = []
    for batch in simulator.iter(num_batches, batch_size, approx):
        assert len(batch) == batch_size, "Incorrect batch_size!"
        batch += [batch]
    assert len(batches) == num_batches, "Incorrect num_batches!"
