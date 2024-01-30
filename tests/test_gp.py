import jax.numpy as jnp

from sps.gp import GP, cholesky, kronecker
from sps.kernels import rbf
from sps.utils import build_grid


# TODO(danj): this isn't close for dims > 1
def test_kronecker_approx():
    grid = build_grid([{"start": 0, "stop": 1, "num": 50}] * 2)
    var, ls = 1.0, 0.1
    L_ch = cholesky(rbf, grid, var, ls)
    L_kr = kronecker(rbf, grid, var, ls, noise=1e-8)
    # print(jnp.max(jnp.abs(L_ch - L_kr)))
    # plt.imshow(L_ch - L_kr, cmap='inferno')
    # plt.colorbar()
    assert jnp.allclose(L_ch, L_kr)


def test_1D_gp_approx():
    batch_size = 3
    grid = build_grid()
    gp = GP(seed=0)
    _, _, mu = gp.simulate(grid, batch_size)
    gp = GP(seed=0)  # reset seed
    _, _, mu_approx = gp.simulate(grid, batch_size, approx=True)
    assert jnp.allclose(mu, mu_approx)


# TODO(danj): this is failing
def test_2D_gp_approx():
    batch_size = 3
    grid = build_grid([{"start": 0, "stop": 1, "num": 50}] * 2)
    gp = GP(seed=0)
    _, _, mu = gp.simulate(grid, batch_size)
    gp = GP(seed=0)  # reset seed
    _, _, mu_approx = gp.simulate(grid, batch_size, approx=True)
    assert jnp.allclose(mu, mu_approx)
