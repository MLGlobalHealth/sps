import itertools as it
from functools import reduce

import jax.numpy as jnp
import pytest
from jax import enable_x64, random

from dl4bi_sps.gp import GP, _kronecker_Ls, _kronecker_mvprod
from dl4bi_sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf
from dl4bi_sps.priors import Prior
from dl4bi_sps.utils import build_grid


@pytest.mark.parametrize("ls", [0.1, 0.5, 1.0])
def test_factorizations(
    ls,
    var=1.0,
    num_dims=2,
    dim_size=32,
    noise=1e-6,
):
    """Verify that dense and Kronecker factorizations recover the same covariance."""
    locations = build_grid([{"start": -1, "stop": 1, "num": dim_size}] * num_dims)
    num_locations = locations.size // locations.shape[-1]
    with enable_x64():
        locations = jnp.float64(locations)
        K = rbf(locations, locations, var, ls) + noise * jnp.eye(num_locations)
        L_ch = jnp.linalg.cholesky(K)
        Ls_kr = _kronecker_Ls(rbf, locations, var, ls, noise / num_dims)
        L_kr = reduce(jnp.kron, Ls_kr)
        K_ch = L_ch @ L_ch.T
        K_kr = L_kr @ L_kr.T
        assert jnp.allclose(K, K_ch)
        assert jnp.allclose(K, K_kr)
        assert jnp.allclose(K_ch, K_kr)


@pytest.mark.parametrize("ls", [0.1, 0.5, 1.0])
def test_kronecker_mvprod(ls, var=1.0, num_dims=2, dim_size=32, seed=7, noise=1e-5):
    """Verify the implicit Kronecker-vector product matches the explicit one."""
    locations = build_grid([{"start": -1, "stop": 1, "num": dim_size}] * num_dims)
    num_locations = locations.size // locations.shape[-1]
    with enable_x64():
        z = random.normal(random.key(seed), (num_locations,))
        Ls_kr = _kronecker_Ls(rbf, locations, var, ls, noise / num_dims)
        Lz_kr = reduce(jnp.kron, Ls_kr) @ z
        Lz_kr_mvprod = _kronecker_mvprod(Ls_kr, z)
        assert jnp.allclose(Lz_kr, Lz_kr_mvprod)
        # K = rbf(locations, locations, var, ls) + noise * jnp.eye(num_locations)
        # Lz_ch = jnp.linalg.cholesky(K) @ z
        # print(jnp.max(jnp.abs(Lz_kr_mvprod - Lz_ch)))
        # assert jnp.allclose(Lz_kr_mvprod, Lz_ch)  # approximation only


@pytest.mark.parametrize("kernel", [matern_1_2, matern_3_2, matern_5_2, periodic, rbf])
def test_gp(kernel, num_dims=1, dim_size=32, batch_size=3, seed=0):
    """Verify GP simulation returns finite samples for each supported kernel."""
    locations = build_grid([{"start": -1, "stop": 1, "num": dim_size}] * num_dims)
    gp = GP(kernel)
    key = random.key(seed)
    f, *_ = gp.simulate(key, locations, batch_size)
    assert jnp.isfinite(f).all()


@pytest.mark.parametrize("ls, num_dims", it.product([0.1, 0.5, 1.0], [1, 2]))
def test_gp_approx(ls, num_dims, dim_size=32, batch_size=3, seed=0):
    """Check that approximate GP samples stay close to dense samples."""
    locations = build_grid([{"start": -1, "stop": 1, "num": dim_size}] * num_dims)
    with enable_x64():
        locations = jnp.float64(locations)
        gp = GP(ls=Prior("fixed", {"value": ls}))
        key = random.key(seed)
        f, *_ = gp.simulate(key, locations, batch_size)
        f_approx, *_ = gp.simulate(key, locations, batch_size, approx=True)
        assert jnp.allclose(f, f_approx, atol=2.5)  # TODO(danj): this feels quite large
