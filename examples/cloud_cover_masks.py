#!/usr/bin/env python3
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import random
from sps.gp import GP
from sps.kernels import matern_1_2
from sps.priors import Prior
from sps.utils import build_grid

rng = random.key(42)
B = 16
var = Prior("normal", {"mu": 1.0, "sigma": 0.2})
ls = Prior("normal", {"mu": 1.0, "sigma": 0.2})
gp = GP(matern_1_2, var, ls)
s = build_grid([dict(start=-1.5, stop=1.5, num=30), dict(start=-2.5, stop=2.5, num=50)])
f, *_ = gp.simulate(rng, s, B)
rot_idx = jnp.arange(1, f.shape[0] + 1).at[-1].set(0)
f_mask = f[rot_idx] > 0.5  # ~30% in N(0, 1)
f_masked = f.at[f_mask].set(jnp.nan)
fig, axes = plt.subplots(B, 1, figsize=(B, 5 * B))
cmap = mpl.colormaps.get_cmap("Spectral_r")
cmap.set_bad("grey")
for i in range(B):
    axes[i].imshow(f_masked[i].squeeze(), cmap=cmap, interpolation="none")
    axes[i].set_title(f"Sample {i+1}")
plt.savefig("masks.png", dpi=150)
plt.tight_layout()
plt.clf()
