#!/usr/bin/env python3
import matplotlib.pyplot as plt
from jax import random

from sps.gp import GP
from sps.kernels import matern_3_2, matern_5_2, rbf
from sps.priors import Prior
from sps.utils import build_grid


def main():
    rng = random.key(42)
    s = build_grid(
        [
            {"start": -1.5, "stop": 1.5, "num": 300},
            {"start": -2.5, "stop": 2.5, "num": 500},
        ]
    )
    batch_size = 1
    approx = True
    lengthscales = [0.1, 0.2, 0.3]
    n = len(lengthscales)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5 * n))
    for i, ls in enumerate(lengthscales):
        gp = GP(matern_5_2, ls=Prior("fixed", {"value": ls}))
        f, *_ = gp.simulate(rng, s, batch_size, approx)
        axes[i].set_title(f"ls={ls}")
        axes[i].imshow(f.squeeze().reshape(300, 500), cmap="Spectral_r")
    plt.tight_layout()
    plt.savefig("2d_gp.png", dpi=150)
    plt.clf()


if __name__ == "__main__":
    main()
