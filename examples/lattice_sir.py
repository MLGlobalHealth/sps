#!/usr/bin/env python3
import matplotlib.pyplot as plt
from jax import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from sps.priors import Prior
from sps.sir import LatticeSIR


def main():
    rng = random.key(42)
    dims = (64, 64)
    num_steps = 25
    beta = Prior("beta", {"a": 2, "b": 8})  # transmission prior
    gamma = Prior("inverse_gamma", {"alpha": 5, "beta": 0.4})
    num_init = Prior("randint", {"minval": 1, "maxval": 5})
    sir = LatticeSIR(beta, gamma)
    steps, beta, gamma, num_init = sir.simulate(rng, dims, num_steps)
    beta, gamma = float(beta[0]), float(gamma[0])  # Extract scalar values
    cmap = ListedColormap(
        ["#004D40", "#1E88E5", "#D81B60"]
    )  # recovered, susceptible, infected
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries for colormap
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    img = ax.imshow(steps[0], cmap=cmap, norm=norm, interpolation="nearest")

    def update(i):
        img.set_data(steps[i])
        ax.set_title(
            f"Time Step: {i} (beta: {beta:.3f}, gamma: {gamma:.3f}, num_init: {num_init})"
        )
        fig.canvas.draw_idle()
        return (img,)

    ani = FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
