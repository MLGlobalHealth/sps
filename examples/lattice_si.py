#!/usr/bin/env python3
import matplotlib.pyplot as plt
from jax import random
from matplotlib.animation import FuncAnimation

from sps.si import LatticeSI


def main():
    rng = random.key(42)
    dims = (64, 64)
    num_steps = 100
    steps, beta, num_init = LatticeSI().simulate(rng, dims, num_steps)
    beta = float(beta[0])
    fig, ax = plt.subplots()
    img = ax.imshow(steps[0], cmap="viridis", interpolation="nearest")

    def update(i):
        img.set_data(steps[i])
        ax.set_title(f"Time Step: {i} (beta: {beta:0.3f}, num_init: {num_init})")
        fig.canvas.draw_idle()
        return (img,)

    ani = FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
