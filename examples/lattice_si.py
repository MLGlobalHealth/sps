#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
from jax import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from dl4bi_sps.priors import Prior
from dl4bi_sps.si import LatticeSI


def main(args):
    """Animate a sampled SI trajectory on a lattice.

    Args:
        args: Parsed command-line arguments.
    """
    rng = random.key(args.seed)
    dims = (args.dim, args.dim)
    beta = Prior("beta", {"a": 2, "b": 8})  # transmission prior
    num_init = Prior("randint", {"minval": args.min_init, "maxval": args.max_init})
    sir = LatticeSI(beta, num_init)
    steps, beta, num_init = sir.simulate(rng, dims, args.num_steps)
    beta = float(beta[0])  # Extract scalar values
    cmap = ListedColormap(
        ["#004D40", "#1E88E5", "#D81B60"]
    )  # recovered, susceptible, infected
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries for colormap
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    img = ax.imshow(steps[0], cmap=cmap, norm=norm, interpolation="nearest")

    def update(i):
        """Update the displayed frame in the lattice animation.

        Args:
            i: Frame index.

        Returns:
            Tuple containing the updated artist.
        """
        img.set_data(steps[i])
        ax.set_title(f"Time Step: {i} (beta: {beta:.3f}, num_init: {num_init})")
        fig.canvas.draw_idle()
        return (img,)

    ani = FuncAnimation(fig, update, frames=args.num_steps, interval=100, blit=True)
    plt.show()


def parse_args(argv):
    """Parse command-line arguments for the SI animation.

    Args:
        argv: Raw command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-n", "--num_steps", type=int, default=25)
    parser.add_argument("-d", "--dim", type=int, default=64)
    parser.add_argument("--min_init", type=int, default=1)
    parser.add_argument("--max_init", type=int, default=5)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
