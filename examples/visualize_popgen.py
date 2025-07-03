#!/usr/bin/env python3
import argparse
import sys

import imageio
import numpy as np
from jax import random

from sps.popgen import PopGen
from sps.priors import Prior


def main(args):
    rng = random.key(args.seed)
    migration = Prior("fixed", {"value": 0.001})
    mutation = Prior("fixed", {"value": 0.0001})
    population = Prior("fixed", {"value": 1000})
    batch_size, wrap_edges = 1, True
    popgen = PopGen(migration, mutation, population)
    prevalences, _ = popgen.simulate(  # [B=1, T, C, H, W]
        rng,
        args.num_warmup,
        args.num_steps,
        args.step_interval,
        batch_size,
        args.dims,
        wrap_edges,
    )
    path = "popgen.gif"
    clips = (prevalences[0, :, 0, ...] * 255.0).astype(np.uint8)
    imageio.mimsave(path, clips, duration=0.2, loop=0)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-nw", "--num_warmup", type=int, default=10000)
    parser.add_argument("-ns", "--num_steps", type=int, default=16)
    parser.add_argument("-si", "--step_interval", type=int, default=50)
    parser.add_argument("-d", "--dims", nargs=2, type=int, default=[32, 32])
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
