#!/usr/bin/env python3
import argparse
import sys

import numpy as np
from jax import random
from tqdm import tqdm

from sps.popgen import PopGen


def main(args):
    rng = random.key(args.seed)
    popgen = PopGen()
    N, B, T, C, (H, W) = args.num_batches, args.batch_size, args.num_steps, 1, args.dims
    mm = np.memmap(args.path, dtype=np.float32, mode="w+", shape=(N, B, T, C, H, W))
    for i in tqdm(range(N), unit="batches"):
        rng_i, rng = random.split(rng)
        prevalences, _ = popgen.simulate(
            rng_i,
            args.num_warmup,
            args.num_steps,
            args.step_interval,
            args.batch_size,
            args.dims,
            wrap_edges=True,
        )
        mm[i : i + 1] = prevalences
        if i % args.flush_every_n == 0:
            mm.flush()
    print(f"Finished! Saved to {args.path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for the entire process.",
    )
    parser.add_argument(
        "-nw",
        "--num_warmup",
        type=int,
        default=1000,
        help="Number of warmup steps (thrown away) for each set of parameters.",
    )
    parser.add_argument(
        "-ns",
        "--num_steps",
        type=int,
        default=16,
        help="Number of steps per batch element.",
    )
    parser.add_argument(
        "-si",
        "--step_interval",
        type=int,
        default=16,
        help="Number of timesteps between kept steps.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "-nb",
        "--num_batches",
        type=int,
        default=100000,
        help="Number of batches to generate.",
    )
    parser.add_argument(
        "-f",
        "--flush_every_n",
        type=int,
        default=100,
        help="Flush batches to disk every `n`.",
    )
    parser.add_argument(
        "-d",
        "--dims",
        type=list,
        default=[32, 32],
        help="Dimensions of surface.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="popgen.npy",
        help="Output filename.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
