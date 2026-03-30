#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random
from numpy.lib.format import open_memmap
from tqdm import tqdm

from dl4bi_sps.popgen import PopGen, PopGenState


def main(args):
    """Generate population genetics trajectories and write them to disk.

    Args:
        args: Parsed command-line arguments.
    """
    rng = random.key(args.seed)
    popgen = PopGen()
    N, B, T, C, (H, W) = (
        args.num_batches_per_state,
        args.batch_size,
        args.num_steps,
        1,
        args.dims,
    )
    dir = Path(args.dir)
    population = jnp.array(args.population)
    prevalence = jnp.zeros((B, C, H, W))
    pbar = tqdm(total=args.num_migration * args.num_mutation * N)
    # TODO(danj): add args for param ranges?
    for migration in jnp.logspace(-3.3, -1.3, args.num_migration):
        for mutation in jnp.logspace(-6, -3, args.num_mutation):
            fname = f"popgen_migration_{migration.item()}_mutation_{mutation.item()}_population_{population.item()}.npy"
            # NOTE: you can load this dataset with np.load(path, mmap_mode='r')
            mm = open_memmap(
                dir / fname,
                dtype=np.float32,
                mode="w+",
                shape=(N, B, C, T, H, W),
            )
            offset = 0
            last_state = PopGenState(migration, mutation, population, prevalence)
            for i in range(args.num_batches_per_state):
                rng, rng_i = random.split(rng)
                prevalences, last_state = popgen.simulate(
                    rng_i,
                    num_warmup=args.num_warmup if i == 0 else 0,
                    num_steps=args.num_steps,
                    step_interval=args.step_interval,
                    batch_size=args.batch_size,
                    dims=args.dims,
                    wrap_edges=True,
                    state=last_state,
                )
                mm[offset] = prevalences
                mm.flush()
                offset += 1
                pbar.update(1)


def parse_args(argv):
    """Parse command-line arguments for dataset generation.

    Args:
        argv: Raw command-line arguments.

    Returns:
        Parsed argument namespace.
    """
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
        default=200000,
        help="Number of warmup steps (thrown away) for each set of parameters.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=1000,
        help="Population per deme.",
    )
    parser.add_argument(
        "--num_migration",
        type=int,
        default=10,
        help="Number of migration parameters to use in default range.",
    )
    parser.add_argument(
        "--num_mutation",
        type=int,
        default=10,
        help="Number of mutation parameters to use in default range.",
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
        default=50,
        help="Number of timesteps between kept steps.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generating data.",
    )
    parser.add_argument(
        "-nb",
        "--num_batches_per_state",
        type=int,
        default=500,
        help="Number of batches to generate per parameter state.",
    )
    parser.add_argument(
        "-d",
        "--dims",
        nargs=2,
        type=int,
        default=[32, 32],
        help="Dimensions of surface.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Output dir.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
