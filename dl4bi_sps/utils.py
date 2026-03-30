from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.typing import ArrayLike


def build_grid(
    axes: Sequence[dict[str, jax.Array | float]] = [
        {"start": 0, "stop": 1, "num": 128}
    ],
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Build a dense mesh grid from axis specifications.

    Args:
        axes: Sequence of dictionaries with `start`, `stop`, and `num`
            arguments for `jnp.linspace`.
        dtype: Dtype used when constructing each axis.

    Returns:
        Mesh grid with shape `[..., D]`.
    """
    pts = [jnp.linspace(**axis, dtype=dtype) for axis in axes]
    return jnp.stack(jnp.meshgrid(*pts, indexing="ij"), axis=-1)


def scale_grid(grid: ArrayLike, factor: int) -> jax.Array:
    """Upsample a grid uniformly along every axis.

    Args:
        grid: Grid with shape `[..., D]`.
        factor: Multiplicative factor applied to each axis resolution.

    Returns:
        Resampled grid with the same bounds as `grid`.
    """
    axes = [
        jnp.linspace(grid[..., dim].min(), grid[..., dim].max(), int(n * factor))
        for dim, n in enumerate(grid.shape[:-1])
    ]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)


def random_subgrid(
    rng: jax.Array,
    axes: Sequence[dict[str, float]] = [{"start": 0, "stop": 1, "num": 32}] * 2,
    min_axes_pct: float = 0.05,
    max_axes_pct: float = 1.0,
):
    """Create a random subgrid at the original axis resolution.

    Args:
        rng: Pseudo-random key.
        axes: Axis specifications defining the full domain.
        min_axes_pct: Minimum side-length fraction for the sampled subgrid.
        max_axes_pct: Maximum side-length fraction for the sampled subgrid.

    Returns:
        Randomly positioned subgrid with the same per-axis sample counts.
    """
    D = len(axes)
    rng_width, rng_shift = random.split(rng)
    u_width = random.uniform(rng_width, (1,), minval=min_axes_pct, maxval=max_axes_pct)
    u_width = u_width[0]
    u_corner = random.uniform(rng_shift, (D,), maxval=1 - u_width)  # bottom left
    u_center = jnp.array([0.5] * D)
    lower_left = jnp.array([d["start"] for d in axes])
    upper_right = jnp.array([d["stop"] for d in axes])
    scale = upper_right - lower_left
    center = (upper_right + lower_left) / 2
    corner = (u_corner - u_center + center) * scale
    width = u_width * scale
    return build_grid(
        [
            {"start": corner[i], "stop": corner[i] + width[i], "num": axes[i]["num"]}
            for i in range(D)
        ]
    )


@partial(jit, static_argnames=("width",))
def inv_dist_sq_kernel(width: int = 7):
    r"""Build an inverse-distance-squared convolution kernel.

    For offsets `(i, j) != (0, 0)`, the kernel entry is proportional to

    $$
    \frac{1}{i^2 + j^2},
    $$

    with the center entry set to zero.

    Args:
        width: Side length of the square kernel.

    Returns:
        Kernel with the center entry set to zero contribution.
    """
    center = width // 2
    x = y = jnp.arange(width) - center
    xx, yy = jnp.meshgrid(x, y)
    dist_sq = jnp.float32(xx**2 + yy**2)
    return 1 / dist_sq.at[center, center].set(jnp.inf)
