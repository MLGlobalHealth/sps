# Stochastic Process Simulators (sps)

## Install
Install with the appropriate command. If JAX isn't installed already, we recommend using one of the `sps[<jax-version>]` installs.
```bash
pip install sps # sps
pip install sps[cpu] # sps + jax for CPU
pip install sps[cuda12] # sps + jax for CUDA-12
pip install sps[cuda13] # sps + jax for CUDA-13
```

## View Documentation (Locally)
```bash
git clone git@github.com:MLGlobalHealth/sps.git
cd sps
uv sync --extra {cpu,cuda12,cuda13}
uv run --with pdoc pdoc --docformat google --math sps
```

## Demo
```python
import matplotlib.pyplot as plt

import jax
from jax import random

from sps.gp import GP
from sps.priors import Prior
from sps.utils import build_grid
from sps.kernels import matern_3_2, matern_5_2

rng = random.key(42)

s_1d = build_grid([{"start": -2, "stop": 2, "num": 128}])
s_2d = build_grid([{"start": -1.5, "stop": 1.5, "num": 300}, {"start": -2.5, "stop": 2.5, "num": 500}])
batch_size = 1
approx = True
lengthscales = [0.05, 0.1, 0.2]
for name, s in zip(["1d", "2d"], [s_1d, s_2d]):
    fig, axes = plt.subplots(len(lengthscales), 1)
    for i, ls in enumerate(lengthscales):
        gp = GP(matern_3_2, ls=Prior("fixed", {"value": ls}))
        f, *_ = gp.simulate(rng, s, batch_size, approx)
        axes[i].set_title(f"ls={ls}")
        if name == "1d":
            axes[i].plot(s, f.squeeze().T)
        else:
            axes[i].imshow(f.squeeze().reshape(300, 500), cmap="Spectral_r")
    plt.tight_layout()
    plt.savefig(f"{name}_gp.png", dpi=150)
    plt.clf()

# create a simple (forever) dataloader
def dataloader(rng, gp, s, batch_size=64, approx=False):
    while True:
        rng_i, rng = random.split(rng)
        yield gp.simulate(rng_i, s, batch_size, approx)


gp = GP(matern_5_2, ls=Prior("beta", {"a": 2.5, "b": 5}))
loader = dataloader(rng, gp, s, batch_size, approx=True)
f, var, ls, period, z = next(loader)


# within IPython, speed test Kronecker (approx) vs. Cholesky methods 
rng, batch_size = random.key(42), 1024
s = build_grid([{"start": 0, "stop": 1, "num": 64}] * 2) # 64x64 grid
%timeit gp.simulate(rng, s, batch_size, approx=True) # ~5 ms
%timeit gp.simulate(rng, s, batch_size, approx=False) # ~50 ms
````
More examples can be found [here](https://github.com/MLGlobalHealth/sps/tree/main/examples).

## Gotchas
Small lengthscales can cause numerical instability. Enabling 64-bit floating
operations often helps, but it roughly doubles memory usage and may reduce
throughput on accelerators.
```python
import jax
# use 64-bit precision globally
jax.config.update("jax_enable_x64", True)
# use 64-bit precision only inside this context manager
with jax.enable_x64():
    # Do something in 64-bit precision
    ...
# Back to default 32-bit precision
```

## Development Setup
- Install [uv](https://docs.astral.sh/uv/).
- Clone the repository and `cd` into it: `git clone git@github.com:MLGlobalHealth/sps.git && cd sps`
- Install the pinned Python version if needed: `uv python install`
- Sync the project, development dependencies, and one JAX extra: `uv sync --extra {cpu,cuda12,cuda13}`
- Run the test suite: `uv run pytest`

`uv sync` creates a local `.venv/` and installs the project in editable mode,
so changes in `sps/` are reflected immediately.

## Build and Publish to PyPI
1. Bump the package version:
```bash
uv version --bump patch --frozen
```

2. Build the source distribution and wheel:
```bash
uv build --no-sources
```

3. Publish to TestPyPI first:
```bash
UV_PUBLISH_TOKEN=$TEST_PYPI_TOKEN uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --check-url https://test.pypi.org/simple/
```

4. After validating the release, publish the same artifacts to PyPI:
```bash
UV_PUBLISH_TOKEN=$PYPI_TOKEN uv publish
```

5. Smoke-test the published install targets in fresh environments:
```bash
uv run --isolated --with "sps==<version>" --no-project -- python -c "import sps"
uv run --isolated --with "sps[cpu]==<version>" --no-project -- python -c "import sps"
uv run --isolated --with "sps[cuda12]==<version>" --no-project -- python -c "import sps"
uv run --isolated --with "sps[cuda13]==<version>" --no-project -- python -c "import sps"
```
