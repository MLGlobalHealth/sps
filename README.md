# Stochastic Process Simulators (sps)

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install the `sps` package from git:
```bash
pip install -U --force-reinstall git+ssh://git@github.com/MLGlobalHealth/sps.git
```

## View Documentation (Locally)
```bash
pip install pdoc
git clone git@github.com:MLGlobalHealth/sps.git
cd sps
pdoc --docformat google --math sps
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
- Small lengthscales can cause numerical instability; enabiling 64-bit floating
operations can often help, but be warned that this will double memory usage.
```python
from jax import config
config.update("jax_enable_x64", True)
```
Or, use the experimental context manager, which restricts 64-bit precision to
the local execution block:
```python
from jax.experimental import enable_x64

with enable_x64():
    # Do something in 64-bit precision
# Back to default 32-bit precision
```

## Development Setup
- Install Python 3.12 with `pyenv`:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
- Create a virtualenv called `sps-dev` using Python 3.12: `pyenv virtualenv 3.12 sps-dev`
- Clone the repository and `cd` into it: `git clone git@github.com:MLGlobalHealth/sps.git && cd sps`
- Inside the `sps` repository, tell `pyenv` to use the `sps-dev` virtualenv: `pyenv local sps-dev`
    - `pyenv local sps-dev` creates a `.python-version` file that tells `pyenv`
        to automatically activate the `sps-dev` virtualenv whenever you are
        working in the `sps` repository, so all `python` and `pip` commands will
        execute within the `sps-dev` virtualenv
- Inside the `sps` directory, install the package to the `sps-dev` virtualenv: `pip install -e .`
    - Installing this package locally means it is installed "live", i.e. it
        immediately reflects any changes you make (this only needs to be done
        once)
