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

## Examples
```python
import matplotlib.pyplot as plt

import jax
from jax import random

from sps.gp import GP
from sps.priors import Prior
from sps.utils import build_grid
from sps.kernels import matern_3_2, matern_5_2


# plot 5 samples from a collection of lengthscales
s = build_grid([{"start": 0, "stop": 1, "num": 128}])
batch_size = 64
approx = True # approx uses Kronecker factorization instead of Cholesky
lengthscales = [0.05, 0.1, 0.2, 0.3, 0.5]
fig, axes = plt.subplots(len(lengthscales), 1)
key = random.key(42)
for i, ls in enumerate(lengthscales):
    gp = GP(matern_3_2, ls=Prior("fixed", {"value": ls}))
    _var, _ls, _z, f = gp.simulate(key, s, batch_size, approx)
    axes[i].plot(s, f.squeeze().T)
    axes[i].set_title(f"ls={ls}")


# create a simple (forever) dataloader
def dataloader(key, gp, s, batch_size=64, approx=False):
    while True:
        rng, key = random.split(key)
        yield gp.simulate(rng, s, batch_size, approx)


gp = GP(matern_5_2, ls=Prior("beta", {"a": 2.5, "b": 5}))
loader = dataloader(key, gp, s, batch_size, approx=True)
var, ls, z, f = next(loader)


# within IPython, speed test Kronecker (approx) vs. Cholesky methods 
key, batch_size = random.key(42), 1024
s = build_grid([{"start": 0, "stop": 1, "num": 64}] * 2) # 64x64 grid
%timeit gp.simulate(key, s, batch_size, approx=True) # ~6 ms
%timeit gp.simulate(key, s, batch_size, approx=False) # ~57 ms
````

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
