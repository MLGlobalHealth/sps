# Stochastic Process Simulators (sps)

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install the `sps` package from git:
```bash
pip install git+ssh://git@github.com/MLGlobalHealth/sps.git
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
locations = build_grid([{"start": 0, "stop": 1, "num": 128}])
batch_size = 64
approx = True # approx uses Kronecker factorization instead of Cholesky
lengthscales = [0.05, 0.1, 0.2, 0.3, 0.5]
fig, axes = plt.subplots(len(lengthscales), 1)
key = random.key(42)
for i, ls in enumerate(lengthscales):
    gp = GP(matern_3_2, ls=Prior("fixed", {"value": ls}))
    _var, _ls, _z, f = gp.simulate(key, locations, batch_size, approx)
    axes[i].plot(f.squeeze().T)
    axes[i].set_title(f"ls={ls}")


# create a simple (forever) dataloader
def dataloader(key, gp, locations, batch_size=64, approx=False):
    while True:
        rng, key = random.split(key)
        yield gp.simulate(rng, locations, batch_size, approx)


gp = GP(matern_5_2, ls=Prior("beta", {"a": 2.5, "b": 5}))
loader = dataloader(key, gp, locations, batch_size, approx=True)
var, ls, z, f = next(loader)


# within IPython, speed test Kronecker (approx) vs. Cholesky methods 
key, batch_size = random.key(42), 1024
locations = build_grid([{"start": 0, "stop": 1, "num": 64}] * 2) # 64x64 grid
%timeit gp.simulate(key, locations, batch_size, approx=True) # ~6 ms
%timeit gp.simulate(key, locations, batch_size, approx=False) # ~57 ms
````

## Development
- Install Python 3.12 (or later):
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
    - Make Python 3.12 your default: `pyenv global 3.12`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd sps && poetry install`
- Run tests: `poetry run pytest`
