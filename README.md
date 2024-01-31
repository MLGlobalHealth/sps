# Stochastic Process Simulators (sps)

## Install
```bash
pip install git+ssh://git@github.com/MLGlobalHealth/sps.git
```

## Examples
```python
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

from sps.gp import GP
from sps.priors import Prior
from sps.utils import build_grid

# plot 5 samples from a collection of lengthscales
locations = build_grid([{"start": 0, "stop": 1, "num": 128}])
batch_size = 5
approx = True # approx uses Kronecker factorization instead of Cholesky
lengthscales = [0.05, 0.1, 0.2, 0.3, 0.5]
fig, axes = plt.subplots(len(lengthscales), 1)
for i, ls in enumerate(lengthscales):
    gp = GP("matern_3_2", lengthscale=Prior("fixed", {"value": ls}))
    _var, _ls, mu = gp.simulate(locations, batch_size, approx)
    axes[i].plot(mu.squeeze().T)
    axes[i].set_title(f"ls={ls}")


# create a simple (forever) dataloader
def dataloader(gp: GP, locations: ArrayLike, batch_size: int, approx: bool):
    while True:
        yield gp.simulate(locations, batch_size, approx)


gp = GP("matern_5_2", lengthscale=Prior("beta", {"a": 2.5, "b": 5}))
loader = dataloader(gp, locations, batch_size, approx=True)
var, ls, mu = next(loader)


# build a 2D grid, 64x64 grid
locations = build_grid([{"start": 0, "stop": 1, "num": 64}] * 2)


# within IPython, speed test Kronecker (approx) vs. Cholesky methods 
%timeit gp.simulate(locations, batch_size, approx=True) # ~7 ms
%timeit gp.simulate(locations, batch_size, approx=False) # ~136 ms
````

## Development
- Install Python 3.11:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.11: `pyenv install 3.11`
    - Make Python 3.11 your default: `pyenv global 3.11`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd sps && poetry install`
- Run tests: `poetry run pytest`
