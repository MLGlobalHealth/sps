# Stochastic Process Simulators (sps)

## Setup
- Install Python 3.11:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.11: `pyenv install 3.11`
    - Make Python 3.11 your default: `pyenv global 3.11`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd prior-cvae && poetry install`
- Run tests: `poetry run pytest --jaxtyping=gp,beartype.beartype`
