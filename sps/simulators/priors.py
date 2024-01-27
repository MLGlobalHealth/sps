from dataclasses import dataclass
from jax import random


@dataclass
class Prior:
    dist: str
    kwargs: dict[str, Any]
