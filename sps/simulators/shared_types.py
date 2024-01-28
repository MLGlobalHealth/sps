from jaxtyping import Float, Array
from collections.abc import Callable

Locations = Float[Array, "... D"]
Variance = Float
Lengthscale = Float
Covariance = Float[Array, "N N"]
LowerTriangular = Float[Array, "N N"]
Kernel = Callable[[Locations, Locations, Variance, Lengthscale], Covariance]
