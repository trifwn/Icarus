from typing import Callable

from interpax import CubicSpline as interpax_CubicSpline
from jaxtyping import Array
from jaxtyping import Float
import jax 

def CubicSpline_factory(
    x: Float[Array, "dim1"],
    y: Float[Array, "dim1"],
) -> Callable[[Float[Array, "dim2"]], Float[Array, 'dim2']]:
    cs = interpax_CubicSpline(x, y, check=False)

    def spline(x: Float[Array, "dim2"]) -> Float[Array, "dim2"]:
        return cs(x)

    return spline
