from typing import Callable

from interpax import CubicSpline as interpax_CubicSpline
from jaxtyping import Array
from jaxtyping import Float


def CubicSpline_factory(
    x: Float[Array, ...],
    y: Float[Array, "{x.shape}"],
) -> Callable[[Float[Array, ...]], Float[Array, ...]]:
    cs = interpax_CubicSpline(x, y, check=False)

    def spline(x: Float[Array, ...]) -> Float[Array, "{x.shape}"]:
        return cs(x)

    return spline
