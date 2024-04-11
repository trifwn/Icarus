from typing import Callable

from interpax import CubicSpline as interpax_CubicSpline
from jaxtyping import Array
from jaxtyping import Float


def CubicSpline_factory(
    x: Float[Array, "dim1"],
    y: Float[Array, "dim1"],
) -> tuple[Callable[[Float[Array, "dim2"]], Float[Array, 'dim2']], str]:
    cs = interpax_CubicSpline(x, y)

    def spline(x: Float[Array, "dim2"]) -> Float[Array, "dim2"]:
        return cs(x)

    title = "Cubic Spline Passes Through\n$"
    for num in cs.x:
        title += f" {num:.2f}, "
    title += "\n"
    for num in y:
        title += f" {num:.2f}, "
    title += "$"
    return spline, title
