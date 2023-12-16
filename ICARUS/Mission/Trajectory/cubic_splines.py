from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline

from ICARUS.Core.types import FloatArray


def CubicSpline_factory(
    x: list[float] | FloatArray,
    y: list[float] | FloatArray,
) -> tuple[Callable[..., float | FloatArray], str]:
    cs = CubicSpline(x, y)

    def spline(x: float | FloatArray) -> float | FloatArray:
        return np.array(cs(x), dtype=np.float64)

    title = "Cubic Spline Passes Through\n$"
    for num in cs.x:
        title += f" {num:.2f}, "
    title += "$"
    return spline, title
