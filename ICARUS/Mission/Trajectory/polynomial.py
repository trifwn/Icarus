from typing import Callable

from ICARUS.Core.types import FloatArray


def h_polynomial_factory(coeffs: list[float] | FloatArray) -> Callable[..., float | FloatArray]:
    def h_polynomial(x: float | FloatArray) -> float | FloatArray:
        h = 0
        for i, c in enumerate(coeffs):
            if i == 0:
                h += c
            else:
                h += (c / 10 ** (i + 1)) * x ** (i + 1)
        return h

    return h_polynomial
