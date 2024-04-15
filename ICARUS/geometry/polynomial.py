from typing import Callable

import jax.numpy as jnp
import numpy as np

from ICARUS.core.types import FloatArray


def h_polynomial_factory(
    coeffs: jnp.ndarray,
) -> Callable[..., jnp.ndarray]:
    def h_polynomial(x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.zeros_like(x)
        for i, c in enumerate(coeffs):
            if i == 0:
                h += c
            else:
                h += (c / 10 ** (i + 1)) * x ** (i + 1)
        return h

    return h_polynomial
