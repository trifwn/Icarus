from typing import Callable

import jax.numpy as jnp


def polynomial_factory(
    coeffs: jnp.ndarray,
) -> Callable[..., jnp.ndarray]:
    def poly(x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.zeros_like(x)
        for i, c in enumerate(coeffs):
            if i == 0:
                h += c
            else:
                h += (c / 10 ** (i + 1)) * x ** (i + 1)
        return h

    return poly
