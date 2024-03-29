import jax.numpy as jnp


class DynamicalSystem:
    # This class is the base class for all dynamical elements
    def __init__(self) -> None:
        raise NotImplementedError

    def jacobian(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
