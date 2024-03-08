from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit

from .base_system import DynamicalSystem


class LinearSystem(DynamicalSystem):
    def __init__(self, A: jnp.ndarray, B: jnp.ndarray) -> None:
        self.A = A
        self.B = B

    def jacobian(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.A

    @partial(jit, static_argnums=(0,))
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.A.squeeze() @ x + self.B


class NonLinearSystem(DynamicalSystem):
    def __init__(
        self,
        f: Callable[[float, jnp.ndarray], jnp.ndarray],
        jac: Callable[[float, jnp.ndarray], jnp.ndarray] | None = None,
    ) -> None:
        self.f = jit(f)
        self.jac = jit(jac) if jac is not None else None

    @partial(jit, static_argnums=(0,))
    def jacobian(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        if self.jac is not None:
            return self.jac(t, x)
        return jax.jacobian(self.f, argnums=1)(t, x)

    @partial(jit, static_argnums=(0,))
    def linearize(self, t: float, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        A = self.jacobian(t, x)
        B = self.f(t, x) - A.squeeze() @ x
        return A, B

    @partial(jit, static_argnums=(0,))
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        # Linearize the system first and then use the linearized system to get the output
        linearized: LinearSystem = LinearSystem(*self.linearize(t, x))
        return linearized(t, x)
