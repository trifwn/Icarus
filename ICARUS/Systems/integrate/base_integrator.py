from typing import Any

import jax.numpy as jnp

from ..base_system import DynamicalSystem


class Integrator:
    def __init__(self, system: DynamicalSystem) -> None:
        self.name: str = "None"
        self.type: str = "None"
        raise NotImplementedError

    def step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def simulate(self, x0: jnp.ndarray, t0: float, tf: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
