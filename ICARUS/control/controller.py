import jax.numpy as jnp

from ICARUS.core.types import FloatArray

from .variable import ControllerVariable


class Controlller:
    def __init__(
        self,
        control_variables: list[ControllerVariable],
    ) -> None:
        self.control_vector: dict[str, float | str] = {}
        for var in control_variables:
            self.control_vector[var.name] = var.value

    def control_law(
        self,
        time: float,
        x: FloatArray | jnp.ndarray,
        v: FloatArray | jnp.ndarray,
    ) -> None:
        raise NotImplementedError("Control law not implemented")

    @property
    def trim_variables(self) -> dict[str, str | float]:
        return {k: v for k, v in self.control_vector.items() if isinstance(v, str) and v == "TRIM_VARIABLE"}
