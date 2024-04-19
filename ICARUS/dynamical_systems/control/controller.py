import jax.numpy as jnp

from ICARUS.core.types import FloatArray
from ICARUS.dynamical_systems.control.variable import ControllerVariable


class MissionController:
    def __init__(
        self,
        control_variables: list[ControllerVariable],
    ) -> None:
        self.control_vector: dict[str, float | str] = {}
        for var in control_variables:
            self.control_vector[var.name] = var.value

    def control_law(self, time: float, x: FloatArray | jnp.ndarray, v: FloatArray | jnp.ndarray) -> None:
        if time < 30:
            self.control_vector["aoa"] = "TRIM"
            self.control_vector["engine_amps"] = 30
        elif time < 120:
            self.control_vector["aoa"] = "TRIM"
            self.control_vector["engine_amps"] = 30
        elif time < 210:
            self.control_vector["aoa"] = "TRIM"
            self.control_vector["engine_amps"] = 30
        else:
            self.control_vector["aoa"] = 0
            self.control_vector["engine_amps"] = 0

    @property
    def trim_variables(self) -> dict[str, str | float]:
        trim_vars = {}
        for var in self.control_vector:
            if self.control_vector[var] == "TRIM":
                trim_vars[var] = self.control_vector[var]
        return trim_vars
