from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.vehicle.airplane import Airplane


class ControlState:
    def __init__(
        self,
        airplane: Airplane,
    ) -> None:
        # Get the airplane control variables
        self.control_vars: set[str] = airplane.control_vars
        self.num_control_vars: int = len(self.control_vars)
        self.control_vector_dict: dict[str, float] = airplane.control_vector
        self.hash_dict: dict[str, int] = {}

    def update(self, control_vector_dict: dict[str, float]) -> None:
        self.control_vector_dict = control_vector_dict

    @property
    def control_vector(self) -> FloatArray:
        return np.array(
            [self.control_vector_dict[key] for key in self.control_vars],
        )

    def __str__(self) -> str:
        string = "Control State: "
        for key in self.control_vars:
            string += f"{key}: {self.control_vector_dict[key]:.3f} "
        return string

    def __hash__(self) -> int:
        """Unique hash for the control state. This is used to generate a unique name for the state.
        It depends on the control variables and their values.

        Returns:
            int: Unique hash value for the control state

        """
        hash_val = hash(frozenset(self.control_vector_dict.items()))
        # Add to the hash dictionary if not already present
        if str(hash_val) not in list(self.hash_dict.keys()):
            self.hash_dict[str(hash_val)] = len(self.hash_dict)

        return hash_val

    def identifier(self) -> int:
        num = self.__hash__()
        return self.hash_dict[str(num)]
