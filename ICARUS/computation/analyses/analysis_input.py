from __future__ import annotations

import io
from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane


@dataclass
class AnalysisInput(ABC):
    """
    Abstract base class for analysis inputs.
    Serves as a data container for a single analysis run.
    """


class Input:
    """Class that represents options for an analysis. It stores the name, value, description and type of the option

    Args:
        name (str): Name of the option.
        description (str): Description of the option.
        value_type (Any): Type of the option.

    Methods:
        __str__(): Returns a string representation of the option.
        __repr__(): Returns a string representation of the option.

    """

    def __init__(self, name: str, description: str, value_type: Any = None) -> None:
        self.name = name
        self.description: str = description
        self.value_type = value_type
        self.value = None

    def __str__(self) -> str:
        ss = io.StringIO()

        ss.write(f"{self.name} :\n")
        ss.write(f"{self.description}\n")
        return ss.getvalue()

    def __repr__(self) -> str:
        ss = io.StringIO()

        ss.write(f"{self.name} :\n")
        ss.write(f"{self.description}\n")
        return ss.getvalue()


class BoolInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, bool)


class IntInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, int)


class FloatInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, float)


class ListInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, list)


class ListFloatInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, list[float])


class NDArrayInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, np.ndarray)


class StrInput(Input):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description, str)


class AirplaneInput(Input):
    def __init__(self) -> None:
        name = "plane"
        description = "Vehicle Airplane Object"
        super().__init__(name, description, Airplane)


class StateInput(Input):
    def __init__(self) -> None:
        name = "state"
        description = "State Object"
        super().__init__(name, description, State)


class AirfoilInput(Input):
    def __init__(self) -> None:
        name = "airfoil"
        description = "Airfoil Object"
        value_type = Airfoil
        super().__init__(name, description, value_type)
