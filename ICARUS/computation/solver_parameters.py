import io
from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class SolverParameters(ABC):
    """Base class for solver parameters."""

    pass


@dataclass
class NoSolverParameters(SolverParameters):
    """Represents a solver with no parameters."""

    pass


class Parameter:
    def __init__(
        self,
        name: str,
        default_value: Any,
        description: str,
        value_type: Any = None,
    ):
        self.name = name
        self.default_value = default_value
        self.description = description
        if value_type is not None:
            self.value_type = value_type
        else:
            self.value_type = type(default_value)
        self.value = default_value

    def __str__(self) -> str:
        ss = io.StringIO()

        ss.write(f"{self.name} : {self.default_value}\n")
        ss.write(f"{self.description}\n")
        return ss.getvalue()

    def __repr__(self) -> str:
        ss = io.StringIO()

        ss.write(f"{self.name} : {self.default_value}\n")
        ss.write(f"{self.description}\n")
        return ss.getvalue()


class BoolParameter(Parameter):
    def __init__(self, name: str, default_value: bool, description: str):
        super().__init__(name, default_value, description, bool)


class IntParameter(Parameter):
    def __init__(self, name: str, default_value: int, description: str):
        super().__init__(name, default_value, description, int)


class IntOrNoneParameter(Parameter):
    def __init__(self, name: str, default_value: int | None, description: str):
        super().__init__(name, default_value, description, int)


class FloatParameter(Parameter):
    def __init__(self, name: str, default_value: float, description: str):
        super().__init__(name, default_value, description, float)


class StrParameter(Parameter):
    def __init__(self, name: str, default_value: str, description: str):
        super().__init__(name, default_value, description, str)
