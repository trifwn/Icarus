from enum import Enum

import numpy as np


class ControlVariableType(Enum):
    TRIM_VARIABLE = 0
    FIXED_VALUE = 1
    FIXED_DERIVATIVE = 2
    NONE = 3

    def __str__(self) -> str:
        return self.name


class ControllerVariable:
    def __init__(
        self,
        name: str,
        type: ControlVariableType | str,
        initial_value: float = 0,
        bounds: tuple[float, float] = (-np.inf, np.inf),
    ) -> None:
        self.name = name

        if isinstance(type, str):
            self.type = ControlVariableType[type]
        else:
            self.type = type

        self.value = initial_value
        self.bounds = bounds

    def __str__(self) -> str:
        return f"{self.name}: {self.value} ({self.type})"


class ObserverVariableType(Enum):
    P = 1
    Int = 2
    D = 3
    PI = 4
    PD = 5
    ID = 6
    PID = 7
    FIXED = 8
    FREE = 9

    def __str__(self) -> str:
        return self.name


class ObserverVariable:
    def __init__(
        self,
        name: str,
        type: ObserverVariableType | str,
        initial_value: float = 0,
    ) -> None:
        self.name = name

        if isinstance(type, str):
            self.type = ObserverVariableType[type]
        else:
            self.type = type

        self.value = initial_value

    def __str__(self) -> str:
        return f"{self.name}: {self.value} ({self.type})"
