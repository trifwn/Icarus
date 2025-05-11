from enum import Enum
from functools import partial
from typing import Callable, Literal

import numpy as np

from ICARUS.core.types import FloatArray


class ControlType(Enum):
    AIRFOIL = "AIRFOIL"


def default_chord_function_factory(
    span_position: float,
    chord_percentage_start: float,
    chord_percentage_end: float,
) -> float:
    return chord_percentage_start + (chord_percentage_end - chord_percentage_start) * span_position


class ControlSurface:
    """Class to represent aerodynamic control surfaces e.g. elevator, rudder, flaps etc."""

    def __init__(
        self,
        name: str,
        control_vector_var: str,
        span_positions: tuple[float, float],
        hinge_chord_percentages: tuple[float, float],
        chord_extension: float = 1.0,
        local_rotation_axis: FloatArray = np.array([0.0, 1.0, 0.0]),
        chord_function: Callable[[float], float] | None = None,
        inverse_symmetric: bool = False,
        constant_chord: float = 0,
        coordinate_system: Literal["local", "global"] = "local",
    ) -> None:
        """Initialize the control surface object.

        Args:
            name (str): Name of the control surface.
            control_vector_var (str): Name of the control vector variable.
            span_positions (tuple[float, float]): Percentage of the span where the control surface starts and ends.
            chord_percentages (tuple[float, float]): Percentage of the chord where the control surface starts and ends.
            chord_extension (float, optional): Chord extension of the control surface. Defaults to 1.0.
            local_rotation_axis (FloatArray): Local rotation axis of the control surface. Defaults to np.array([0.0, 1.0, 0.0]) which is the y-axis.
            chord_function (Callable[[float], float] | None, optional): Function to calculate the chord length. Defaults to None.
            inverse_symmetric (bool, optional): If True, the control surface is inverted. Defaults to False.
            constant_chord (float, optional): If not 0, the chord length is constant. Defaults to 0.
            coordinate_system (Literal["local", "global"], optional): Coordinate system of the control surface. Defaults to "local".
        """
        self.name = name
        self.type = ControlType.AIRFOIL
        self.control_var = control_vector_var
        self.span_position_start = span_positions[0]
        self.span_position_end = span_positions[1]
        self.coordinate_system = coordinate_system

        self.chord_percentage_start = hinge_chord_percentages[0]
        # In between the chord percentages we should take a
        self.chord_percentage_end = hinge_chord_percentages[1]
        self.chord_extension = chord_extension

        self.local_rotation_axis = local_rotation_axis

        if chord_function is None:
            self.chord_function: Callable[[float], float] = partial(
                default_chord_function_factory,
                self.chord_percentage_start,
                self.chord_percentage_end,
            )
        else:
            self.chord_function = chord_function

        self.constant_chord = constant_chord
        self.inverse_symmetric = inverse_symmetric

    def __str__(self) -> str:
        return f"ControlSurface(name={self.name}, type={self.type}, control_var={self.control_var}, span_position_start={self.span_position_start}, span_position_end={self.span_position_end}, chord_percentage_start={self.chord_percentage_start}, chord_percentage_end={self.chord_percentage_end}, chord_extension={self.chord_extension})"

    def __repr__(self) -> str:
        return self.__str__()


NoControl = ControlSurface(
    name="none",
    control_vector_var="none",
    span_positions=(0.0, 0.0),
    hinge_chord_percentages=(0.0, 0.0),
    local_rotation_axis=np.array([0.0, 0.0, 0.0]),
)


class Elevator(ControlSurface):
    def __init__(
        self,
        span_positions: tuple[float, float],
        hinge_chord_percentages: tuple[float, float],
        coordinate_system: Literal["local", "global"] = "local",
    ) -> None:
        super().__init__(
            name="elevator",
            control_vector_var="delta_e",
            span_positions=span_positions,
            hinge_chord_percentages=hinge_chord_percentages,
            local_rotation_axis=np.array([0.0, 1.0, 0.0]),
            chord_extension=1.0,
            coordinate_system=coordinate_system,
        )


class Rudder(ControlSurface):
    def __init__(
        self,
        span_positions: tuple[float, float],
        hinge_chord_percentages: tuple[float, float],
        coordinate_system: Literal["local", "global"] = "local",
    ) -> None:
        super().__init__(
            name="rudder",
            control_vector_var="delta_r",
            span_positions=span_positions,
            hinge_chord_percentages=hinge_chord_percentages,
            local_rotation_axis=np.array([0.0, 1.0, 0.0]),
            chord_extension=1.0,
            coordinate_system=coordinate_system,
        )


class Aileron(ControlSurface):
    def __init__(
        self,
        span_positions: tuple[float, float],
        hinge_chord_percentages: tuple[float, float],
        coordinate_system: Literal["local", "global"] = "local",
    ) -> None:
        super().__init__(
            name="aileron",
            control_vector_var="delta_a",
            span_positions=span_positions,
            hinge_chord_percentages=hinge_chord_percentages,
            local_rotation_axis=np.array([0.0, 1.0, 0.0]),
            chord_extension=1.0,
            inverse_symmetric=True,
            coordinate_system=coordinate_system,
        )


class Flap(ControlSurface):
    def __init__(
        self,
        span_positions: tuple[float, float],
        hinge_chord_percentages: tuple[float, float],
        chord_extension: float,
        coordinate_system: Literal["local", "global"] = "local",
    ) -> None:
        super().__init__(
            name="flap",
            control_vector_var="delta_f",
            span_positions=span_positions,
            hinge_chord_percentages=hinge_chord_percentages,
            local_rotation_axis=np.array([0.0, 1.0, 0.0]),
            chord_extension=chord_extension,
            coordinate_system=coordinate_system,
        )
