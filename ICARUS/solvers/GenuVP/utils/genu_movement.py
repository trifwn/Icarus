from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import numpy as np

from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import Disturbance
    from ICARUS.vehicle import WingSurface


class GNVP_Movement:
    """Class to specify a generalized Movement as defined in the gnvp3 manual."""

    def __init__(
        self,
        name: str,
        rotation: dict[str, Any],
        translation: dict[str, Any],
    ) -> None:
        """Initialize the Movement Class

        Args:
            name (str): Name of the Movement
            Rotation (dict[str,Any]): Rotation Parameters. They must include:\
                                        1) type: int \
                                        2) axis: int \
                                        3) t1: float \
                                        4) t2: float \
                                        5) a1: float \
                                        6) a2: float \
            Translation (dict[str,Any]): Translation Parameters. They must include: \
                                        1) type: int \
                                        2) axis: int \
                                        3) t1: float \
                                        4) t2: float \
                                        5) a1: float \
                                        6) a2: float \

        """
        self.name: str = name
        self.rotation_type: int = rotation["type"]
        self.rotation_axis: int = rotation["axis"]

        self.rot_t1: float = rotation["t1"]
        self.rot_t2: float = rotation["t2"]

        self.rot_a1: float = rotation["a1"]
        self.rot_a2: float = rotation["a2"]

        if "str" in rotation.keys():
            self.rotation_str: str = rotation["str"]
        else:
            self.rotation_str = ""

        self.translation_type: int = translation["type"]
        self.translation_axis: int = translation["axis"]

        self.translation_t1: float = translation["t1"]
        self.translation_t2: float = translation["t2"]

        self.translation_a1: float = translation["a1"]
        self.translation_a2: float = translation["a2"]

        if "str" in translation.keys():
            self.translation_str: str = translation["str"]
        else:
            self.translation_str = ""


def disturbance2movement(disturbance: Disturbance) -> GNVP_Movement:
    """Converts a disturbance to a movement

    Args:
        disturbance (Disturbance): Disturbance Object

    Raises:
        ValueError: If the disturbance type is not supported

    Returns:
        Movement: Movement Object

    """
    if disturbance.type == "Derivative":
        t1: float = -1
        t2: float = 0
        a1: float | None = 0
        a2: float | None = disturbance.amplitude
        distType = 100
    elif disturbance.type == "Value":
        t1 = -1
        t2 = 0.0
        a1 = disturbance.amplitude
        a2 = disturbance.amplitude
        distType = 1
    else:
        raise ValueError

    undisturbed: dict[str, Any] = {
        "type": 1,
        "axis": disturbance.axis,
        "t1": -1,
        "t2": 0,
        "a1": 0,
        "a2": 0,
    }

    disturbed: dict[str, Any] = {
        "type": distType,
        "axis": disturbance.axis,
        "t1": t1,
        "t2": t2,
        "a1": a1,
        "a2": a2,
        "str": f"{disturbance.name} {disturbance.axis} {disturbance.amplitude} {disturbance.type}",
    }

    # We need to flip the disturbances as the refer to the stability axes
    if disturbance.axis == 1 or disturbance.axis == 3:
        disturbed["a1"] = -disturbed["a1"]
        disturbed["a2"] = -disturbed["a2"]

    if disturbance.is_rotational:
        # Conver Rad 2 Deg
        disturbed["a1"] = np.rad2deg(disturbed["a1"])
        disturbed["a2"] = np.rad2deg(disturbed["a2"])
        Rotation: dict[str, Any] = disturbed
        Translation: dict[str, Any] = undisturbed
    else:
        Rotation = undisturbed
        Translation = disturbed

    return GNVP_Movement(disturbance.name, Rotation, Translation)


def define_movements(
    surfaces: Sequence[WingSurface],
    CG: FloatArray,
    orientation: FloatArray | list[float],
    disturbances: list[Disturbance] = [],
) -> list[list[GNVP_Movement]]:
    """Define Movements for the surfaces.

    Args:
        surfaces (list[Wing]): List of Wing Objects
        CG (FloatArray): Center of Gravity
        orientation (FloatArray | list[float]): Orientation of the plane
        disturbances (list[Disturbance]): List of possible Disturbances. Defaults to empty list.

    Returns:
        list[list[Movement]]: A list of movements for each surface of the plane so that the center of gravity is at the origin.

    """
    movement: list[list[GNVP_Movement]] = []
    all_axes = ("pitch", "roll", "yaw")
    all_ax_ids = (2, 1, 3)
    for _ in surfaces:
        sequence: list[GNVP_Movement] = []
        for name, axis in zip(all_axes, all_ax_ids):
            Rotation: dict[str, Any] = {
                "type": 1,
                "axis": axis,
                "t1": -0.1,
                "t2": 0.0,
                "a1": orientation[axis - 1],
                "a2": orientation[axis - 1],
            }
            Translation: dict[str, Any] = {
                "type": 1,
                "axis": axis,
                "t1": -0.1,
                "t2": 0.0,
                "a1": -CG[axis - 1],
                "a2": -CG[axis - 1],
            }

            obj = GNVP_Movement(name, Rotation, Translation)
            sequence.append(obj)

        for disturbance in disturbances:
            if disturbance.type is not None:
                sequence.append(disturbance2movement(disturbance))

        movement.append(sequence)
    return movement
