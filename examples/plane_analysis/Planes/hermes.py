"""This module defines the hermes plane object."""

import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.surface import WingSurface
from ICARUS.vehicle.utils import SymmetryAxes
from ICARUS.vehicle.wing_segment import WingSegment


def hermes(name: str) -> Airplane:
    """
    Function to get the hermes plane.
    Consisting of the main wing, elevator rudder and masses as constructed.

    Args:
        name (str): Name of the plane

    Returns:
        Airplane: hermes Airplane object
    """
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)

    wing_position: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [2.8, 0.0, 0.0],
        dtype=float,
    )

    main_wing = WingSegment(
        name="wing",
        root_airfoil="NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 1.130,
        sweep_offset=0,
        root_chord=0.159,
        tip_chord=0.072,
        N=30,
        M=10,
        mass=0.670,
    )

    elevator_pos: FloatArray = np.array(
        [0.54, 0.0, 0.0],
        dtype=float,
    )
    elevator_orientantion: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    elevator = WingSegment(
        name="elevator",
        root_airfoil="NACA0008",
        origin=origin + elevator_pos,
        orientation=elevator_orientantion,
        symmetries=SymmetryAxes.Y,
        span=2 * 0.169,
        sweep_offset=0,
        root_dihedral_angle=0,
        root_chord=0.130,
        tip_chord=0.03,
        N=10,
        M=5,
        mass=0.06,
    )

    rudder_position: FloatArray = np.array(
        [0.47, 0.0, 0.01],
        dtype=float,
    )
    rudder_orientation: FloatArray = np.array(
        [0.0, 0.0, 90.0],
        dtype=float,
    )

    rudder = WingSegment(
        name="rudder",
        root_airfoil="NACA0008",
        origin=origin + rudder_position,
        orientation=rudder_orientation,
        symmetries=SymmetryAxes.NONE,
        span=0.160,
        sweep_offset=0.1,
        root_dihedral_angle=0,
        root_chord=0.2,
        tip_chord=0.1,
        N=10,
        M=5,
        mass=0.04,
    )

    lifting_surfaces: list[WingSurface] = [main_wing, elevator, rudder]

    point_masses = [
        (0.500, np.array([-0.40, 0.0, 0.0], dtype=float), "engine"),  # Engine
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "structure"),  # Structure
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float), "payload"),  # Payload
        # (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "battery"),  # Battery
    ]
    airplane: Airplane = Airplane(name, lifting_surfaces)
    airplane.add_point_masses(point_masses)

    return airplane


if __name__ == "__main__":
    pln = hermes("hermes")
    pln.save()
