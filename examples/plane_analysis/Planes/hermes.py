"""This module defines the hermes plane object."""

import numpy as np

from ICARUS.airfoils import NACA4
from ICARUS.core.types import FloatArray
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import Mass
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import WingSegment


def hermes(name: str) -> Airplane:
    """Function to get the hermes plane.
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
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    main_wing = WingSegment(
        name="wing",
        root_airfoil=NACA4(M=0.04, P=0.4, XX=0.15),  # "NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 1.130,
        sweep_offset=0,
        root_chord=0.159,
        tip_chord=0.072,
        N=20,
        M=15,
        mass=0.670,
    )

    elevator_pos: FloatArray = np.array(
        [0.54, 0.0, 0.0],
        dtype=float,
    )
    elevator_orientantion: FloatArray = np.array(
        [-4.0, 0.0, 0.0],
        dtype=float,
    )

    elevator = WingSegment(
        name="elevator",
        root_airfoil=NACA4(M=0.0, P=0.0, XX=0.08),  # "NACA0008",
        origin=origin + elevator_pos,
        orientation=elevator_orientantion,
        symmetries=SymmetryAxes.Y,
        span=2 * 0.169,
        sweep_offset=0,
        root_dihedral_angle=0,
        root_chord=0.130,
        tip_chord=0.03,
        N=15,
        M=10,
        mass=0.06,
    )

    rudder_position: FloatArray = np.array(
        [0.47, 0.0, 0.1],
        dtype=float,
    )
    rudder_orientation: FloatArray = np.array(
        [0.0, 90.0, 0.0],
        dtype=float,
    )

    rudder = WingSegment(
        name="rudder",
        root_airfoil=NACA4(M=0.0, P=0.0, XX=0.08),  # "NACA0008",
        origin=origin + rudder_position,
        orientation=rudder_orientation,
        symmetries=SymmetryAxes.NONE,
        span=0.160,
        sweep_offset=0.1,
        root_dihedral_angle=0,
        root_chord=0.2,
        tip_chord=0.1,
        N=15,
        M=10,
        mass=0.04,
    )

    point_masses = [
        Mass(
            mass=0.500,
            position=np.array([-0.40, 0.0, 0.0], dtype=float),
            name="engine",
        ),  # Engine
        Mass(
            mass=1.000,
            position=np.array([0.090, 0.0, 0.0], dtype=float),
            name="structure",
        ),  # Structure
        Mass(
            mass=0.900,
            position=np.array([0.130, 0.0, 0.0], dtype=float),
            name="payload",
        ),  # Payload
        # Mass(mass= 1.000, position=np.array([0.090, 0.0, 0.0], dtype=float), name="battery"),  # Battery
    ]
    airplane: Airplane = Airplane(
        name,
        main_wing=main_wing,
        other_wings=[elevator, rudder],
    )
    airplane.add_point_masses(point_masses)

    return airplane


if __name__ == "__main__":
    pln = hermes("hermes")
    pln.save()
