"""This module defines the hermes plane object."""

import os

import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.solvers.XFLR5.polars import read_polars_2d
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import Wing
from ICARUS.vehicle import WingSegment


def e190_takeoff_generator(
    name: str,
    flap_hinge: float = 0.75,
    chord_extension: float = 1.3,
    flap_angle: float = 35,
) -> Airplane:
    """Function to get the embraer e190 plane.
    Consisting of the main wing, elevator rudder and masses as constructed.

    Args:
        name (str): Name of the plane

    Returns:
        Airplane: hermes Airplane object

    """
    DB = Database("./Data")
    read_polars_2d(os.path.join(DB.EXTERNAL_DB, "2D"))
    from ICARUS.airfoils import Airfoil

    naca64418: Airfoil = DB.get_airfoil("NACA64418")
    naca64418_fl: Airfoil = naca64418.flap(
        flap_hinge_chord_percentage=flap_hinge,
        chord_extension=chord_extension,
        flap_angle=flap_angle,
    )

    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)

    wing_position: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [1.5, 0.0, 0.0],  # [pitch , yaw , Roll]
        dtype=float,
    )

    wing_1 = WingSegment(
        name="wing_1",
        root_airfoil=naca64418_fl,
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 4.180,
        sweep_offset=2.0,
        root_chord=5.6,
        tip_chord=3.7,
        N=15,  # Spanwise
        M=10,  # Chordwise
        mass=1,
    )
    # main_wing.plotWing()

    wing_2_pos: FloatArray = np.array(
        [2.0, 4.18, 0.0],
        dtype=float,
    )
    wing_2_or: FloatArray = np.array(
        [1.5, 0.0, 0.0],
        dtype=float,
    )

    wing_2 = WingSegment(
        name="wing_2",
        root_airfoil=naca64418_fl,
        origin=origin + wing_2_pos,
        orientation=wing_2_or,
        symmetries=SymmetryAxes.Y,
        span=2 * (10.500 - 4.180),
        sweep_offset=5.00 - 2,
        root_dihedral_angle=5,
        tip_dihedral_angle=5,
        root_chord=3.7,
        tip_chord=2.8,
        N=10,
        M=10,
        mass=1,
    )
    # elevator.plotWing()

    wing_3_pos: FloatArray = np.array(
        [3 + 2, 10.5, np.sin(5 * np.pi / 180) * (10.500 - 4.180)],
        dtype=float,
    )
    wing_3_or: FloatArray = np.array(
        [1.5, 0.0, 0],
        dtype=float,
    )

    wing_3 = WingSegment(
        name="wing_3",
        root_airfoil=naca64418,
        origin=origin + wing_3_pos,
        orientation=wing_3_or,
        symmetries=SymmetryAxes.Y,
        span=2 * (14.36 - 10.5),
        sweep_offset=6.75 - 5,
        root_dihedral_angle=5,
        tip_dihedral_angle=5,
        root_chord=2.8,
        tip_chord=2.3,
        N=10,
        M=10,
        mass=1.0,
    )
    # rudder.plotWing()

    main_wing = Wing(name="main_wing", wing_segments=[wing_1, wing_2, wing_3])
    airplane: Airplane = Airplane(name, main_wing=main_wing)
    return airplane
