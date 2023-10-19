"""This module defines the hermes plane object."""
from copy import deepcopy
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct
from ICARUS.Database import XFLRDB
from ICARUS.Database.db import DB
from ICARUS.Input_Output.XFLR5.polars import read_polars_2d
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing_segment import define_linear_chord
from ICARUS.Vehicle.wing_segment import define_linear_span
from ICARUS.Vehicle.wing_segment import Wing_Segment


def e190_takeoff_generator(
    name: str,
    flap_hinge=0.75,
    chord_extension=1.3,
    flap_angle=35,
) -> Airplane:
    """
    Function to get the embraer e190 plane.
    Consisting of the main wing, elevator rudder and masses as constructed.

    Args:
        airfoils (Struct): Struct containing the airfoils
        name (str): Name of the plane

    Returns:
        Airplane: hermes Airplane object
    """
    db = DB()
    foildb = db.foilsDB
    foildb.load_data()
    read_polars_2d(foildb, XFLRDB)
    airfoils: Struct = foildb.set_available_airfoils()

    from ICARUS.Airfoils.airfoilD import AirfoilD

    naca64418: AirfoilD = db.foilsDB.set_available_airfoils()["NACA64418"]
    naca64418_fl: AirfoilD = naca64418.flap_airfoil(
        flap_hinge=flap_hinge,
        chord_extension=chord_extension,
        flap_angle=flap_angle,
        plotting=False,
    )

    origin: ndarray[Any, dtype[floating[Any]]] = np.array([0.0, 0.0, 0.0], dtype=float)

    wing_position: ndarray[Any, dtype[floating[Any]]] = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: ndarray[Any, dtype[floating[Any]]] = np.array(
        [1.5, 0.0, 0.0],
        dtype=float,
    )

    wing_1 = Wing_Segment(
        name="wing_1",
        airfoil=naca64418_fl,
        origin=origin + wing_position,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 4.180,
        sweep_offset=2.0,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=np.array([5.6, 3.7], dtype=float),
        span_fun=define_linear_span,
        N=15,
        M=10,
        mass=1,
    )
    # main_wing.plotWing()

    wing_2_pos: ndarray[Any, dtype[floating[Any]]] = np.array(
        [2.0, 4.18, 0.0],
        dtype=float,
    )
    wing_2_or: ndarray[Any, dtype[floating[Any]]] = np.array(
        [1.5, 0.0, 0.0],
        dtype=float,
    )

    wing_2 = Wing_Segment(
        name="wing_2",
        airfoil=naca64418_fl,
        origin=origin + wing_2_pos,
        orientation=wing_2_or,
        is_symmetric=True,
        span=2 * (10.500 - 4.180),
        sweep_offset=5.00 - 2,
        dih_angle=5,
        chord_fun=define_linear_chord,
        chord=np.array([3.7, 2.8]),
        span_fun=define_linear_span,
        N=10,
        M=10,
        mass=1,
    )
    # elevator.plotWing()

    wing_3_pos: ndarray[Any, dtype[floating[Any]]] = np.array(
        [3 + 2, 10.5, np.sin(5 * np.pi / 180) * (10.500 - 4.180)],
        dtype=float,
    )
    wing_3_or: ndarray[Any, dtype[floating[Any]]] = np.array(
        [1.5, 0.0, 0],
        dtype=float,
    )

    wing_3 = Wing_Segment(
        name="wing_3",
        airfoil=naca64418,
        origin=origin + wing_3_pos,
        orientation=wing_3_or,
        is_symmetric=True,
        span=2 * (14.36 - 10.5),
        sweep_offset=6.75 - 5,
        dih_angle=5,
        chord_fun=define_linear_chord,
        chord=np.array([2.8, 2.3]),
        span_fun=define_linear_span,
        N=10,
        M=10,
        mass=1.0,
    )
    # rudder.plotWing()

    lifting_surfaces: list[Wing_Segment] = [wing_1, wing_2, wing_3]
    airplane: Airplane = Airplane(name, lifting_surfaces)

    # Define the surface area of the main wing
    airplane.S = wing_1.S + wing_2.S + wing_3.S

    # from ICARUS.Database import DB3D
    # airplane.accessDB(HOMEDIR, DB3D)
    # airplane.visAirplane()

    return airplane
