"""This module defines the hermes plane object."""
import numpy as np
from numpy import dtype, ndarray, floating
from typing import Any

from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing import define_linear_chord
from ICARUS.Vehicle.wing import define_linear_span
from ICARUS.Vehicle.wing import Wing


def hermes(airfoils, name) -> Airplane:
    origin: ndarray[Any, dtype[floating]] = np.array([0.0, 0.0, 0.0], dtype=float)

    wing_position: ndarray[Any, dtype[floating]] = np.array([0.0 - 0.159 / 4, 0.0, 0.0], dtype=float)
    wing_orientation: ndarray[Any, dtype[floating]] = np.array([2.8, 0.0, 0.0], dtype=float)

    main_wing = Wing(
        name="wing",
        airfoil=airfoils["NACA4415"],
        origin=origin + wing_position,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 1.130,
        sweep_offset=0,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=np.array([0.159, 0.072], dtype=float),
        span_fun=define_linear_span,
        N=30,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()

    elevatorPos: ndarray[Any, dtype[floating]] = np.array([0.54 - 0.130 / 4, 0.0, 0.0], dtype=float)
    elevatorOrientantion: ndarray[Any, dtype[floating]] = np.array([0.0, 0.0, 0.0], dtype=float)

    elevator = Wing(
        name="tail",
        airfoil=airfoils["NACA0008"],
        origin=origin + elevatorPos,
        orientation=elevatorOrientantion,
        is_symmetric=True,
        span=2 * 0.169,
        sweep_offset=0,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=np.array([0.130, 0.03]),
        span_fun=define_linear_span,
        N=15,
        M=5,
        mass=0.06,
    )
    # elevator.plotWing()

    rudder_position: ndarray[Any, dtype[floating]] = np.array([0.47 - 0.159 / 4, 0.0, 0.01], dtype=float)
    rudder_orientation: ndarray[Any, dtype[floating]] = np.array([0.0, 0.0, 90.0], dtype=float)

    rudder = Wing(
        name="rudder",
        airfoil=airfoils["NACA0008"],
        origin=origin + rudder_position,
        orientation=rudder_orientation,
        is_symmetric=False,
        span=0.160,
        sweep_offset=0.1,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=np.array([0.2, 0.1]),
        span_fun=define_linear_span,
        N=15,
        M=5,
        mass=0.04,
    )
    # rudder.plotWing()

    lifting_surfaces = [main_wing, elevator, rudder]

    point_masses = [
        (0.500, np.array([-0.40, 0.0, 0.0], dtype=float)),  # Motor
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float)),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float)),  # Payload
    ]
    airplane: Airplane = Airplane(name, lifting_surfaces)

    # from ICARUS.Database import DB3D
    # airplane.accessDB(HOMEDIR, DB3D)
    # airplane.visAirplane()
    airplane.addMasses(point_masses)

    return airplane
