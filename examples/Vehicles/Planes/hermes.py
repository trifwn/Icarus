"""This module defines the hermes plane object."""
import numpy as np

from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import define_linear_chord
from ICARUS.Vehicle.utils import define_linear_span


def hermes(name: str) -> Airplane:
    """
    Function to get the hermes plane.
    Consisting of the main wing, elevator rudder and masses as constructed.

    Args:
        airfoils (Struct): Struct containing the airfoils
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

    main_wing = Lifting_Surface(
        name="wing",
        airfoil="NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 1.130,
        sweep_offset=0,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=np.array([0.159, 0.072], dtype=float),
        span_fun=define_linear_span,
        N=25,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()

    elevator_pos: FloatArray = np.array(
        [0.54, 0.0, 0.0],
        dtype=float,
    )
    elevator_orientantion: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    elevator = Lifting_Surface(
        name="elevator",
        airfoil="NACA0008",
        origin=origin + elevator_pos,
        orientation=elevator_orientantion,
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

    rudder_position: FloatArray = np.array(
        [0.47, 0.0, 0.01],
        dtype=float,
    )
    rudder_orientation: FloatArray = np.array(
        [0.0, 0.0, 90.0],
        dtype=float,
    )

    rudder = Lifting_Surface(
        name="rudder",
        airfoil="NACA0008",
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

    lifting_surfaces: list[Lifting_Surface] = [main_wing, elevator, rudder]

    point_masses = [
        (0.500, np.array([-0.40, 0.0, 0.0], dtype=float), "engine"),  # Engine
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "structure"),  # Structure
        # (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "battery"),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float), "payload"),  # Payload
    ]
    airplane: Airplane = Airplane(name, lifting_surfaces)
    airplane.add_point_masses(point_masses)

    return airplane
