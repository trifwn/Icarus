"""This module defines the hermes plane object."""
import numpy as np

from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import define_linear_chord
from ICARUS.Vehicle.utils import define_linear_span

############################## AIRPLANE GENERATOR ########################################
# The code below is a demo function used to generate a conventional airplane.
############################## AIRPLANE GENERATOR ########################################


def airplane_generator(name: str, plotting=False) -> Airplane:
    """
    Function to generate a conventional airplane, consisting of:
        1) main wing
        2) elevator
        3) rudder
        4) Point Masses

    Args:
        name (str): Name of the plane

    Returns:
        Airplane: Airplane object
    """

    # Coordinate system to which the airplane is attached
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)

    ########################## MAIN WING ########################################

    # Positioning the wing in the coordinate system

    # Position of the main wing in the coordinate system. I typically place the
    # quarter chord of the root chord at the origin.
    wing_position: FloatArray = np.array(
        [0.0 - 0.159 / 4, 0.0, 0.0],
        dtype=float,
    )

    # The wing orientation defined through rotations about the x, y and z axis
    # The first number is the incidence angle, the second is almost never used
    # in aircraft design and the third is rotation around the x axis.
    wing_orientation: FloatArray = np.array(
        [2.8, 0.0, 0.0],
        dtype=float,
    )

    main_wing = Lifting_Surface(
        name="wing",  # Name of the wing
        airfoil="4415",  # Airfoil name
        origin=origin + wing_position,  # Position of the wing in the coordinate system
        orientation=wing_orientation,  # Orientation of the wing
        is_symmetric=True,  # Is the wing symmetric about the x-z plane?
        span=2 * 1.130,  # Span of the wing
        sweep_offset=0,  # Sweep offset of the wing
        dih_angle=0,  # Dihedral angle of the wing
        chord_fun=define_linear_chord,  # Function to define the chord distribution
        chord=np.array([0.159, 0.072], dtype=float),  # Arguements to the chord function.
        span_fun=define_linear_span,  # Span Discretization function
        N=25,  # Number of chordwise panels
        M=5,  # Number of spanwise panels
        mass=0.670,  # Mass of the wing in kg
    )
    ########################## MAIN WING ########################################

    ########################## ELEVATOR ########################################

    # Position of the elevator in the coordinate system
    elevator_pos: FloatArray = np.array(
        [0.54 - 0.130 / 4, 0.0, 0.0],
        dtype=float,
    )

    # Orientation of the elevator in the coordinate system
    elevator_orientantion: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    # Elevator Definition
    elevator = Lifting_Surface(
        name="elevator",
        airfoil="0008",
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
    ########################## ELEVATOR ########################################

    ########################## RUDDER ########################################

    # Position of the rudder in the coordinate system
    rudder_position: FloatArray = np.array(
        [0.47 - 0.159 / 4, 0.0, 0.01],
        dtype=float,
    )
    # Orientation of the rudder in the coordinate system
    rudder_orientation: FloatArray = np.array(
        [0.0, 0.0, 90.0],
        dtype=float,
    )

    # Rudder Definition
    rudder = Lifting_Surface(
        name="rudder",
        airfoil="0008",
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
    ########################## RUDDER ########################################

    ########################## POINT MASSES ########################################
    point_masses = [
        (0.500, np.array([-0.40, 0.0, 0.0], dtype=float), "engine"),  # Engine
        # (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "battery"), # Battery
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "structure"),  # Structure
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float), "payload"),  # Payload
    ]

    ########################## POINT MASSES ########################################

    lifting_surfaces: list[Lifting_Surface] = [main_wing, elevator, rudder]
    airplane: Airplane = Airplane(name, lifting_surfaces)
    airplane.add_point_masses(point_masses)

    ########################## VISUALIZATION ########################################
    if plotting:
        main_wing.plot()  # Plot the wing
        rudder.plot()
        elevator.plot()
        airplane.visualize()
    ########################## VISUALIZATION ########################################

    return airplane


if __name__ == "__main__":
    airplane_generator("aeroplano", True)
