"""This module defines the hermes plane object."""
import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.surface import WingSurface
from ICARUS.vehicle.utils import SymmetryAxes
from ICARUS.vehicle.wing_segment import WingSegment

############################## AIRPLANE GENERATOR ########################################
# The code below is a demo function used to generate a conventional airplane.
############################## AIRPLANE GENERATOR ########################################


def airplane_generator(name: str, plotting: bool = False) -> Airplane:
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

    main_wing = WingSegment(
        name="wing",  # Name of the wing
        root_airfoil="4415",  # Airfoil name
        origin=origin + wing_position,  # Position of the wing in the coordinate system
        orientation=wing_orientation,  # Orientation of the wing
        symmetries=SymmetryAxes.Y,  # Is the wing symmetric about the x-z plane?
        span=2 * 1.130,  # Span of the wing
        sweep_offset=0,  # Sweep offset of the wing
        root_chord=0.159,  # Root chord of the wing
        tip_chord=0.072,  # Tip chord of the wing
        twist_root=0,  # Twist at the root of the wing
        twist_tip=0,  # Twist at the tip of the wing
        root_dihedral_angle=0,  # Dihedral angle at the root of the wing
        tip_dihedral_angle=0,  # Dihedral angle at the tip of the wing
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

    elevator = WingSegment(
        name="elevator",  # Name of the wing
        root_airfoil="0008",  # Airfoil name
        origin=origin + elevator_pos,  # Position of the wing in the coordinate system
        orientation=elevator_orientantion,  # Orientation of the wing
        symmetries=SymmetryAxes.Y,  # Is the wing symmetric about the x-z plane?
        span=2 * 0.169,  # Span of the wing
        sweep_offset=0,  # Sweep offset of the wing
        root_chord=0.13,  # Root chord of the wing
        tip_chord=0.03,  # Tip chord of the wing
        twist_root=0,  # Twist at the root of the wing
        twist_tip=0,  # Twist at the tip of the wing
        root_dihedral_angle=0,  # Dihedral angle at the root of the wing
        tip_dihedral_angle=0,  # Dihedral angle at the tip of the wing
        N=15,  # Number of chordwise panels
        M=5,  # Number of spanwise panels
        mass=0.06,  # Mass of the wing in kg
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
    rudder = WingSegment(
        name="rudder",
        root_airfoil="0008",
        origin=origin + rudder_position,
        orientation=rudder_orientation,
        symmetries=SymmetryAxes.NONE,
        span=0.160,
        sweep_offset=0.1,
        root_dihedral_angle=0,
        tip_dihedral_angle=0,
        root_chord=0.2,
        tip_chord=0.1,
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

    lifting_surfaces: list[WingSurface] = [main_wing, elevator, rudder]
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
