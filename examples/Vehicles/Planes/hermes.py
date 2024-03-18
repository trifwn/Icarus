"""This module defines the hermes plane object."""
import numpy as np

from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import SymmetryAxes
from ICARUS.Vehicle.wing_segment import Wing_Segment

# 10 20 m turn radius ???


def hermes(
    name: str,
    motor_x,
    elevator_x,
    main_wing_x,
) -> Airplane:
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

    main_wing = Wing_Segment(
        name="wing",
        root_airfoil="NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 1.130,
        sweep_offset=0,
        root_chord=0.159,
        tip_chord=0.072,
        twist_root=0,
        twist_tip=30,
        N=25,
        M=5,
        mass=0.670,
    )

    # main_wing.plotWing()

    elevator_pos: FloatArray = np.array(
        [0.565 - 0.27 + elevator_x, 0.0, 0.0],  # 0.565-0.27+200..800
        dtype=float,
    )
    elevator_orientantion: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    elevator = Wing_Segment(
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

    rudder = Wing_Segment(
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
        N=15,
        M=5,
        mass=0.04,
    )
    # rudder.plotWing()

    lifting_surfaces: list[Lifting_Surface] = [main_wing, elevator, rudder]

    point_masses = [
        (
            0.205,
            np.array([-motor_x, 0.0, 0.0], dtype=float),
            "engine",
        ),  # Engine (200-400?)
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "structure"),  # Structure
        # (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "battery"),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float), "payload"),  # Payload
    ]
    for i in range(0, 5):
        point_masses.append((0.175, np.array([0.06 * i, 0.04, 0.0], dtype=float), f"ball_{i+1}"))
    for i in range(0, 5):
        point_masses.append((0.175, np.array([0.06 * i, -0.04, 0.0], dtype=float), f"ball_{5+i+1}"))
    for i in range(0, 4):
        point_masses.append((0.175, np.array([-0.06 * i, 0.04, 0.0], dtype=float), f"ball_{i+1+10}"))
    for i in range(0, 4):
        point_masses.append((0.175, np.array([-0.06 * i, -0.04, 0.0], dtype=float), f"ball_{14+i+1}"))
    airplane: Airplane = Airplane(name, lifting_surfaces)
    airplane.add_point_masses(point_masses)

    return airplane


if __name__ == "__main__":
    airplane = hermes("hermes", 0, 0, 0)
    airplane.visualize()
