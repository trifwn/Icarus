"""This module defines the hermes plane object."""
import numpy as np

import ICARUS.Vehicle.wing as wing
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg


def hermes(airfoils: dict, name: str) -> Plane:
    """Get the hermes plane object.

    Args:
        airfoils (dict): _description_
        name (str): _description_

    Returns:
        Plane: _description_
    """
    origin = np.array([0.0, 0.0, 0.0])

    wing_pos = np.array([0.0 - 0.159 / 4, 0.0, 0.0])
    wing_orientation = np.array([2.8, 0.0, 0.0])

    main_wing = wg(
        name='wing',
        airfoil=airfoils["NACA4415"],
        origin=origin + wing_pos,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 1.130,
        sweep_offset=0,
        dih_angle=0,
        chord_fun=wing.define_linear_chord,
        chord=[0.159, 0.072],
        span_fun=wing.define_linear_span,
        N=20,
        M=5,
        mass=0.670,
    )
    
    elevator_position = np.array([0.54 - 0.130 / 4, 0.0, 0.0])
    elevator_orientantion = np.array([0.0, 0.0, 0.0])

    elevator = wg(
        name="tail",
        airfoil=airfoils["NACA0008"],
        origin=origin + elevator_position,
        orientation=elevator_orientantion,
        is_symmetric=True,
        span=2 * 0.169,
        sweep_offset=0,
        dih_angle=0,
        chord_fun=wing.define_linear_chord,
        chord=[0.130, 0.03],
        span_fun=wing.define_linear_span,
        N=15,
        M=5,
        mass=0.06,
    )
    # elevator.plotWing()

    rudder_position = np.array([0.47 - 0.159 / 4, 0.0, 0.01])
    rudder_orientation = np.array([0.0, 0.0, 90.0])

    rudder = wg(
        name="rudder",
        airfoil=airfoils["NACA0008"],
        origin=origin + rudder_position,
        orientation=rudder_orientation,
        is_symmetric=False,
        span=0.160,
        sweep_offset=0.1,
        dih_angle=0,
        chord_fun=wing.define_linear_chord,
        chord=[0.2, 0.1],
        span_fun=wing.define_linear_span,
        N=15,
        M=5,
        mass=0.04,
    )
    # rudder.plotWing()

    lifting_surfaces = [main_wing, elevator, rudder]

    point_masses = [
        (0.500, np.array([-0.40, 0.0, 0.0])),  # Motor
        (1.000, np.array([0.090, 0.0, 0.0])),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0])),  # Payload
    ]
    airplane = Plane(name, lifting_surfaces)

    # from ICARUS.Database import DB3D
    # airplane.accessDB(HOMEDIR, DB3D)
    # airplane.visAirplane()
    airplane.addMasses(point_masses)

    return airplane
