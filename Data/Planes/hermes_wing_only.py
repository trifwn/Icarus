import numpy as np

import ICARUS.Vehicle.wing as wing
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg


def hermesMainWing(airfoils, name: str) -> Plane:
    """Function to get a plane Consisting only of the main wing of the hermes plane

    Args:
        airfoils (_type_): _description_
        name (str): _description_

    Returns:
        _type_: _description_
    """
    origin = np.array([0.0, 0.0, 0.0])

    wing_pos = np.array([0.0, 0.0, 0.0])
    wing_orientation = np.array([2.8, 0.0, 0.0])

    main_wing = wg(
        name="wing",
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
    # main_wing.plotWing()

    lifting_surfaces = [main_wing]
    airplane = Plane(name, lifting_surfaces)

    # airplane.visAirplane()

    # point_masses = [
    #     (0.500 , np.array([-0.40, 0.0, 0.0])), # Motor
    #     (1.000 , np.array([0.090, 0.0, 0.0])), # Battery
    #     (0.900 , np.array([0.130, 0.0, 0.0])), # Payload
    #     ]
    # airplane.addMasses(point_masses)
    return airplane
