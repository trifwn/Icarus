""" Variations of the wing of the hermes plane

Returns:
    _type_: _description_
"""
    
import numpy as np

import ICARUS.Vehicle.wing as wing
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg


def wing_var_chord_offset(airfoils, name: str, chords: list, offset: float):
    """Function to get a plane consisting only of the main wing of the hermes plane

    Args:
        airfoils (_type_): _description_
        name (str): _description_
        chords (list): _description_
        offset (float): _description_

    Returns:
        _type_: _description_
    """
    
    origin = np.array([0.0, 0.0, 0.0])
    wing_pos = np.array([0.0 - 0.159 / 4, 0.0, 0.0])
    wing_orientation = np.array([2.8, 0.0, 0.0])

    main_wing = wg(
        name="wing",
        airfoil=airfoils["NACA4415"],
        origin=origin + wing_pos,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 1.130,
        sweep_offset=offset,
        dih_angle=0,
        chord_fun=wing.define_linear_chord,
        chord=chords,  #  [0.159, 0.072],
        span_fun=wing.define_linear_span,
        N=30,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()
    lifting_surfaces = [main_wing]

    point_masses = [
        (0.500, np.array([-0.40, 0.0, 0.0])),  # Motor
        (1.000, np.array([0.090, 0.0, 0.0])),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0])),  # Payload
    ]
    airplane = Plane(name, lifting_surfaces)
    airplane.addMasses(point_masses)
    return airplane
