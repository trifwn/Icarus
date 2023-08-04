from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.types import DataDict
from ICARUS.Core.types import FloatOrListArray
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing import define_linear_chord
from ICARUS.Vehicle.wing import define_linear_span
from ICARUS.Vehicle.wing import Wing


def wing_var_chord_offset(
    airfoils: DataDict,
    name: str,
    chords: FloatOrListArray,
    offset: float,
) -> Airplane:
    """Defines an airplane consisting only of a wing with a variable chord and offset.
    The rest of the parameters are the same with the hermes Plane;

    Args:
        airfoils (DataDict): _description_
        name (str): _description_
        chords (NumericArray): _description_
        offset (float): _description_

    Returns:
        Airplane: _description_
    """
    origin: ndarray[Any, dtype[floating[Any]]] = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: ndarray[Any, dtype[floating[Any]]] = np.array(
        [0.0 - 0.159 / 4, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: ndarray[Any, dtype[floating[Any]]] = np.array(
        [2.8, 0.0, 0.0],
        dtype=float,
    )

    main_wing = Wing(
        name="wing",
        airfoil=airfoils["NACA4415"],
        origin=origin + wing_position,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 1.130,
        sweep_offset=offset,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=chords,  # [0.159, 0.072],
        span_fun=define_linear_span,
        N=25,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()
    lifting_surfaces: list[Wing] = [main_wing]
    # main_wing.plotWing()

    addedMasses: list[tuple[float, np.ndarray[Any, np.dtype[Any]]]] = [
        (0.500, np.array([-0.40, 0.0, 0.0], dtype=float)),  # Motor
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float)),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float)),  # Payload
    ]
    airplane = Airplane(name, lifting_surfaces)
    airplane.add_point_masses(addedMasses)
    return airplane
