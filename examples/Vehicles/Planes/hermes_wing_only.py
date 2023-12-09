import numpy as np

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import define_linear_chord
from ICARUS.Vehicle.utils import define_linear_span


def hermes_main_wing(airfoils: Struct, name: str) -> Airplane:
    """Function to get a plane Consisting only of the main wing of the hermes plane

    Args:
        airfoils (Struct): Data of all airfoils
        name (str): Name assinged to plane
    Returns:
        Airplane: Plane consisting of hermes V main wing
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
        N=20,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()

    lifting_surfaces: list[Lifting_Surface] = [main_wing]
    airplane = Airplane(name, lifting_surfaces)

    # airplane.visAirplane()

    # point_masses = [
    #     (0.500 , np.array([-0.40, 0.0, 0.0], dtype = float)), # Motor
    #     (1.000 , np.array([0.090, 0.0, 0.0], dtype = float)), # Battery
    #     (0.900 , np.array([0.130, 0.0, 0.0], dtype = float)), # Payload
    #     ]
    # airplane.addMasses(point_masses)
    return airplane
