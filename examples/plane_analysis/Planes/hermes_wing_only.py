import numpy as np

from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.database import DB
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.surface import WingSurface
from ICARUS.vehicle.utils import SymmetryAxes
from ICARUS.vehicle.wing_segment import WingSegment


def hermes_main_wing(name: str) -> Airplane:
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

    main_wing = WingSegment(
        name="wing",
        root_airfoil=DB.get_airfoil("NACA4415"),
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 1.130,
        sweep_offset=0,
        root_chord=0.159,
        tip_chord=0.072,
        N=20,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()

    lifting_surfaces: list[WingSurface] = [main_wing]
    airplane = Airplane(name, lifting_surfaces)

    # airplane.visAirplane()

    # point_masses = [
    #     (0.500 , np.array([-0.40, 0.0, 0.0], dtype = float)), # Motor
    #     (1.000 , np.array([0.090, 0.0, 0.0], dtype = float)), # Battery
    #     (0.900 , np.array([0.130, 0.0, 0.0], dtype = float)), # Payload
    #     ]
    # airplane.addMasses(point_masses)
    return airplane
