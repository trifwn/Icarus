import numpy as np

from ICARUS.airfoils import NACA4
from ICARUS.core.types import FloatArray
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import WingSegment


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
        root_airfoil=NACA4(M=4, P=4, XX=15),  # "NACA4415",
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
    airplane = Airplane(name, main_wing=main_wing)
    return airplane
