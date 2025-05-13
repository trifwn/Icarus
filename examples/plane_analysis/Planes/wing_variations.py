import numpy as np

from ICARUS.core.types import DataDict
from ICARUS.core.types import FloatArray
from ICARUS.vehicle.airplane import Airplane
from ICARUS.vehicle.surface import WingSurface
from ICARUS.vehicle.utils import SymmetryAxes
from ICARUS.vehicle.wing_segment import WingSegment


def wing_var_chord_offset(
    airfoils: DataDict,
    name: str,
    root_chord: float,
    tip_chord: float,
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
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0.0 - 0.159 / 4, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [2.8, 0.0, 0.0],
        dtype=float,
    )

    main_wing = WingSegment(
        name="wing",
        root_airfoil=airfoils["NACA4415"],
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 1.130,
        sweep_offset=offset,
        root_chord=root_chord,
        tip_chord=tip_chord,
        N=25,
        M=5,
        mass=0.670,
    )
    # main_wing.plotWing()
    lifting_surfaces: list[WingSurface] = [main_wing]
    # main_wing.plotWing()

    added_masses: list[tuple[float, FloatArray, str]] = [
        (0.500, np.array([-0.40, 0.0, 0.0], dtype=float), "motor"),  # Motor
        (1.000, np.array([0.090, 0.0, 0.0], dtype=float), "battery"),  # Battery
        (0.900, np.array([0.130, 0.0, 0.0], dtype=float), "payload"),  # Payload
    ]
    airplane = Airplane(name, lifting_surfaces)
    airplane.add_point_masses(added_masses)
    return airplane
