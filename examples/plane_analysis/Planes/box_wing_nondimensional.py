import os

import numpy as np

from ICARUS.computation.solvers.XFLR5.polars import read_polars_2d
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import DiscretizationType
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import WingSegment


def get_box_wing(name: str, AR: float = 9, airfoil: str = "NACA0012") -> Airplane:
    from ICARUS import INSTALL_DIR

    database_folder = os.path.join(INSTALL_DIR, "Data")
    DB = Database(database_folder)

    read_polars_2d(os.path.join(DB.EXTERNAL_DB, "2D"))
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    box_wing = WingSegment(
        name=name,
        root_airfoil=DB.get_airfoil(airfoil),
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 0.5,
        sweep_offset=0.0,
        root_chord=1.0 / AR,
        tip_chord=1.0 / AR,
        N=15,
        M=8,
        mass=1,
        span_spacing=DiscretizationType.LINEAR,
        chord_spacing=DiscretizationType.LINEAR,
    )
    airplane = Airplane(box_wing.name, main_wing=box_wing)
    return airplane


if __name__ == "__main__":
    airplane = get_box_wing("benchmark_plane")
    # airplane.plot()
    print(airplane.span)
    print(airplane.aspect_ratio)
