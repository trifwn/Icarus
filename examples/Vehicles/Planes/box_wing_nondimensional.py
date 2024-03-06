import numpy as np

from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_2d
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import EXTERNAL_DB
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import DiscretizationType
from ICARUS.Vehicle.utils import SymmetryAxes
from ICARUS.Vehicle.wing_segment import Wing_Segment


def get_box_wing(name: str, AR: float = 9, naca: str = "0012") -> Airplane:
    read_polars_2d(EXTERNAL_DB)
    airfoils: Struct = DB.foils_db.set_available_airfoils()

    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    box_wing = Wing_Segment(
        name=name,
        root_airfoil=airfoils[f"NACA{naca}"],
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
        span_spacing=DiscretizationType.EQUAL,
        chord_spacing=DiscretizationType.EQUAL,
    )
    airplane = Airplane(box_wing.name, [box_wing])
    # print(airplane.CG)
    # airplane.CG = np.array([0.337, 0, 0])

    return airplane


if __name__ == "__main__":
    airplane = get_box_wing("benchmark_plane")
    # airplane.visualize()
    print(airplane.span)
    print(airplane.aspect_ratio)
