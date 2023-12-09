import re
from typing import Any

import numpy as np

from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_2d
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import EXTERNAL_DB
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing_segment import define_linear_chord
from ICARUS.Vehicle.wing_segment import define_linear_span
from ICARUS.Vehicle.wing_segment import Wing_Segment


def get_bmark_plane(name: str):
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

    Simplewing = Wing_Segment(
        name=name,
        airfoil=airfoils["NACA0015"],
        origin=origin + wing_position,
        orientation=wing_orientation,
        is_symmetric=True,
        span=2 * 2.5,
        sweep_offset=0.0,
        dih_angle=0,
        chord_fun=define_linear_chord,
        chord=np.array([0.8, 0.8]),
        span_fun=define_linear_span,
        N=10,
        M=5,
        mass=1,
    )
    airplane = Plane(Simplewing.name, [Simplewing])
    # print(airplane.CG)
    airplane.CG = np.array([0.337, 0, 0])

    return airplane
