from typing import Any

import numpy as np

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import XFLRDB
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Input_Output.XFLR5.polars import read_polars_2d
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing_segment import define_linear_chord
from ICARUS.Vehicle.wing_segment import define_linear_span
from ICARUS.Vehicle.wing_segment import Wing_Segment

db: DB = DB()
db.load_data()
db2d: Database_2D = db.foilsDB
read_polars_2d(db2d, XFLRDB)
airfoils: Struct = db2d.set_available_airfoils()

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
    name="bmark",
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
    N=20,
    M=5,
    mass=1,
)
airplane = Plane(Simplewing.name, [Simplewing])
# print(airplane.CG)
airplane.CG = np.array([0.337, 0, 0])
