import numpy as np
from typing import Any
from numpy import ndarray, floating, dtype

from ICARUS.Database import XFLRDB
from ICARUS.Database.db import DB
from ICARUS.Core.struct import Struct
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Software.XFLR5.polars import readPolars2D
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import define_linear_chord
from ICARUS.Vehicle.wing import define_linear_span
from ICARUS.Vehicle.wing import Wing

db = DB()
db.loadData()
db2d: Database_2D = db.foilsDB
readPolars2D(db2d, XFLRDB)
airfoils: Struct = db2d.getAirfoils()

origin: ndarray[Any, dtype[floating]] = np.array([0.0, 0.0, 0.0], dtype=float)
wing_position: ndarray[Any, dtype[floating]] = np.array([-0.2, 0.0, 0.0], dtype=float)
wing_orientation: ndarray[Any, dtype[floating]] = np.array([0.0, 0.0, 0.0], dtype=float)

Simplewing = Wing(
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
airplane.CG = [0.337, 0, 0]
