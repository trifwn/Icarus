import numpy as np

from ICARUS.Database import XFLRDB
from ICARUS.Database.db import DB
from ICARUS.Software.XFLR5.polars import readPolars2D
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg
from ICARUS.Vehicle.wing import define_linear_chord, define_linear_span

db = DB()
db.loadData()
db2d = db.foilsDB
readPolars2D(db2d, XFLRDB)
airfoils = db2d.getAirfoils()

origin = np.array([0.0, 0.0, 0.0])
wing_pos = np.array([-0.2, 0.0, 0.0])
wing_orientation = np.array([0.0, 0.0, 0.0])

Simplewing = wg(
    name="bmark",
    airfoil=airfoils["NACA0015"],
    origin=origin + wing_pos,
    orientation=wing_orientation,
    is_symmetric=True,
    span=2 * 2.5,
    sweep_offset=0.0,
    dih_angle=0,
    chord_fun=define_linear_chord,
    chord=[0.8, 0.8],
    span_fun=define_linear_span,
    N=20,
    M=5,
    mass=1,
)
airplane = Plane(Simplewing.name, [Simplewing])
airplane.CG = [0.337, 0, 0]
