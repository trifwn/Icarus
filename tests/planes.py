from ICARUS.Vehicle.wing import Wing as wg
from ICARUS.Vehicle.wing import linearChord, linSpan
from ICARUS.Vehicle.plane import Airplane as Plane

from ICARUS.Database.db import DB
from ICARUS.Software.XFLR5.polars import readPolars2D
from ICARUS.Database import XFLRDB

import numpy as np

db = DB()
db.loadData()
db2d = db.foilsDB
readPolars2D(db2d, XFLRDB)
airfoils = db2d.getAirfoils()

Origin = np.array([0., 0., 0.])
wingPos = np.array([-0.2, 0.0, 0.0])
wingOrientation = np.array([0.0, 0.0, 0.0])

Simplewing = wg(name="bmark",
                airfoil=airfoils['NACA0015'],
                Origin=Origin + wingPos,
                Orientation=wingOrientation,
                isSymmetric=True,
                span= 2 * 2.5,
                sweepOffset= 0.,
                dihAngle= 0,
                chordFun= linearChord,
                chord= [0.8, 0.8],
                spanFun= linSpan,
                N= 20,
                M= 5,
                mass= 1
            )
ap = Plane(Simplewing.name, [Simplewing])
ap.CG = [0.337, 0, 0]