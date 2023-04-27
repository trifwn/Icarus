from ICARUS.Vehicle.wing import Wing as wg
import ICARUS.Vehicle.wing as wing
from ICARUS.Vehicle.plane import Airplane as Plane

from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Software.XFLR5.polars import readPolars2D
from ICARUS.Database import DB3D,XFLRDB

import numpy as np
import os

dbMASTER = DB()
dbMASTER.loadData()
db = dbMASTER.foilsDB
readPolars2D(db, XFLRDB)
airfoils = db.getAirfoils()

Origin = np.array([0., 0., 0.])
wingPos = np.array([-0.2, 0.0, 0.0])
wingOrientation = np.array([0.0, 0.0, 0.0])

Simplewing = wg(name="bmark",
                airfoil=airfoils['NACA0015'],
                Origin=Origin + wingPos,
                Orientation=wingOrientation,
                isSymmetric=True,
                span=2 * 2.5,
                sweepOffset=0.,
                dihAngle=0,
                chordFun=wing.linearChord,
                chord=[0.8, 0.8],
                spanFun=wing.linSpan,
                N=20,
                M=5,
                mass=1)
ap = Plane(Simplewing.name, [Simplewing])
ap.CG = [0.337, 0, 0]
# ap.accessDB(HOMEDIR, DB3D)
