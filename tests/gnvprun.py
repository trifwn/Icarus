import numpy as np

from ICARUS.Software.GenuVP3 import runGNVP as gnvp3
from ICARUS.Software.GenuVP3.checkRuns import checkRuns

from ICARUS.Database import BASEGNVP3


def gnvprun():
    print("Testing GNVP Running...")

    from tests.planes import ap, db, HOMEDIR

    polars2D = db.Data
    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles = np.linspace(AoAmin, AoAmax, NoAoA)
    ang = []
    for a in angles:
        if a != 0:
            ang.append(a)
    angles = ang
    Uinf = 20
    maxiter = 20
    timestep = 10
    genuBatchArgs = [ap, BASEGNVP3, polars2D, "XFLR",
                     maxiter, timestep, Uinf, angles]
    ap.runSolver(gnvp3.runGNVPangles, genuBatchArgs)
    genuPolarArgs = [ap.CASEDIR, HOMEDIR]
    ap.defineSim(Uinf, 1.225)
    ap.makePolars(gnvp3.makePolar, genuPolarArgs)
    ap.save()
    checkRuns(ap.CASEDIR, angles)
