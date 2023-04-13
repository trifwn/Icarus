import numpy as np

from ICARUS.Software.GenuVP3 import angles as gnvp3
from ICARUS.Software.GenuVP3.checkRuns import checkRuns

from ICARUS.Database import BASEGNVP3
import time


def gnvprun(mode='Parallel'):
    print("Testing GNVP Running...")

    from tests.planes import ap, db, HOMEDIR

    polars2D = db.Data
    AoAmin = -6
    AoAmax = 10
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
    genuBatchArgs = [ap, polars2D, "XFLR",
                     maxiter, timestep, Uinf, angles]
    start_time = time.perf_counter()
    if mode == 'Parallel':
        ap.runSolver(gnvp3.runGNVPanglesParallel, genuBatchArgs)
    elif mode == 'Serial':
        ap.runSolver(gnvp3.runGNVPangles, genuBatchArgs)
    end_time = time.perf_counter()

    print(f"GNVP {mode} Run took: --- %s seconds ---" %
          (end_time - start_time))

    genuPolarArgs = [ap.CASEDIR, HOMEDIR]
    ap.defineSim(Uinf, 1.225)
    ap.makePolars(gnvp3.makePolar, genuPolarArgs)
    ap.save()
    checkRuns(ap.CASEDIR, angles)
