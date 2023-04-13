import numpy as np

from ICARUS.Software.GenuVP3.angles import runGNVPangles, runGNVPanglesParallel
from ICARUS.Software.GenuVP3.filesInterface import makePolar
from ICARUS.Software.GenuVP3.checkRuns import checkRuns

import time


def gnvprun(mode='Parallel'):
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
    genuBatchArgs = [ap, polars2D, "XFLR",
                     maxiter, timestep, Uinf, angles]
    start_time = time.perf_counter()
    if mode == 'Parallel':
        ap.runAnalysis(runGNVPanglesParallel, genuBatchArgs)
    elif mode == 'Serial':
        ap.runAnalysis(runGNVPangles, genuBatchArgs)
    end_time = time.perf_counter()

    print(f"GNVP {mode} Run took: --- %s seconds ---" %
          (end_time - start_time))

    genuPolarArgs = [ap.CASEDIR, HOMEDIR]
    ap.defineSim(Uinf, 1.225)
    ap.setPolars(makePolar, genuPolarArgs)
    ap.save()
    checkRuns(ap.CASEDIR, angles)
