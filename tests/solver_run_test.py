import time

import numpy as np


def gnvprun(mode="Parallel"):
    print("Testing GNVP Running...")

    # Get Plane, DB
    from Data.Planes.simple_wing import ap, db

    # Get Environment
    from ICARUS.Enviroment.definition import EARTH

    # Get Solver
    from ICARUS.Software.GenuVP3.gnvp3 import get_gnvp3

    gnvp3 = get_gnvp3(db)

    ## Set Analysis
    if mode == "Parallel":
        # gnvp3.setParallel(True)
        analysis = gnvp3.getAvailableAnalyses()[2]
    else:
        analysis = gnvp3.getAvailableAnalyses()[1]

    gnvp3.setAnalysis(analysis)

    ## Set Options
    options = gnvp3.getOptions(analysis)

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles = np.linspace(AoAmin, AoAmax, NoAoA)
    angles = [ang for ang in angles if ang != 0]
    Uinf = 20
    maxiter = 20
    timestep = 10

    ap.defineSim(Uinf, EARTH.AirDensity)

    options.plane.value = ap
    options.environment.value = EARTH
    options.db.value = db
    options.solver2D.value = "XFLR"
    options.maxiter.value = maxiter
    options.timestep.value = timestep
    options.Uinf.value = Uinf
    options.angles.value = angles

    _ = gnvp3.getOptions(verbose=True)
    start_time = time.perf_counter()

    gnvp3.run()

    end_time = time.perf_counter()
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    polars = gnvp3.getResults()
    ap.save()
    # getRunOptions()
