import time
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct
from ICARUS.Workers.solver import Solver


def gnvprun(mode: str = "Parallel") -> None:
    print("Testing GNVP Running...")

    # Get Plane, DB
    from Data.Planes.simple_wing import airplane, db

    # Get Environment
    from ICARUS.Enviroment.definition import EARTH

    # Get Solver
    from ICARUS.Software.GenuVP3.gnvp3 import get_gnvp3

    gnvp3: Solver = get_gnvp3(db)

    # Set Analysis
    if mode == "Parallel":
        analysis: str = gnvp3.available_analyses_names()[2]
    else:
        analysis = gnvp3.available_analyses_names()[1]

    gnvp3.set_analyses(analysis)

    # Set Options
    options: Struct = gnvp3.get_analysis_options(verbose=True)
    solver_parameters: Struct = gnvp3.get_solver_parameters()

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles_all: ndarray[Any, dtype[floating[Any]]] = np.linspace(AoAmin, AoAmax, NoAoA)
    angles: list[float] = [ang for ang in angles_all if ang != 0]
    u_freestream = 20
    maxiter = 10
    timestep = 10

    airplane.define_dynamic_pressure(u_freestream, EARTH.air_density)

    options.plane.value = airplane
    options.environment.value = EARTH
    options.db.value = db
    options.solver2D.value = "XFLR"
    options.maxiter.value = maxiter
    options.timestep.value = timestep
    options.u_freestream.value = u_freestream
    options.angles.value = angles

    solver_parameters.Split_Symmetric_Bodies.value = True
    solver_parameters.Use_Grid.value = True

    _ = gnvp3.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    gnvp3.run()

    end_time: float = time.perf_counter()
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = gnvp3.get_results()
    airplane.save()
    # getRunOptions()
