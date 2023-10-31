import time

import numpy as np

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Workers.solver import Solver


def gnvp7_run(mode: str = "Parallel") -> None:
    print("Testing GNVP Running...")

    # Get Plane, DB
    from examples.Vehicles.Planes.simple_wing import airplane, db

    # Get Environment
    from ICARUS.Environment.definition import EARTH_ISA

    # Get Solver
    from ICARUS.Solvers.Airplane.gnvp7 import get_gnvp7

    gnvp7: Solver = get_gnvp7(db)

    # Set Analysis
    if mode == "Parallel":
        analysis: str = gnvp7.available_analyses_names()[2]
    else:
        analysis = gnvp7.available_analyses_names()[1]

    gnvp7.set_analyses(analysis)

    # Set Options
    options: Struct = gnvp7.get_analysis_options(verbose=True)
    solver_parameters: Struct = gnvp7.get_solver_parameters()

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles_all: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)
    angles: list[float] = [ang for ang in angles_all if ang != 0]
    u_freestream = 20
    maxiter = 20
    timestep = 10

    airplane.define_dynamic_pressure(u_freestream, EARTH_ISA.air_density)

    options.plane.value = airplane
    options.environment.value = EARTH_ISA
    options.db.value = db
    options.solver2D.value = "XFLR"
    options.maxiter.value = maxiter
    options.timestep.value = timestep
    options.u_freestream.value = u_freestream
    options.angles.value = angles

    solver_parameters.Split_Symmetric_Bodies.value = False
    solver_parameters.Use_Grid.value = True

    _ = gnvp7.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    gnvp7.run()

    end_time: float = time.perf_counter()
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = gnvp7.get_results()
    airplane.save()
    # getRunOptions()
