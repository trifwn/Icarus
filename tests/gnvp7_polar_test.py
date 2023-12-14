import time

import numpy as np

from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.plane import Airplane


def gnvp7_run(mode: str = "Parallel") -> None:
    print("Testing GNVP7 Running...")

    # Get Plane, DB
    from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane

    airplane: Airplane = get_bmark_plane("bmark")

    # Get Environment
    from ICARUS.Environment.definition import EARTH_ISA

    # Get Solver
    from ICARUS.Computation.Solvers.GenuVP.gnvp7 import get_gnvp7

    gnvp7: Solver = get_gnvp7()

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
    angles: list[float] = [ang for ang in angles_all]
    u_freestream = 20
    maxiter = 30
    timestep = 0.004

    options.plane.value = airplane
    options.environment.value = EARTH_ISA
    options.solver2D.value = "XFLR"
    options.maxiter.value = maxiter
    options.timestep.value = timestep
    options.u_freestream.value = u_freestream
    options.angles.value = angles

    solver_parameters.Split_Symmetric_Bodies.value = False
    solver_parameters.Use_Grid.value = True

    # DEFORMATION
    solver_parameters.Bound_Vorticity_Cutoff.value = 0.003  # EPSFB
    solver_parameters.Wake_Vorticity_Cutoff.value = 0.003  # EPSFW
    solver_parameters.Vortex_Cutoff_Length_f.value = 1e-1  # EPSVR
    solver_parameters.Vortex_Cutoff_Length_i.value = 1e-1  # EPSO
    _ = gnvp7.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    gnvp7.run()

    end_time: float = time.perf_counter()
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = gnvp7.get_results()
    airplane.save()
    # getRunOptions()
