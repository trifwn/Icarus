import time

import numpy as np

from ICARUS.computation.solvers.solver import Solver
from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray


def gnvp7_run(run_parallel: bool = True) -> None:
    print("Testing GNVP7 Running...")

    # Get Plane, DB
    from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane

    airplane, state = get_bmark_plane("bmark")

    # Get Solver
    from ICARUS.computation.solvers.GenuVP.gnvp7 import GenuVP7

    gnvp7: Solver = GenuVP7()

    # Set Analysis
    analysis: str = gnvp7.get_analyses_names()[0]
    gnvp7.select_analysis(analysis)

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

    options.plane = airplane
    options.state = state
    options.solver2D = "Xfoil"
    options.maxiter = maxiter
    options.timestep = timestep
    options.angles = angles

    solver_parameters.Split_Symmetric_Bodies = False
    solver_parameters.Use_Grid = True

    # DEFORMATION
    solver_parameters.Bound_Vorticity_Cutoff = 0.003  # EPSFB
    solver_parameters.Wake_Vorticity_Cutoff = 0.003  # EPSFW
    solver_parameters.Vortex_Cutoff_Length_f = 1e-1  # EPSVR
    solver_parameters.Vortex_Cutoff_Length_i = 1e-1  # EPSO

    gnvp7.define_analysis(options, solver_parameters)
    _ = gnvp7.get_analysis_options(verbose=True)

    start_time: float = time.perf_counter()
    gnvp7.execute(parallel=run_parallel)

    end_time: float = time.perf_counter()
    mode = "Parallel" if run_parallel else "Serial"
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = gnvp7.get_results()
    airplane.save()
    # getRunOptions()
