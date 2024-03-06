import time

import numpy as np

from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def gnvp3_run(run_parallel: bool = True) -> None:
    print("Testing GNVP3 Running...")

    # Get Plane, DB
    from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane

    bmark, state = get_bmark_plane("bmark")

    # Get Solver
    from ICARUS.Computation.Solvers.GenuVP.gnvp3 import GenuVP3

    gnvp3: Solver = GenuVP3()

    # Set Analysis
    polar_analysis: str = gnvp3.get_analyses_names()[0]
    gnvp3.select_analysis(polar_analysis)

    # Set Options
    options: Struct = gnvp3.get_analysis_options(verbose=True)
    solver_parameters: Struct = gnvp3.get_solver_parameters()

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles_all: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)
    angles: list[float] = [ang for ang in angles_all]
    maxiter = 30
    timestep = 0.004

    options.plane = bmark
    options.state = state
    options.solver2D = "XFLR"
    options.maxiter = maxiter
    options.timestep = timestep
    options.angles = angles

    solver_parameters.Split_Symmetric_Bodies = False
    solver_parameters.Use_Grid = True

    # Deformation
    solver_parameters.Bound_Vorticity_Cutoff = 0.003  # EPSFB
    solver_parameters.Wake_Vorticity_Cutoff = 0.003  # EPSFW
    solver_parameters.Vortex_Cutoff_Length_f = 1e-1  # EPSVR
    solver_parameters.Vortex_Cutoff_Length_i = 1e-1  # EPSO

    gnvp3.define_analysis(options, solver_parameters)
    _ = gnvp3.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    gnvp3.execute(parallel=run_parallel)

    end_time: float = time.perf_counter()
    if run_parallel:
        mode = "Parallel"
    else:
        mode = "Serial"
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = gnvp3.get_results()
    bmark.save()


if __name__ == "__main__":
    gnvp3_run()