import time

import numpy as np

from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def gnvp3_run(mode: str = "Parallel") -> None:
    print("Testing GNVP3 Running...")

    # Get Plane, DB
    from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane

    bmark: Airplane = get_bmark_plane("bmark")
    # Get Environment
    from ICARUS.Environment.definition import EARTH_ISA

    # Set State
    u_freestream = 20.0
    state = State("Unstick", bmark, EARTH_ISA, u_freestream)

    # Get Solver
    from ICARUS.Computation.Solvers.GenuVP.gnvp3 import get_gnvp3

    gnvp3: Solver = get_gnvp3()

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
    angles_all: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)
    angles: list[float] = [ang for ang in angles_all]
    maxiter = 30
    timestep = 0.004

    options.plane.value = bmark
    options.state.value = state
    options.solver2D.value = "XFLR"
    options.maxiter.value = maxiter
    options.timestep.value = timestep
    options.angles.value = angles

    solver_parameters.Split_Symmetric_Bodies.value = False
    solver_parameters.Use_Grid.value = True

    # Deformation
    solver_parameters.Bound_Vorticity_Cutoff.value = 0.003  # EPSFB
    solver_parameters.Wake_Vorticity_Cutoff.value = 0.003  # EPSFW
    solver_parameters.Vortex_Cutoff_Length_f.value = 1e-1  # EPSVR
    solver_parameters.Vortex_Cutoff_Length_i.value = 1e-1  # EPSO

    _ = gnvp3.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    gnvp3.run()

    end_time: float = time.perf_counter()
    print(f"GNVP {mode} Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = gnvp3.get_results()
    bmark.save()
