import time

import numpy as np

from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def avl_run() -> None:
    print("Testing AVL Running ...")
    # Get Plane, DB
    from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane

    bmark: Airplane = get_bmark_plane("bmark")
    # Get Environment
    from ICARUS.Environment.definition import EARTH_ISA

    # Set State
    u_freestream = 20.0
    state = State("Unstick", bmark, EARTH_ISA, u_freestream)

    # Get Solver
    from ICARUS.Computation.Solvers.AVL.avl import get_avl

    avl: Solver = get_avl()
    analysis: str = avl.available_analyses_names()[0]

    avl.set_analyses(analysis)

    # Set Options
    options: Struct = avl.get_analysis_options(verbose=True)

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles_all: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)
    angles: list[float] = [ang for ang in angles_all]

    options.plane.value = bmark
    options.state.value = state
    options.solver2D.value = "XFLR"
    options.angles.value = angles

    _ = avl.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    avl.run()

    end_time: float = time.perf_counter()
    print(f"AVL Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing AVL Running... Done")

    _ = avl.get_results()
    bmark.save()
