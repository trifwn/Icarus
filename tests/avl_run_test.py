import time

import numpy as np

from ICARUS.computation.solvers.solver import Solver
from ICARUS.core.struct import Struct


def avl_run() -> None:
    print("Testing AVL Running ...")
    # Get Plane, DB
    from .benchmark_plane_test import get_bmark_plane

    bmark, state = get_bmark_plane("bmark")

    # Get Solver
    from ICARUS.computation.solvers.AVL.avl import AVL

    avl: Solver = AVL()
    analysis: str = avl.get_analyses_names()[0]

    avl.select_analysis(analysis)

    # Set Options
    options: Struct = avl.get_analysis_options(verbose=True)
    solver_parameters: Struct = avl.get_solver_parameters(verbose=True)

    AoAmin = -10
    AoAmax = 10
    NoAoA = (AoAmax - AoAmin) + 1
    angles = np.linspace(AoAmin, AoAmax, NoAoA)

    options.plane = bmark
    options.state = state
    options.solver2D = "Xfoil"
    options.angles = angles

    solver_parameters.use_avl_control = False

    avl.define_analysis(options=options, solver_parameters=solver_parameters)
    _ = avl.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    avl.execute()

    end_time: float = time.perf_counter()
    print("AVL Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing AVL Running... Done")

    _ = avl.get_results()
    bmark.save()
