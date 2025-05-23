import time

import numpy as np

from ICARUS.computation.solvers import Solver
from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray


def lspt_run() -> None:
    print("Testing LSPT Running...")

    # Get Plane, DB
    from .benchmark_plane_test import get_bmark_plane

    airplane, state = get_bmark_plane("bmark")

    # Get Solver
    from ICARUS.computation.solvers.Icarus_LSPT import LSPT

    lspt: Solver = LSPT()

    # Set Analysis
    analysis: str = lspt.get_analyses_names()[0]

    lspt.select_analysis(analysis)

    # Set Options
    options: Struct = lspt.get_analysis_options(verbose=True)
    solver_parameters: Struct = lspt.get_solver_parameters()

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)

    options.plane = airplane
    options.state = state
    options.solver2D = "Xfoil"
    options.angles = angles

    solver_parameters.Ground_Effect = True
    solver_parameters.Wake_Geom_Type = "TE-Geometrical"

    lspt.define_analysis(options, solver_parameters)
    _ = lspt.get_analysis_options(verbose=True)

    start_time: float = time.perf_counter()
    lspt.execute()

    end_time: float = time.perf_counter()
    print("LSPT Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = lspt.get_results()
    airplane.save()
