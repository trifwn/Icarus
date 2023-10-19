import time
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct
from ICARUS.Workers.solver import Solver


def lspt_run() -> None:

    print("Testing GNVP Running...")

    # Get Plane, DB
    from examples.Planes.simple_wing import airplane, db

    # Get Environment
    from ICARUS.Enviroment.definition import EARTH_ISA

    # Get Solver
    from ICARUS.Solvers.Airplane.lspt import get_lspt

    lspt: Solver = get_lspt(db)

    # Set Analysis
    analysis: str = lspt.available_analyses_names()[0]


    lspt.set_analyses(analysis)

    # Set Options
    options: Struct = lspt.get_analysis_options(verbose=True)
    solver_parameters: Struct = lspt.get_solver_parameters()

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(AoAmin, AoAmax, NoAoA)
    u_freestream = 20

    airplane.define_dynamic_pressure(u_freestream, EARTH_ISA.air_density)

    options.plane.value = airplane
    options.environment.value = EARTH_ISA
    options.db.value = db
    options.solver2D.value = "Xfoil"
    options.u_freestream.value = u_freestream
    options.angles.value = angles

    solver_parameters.Ground_Effect.value = True
    solver_parameters.Wake_Geom_Type.value = "TE-Geometrical"

    _ = lspt.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    lspt.run()

    end_time: float = time.perf_counter()
    print(f"LSPT Run took: --- %s seconds ---" % (end_time - start_time))
    print("Testing GNVP Running... Done")

    _ = lspt.get_results()
    airplane.save()

