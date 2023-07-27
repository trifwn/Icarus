from typing import Any

from ICARUS.Database.db import DB
from ICARUS.Software.Xfoil.analyses.angles import multiple_reynolds_parallel
from ICARUS.Software.Xfoil.analyses.angles import multiple_reynolds_serial
from ICARUS.Workers.analysis import Analysis
from ICARUS.Workers.solver import Solver


def get_xfoil(db: DB) -> Solver:
    xfoil = Solver(name="xfoil", solver_type="2D-IBLM", fidelity=2, db=db)

    options: dict[str, Any] = {
        "db": "Database to save results",
        "airfoil": "Airfoil to run",
        "reynolds": "List of Reynolds numbers to run",
        "mach": "Mach number",
        "min_aoa": "Minimum angle of attack",
        "max_aoa": "Maximum angle of attack",
        "aoa_step": "Step between each angle of attack",
    }

    solver_options: dict[str, tuple[Any, str]] = {
        "max_iter": (
            400,
            "Maximum number of iterations",
        ),
        "Ncrit": (
            1e-3,
            "Timestep between each iteration",
        ),
        "xtr": (
            (0.1, 0.1),
            "Transition points: Lower and upper",
        ),
        "print": (
            False,
            "Print xfoil output",
        ),
    }

    aseq_multiple_reynolds_serial: Analysis = Analysis(
        solver_name="xfoil",
        analysis_name="Aseq for Multiple Reynolds Sequentially",
        run_function=multiple_reynolds_serial,
        options=options,
        solver_options=solver_options,
        unhook=None,
    )

    aseq_multiple_reynolds_parallel: Analysis = aseq_multiple_reynolds_serial << {
        "name": "Aseq for Multiple Reynolds in Parallel",
        "execute": multiple_reynolds_parallel,
        "unhook": None,
    }

    xfoil.add_analyses(
        [
            aseq_multiple_reynolds_parallel,
            aseq_multiple_reynolds_serial,
        ],
    )

    return xfoil
