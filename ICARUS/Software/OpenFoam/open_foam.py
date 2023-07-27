from typing import Any

from ICARUS.Database.db import DB
from ICARUS.Software.OpenFoam.analyses.angles import angles_parallel
from ICARUS.Software.OpenFoam.analyses.angles import angles_serial
from ICARUS.Software.OpenFoam.filesOpenFoam import MeshType
from ICARUS.Workers.analysis import Analysis
from ICARUS.Workers.solver import Solver


def get_open_foam(db: DB) -> Solver:
    open_foam = Solver(name="open_foam", solver_type="CFD", fidelity=3, db=db)

    options: dict[str, Any] = {
        "db": "Database",
        "airfoil": "Airfoil to run",
        "reynolds": "List of Reynolds numbers to run",
        "mach": "Mach number",
        "angles": "angles to run",
    }

    solver_options: dict[str, tuple[Any, str]] = {
        "mesh_type": (
            MeshType.structAirfoilMesher,
            "Type of mesh to use",
        ),
        "max_iterations": (
            400,
            "Maximum number of iterations",
        ),
        "silent": (
            1e-3,
            "Whether to print progress or not",
        ),
    }

    aoa_simple_struct_serial: Analysis = Analysis(
        solver_name="open_foam",
        analysis_name="SIMPLE with structured Mesh sequentially",
        run_function=angles_serial,
        options=options,
        solver_options=solver_options,
        unhook=None,
    )

    aoa_simple_struct_parallel: Analysis = Analysis(
        solver_name="open_foam",
        analysis_name="Simple with structured Mesh in parallel",
        run_function=angles_parallel,
        options=options,
        solver_options=solver_options,
        unhook=None,
    )

    open_foam.add_analyses(
        [
            aoa_simple_struct_parallel,
            aoa_simple_struct_serial,
        ],
    )

    return open_foam
