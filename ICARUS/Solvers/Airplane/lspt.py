from typing import Any
from ICARUS.Aerodynamics.Potential.lifting_line import run_lstp_angles

from ICARUS.Database.db import DB
from ICARUS.Enviroment.definition import Environment
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Input_Output.GenuVP.analyses.angles import process_gnvp_angles_run, run_gnvp3_angles
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Workers.analysis import Analysis
from ICARUS.Workers.solver import Solver


def get_lspt(db: DB) -> Solver:
    """
    Returns a Solver object for the ICARUS lifting surface potential theory
    solver.

    Args:
        db (DB): Database

    Returns:
        Solver: Solver object
    """
    lspt = Solver(name="lspt", solver_type="3D", fidelity=1, db=db)

    options: dict[str, Any] = {
        "plane": (
            "Plane Object",
            Airplane,
        ),
        "environment": (
            "Environment",
            Environment,
        ),
        "db": (
            "Database",
            DB,
        ),
        "solver2D": (
            "2D Solver",
            str,
        ),
        "u_freestream": (
            "Velocity Magnitude",
            float,
        ),
        "angles": (
            "Angles to run",
            list[float],
        ),
    }

    solver_options: dict[str, tuple[Any, str, Any]] = {
        "Ground_Effect": (
            None,
            "Distance From Ground (m). None for no ground effect",
            int,
        ),
        "Wake_Geom_Type": (
            "TE-Geometrical",
            "Type of wake geometry. The options are: -TE-Geometrical -Inflow-Uniform -Inflow-TE",
            str,
        ),
    }

    angles: Analysis = Analysis(
        "lspt",
        "Angles_Serial",
        run_lstp_angles,
        options,
        solver_options,
        unhook= None,
    )

    lspt.add_analyses(
        [
            angles,
            # pertrubations,
        ],
    )

    return lspt


# # EXAMPLE USAGE
if __name__ == "__main__":
    from ICARUS.Database.utils import angle_to_case
    import os

    HOMEDIR = os.getcwd()
    db = DB()
    db.load_data()
    lspt = get_lspt(db)
    analysis = lspt.available_analyses_names()[0]
    lspt.set_analyses(analysis)
    options = lspt.get_analysis_options()

    plane = list(db.vehiclesDB.planes.items())[0][1]
    CASEDIR = plane.CASEDIR + "/" + angle_to_case(0.0) + "/"
    options["HOMEDIR"].value = HOMEDIR
    options["CASEDIR"].value = CASEDIR
    # gnvp3.run()
