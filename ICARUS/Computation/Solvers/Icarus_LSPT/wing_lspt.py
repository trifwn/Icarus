from typing import Any

from ICARUS.Aerodynamics.Potential.lifting_surfaces import run_lstp_angles
from ICARUS.Computation.Analyses.analysis import Analysis
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def get_lspt() -> Solver:
    """
    Returns a Solver object for the ICARUS lifting surface potential theory
    solver.


    Returns:
        Solver: Solver object
    """
    lspt = Solver(name="lspt", solver_type="3D", fidelity=1)

    options: dict[str, Any] = {
        "plane": (
            "Plane Object",
            Airplane,
        ),
        "state": (
            "State Object",
            State,
        ),
        "solver2D": (
            "2D Solver",
            str,
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
        unhook=None,
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
    from ICARUS.Database import DB
    import os

    HOMEDIR = os.getcwd()
    lspt = get_lspt()
    analysis = lspt.available_analyses_names()[0]
    lspt.set_analyses(analysis)
    options = lspt.get_analysis_options()

    plane = list(DB.vehicles_db.planes.items())[0][1]
    CASEDIR = plane.CASEDIR + "/" + angle_to_case(0.0) + "/"
    options["HOMEDIR"].value = HOMEDIR
    options["CASEDIR"].value = CASEDIR
    # gnvp3.run()
