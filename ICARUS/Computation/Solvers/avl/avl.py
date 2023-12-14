from ICARUS.Computation.Analyses.analysis import Analysis
from ICARUS.Computation.Solvers.AVL.analyses.pertrubations import avl_dynamic_analysis_fd
from ICARUS.Computation.Solvers.AVL.analyses.pertrubations import avl_dynamic_analysis_implicit
from ICARUS.Computation.Solvers.AVL.analyses.pertrubations import process_avl_fd_res
from ICARUS.Computation.Solvers.AVL.analyses.pertrubations import process_avl_impl_res
from ICARUS.Computation.Solvers.AVL.analyses.polars import avl_angle_run
from ICARUS.Computation.Solvers.AVL.analyses.polars import process_avl_angles_run
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def get_avl() -> Solver:
    avl = Solver(name="avl", solver_type="3D", fidelity=1)

    # # Define GNVP3 Analyses
    options = {
        "plane": (
            "Plane Object",
            Airplane,
        ),
        "state": (
            "State of the Ariplane",
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

    polar_run: Analysis = Analysis(
        "avl",
        "Polar Run",
        avl_angle_run,
        options,
        unhook=process_avl_angles_run,
    )

    # trim_analysis: Analysis = Analysis(
    #     "avl",
    #     "Trim Analysis",
    #     run_gnvp3_angles,
    #     options,
    #     unhook=process_gnvp_angles_run_3,
    # )

    options = {
        "plane": (
            "Plane Object",
            Airplane,
        ),
        "state": (
            "Dynamic State of the airplane",
            State,
        ),
        "solver2D": (
            "2D Solver",
            str,
        ),
    }

    dynamic_analysis_implicit: Analysis = Analysis(
        "avl",
        "Dynamic Analysis With Implicit Differentiation",
        avl_dynamic_analysis_implicit,
        options,
        unhook=process_avl_impl_res,
    )

    dynamic_analysis_fd: Analysis = dynamic_analysis_implicit << {
        "name": "Dynamic Analysis with Finite Differences",
        "execute": avl_dynamic_analysis_fd,
        "unhook": process_avl_fd_res,
    }

    avl.add_analyses(
        [
            polar_run,
            dynamic_analysis_implicit,
            dynamic_analysis_fd,
        ],
    )

    return avl


# # EXAMPLE USAGE
if __name__ == "__main__":
    from ICARUS.Database.utils import angle_to_case
    import os

    HOMEDIR = os.getcwd()
    avl = get_avl()
    analysis = avl.available_analyses_names()[0]
    avl.set_analyses(analysis)
    options = avl.get_analysis_options()
    # etc...
