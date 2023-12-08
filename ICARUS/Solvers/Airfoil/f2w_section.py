from typing import Any

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Input_Output.F2Wsection.analyses.angles import process_f2w_run
from ICARUS.Input_Output.F2Wsection.analyses.angles import run_multiple_reynolds_parallel
from ICARUS.Input_Output.F2Wsection.analyses.angles import run_multiple_reynolds_sequentially
from ICARUS.Input_Output.F2Wsection.analyses.angles import run_single_reynolds
from ICARUS.Workers.analysis import Analysis
from ICARUS.Workers.solver import Solver


def get_f2w_section() -> Solver:
    f2w_section = Solver(name="f2w_section", solver_type="2D-IBLM", fidelity=2)

    options: dict[str, tuple[str, Any]] = {
        "airfoil": (
            "Airfoil to run",
            Airfoil,
        ),
        "reynolds": (
            "List of Reynolds numbers to run",
            list[float],
        ),
        "mach": (
            "Mach number",
            float,
        ),
        "angles": (
            "All angles to run",
            list[float],
        ),
    }

    solver_options: dict[str, tuple[Any, str, Any]] = {
        "max_iter": (
            251,
            "NTIMEM | Maximum number of iterations",
            int,
        ),
        "timestep": (
            0.01,
            "DT1 | Simulation timestep",
            float,
        ),
        "f_trip_low": (
            0.1,
            "TRANSLO | Transition points for positive and negative angles for the lower surface",
            float,
        ),
        "f_trip_upper": (
            0.1,
            "TRANSUP | Transition points for positive and negative angles for the upper surface",
            float,
        ),
        "Ncrit": (
            9,
            "N critical value for transition according to e to N method",
            float,
        ),
        "boundary_layer_solve_time": (
            250,
            "NTIME_bl | When to start solving the boundary layer",
            int,
        ),
        "trailing_edge_angle": (0.0, "TEANGLE (deg) | Trailing edge angle", float),
        "u_freestrem": (1.0, "UINF | Freestream velocity", float),
        "Cuttoff_1": (0.1, "EPS1 | ...", float),
        "Cuttoff_2": (0.1, "EPS2 | ...", float),
        "EPSCOE": (1.0, "EPSCOE | ...", float),
        "NWS": (3, "NWS | ...", int),
        "CCC1": (0.03, "CCC1 | ...", float),
        "CCC2": (0.03, "CCC2 | ...", float),
        "CCGON1": (30.0, "CCGON1 | ...", float),
        "CCGON2": (30.0, "CCGON2 | ...", float),
        "IMOVE": (1, "IMOVE | ...", int),
        "A0": (0.0, "A0 | ...", float),
        "AMPL": (0.0, "AMPL | ...", float),
        "APHASE": (0.0, "APHASE | ...", float),
        "AKF": (0.0, "AKF | ...", float),
        "Chord_hinge": (0.25, "XC | Point from where to pitch the airfoil. 0.25 is the quarter chord", float),
        "ITEFLAP": (1, "ITEFLAP | Whether to use flap or not. 1: use flap, 0: don't use flap", int),
        "XEXT": (0.0, "XEXT | ...", float),
        "YEXT": (0.0, "YEXT | ...", float),
        "NTEWT": (9, "NTEWT | ...", int),
        "NTEST": (9, "NTEST | ...", int),
        "IBOUNDL": (1, "IBOUNDL | Whether to use solve the boundary layer or not. 1: solve, 0: don't solve", int),
        "IYNEXTERN": (0, "IYNEXTERN | ...", int),
        "ITSEPAR": (0, "ITSEPAR | ...", int),
        "ISTEADY": (1, "ISTEADY | ...", int),
    }

    multi_reyn_parallel: Analysis = Analysis(
        solver_name="f2w_section",
        analysis_name="Multiple_Reynolds_Parallel",
        run_function=run_multiple_reynolds_parallel,
        options=options,
        solver_options=solver_options,
        unhook=process_f2w_run,
    )

    multi_reyn_serial: Analysis = multi_reyn_parallel << {
        "name": "Multiple_Reynolds_Serial",
        "execute": run_multiple_reynolds_sequentially,
        "unhook": process_f2w_run,
    }

    options = {
        "airfoil": (
            "Airfoil to run",
            Airfoil,
        ),
        "reynolds": (
            "Reynolds number to run",
            float,
        ),
        "mach": (
            "Mach number",
            float,
        ),
        "angles": (
            "All angles to run",
            list[float],
        ),
    }

    signle_reyn: Analysis = Analysis(
        solver_name="f2w_section",
        analysis_name="Single_Reynolds",
        run_function=run_single_reynolds,
        options=options,
        solver_options=solver_options,
        unhook=process_f2w_run,
    )

    f2w_section.add_analyses(
        [
            multi_reyn_parallel,
            multi_reyn_serial,
            signle_reyn,
        ],
    )

    return f2w_section


if __name__ == "__main__":
    f2w_section = get_f2w_section()
