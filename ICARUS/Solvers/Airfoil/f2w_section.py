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
        # "flap"
    }

    solver_options: dict[str, tuple[Any, str, Any]] = {
        "max_iter": (
            200,
            "NTIMEM | Maximum number of iterations",
            int,
        ),
        "timestep": (
            0.001,
            "Simulation timestep (DT1)",
            float,
        ),
        "f_trip_low": (
            0.1,
            "Transition points for positive and negative angles for the lower surface",
            float,
        ),
        "f_trip_upper": (
            0.1,
            "Transition points for positive and negative angles for the upper surface",
            float,
        ),
        "Ncrit": (
            9,
            "N critical value for transition according to e to N method",
            float,
        ),
        # 0.                       ! TEANGLE (deg)
        # 1.                       ! UINF
        # 201                      ! NTIMEM
        # 0.010                    ! DT1
        # 50000                    ! DT2
        # 0.025                    ! EPS1
        # 0.025                    ! EPS2
        # 1.00                    ! EPSCOE
        # 3                        ! NWS
        # 0.015                    ! CCC1
        # 0.015                    ! CCC2
        # 20.                      ! CCGON1
        # 20.                      ! CCGON2
        # 1                        ! IMOVE
        # 0.000                   ! A0
        # 0.000                   ! AMPL
        # 0.000                   ! APHASE
        # 0.000                   ! AKF
        # 0.25                     ! XC
        # 1                        ! ITEFLAP
        # 0.9                     ! XEXT
        # 0.                      ! YEXT
        # 7                        ! NTEWT
        # 7                        ! NTEST
        # 1                        ! IBOUNDL ---> DO NOT CHANGE
        # 200                      ! NTIME_bl
        # 0                        ! IYNEXTERN
        # 1.96e+05                 ! Reynolds
        # 0.08815750808110491      ! Mach     Number
        # 0.1    1                 ! TRANSLO
        # 0.2    2                 ! TRANSLO
        # 9.                       ! AMPLUP_tr
        # 9.                       ! AMPLLO_tr
        # 0                        ! ITSEPAR (1: 2 wake calculation) ---> DO NOT CHANGE
        # 1                        ! ISTEADY (1: steady calculation) ---> DO NOT CHANGE
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
