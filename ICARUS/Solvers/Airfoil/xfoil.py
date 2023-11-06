from typing import Any

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Workers.analysis import Analysis
from ICARUS.Workers.solver import Solver


def get_xfoil() -> Solver:
    xfoil = Solver(name="xfoil", solver_type="2D-IBLM", fidelity=2)

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
        "min_aoa": (
            "Minimum angle of attack",
            float,
        ),
        "max_aoa": (
            "Maximum angle of attack",
            float,
        ),
        "aoa_step": (
            "Step between each angle of attack",
            float,
        ),
    }

    solver_options: dict[str, tuple[Any, str, Any]] = {
        "max_iter": (
            100,
            "Maximum number of iterations",
            int,
        ),
        "Ncrit": (
            1e-3,
            "Timestep between each iteration",
            float,
        ),
        "xtr": (
            (0.1, 0.1),
            "Transition points: Lower and upper",
            tuple[float],
        ),
        "print": (
            False,
            "Print xfoil output",
            bool,
        ),
    }

    from ICARUS.Input_Output.Xfoil.analyses.angles import multiple_reynolds_serial
    from ICARUS.Input_Output.Xfoil.analyses.angles import multiple_reynolds_parallel
    from ICARUS.Input_Output.Xfoil.analyses.angles import multiple_reynolds_serial_seq
    from ICARUS.Input_Output.Xfoil.analyses.angles import multiple_reynolds_parallel_seq

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

    options = {
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
            "List or numpy array of angles to run",
            list[float] | FloatArray,
        ),
    }

    aseq_multiple_reynolds_parallel_2: Analysis = Analysis(
        solver_name="xfoil",
        analysis_name="Aseq for Multiple Reynolds Parallel 2",
        run_function=multiple_reynolds_parallel_seq,
        options=options,
        solver_options=solver_options,
        unhook=None,
    )

    aseq_multiple_reynolds_serial_2: Analysis = aseq_multiple_reynolds_parallel_2 << {
        "name": "Aseq for Multiple Reynolds Sequentially 2",
        "execute": multiple_reynolds_serial_seq,
        "unhook": None,
    }

    xfoil.add_analyses(
        [
            aseq_multiple_reynolds_parallel,
            aseq_multiple_reynolds_serial,
            aseq_multiple_reynolds_parallel_2,
            aseq_multiple_reynolds_serial_2,
        ],
    )

    return xfoil


if __name__ == "__main__":
    f2w_section = get_xfoil()
