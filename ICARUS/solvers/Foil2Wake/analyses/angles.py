from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.solvers.Foil2Wake.files import run_aseq
from ICARUS.solvers.Foil2Wake.files import run_case
from ICARUS.solvers.Foil2Wake.utils import separate_angles

if TYPE_CHECKING:
    from ICARUS.solvers.Foil2Wake import Foil2WakeSolverParameters


def f2w(
    airfoil: Airfoil,
    reynolds: float,
    mach: float,
    angles: float,
    solver_parameters: Foil2WakeSolverParameters,
) -> None:
    """
    Function exists to rename parameter `angles` to `aoa` for consistency with other analyses.

    Args:
        airfoil (Airfoil): Airfoil object to be analyzed.
        reynolds (float): Reynolds number for the analysis.
        mach (float): Mach number for the analysis.
        angles (float): Angle of attack in degrees.
        solver_parameters (Foil2WakeSolverParameters): Solver parameters for the Foil2Wake analysis.
    """
    run_case(
        airfoil=airfoil,
        aoa=angles,
        reynolds=reynolds,
        mach=mach,
        solver_parameters=solver_parameters,
    )


def f2w_aseq(
    airfoil: Airfoil,
    reynolds: float,
    mach: float,
    angles: list[float] | FloatArray,
    solver_parameters: Foil2WakeSolverParameters,
) -> None:
    nangles, pangles = separate_angles(angles)

    runs = []
    if pangles:
        runs.append(("pos", pangles))
    if nangles:
        runs.append(("neg", nangles))

    if len(runs) == 1:
        _, selected_angles = runs[0]
        run_aseq(
            airfoil=airfoil,
            angles=selected_angles,
            reynolds=reynolds,
            mach=mach,
            solver_parameters=solver_parameters,
        )
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_aseq,
                    airfoil=airfoil,
                    angles=selected_angles,
                    reynolds=reynolds,
                    mach=mach,
                    solver_parameters=solver_parameters,
                )
                for name, selected_angles in runs
            ]

            for future in futures:
                future.result()
