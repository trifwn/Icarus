from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.solvers.Foil2Wake.files_interface import sequential_run
from ICARUS.solvers.Foil2Wake.utils import separate_angles

if TYPE_CHECKING:
    from ICARUS.solvers.Foil2Wake.f2w_section import Foil2WakeSolverParameters


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
        name, selected_angles = runs[0]
        sequential_run(
            airfoil=airfoil,
            name=name,
            angles=selected_angles,
            reynolds=reynolds,
            mach=mach,
            solver_parameters=solver_parameters,
        )
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    sequential_run,
                    airfoil=airfoil,
                    name=name,
                    angles=selected_angles,
                    reynolds=reynolds,
                    mach=mach,
                    solver_parameters=solver_parameters,
                )
                for name, selected_angles in runs
            ]

            for future in futures:
                future.result()
