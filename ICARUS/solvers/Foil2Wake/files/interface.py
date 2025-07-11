from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

from ICARUS import Foil_exe
from ICARUS.airfoils import Airfoil
from ICARUS.database import Database

from .files_f2w import boundary_layer_file
from .files_f2w import design_file

if TYPE_CHECKING:
    from ICARUS.solvers.Foil2Wake import Foil2WakeSolverParameters


def run_aseq(
    airfoil: Airfoil,
    angles: list[float],
    reynolds: float,
    mach: float,
    solver_parameters: Foil2WakeSolverParameters,
) -> None:
    DB = Database.get_instance()
    _, _, ANGLEDIRS = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )

    for angle_dir in ANGLEDIRS:
        # Remove f2w.out if it exists
        file_name_out = os.path.join(angle_dir, "f2w.out")
        if os.path.exists(file_name_out):
            os.remove(file_name_out)

    for i, angle in enumerate(angles):
        if i == 0:
            solver_parameters.IRSOL = 0
        else:
            prev_solution_file = os.path.join(ANGLEDIRS[i - 1], "SOL_IN")
            # Check if the previous solution file exists and is not empty
            if not os.path.exists(prev_solution_file) or os.path.getsize(
                prev_solution_file,
            ):
                solver_parameters.IRSOL = 0
            else:
                solver_parameters.IRSOL = 2
                shutil.copy(
                    prev_solution_file,
                    os.path.join(ANGLEDIRS[i], "SOL_IN"),
                )

        run_case(
            airfoil=airfoil,
            aoa=angle,
            reynolds=reynolds,
            mach=mach,
            solver_parameters=solver_parameters,
        )


def run_case(
    airfoil: Airfoil,
    aoa: float,
    reynolds: float,
    mach: float,
    solver_parameters: Foil2WakeSolverParameters,
) -> None:
    DB = Database.get_instance()

    _, REYNDIR, ANGLEDIRS = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=aoa,
    )

    case_directory = ANGLEDIRS[0]
    airfoil.save_selig(directory=case_directory, inverse=True)

    design_file(
        directory=case_directory,
        airfoil_file_name=airfoil.file_name,
        aoa=aoa,
        solver_parameters=solver_parameters,
    )

    boundary_layer_file(
        directory=case_directory,
        reynolds=reynolds,
        mach=mach,
        solver_parameters=solver_parameters,
    )

    # RUN Files
    file_name_out = os.path.join(case_directory, "f2w.out")
    if os.path.exists(file_name_out):
        os.remove(file_name_out)
    with open(file_name_out, "w") as fout:
        subprocess.call(
            [Foil_exe],
            stdout=fout,
            stderr=fout,
            cwd=case_directory,
        )

    try:
        pass
        # os.rmdir(os.path.join(directory, "TMP.dir"))
        # os.remove(os.path.join(directory, "SOLOUTI*"))
    except (FileNotFoundError, OSError):
        pass
