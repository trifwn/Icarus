from __future__ import annotations
import os
import subprocess
from time import sleep
from typing import TYPE_CHECKING

from ICARUS import Foil_Section_exe
from ICARUS.airfoils import Airfoil
from ICARUS.database import Database
from .files_f2w import io_file, design_file, input_file

if TYPE_CHECKING:
    from ICARUS.solvers.Foil2Wake.f2w_section import Foil2WakeSolverParameters


def sequential_run(
    airfoil: Airfoil,
    name: str,
    angles: list[float],
    reynolds: float,
    mach: float,
    solver_parameters: Foil2WakeSolverParameters,
) -> None:
    DB = Database.get_instance()
    _, REYNDIR, _ = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )

    # IO FILES
    io_file(REYNDIR, airfoil.file_name, name)

    # DESIGN.INP
    design_file(
        directory=REYNDIR,
        angles=angles,
        name=name,
    )

    # F2W.INP
    input_file(
        directory=REYNDIR,
        reynolds=reynolds,
        mach=mach,
        name=name,
        solver_parameters=solver_parameters,
    )

    # RUN Files
    file_name_out = os.path.join(REYNDIR, f"{name}.out")
    file_name_in = os.path.join(REYNDIR, f"io_{name}.files")
    with open(file_name_out, "w") as fout:
        with open(file_name_in) as fin:
            subprocess.call(
                [Foil_Section_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
                cwd=REYNDIR,
            )
    try:
        pass
        # os.rmdir(os.path.join(directory, "TMP.dir"))
        # os.remove(os.path.join(directory, "SOLOUTI*"))
    except (FileNotFoundError, OSError):
        pass

    sleep(1.0)
