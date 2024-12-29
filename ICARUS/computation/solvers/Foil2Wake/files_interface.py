import os
import subprocess
from time import sleep
from typing import Any

from ICARUS import Foil_Section_exe
from ICARUS.computation.solvers.Foil2Wake import files_f2w as ff2w


def sequential_run(
    CASEDIR: str,
    HOMEDIR: str,
    airfile: str,
    name: str,
    angles: list[float],
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:
    os.chdir(CASEDIR)
    num_of_angles: int = len(angles)

    # unpack solver options to args
    max_iter: int = solver_options["max_iter"]
    timestep: float = solver_options["timestep"]
    f_trip_low: float = solver_options["f_trip_low"]
    f_trip_upper: float = solver_options["f_trip_upper"]
    Ncrit: float = solver_options["Ncrit"]

    # IO FILES
    ff2w.io_file(airfile, name)

    # DESIGN.INP
    ff2w.design_file(
        number_of_angles=num_of_angles,
        angles=angles,
        name=name,
    )

    # F2W.INP
    ff2w.input_file(
        reynolds=reynolds,
        mach=mach,
        max_iter=max_iter,
        timestep=timestep,
        ftrip_low=f_trip_low,
        ftrip_upper=f_trip_upper,
        Ncrit=Ncrit,
        name=name,
        solver_options=solver_options,
    )

    # RUN Files
    with open(f"{name}.out", "w") as fout:
        with open(f"io_{name}.files") as fin:
            subprocess.call(
                [Foil_Section_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
    try:
        # os.rmdir("TMP.dir")
        pass
        # os.remove("SOLOUTI*")
    except (FileNotFoundError, OSError):
        pass

    sleep(1.0)
    os.chdir(HOMEDIR)
