import subprocess
from time import sleep
from typing import Any

from ICARUS import Foil_Section_exe
from ICARUS.solvers.Foil2Wake import files_f2w as ff2w


def sequential_run(
    directory: str,
    airfile: str,
    name: str,
    angles: list[float],
    reynolds: float,
    mach: float,
    solver_parameters: dict[str, Any],
) -> None:
    num_of_angles: int = len(angles)

    # unpack solver options to args
    max_iter: int = solver_parameters["max_iter"]
    timestep: float = solver_parameters["timestep"]
    f_trip_low: float = solver_parameters["f_trip_low"]
    f_trip_upper: float = solver_parameters["f_trip_upper"]
    Ncrit: float = solver_parameters["Ncrit"]

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
        solver_parameters=solver_parameters,
    )

    # RUN Files
    with open(f"{name}.out", "w") as fout:
        with open(f"io_{name}.files") as fin:
            subprocess.call(
                [Foil_Section_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
                cwd=directory,
            )
    try:
        pass
        # os.rmdir(os.path.join(directory, "TMP.dir"))
        # os.remove(os.path.join(directory, "SOLOUTI*"))
    except (FileNotFoundError, OSError):
        pass

    sleep(1.0)
