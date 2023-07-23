from typing import Any
from ICARUS.Software.F2Wsection import files_f2w as ff2w
from ICARUS.Software.F2Wsection.utils import separate_angles
from ICARUS.Database import BASEFOIL2W as F2WBASE

import os
import shutil
import subprocess
from time import sleep

def sequential_run(
    CASEDIR: str,
    HOMEDIR: str,
    airfile: str,
    name: str,
    angles: list[float],
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any]
) -> None:
    
    os.chdir(CASEDIR)
    num_of_angles: int = len(angles)
    
    # unpack solver options to args
    max_iter: int = solver_options['max_iter']
    timestep: float= solver_options['timestep']
    f_trip_low: float = solver_options['f_trip_low']
    f_trip_upper: float = solver_options['f_trip_upper']
    Ncrit: float= solver_options['Ncrit']

    # Create files from mock
    ff2w.setup_f2w(
        F2WBASE= F2WBASE,
        HOMEDIR= HOMEDIR,
        CASEDIR= CASEDIR,
    )

    # IO FILES
    ff2w.io_file(airfile)

    # DESIGN.INP
    ff2w.design_file(
        number_of_angles = num_of_angles,
        angles = angles,
        name= name,
    )

    # F2W.INP
    ff2w.input_file(
        reynolds=reynolds,
        mach=mach,
        max_iter = max_iter,
        timestep= timestep,
        ftrip_low = f_trip_low,
        ftrip_upper= f_trip_upper,
        Ncrit= Ncrit,
        name= name,
    )

    # RUN Files
    shutil.copy(f"design_{name}.inp", "design.inp")
    shutil.copy(f"f2w_{name}.inp", "f2w.inp")
    print(f"\t\t\tRunning {angles}")
    with open(f"{name}.out", "w") as f:
        subprocess.call([os.path.join(CASEDIR, "foil_section")], stdout=f, stderr=f)
    try:
        os.rmdir("TMP.dir")
        os.remove("SOLOUTI*")
    except FileNotFoundError:
        pass

    sleep(1)
    os.chdir(HOMEDIR)

    pass


