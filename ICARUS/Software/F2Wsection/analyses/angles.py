from typing import Any
import numpy as np
import os
import shutil

from pandas import DataFrame

from ICARUS.Software.F2Wsection.files_interface import sequential_run
from ICARUS.Software.F2Wsection.post_process.polars import make_polars
from ICARUS.Software.F2Wsection.utils import separate_angles
from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Database.db import DB


def run_single_reynolds(
    db: DB,
    airfoil: AirfoilD,
    reyn: float,
    mach: float,
    angles: list[float],
    solver_options: dict[str, Any],
) -> None:
    os.chdir(db.foilsDB.DATADIR)
    AFDIR: str = os.path.join(
        db.foilsDB.DATADIR,
        f"NACA{airfoil.name}"
    )
    os.makedirs(AFDIR, exist_ok=True)
    exists = False
    for i in os.listdir():
        if i.startswith("naca"):
            exists = True
    if not exists:
            airfoil.save(AFDIR)
    os.chdir(AFDIR)

    nangles, pangles = separate_angles(angles)
    print(f"\t\tAt Reynolds {reyn}")
    reynolds_str: str = np.format_float_scientific(
        reyn,
        sign=False,
        precision=3,
    )
    
    REYNDIR: str = os.path.join(
        AFDIR,
        f"Reynolds_{reynolds_str.replace('+', '')}",
    )
    try:
        os.makedirs(REYNDIR, exist_ok=True)
        airfile = os.path.join(
                AFDIR,
                airfoil.file_name,
        )
        shutil.copy(airfile, REYNDIR)
    except AttributeError as e:
        print(e)

    for name, angles in zip(["pos", "neg"], [pangles, nangles]):
        sequential_run(
            CASEDIR = REYNDIR,
            HOMEDIR = db.HOMEDIR,
            airfile = airfoil.file_name,
            name = name,
            angles = angles,
            reynolds = reyn,
            mach = mach,
            solver_options = solver_options
        )

def run_multiple_reynolds_parallel(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, float],
) -> None:
    from multiprocessing import Pool

    print("Running in parallel")
    with Pool(12) as pool:
        args_list = [
            (
                db,
                airfoil,
                reyn,
                mach,
                angles,
                solver_options,
            )
            for reyn in reynolds
        ]
        _ = pool.starmap(run_single_reynolds, args_list)

def run_multiple_reynolds_sequentially(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, float],
) -> None:

    print("Running Sequentially")
    for reyn in reynolds:
        run_single_reynolds(
            db,
            airfoil,
            reyn,
            mach,
            angles,
            solver_options,
        )

def process_f2w_run(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float] | float,
)-> dict[str,DataFrame]:

    polars: dict[str, DataFrame] = {}

    if isinstance(reynolds, float):
        reynolds = [reynolds]

    for reyn in reynolds:
        reynolds_str: str = np.format_float_scientific(
            reyn,
            sign=False,
            precision=3,
        )

        CASEDIR: str = os.path.join(
            db.foilsDB.DATADIR,
            f"NACA{airfoil.name}",
            f"Reynolds_{reynolds_str.replace('+', '')}",
        )

        polars[reynolds_str] = make_polars(CASEDIR, db.HOMEDIR)

    return polars
