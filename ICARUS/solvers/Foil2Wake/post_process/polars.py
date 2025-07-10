import os

import numpy as np
from pandas import DataFrame

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils import AirfoilOperatingConditions
from ICARUS.airfoils import AirfoilOperatingPointMetrics
from ICARUS.airfoils import AirfoilPolar
from ICARUS.airfoils.metrics.aerodynamic_dataclasses import AirfoilPressure
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.database import directory_to_angle


def process_f2w_run(
    airfoil: Airfoil | list[Airfoil],
    reynolds: FloatArray | list[float] | float,
) -> dict[float, DataFrame]:
    polars: dict[float, DataFrame] = {}
    DB = Database.get_instance()

    if isinstance(airfoil, Airfoil):
        airfoil_list = [airfoil]
    elif isinstance(airfoil, list):
        airfoil_list = airfoil
    else:
        raise TypeError(
            f"airfoil must be either an Airfoil object or a list of Airfoil objects. Got {type(airfoil)}",
        )

    if isinstance(reynolds, float):
        reynolds_list: list[float] = [reynolds]
    elif isinstance(reynolds, list):
        reynolds_list = reynolds
    elif isinstance(reynolds, np.ndarray):
        reynolds_list = reynolds.tolist()
    else:
        raise TypeError(
            f"reynolds must be either a float or a list of floats. Got {type(reynolds)}",
        )

    for airfoil in airfoil_list:
        for reyn in reynolds_list:
            _, REYNDIR, _ = DB.generate_airfoil_directories(
                airfoil=airfoil,
                reynolds=reyn,
            )

            try:
                polar = make_polars(
                    case_directory=REYNDIR,
                    reynolds=reyn,
                    mach=0.0,  # Mach number is not used in Foil2Wake)
                )
                polar.save(REYNDIR, "polar.xfoil")
            except ValueError:
                continue

        DB.load_airfoil_data(airfoil)
    return polars


def make_polars(
    case_directory: str,
    reynolds: float,
    mach: float,
) -> AirfoilPolar:
    """Make the polars from the forces and return a dataframe with them

    Args:
        case_directory (str): Path to the case directory containing the results
    Returns:
        DataFrame: Dataframe Containing CL, CD, CM for all angles

    """
    folders: list[str] = next(os.walk(case_directory))[1]

    metrics: list[AirfoilOperatingPointMetrics] = []
    for folder in folders:
        load_file = "aerloa.dat"
        pressure_file = "cp.dat"
        folder_path = os.path.join(case_directory, folder)

        if load_file not in next(os.walk(folder_path))[2]:
            continue

        if pressure_file not in next(os.walk(folder_path))[2]:
            continue

        aoa = directory_to_angle(folder)
        op = AirfoilOperatingConditions(
            aoa=aoa,
            reynolds_number=reynolds,
            mach_number=mach,
        )

        load_file = os.path.join(folder_path, load_file)
        with open(load_file, encoding="UTF-8") as f:
            data: list[str] = f.readlines()
        if len(data) < 2:
            continue

        values = [x.strip() for x in data[-1].split(" ") if x != ""]
        cl, cd, cm, aoa = (values[7], values[8], values[11], values[17])

        load_file = os.path.join(folder_path, pressure_file)
        pressure_data = np.loadtxt(load_file).T

        x = pressure_data[0]
        y = pressure_data[3]
        cp = pressure_data[1]

        cp_distribution = AirfoilPressure(x=x, y=y, cp=cp)

        metric = AirfoilOperatingPointMetrics(
            operating_conditions=op,
            Cl=float(cl),
            Cd=float(cd),
            Cm=float(cm),
            Cp_min=np.min(cp),
            Cp_distribution=cp_distribution,
        )
        metrics.append(metric)

    return AirfoilPolar.from_airfoil_metrics(metrics)
