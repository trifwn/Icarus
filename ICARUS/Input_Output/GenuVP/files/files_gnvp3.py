import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame

from ICARUS.Core.formatting import ff2
from ICARUS.Core.formatting import ff3
from ICARUS.Core.formatting import ff4
from ICARUS.Core.struct import Struct
from ICARUS.Database.Database_2D import Database_2D as foilsdb
from ICARUS.Input_Output.GenuVP.utils.genu_movement import Movement
from ICARUS.Input_Output.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.Input_Output.GenuVP.utils.genu_surface import GenuSurface


def input_file() -> None:
    """
    Creates the input file for GNVP3
    """

    fname: str = "input"
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()
    data[0] = "dfile.yours\n"
    data[1] = "0\n"
    data[2] = "1\n"
    data[3] = "1.     ! NTIMER\n"
    data[4] = "1.     ! NTIMEHYB\n"
    data[5] = "1.     ! ITERWAK\n"
    data[6] = "1.     ! ITERVEL\n"
    data[7] = "1.\n"
    data[8] = "1.\n"
    data[9] = "1.\n"
    data[10] = "1.\n"
    data[11] = "1.\n"

    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def dfile(params: GenuParameters) -> None:
    """Create Dfile for GNVP3

    Args:
        params (GenuParameters): An object containing all parameter values
    """
    fname: str = "dfile.yours"
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()

    data[27] = f"{int(params.nBods)}           NBODT      number of bodies\n"
    data[28] = f"{int(params.nBlades)}           NBLADE     number of blades\n"
    data[35] = f"{int(params.maxiter)}         NTIMER     number of the last time step to be performed\n"
    data[36] = f"{params.timestep}        DT         time step\n"
    data[55] = "4           NLEVELT    number of movements levels  ( 15 if tail rotor is considered ) \n"
    data[59] = f"{ff2(params.u_freestream[0])}       UINF(1)    the velocity at infinity\n"
    data[60] = f"{ff2(params.u_freestream[1])}       UINF(2)    .\n"
    data[61] = f"{ff2(params.u_freestream[2])}       UINF(3)    .\n"

    DX: float = float(1.5 * np.linalg.norm(params.u_freestream) * params.timestep)
    if DX < 0.005:
        data[94] = f"{ff2(DX)}       EPSVR      Cut-off length for the free vortex particles (final)\n"
        data[95] = f"{ff2(DX)}       EPSO       Cut-off length for the free vortex particles (init.)\n"

        DX = DX / 100
        data[90] = f"{ff2(DX)}       EPSFB      Cut-off length for the bound vorticity\n"
        data[91] = f"{ff2(DX)}       EPSFW      Cut-off length for the near-wake vorticity\n"
        data[92] = f"{ff2(DX)}       EPSSR      Cut-off length for source distributions\n"
        data[93] = f"{ff2(DX)}       EPSDI      Cut-off length for source distributions\n"

    data[110] = f"1           ITERPRE    Write forces every ... time steps\n"
    data[111] = f"1           ITERWAK    Write wake geometry every ... time steps\n"
    data[112] = f"10000       ITERVEL    Write inflow velocities\n"
    data[113] = f"10000       ITERREC    Take back-up every ... time steps\n"
    data[114] = f"100         ITERLOA    Write loads every ... time steps\n"
    data[115] = f"1           ITERCHW    Check the wake calculations every ... time steps\n"

    data[119] = f"{params.rho}       AIRDEN     Fluid density\n"
    data[120] = f"{params.visc}   VISCO      Kinematic viscosity\n"
    data[130] = "hermes.geo   FILEGEO    the data file for the geometry of the configuration\n"

    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def geofile(
    movements: list[list[Movement]],
    bodies_dicts: list[GenuSurface],
) -> None:
    """Create Geo file for GNVP3

    Args:
        movements (list[list[Movement]]): List of Movements for each body
        bd_dicts (GenuSurface): List of Bodies in GenuSurface format
    """
    fname = "hermes.geo"
    # with open(fname, "r") as file:
    #     data = file.readlines()
    data: list[str] = []
    data.append("READ THE FLOW AND GEOMETRICAL DATA FOR EVERY SOLID BODY\n")
    data.append("               <blank>\n")

    for i, bod in enumerate(bodies_dicts):
        data.append("               <blank>\n")
        NB: int = bod.NB
        geo_body_header(data, bod, NB)
        data.append(f"{len(movements[i])+1}           LEVEL  the level of movement\n")
        data.append("               <blank>\n")
        data.append("Give  data for every level\n")
        # PITCH, ROLL, YAW, Movements to CG with angular velocity
        for j, mov in enumerate(movements[i]):
            geo_body_movements(data, mov, len(movements[i]) - j, NB)

        data.append(
            "-----<end of movement data>----------------------------------------------------\n",
        )
        data.append("               <blank>\n")
        data.append("Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.\n")
        data.append("1           IYNVCR(1)\n")
        data.append(
            f"{bod.cld_fname}      FLCLCD      file name wherefrom Cl, Cd are read\n",
        )
        data.append("               <blank>\n")
        data.append("Give the file name for the geometrical distributions\n")
        data.append(f"{bod.bld_fname}\n")
    data.append("               <blank>\n")
    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def geo_body_header(data: list[str], body: GenuSurface, NB: int) -> None:
    """
    Crates the header of each body in the geo file.

    Args:
        data (list[str]): List of string to append to and write to file
        body (GenuSurface): Body in GenuSurface format
        NB (int): Body Number
    """
    data.append(f"Body Number   NB = {NB}\n")
    data.append("               <blank>\n")
    data.append("2           NLIFT\n")
    data.append("0           IYNELSTB   \n")
    data.append("1           NBAER2ELST \n")
    data.append(f"{body.NNB}          NNBB\n")
    data.append(f"{body.NCWB}          NCWB\n")
    data.append("2           ISUBSCB\n")
    data.append("2\n")
    data.append("3           NLEVELSB\n")
    data.append("0           IYNTIPS \n")
    data.append("0           IYNLES  \n")
    data.append("0           NELES   \n")
    data.append("0           IYNCONTW\n")
    data.append("3           IDIRMOB  direction for the torque calculation\n")
    data.append("               <blank>\n")


def geo_body_movements(data: list[str], mov: Movement, i: int, NB: int) -> None:
    """
    Add Movement Data to Geo File.

    Args:
        data (list[str]): Data to append to
        mov (Movement): Movement Object
        i (int): Index of Movement
        NB (int): Body Number
    """
    data.append(f"NB={NB}, lev={i}  ( {mov.name} )\n")
    data.append("Rotation\n")
    data.append(f"{int(mov.rotation_type)}           IMOVEAB  type of movement\n")
    data.append(
        f"{int(mov.rotation_axis)}           NAXISA   =1,2,3 axis of rotation\n",
    )
    data.append(f"{ff3(mov.rot_t1)}    TMOVEAB  -1  1st time step\n")
    data.append(f"{ff3(mov.rot_t2)}    TMOVEAB  -2  2nd time step\n")
    data.append("0.          TMOVEAB  -3  3d  time step\n")
    data.append("0.          TMOVEAB  -4  4th time step!---->omega\n")
    data.append(f"{ff3(mov.rot_a1)}    AMOVEAB  -1  1st value of amplitude\n")
    data.append(f"{ff3(mov.rot_a2)}    AMOVEAB  -2  2nd value of amplitude\n")
    data.append("0.          AMOVEAB  -3  3d  value of amplitude\n")
    data.append("0.          AMOVEAB  -4  4th value of amplitude!---->phase\n")
    data.append("            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")
    data.append("Translation\n")
    data.append(f"{int(mov.translation_type)}           IMOVEUB  type of movement\n")
    data.append(
        f"{int(mov.translation_axis)}           NAXISU   =1,2,3 axis of translation\n",
    )
    data.append(f"{ff3(mov.translation_t1)}    TMOVEUB  -1  1st time step\n")
    data.append(f"{ff3(mov.translation_t2)}    TMOVEUB  -2  2nd time step\n")
    data.append("0.          TMOVEUB  -3  3d  time step\n")
    data.append("0.          TMOVEUB  -4  4th time step\n")
    data.append(f"{ff3(mov.translation_a1)}    AMOVEUB  -1  1st value of amplitude\n")
    data.append(f"{ff3(mov.translation_a2)}    AMOVEUB  -2  2nd value of amplitude\n")
    data.append("0.          AMOVEUB  -3  3d  value of amplitude\n")
    data.append("0.          AMOVEUB  -4  4th value of amplitude\n")
    data.append("            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")


def cldFiles(foil_dat: Struct, bodies: list[GenuSurface], solver: str) -> None:
    """
    Create Polars CL-CD-Cm files for each airfoil

    Args:
        foil_dat (Struct): Foil Database Data containing all airfoils, solvers and reynolds
        bodies (list[GenuSurface]): list of bodies in GenuSurface format
        solver (str): preferred solver
    """
    for bod in bodies:
        fname: str = f"{bod.cld_fname}"
        try:
            polars: dict[str, DataFrame] = foil_dat[bod.airfoil_name][solver]
        except KeyError:
            try:
                polars = foil_dat[f"NACA{bod.airfoil_name}"][solver]
            except KeyError:
                print(foil_dat.keys())
                print(foil_dat[f"NACA{bod.airfoil_name}"].keys())
                raise KeyError(f"Airfoil {bod.airfoil_name} not found in database")

        # GET FILE
        with open(fname) as file:
            data: list[str] = file.readlines()

        # WRITE MACH NUMBERS !! ITS NOT GOING TO BE USED !!
        data[4] = f"{len(polars)}  ! Mach numbers for which CL-CD are given\n"
        for i in range(0, len(polars)):
            data[5 + i] = "0.08\n"

        # WRITE REYNOLDS NUMBERS !! ITS GOING TO BE USED !!
        data[5 + len(polars)] = "! Reyn numbers for which CL-CD are given\n"
        for i, reyn in enumerate(list(polars.keys())):
            data[6 + len(polars) + i] = f"{reyn.zfill(5)}\n"
        data[6 + 2 * len(polars)] = "\n"
        data = data[: 6 + 2 * len(polars) + 1]

        # GET ALL 2D airfoil POLARS IN ONE TABLE
        keys: list[str] = list(polars.keys())
        df: DataFrame = polars[keys[0]].astype("float32").dropna(axis=0, how="all")
        df.rename(
            {"CL": f"CL_{keys[0]}", "CD": f"CD_{keys[0]}", "Cm": f"Cm_{keys[0]}"},
            inplace=True,
            axis="columns",
        )
        for reyn in keys[1:]:
            df2: DataFrame = polars[reyn].astype("float32").dropna(axis=0, how="all")
            try:
                df2 = df2.rename(
                    {"CL": f"CL_{reyn}", "CD": f"CD_{reyn}", "Cm": f"Cm_{reyn}"},
                    errors="raise",
                    axis="columns",
                )
            except KeyError as e:
                print("------------------------")
                print(f"KeyError: {e}")
                print(bod.cld_fname)
                print(reyn)
                continue
            df = pd.merge(df, df2, on="AoA", how="outer")
        # SORT BY AoA
        df = df.sort_values("AoA")
        # FILL NaN Values By neighbors
        df = foilsdb.fill_polar_table(df)

        # Get Angles
        angles = df["AoA"].to_numpy()
        anglenum: int = len(angles)

        # FILL FILE
        for radpos in 0, 1:
            if radpos == 0:
                data.append("-100.       ! Radial Position\n")
            else:
                data.append("100.       ! Radial Position\n")
            data.append(
                f"{anglenum}         ! Number of Angles / Airfoil NACA {bod.airfoil_name}\n",
            )
            data.append(
                "   ALPHA   CL(M=0.0)   CD       CM    CL(M=1)   CD       CM \n",
            )
            for i, ang in enumerate(angles):
                string: str = ""
                nums = df.loc[df["AoA"] == ang].to_numpy().squeeze()
                for num in nums:
                    string = string + ff2(num) + "  "
                data.append(f"{string}\n")
            data.append("\n")
        with open(fname, "w") as file:
            file.writelines(data)


def bldFiles(bodies: list[GenuSurface], params: GenuParameters) -> None:
    """Create BLD files for each body

    Args:
        bodies (list[GenuSurface]): list of bodies in GenuSurface format
        params (GenuParameters): Genu Parameters object containing all parameters
    """
    for bod in bodies:
        fname: str = bod.bld_fname
        with open(fname) as file:
            data: list[str] = file.readlines()

        step: float = round(
            (bod.Root_chord - bod.Tip_chord) / (bod.y_end - bod.y_0),
            ndigits=5,
        )
        offset: float = round(bod.Offset / (bod.y_end - bod.y_0), ndigits=5)
        # Check Whether to split a symmetric body into two parts
        if not params.Use_Grid:
            data[3] = f'1          {"".join(char for char in bod.NACA if char.isdigit())}       {bod.name}.WG\n'
        else:
            # WRITE GRID FILE Since Symmetric objects cant be defined parametrically
            with open(f"{bod.name}.WG", "w") as file:
                grid: ndarray[Any, dtype[floating[Any]]] = bod.Grid
                for n_strip in grid:  # For each strip
                    file.write("\n")
                    for m_point in n_strip:  # For each point in the strip
                        # Grid Coordinates
                        file.write(f"{m_point[0]} {m_point[1]} {m_point[2]}\n")
            # Specify option 0 to read the file
            data[3] = f'0          {"".join(char for char in bod.NACA if char.isdigit())}       {bod.name}.WG\n'
        data[6] = "0          0          1\n"
        data[9] = f"{bod.name}.FL   {bod.name}.DS   {bod.name}OUT.WG\n"
        data[12] = f"{ff4(bod.x_0)} {ff4(bod.y_0)} {ff4(bod.z_0)}\n"
        data[15] = f"{ff4(bod.pitch)} {ff4(bod.cone)} {ff4(bod.wngang)}\n"
        data[18] = "1                      0.         1.         \n"  # KSI
        data[21] = f"1                      0.         {bod.y_end - bod.y_0}\n"
        data[
            24
        ] = f"4                      {ff4(bod.Root_chord)} {ff4(-step)}     0.         0.         0.         0.\n"
        data[30] = f"4                      {ff4(0.)} {ff4( offset )}     0.         0.         0.         0.\n"

        with open(fname, "w") as file:
            file.writelines(data)


def make_input_files(
    ANGLEDIR: str,
    HOMEDIR: str,
    GENUBASE: str,
    movements: list[list[Movement]],
    bodies: list[GenuSurface],
    params: GenuParameters,
    airfoils: list[str],
    foil_dat: Struct,
    solver: str,
) -> None:
    os.chdir(ANGLEDIR)

    # COPY FROM BASE
    filesNeeded: list[str] = [
        "dfile.yours",
        "hermes.geo",
        "hyb.inf",
        "input",
        "name.cld",
        "name.bld",
    ]
    for item in filesNeeded:
        item_location: str = os.path.join(GENUBASE, item)
        shutil.copy(item_location, ANGLEDIR)

    # EMPTY BLD FILES
    # EMPTY CLD FILES
    for body in bodies:
        shutil.copy("name.bld", f"{body.bld_fname}")
        shutil.copy("name.cld", f"{body.cld_fname}")
    os.remove("name.bld")
    os.remove("name.cld")

    # Input File
    input_file()
    # DFILE
    dfile(params)
    # HERMES.GEO
    geofile(movements, bodies)
    # BLD FILES
    bldFiles(bodies, params)
    # CLD FILES
    cldFiles(foil_dat, bodies, solver)
    if "gnvp3" not in next(os.walk("."))[2]:
        src: str = os.path.join(HOMEDIR, "ICARUS", "gnvp3")
        dst: str = os.path.join(ANGLEDIR, "gnvp3")
        os.symlink(src, dst)
    os.chdir(HOMEDIR)


def remove_results(CASEDIR: str, HOMEDIR: str) -> None:
    """Removes the simulation results from a GNVP3 case

    Args:
        CASEDIR (str): _description_
        HOMEDIR (str): _description_
    """
    os.chdir(CASEDIR)
    os.remove("strip*")
    os.remove("x*")
    os.remove("YOURS*")
    os.remove("refstate*")
    os.chdir(HOMEDIR)
