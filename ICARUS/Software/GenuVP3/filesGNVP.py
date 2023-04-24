import os
import numpy as np
import pandas as pd
import shutil

from ICARUS.Core.formatting import ff, ff2, ff3, ff4


def inputF():
    # INPUT File
    fname = "input"
    with open(fname, "r") as file:
        data = file.readlines()
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

    with open(fname, "w") as file:
        file.writelines(data)


def dfile(params):
    fname = "dfile.yours"
    with open(fname, "r") as file:
        data = file.readlines()

    data[27] = f'{int(params["nBods"])}           NBODT      number of bodies\n'
    data[28] = f'{int(params["nBlades"])}           NBLADE     number of blades\n'
    data[
        35
    ] = f'{int(params["maxiter"])}         NTIMER     number of the last time step to be performed\n'
    data[36] = f'{params["timestep"]}        DT         time step\n'
    data[
        55
    ] = "5           NLEVELT    number of movements levels  ( 15 if tail rotor is considered ) \n"
    data[59] = f'{ff2(params["Uinf"][0])}       UINF(1)    the velocity at infinity\n'
    data[60] = f'{ff2(params["Uinf"][1])}       UINF(2)    .\n'
    data[61] = f'{ff2(params["Uinf"][2])}       UINF(3)    .\n'
    data[119] = f'{params["rho"]}       AIRDEN     Fluid density\n'
    data[120] = f'{params["visc"]}   VISCO      Kinematic viscosity\n'
    data[
        130
    ] = f"hermes.geo   FILEGEO    the data file for the geometry of the configuration\n"

    with open(fname, "w") as file:
        file.writelines(data)


def geofile(movements, bodies):
    fname = "hermes.geo"
    # with open(fname, "r") as file:
    #     data = file.readlines()
    data = []
    data.append("READ THE FLOW AND GEOMETRICAL DATA FOR EVERY SOLID BODY\n")
    data.append("               <blank>\n")

    for i, bod in enumerate(bodies):
        data.append("               <blank>\n")
        NB = bod["NB"]
        geoBodyHeader(data, bod, NB)
        data.append(
            f"{len(movements[i])+1}           LEVEL  the level of movement\n")
        data.append("               <blank>\n")
        data.append("Give  data for every level\n")
        # PITCH, ROLL, YAW, Movements to CG with angular velocity
        for j, mov in enumerate(movements[i]):
            geoBodyMovements(data, mov, len(movements[i]) - j, NB)

        data.append(
            "-----<end of movement data>----------------------------------------------------\n")
        data.append("               <blank>\n")
        data.append("Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.\n")
        data.append("1           IYNVCR(1)\n")
        data.append(
            f'{bod["cld"]}      FLCLCD      file name wherefrom Cl, Cd are read\n')
        data.append("               <blank>\n")
        data.append("Give the file name for the geometrical distributions\n")
        data.append(f'{bod["bld"]}\n')
    data.append("               <blank>\n")
    with open(fname, "w") as file:
        file.writelines(data)


def geoBodyHeader(data, body, NB):
    data.append(f'Body Number   NB = {NB}\n')
    data.append("               <blank>\n")
    data.append("2           NLIFT\n")
    data.append("0           IYNELSTB   \n")
    data.append("1           NBAER2ELST \n")
    data.append(f'{body["NNB"]}          NNBB\n')
    data.append(f'{body["NCWB"]}          NCWB\n')
    data.append("2           ISUBSCB\n")
    data.append("2\n")
    data.append("3           NLEVELSB\n")
    data.append("1           IYNTIPS \n")
    data.append("0           IYNLES  \n")
    data.append("0           NELES   \n")
    data.append("0           IYNCONTW\n")
    data.append("3           IDIRMOB  direction for the torque calculation\n")
    data.append("               <blank>\n")


def geoBodyMovements(data, mov, i, NB):
    data.append(f"NB={NB}, lev={i}  ( {mov.name} )\n")
    data.append(f"Rotation\n")
    data.append(f"{int(mov.Rtype)}           IMOVEAB  type of movement\n")
    data.append(
        f"{int(mov.Raxis)}           NAXISA   =1,2,3 axis of rotation\n")
    data.append(f"{ff3(mov.Rt1)}    TMOVEAB  -1  1st time step\n")
    data.append(f"{ff3(mov.Rt2)}    TMOVEAB  -2  2nd time step\n")
    data.append(f"0.          TMOVEAB  -3  3d  time step\n")
    data.append(f"0.          TMOVEAB  -4  4th time step!---->omega\n")
    data.append(f"{ff3(mov.Ra1)}    AMOVEAB  -1  1st value of amplitude\n")
    data.append(f"{ff3(mov.Ra2)}    AMOVEAB  -2  2nd value of amplitude\n")
    data.append(f"0.          AMOVEAB  -3  3d  value of amplitude\n")
    data.append(f"0.          AMOVEAB  -4  4th value of amplitude!---->phase\n")
    data.append(f"            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")
    data.append(f"Translation\n")
    data.append(f"{int(mov.Ttype)}           IMOVEUB  type of movement\n")
    data.append(
        f"{int(mov.Taxis)}           NAXISU   =1,2,3 axis of translation\n")
    data.append(f"{ff3(mov.Tt1)}    TMOVEUB  -1  1st time step\n")
    data.append(f"{ff3(mov.Tt2)}    TMOVEUB  -2  2nd time step\n")
    data.append(f"0.          TMOVEUB  -3  3d  time step\n")
    data.append(f"0.          TMOVEUB  -4  4th time step\n")
    data.append(f"{ff3(mov.Ta1)}    AMOVEUB  -1  1st value of amplitude\n")
    data.append(f"{ff3(mov.Ta2)}    AMOVEUB  -2  2nd value of amplitude\n")
    data.append(f"0.          AMOVEUB  -3  3d  value of amplitude\n")
    data.append(f"0.          AMOVEUB  -4  4th value of amplitude\n")
    data.append(f"            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")


def cldFiles(AeroData, airfoils, solver):
    for airf in airfoils:
        fname = f"{airf[4:]}.cld"
        polars = AeroData[airf][solver]

        # GET FILE
        with open(fname, "r") as file:
            data = file.readlines()

        # WRITE MACH NUMBERS !! ITS NOT GOING TO BE USED !!
        data[4] = f"{len(polars)}  ! Mach numbers for which CL-CD are given\n"
        for i in range(0, len(polars)):
            data[5 + i] = f"0.08\n"

        # WRITE REYNOLDS NUMBERS !! ITS GOING TO BE USED !!
        data[5 + len(polars)] = "! Reyn numbers for which CL-CD are given\n"
        for i, Reyn in enumerate(list(polars.keys())):
            data[6 + len(polars) + i] = f"{Reyn.zfill(5)}\n"
        data[6 + 2 * len(polars)] = "\n"
        data = data[:6 + 2 * len(polars) + 1]

        # GET 2D AIRFOIL POLARS IN ONE TABLE
        keys = list(polars.keys())
        df = polars[keys[0]].astype(
            'float32').dropna(axis=0, how="all")
        for reyn in keys[1:]:
            df2 = polars[reyn].astype(
                'float32').dropna(axis=0, how="all")
            df.rename(columns={
                      "CL": f"CL_{reyn}", "CD": f"CD_{reyn}", "Cm": f"Cm_{reyn}"}, inplace=True)
            df = pd.merge(df, df2, on="AoA", how='outer')

        # SORT BY AoA
        df = df.sort_values("AoA")
        # FILL NaN Values By neighbors
        df = filltable(df)

        # Get Angles
        angles = df["AoA"].values
        anglenum = len(angles)

        # FILL FILE
        for radpos in 0, 1:
            if radpos == 0:
                data.append("-10.       ! Radial Position\n")
            else:
                data.append("10.       ! Radial Position\n")
            data.append(
                f"{anglenum}         ! Number of Angles / Airfoil NACA {airf}\n")
            data.append(
                f"   ALPHA   CL(M=0.0)   CD       CM    CL(M=1)   CD       CM \n")
            for i, ang in enumerate(angles):
                string = ""
                nums = df.loc[df["AoA"] == ang].to_numpy().squeeze()
                for num in nums:
                    string = string + ff2(num) + "  "
                data.append(f"{string}\n")
            data.append("\n")
        with open(fname, "w") as file:
            file.writelines(data)
    return df


def bldFiles(bodies):
    for bod in bodies:
        fname = bod["bld"]
        with open(fname, "r") as file:
            data = file.readlines()

        step = round(
            (bod["Root_chord"] - bod["Tip_chord"]) /
            (bod["y_end"] - bod["y_0"]),
            ndigits=5
        )
        offset = round(
            bod['Offset'] /
            (bod["y_end"] - bod["y_0"]),
            ndigits=5
        )
        data[3] = f'1          {bod["NACA"]}\n'
        data[6] = f"0          0          0\n"
        data[9] = f'{bod["name"]}.FL   {bod["name"]}.DS   {bod["name"]}.WG\n'
        data[12] = f'{ff4(bod["x_0"])}     {ff4(bod["y_0"])}     {ff4(bod["z_0"])}\n'
        data[15] = f'{ff4(bod["pitch"])}     {ff4(bod["cone"])}     {ff4(bod["wngang"])}\n'
        data[18] = f"1                      0.         1.         \n"  # KSI
        data[21] = f'1                      0.         {bod["y_end"] - bod["y_0"]}\n'
        data[24] = \
            f'4                      {ff4(bod["Root_chord"])}     {ff4(-step)}     0.         0.         0.         0.\n'
        data[30] = \
            f'4                      {ff4(0.)}     {ff4( offset )}     0.         0.         0.         0.\n'

        with open(fname, "w") as file:
            file.writelines(data)


def makeInput(ANGLEDIR, HOMEDIR, GENUBASE, movements, bodies, params, airfoils, AeroData, solver):
    os.chdir(ANGLEDIR)

    # COPY FROM BASE
    filesNeeded = ['dfile.yours', 'hermes.geo', 'hyb.inf',
                   'input', 'name.cld', 'wing.bld']
    for item in filesNeeded:
        itemLOC = os.path.join(GENUBASE, item)
        shutil.copy(itemLOC, ANGLEDIR)

    # EMPTY BLD FILES
    for body in bodies:
        shutil.copy('wing.bld', f'{body["name"]}.bld')
    if "wing" not in [bod["name"] for bod in bodies]:
        os.remove('wing.bld')
    # EMPTY CLD FILES
    for airf in airfoils:
        shutil.copy('name.cld', f'{airf[4:]}.cld')
    os.remove("name.cld")

    # Input File
    inputF()
    # DFILE
    dfile(params)
    # HERMES.GEO
    geofile(movements, bodies)
    # BLD FILES
    bldFiles(bodies)
    # CLD FILES
    cldFiles(AeroData, airfoils, solver)
    if 'gnvp3' not in next(os.walk('.'))[2]:
        src = os.path.join(HOMEDIR, 'ICARUS', 'gnvp3')
        dst = os.path.join(ANGLEDIR, 'gnvp3')
        os.symlink(src, dst)
    os.chdir(HOMEDIR)


def filltable(df):
    """Fill Nan Values of Panda Dataframe Row by Row
    substituting first backward and then forward

    Args:
        df (pandas.DataFrame): Dataframe with NaN values
    """
    CLs = []
    CDs = []
    CMs = []
    for item in list(df.keys()):
        if item.startswith("CL"):
            CLs.append(item)
        if item.startswith("CD"):
            CDs.append(item)
        if item.startswith("Cm") or item.startswith("CM"):
            CMs.append(item)
    for colums in [CLs, CDs, CMs]:
        df[colums] = df[colums].interpolate(method='linear',
                                            limit_direction='backward',
                                            axis=1)
        df[colums] = df[colums].interpolate(method='linear',
                                            limit_direction='forward',
                                            axis=1)
    return df


def removeResults(ANGLEDIR, HOMEDIR):
    os.chdir(ANGLEDIR)
    os.remove("strip*")
    os.remove("x*")
    os.remove("YOURS*")
    os.remove("refstate*")
    os.chdir(HOMEDIR)
