import os
import numpy as np
import pandas as pd
import shutil


def ff(num):
    return np.format_float_scientific(num, sign=False, precision=2).zfill(5)


def ff2(num):
    if num >= 0:
        return "{:2.5f}".format(num)
    else:
        return "{:2.4f}".format(num)


def ff3(num):
    if num >= 10:
        return "{:2.5f}".format(num)
    elif num >= 0:
        return "{:2.6f}".format(num)
    else:
        return "{:2.5f}".format(num)


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
    ] = "4           NLEVELT    number of movements levels  ( 15 if tail rotor is considered ) \n"
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
            geoBodyMovements(data, mov, len(movements[i])  - j, NB)

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
    for airfoil in airfoils:
        fname = f"{airfoil[4:]}.cld"
        polars = AeroData[airfoil]

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
        df = polars[keys[0]][solver].astype(
            'float32').dropna(axis=0, how="all")
        for reyn in keys[1:]:
            df2 = polars[reyn][solver].astype(
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
                f"{anglenum}         ! Number of Angles / Airfoil NACA {airfoil}\n")
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
            ndigits=5,
        )
        data[3] = f'1          {bod["NACA"]}\n'
        data[6] = f"0          0          0\n"
        data[9] = f'{bod["name"]}.FL   {bod["name"]}.DS   {bod["name"]}.WG\n'
        data[12] = f'{bod["x_0"]}        {bod["y_0"]}        {bod["z_0"]}\n'
        data[15] = f'{bod["pitch"]}        {bod["cone"]}        {bod["wngang"]}\n'
        data[18] = f"1                      0.         1.         \n"  # KSI
        data[21] = f'1                      0.         {bod["y_end"] - bod["y_0"]}\n'
        data[
            24
        ] = f'4                      {bod["Root_chord"]}        {-step}   0.         0.         0.         0.\n'

        with open(fname, "w") as file:
            file.writelines(data)


def makeInput(ANGLEDIR, HOMEDIR, GENUBASE, movements, bodies, params, airfoils, AeroData, solver):
    os.chdir(ANGLEDIR)

    # COPY FROM BASE
    filesNeeded = ['dfile.yours', 'hermes.geo', 'hyb.inf',
                   'input', 'name.cld', 'Lwing.bld']
    for item in filesNeeded:
        shutil.copy(f'{GENUBASE}/{item}', f'{ANGLEDIR}/')

    # EMPTY BLD FILES
    for body in bodies:
        if body["name"] != 'Lwing':
            os.system(f'cp Lwing.bld {body["name"]}.bld')

    # EMPTY CLD FILES
    for airfoil in airfoils:
        os.system(f'cp name.cld {airfoil[4:]}.cld')
    os.system("rm name.cld")

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
    if 'gnvp' not in next(os.walk('.'))[2]:
        os.system(f'ln -sv {HOMEDIR}/gnvp {ANGLEDIR}/gnvp')
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


def makePolar(CASEDIR, HOMEDIR):
    os.chdir(CASEDIR)
    folders = next(os.walk('.'))[1]
    print('Making Polars')
    pols = []
    for folder in folders:
        os.chdir(f"{CASEDIR}/{folder}")
        files = next(os.walk('.'))[2]
        if "LOADS_aer.dat" in files:
            if folder.startswith("m"):
                a = [- float(folder[1:]), *np.loadtxt("LOADS_aer.dat")]
            else:
                a = [float(folder), *np.loadtxt("LOADS_aer.dat")]
            pols.append(a)
        os.chdir(f"{CASEDIR}")
    df = pd.DataFrame(pols, columns=cols)
    df.pop('TTIME')
    df.pop("PSIB")

    df = df.sort_values("AoA")
    df.to_csv('clcd.genu', index=False)
    os.chdir(HOMEDIR)
    return df


cols = ["AoA",
        "TTIME",
        "PSIB",
        "TFORC(1)",
        "TFORC(2)",
        "TFORC(3)",
        "TAMOM(1)",
        "TAMOM(2)",
        "TAMOM(3)",
        "TFORC2D(1)",
        "TFORC2D(2)",
        "TFORC2D(3)",
        "TAMOM2D(1)",
        "TAMOM2D(2)",
        "TAMOM2D(3)",
        "TFORCDS2D(1)",
        "TFORCDS2D(2)",
        "TFORCDS2D(3)",
        "TAMOMDS2D(1)",
        "TAMOMDS2D(2)",
        "TAMOMDS2D(3)"]


def removeResults(ANGLEDIR, HOMEDIR):
    os.chdir(ANGLEDIR)
    os.system("rm  strip*")
    os.system("rm  x*")
    os.system("rm YOURS*")
    os.system("rm refstate*")
    # os.system('rm ing.WG')
    os.chdir(HOMEDIR)
