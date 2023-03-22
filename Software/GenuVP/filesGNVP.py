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
    ] = "4           NLEVELT    number of movement levels  ( 15 if tail rotor is considered ) \n"
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


def geofile(airMovement, bodies):
    fname = "hermes.geo"
    with open(fname, "r") as file:
        data = file.readlines()

    for i, bod in enumerate(bodies):
        data[2 + 103 * i + 1] = f'Body Number   NB = {bod["NB"]}\n'
        data[2 + 103 * i + 3] = "2           NLIFT\n"
        data[2 + 103 * i + 6] = f'{bod["NNB"]}          NNBB\n'
        data[2 + 103 * i + 7] = f'{bod["NCWB"]}          NCWB\n'
        data[2 + 103 * i + 17] = "4           LEVEL  the level of movement\n"
        # PITCH CONTROL
        data[2 + 103 * i + 22] = "1           IMOVEAB  type of movement\n"
        data[2 + 103 * i + 24] = f"-0.000001   TMOVEAB  -1  1st time step\n"
        data[2 + 103 * i + 25] = f"10.         TMOVEAB  -2  2nd time step\n"
        data[2 + 103 * i + 26] = f"0.          TMOVEAB  -3  3d  time step\n"
        data[2 + 103 * i + 27] = f"0.          TMOVEAB  -4  4th time step!---->omega\n"
        data[
            2 + 103 * i + 28
        ] = f'{airMovement["alpha_s"]}          AMOVEAB  -1  1st value of amplitude\n'
        data[
            2 + 103 * i + 29
        ] = f'{airMovement["alpha_s"]}          AMOVEAB  -2  2nd value of amplitude\n'
        data[2 + 103 * i + 30] = f"0.          AMOVEAB  -3  3d  value of amplitude\n"
        data[
            2 + 103 * i + 31
        ] = f"0.          AMOVEAB  -4  4th value of amplitude!---->phase\n"

        # ROLL CONTROL
        data[2 + 103 * i + 47] = f"1           IMOVEAB  type of movement\n"
        data[2 + 103 * i + 49] = f"-0.000001   TMOVEAB  -1  1st time step\n"
        data[2 + 103 * i + 50] = f"10.         TMOVEAB  -2  2nd time step\n"
        data[2 + 103 * i + 51] = f"0.          TMOVEAB  -3  3d  time step\n"
        data[2 + 103 * i + 52] = f"0.          TMOVEAB  -4  4th time step\n"
        data[
            2 + 103 * i + 53
        ] = f'{airMovement["phi_s"]}         AMOVEAB  -1  1st value of amplitude\n'
        data[
            2 + 103 * i + 54
        ] = f'{airMovement["phi_e"]}         AMOVEAB  -2  2nd value of amplitude\n'
        data[2 + 103 * i + 55] = f"0.          AMOVEAB  -3  3d  value of amplitude\n"
        data[2 + 103 * i + 56] = f"0.          AMOVEAB  -4  4th value of amplitude\n"

        # YAW CONTROL
        data[2 + 103 * i + 72] = f"1           IMOVEAB  type of movement\n"
        data[2 + 103 * i + 74] = f"-0.000001   TMOVEAB  -1  1st time step\n"
        data[2 + 103 * i + 75] = f"10.         TMOVEAB  -2  2nd time step\n"
        data[2 + 103 * i + 76] = f"0.          TMOVEAB  -3  3d  time step\n"
        data[2 + 103 * i + 77] = f"0.          TMOVEAB  -4  4th time step\n"
        data[
            2 + 103 * i + 78
        ] = f'{airMovement["beta_s"]}          AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)\n'
        data[
            2 + 103 * i + 79
        ] = f'{airMovement["beta_e"]}          AMOVEAB  -2  2nd value of amplitude\n'
        data[2 + 103 * i + 80] = f"0.          AMOVEAB  -3  3d  value of amplitude\n"
        data[2 + 103 * i + 81] = f"0.          AMOVEAB  -4  4th value of amplitude\n"

        data[
            2 + 103 * i + 99
        ] = f'{bod["cld"]}      FLCLCD      file name wherefrom Cl, Cd are read\n'
        data[2 + 103 * i + 102] = f'{bod["bld"]}\n'

    with open(fname, "w") as file:
        file.writelines(data)


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
                nums = df.loc[df["AoA"] == 5].to_numpy().squeeze()
                for num in nums:
                    string = string + ff2(num) + "  "
                data.append(f"{ff3(ang)}   {string}\n")
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
        data[21] = f'1                      0.         {bod["y_end"]}\n'
        data[
            24
        ] = f'4                      {bod["Root_chord"]}       {-step}   0.         0.         0.         0.\n'

        with open(fname, "w") as file:
            file.writelines(data)


def makeInput(ANGLEDIR, HOMEDIR, GENUBASE, airMovement, bodies, params, airfoils, AeroData, solver):
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
    geofile(airMovement, bodies)
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
    for cols in [CLs, CDs, CMs]:
        df[cols] = df[cols].interpolate(method='linear',
                                        limit_direction='backward',
                                        axis=1)
        df[cols] = df[cols].interpolate(method='linear',
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
