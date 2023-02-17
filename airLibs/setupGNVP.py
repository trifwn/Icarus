import os
import numpy as np


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


def runGNVP(airMovement, bodies, params, airfoils, AeroData, Reynolds, angles, CASE):
    masterDir = os.getcwd()
    os.chdir(CASE)
    # INPUT
    fname = "input"
    with open(fname, "r") as file:
        data = file.readlines()
    data[0] = 'dfile.yours\n'
    data[1] = '0\n'
    data[2] = '1\n'
    data[3] = '1.     ! NTIMER\n'
    data[4] = '1.     ! NTIMEHYB\n'
    data[5] = '1.     ! ITERWAK\n'
    data[6] = '1.     ! ITERVEL\n'
    data[7] = '1.\n'
    data[8] = '1.\n'
    data[9] = '1.\n'
    data[10] = '1.\n'
    data[11] = '1.\n'

    with open(fname, "w") as file:
        file.writelines(data)

    # DFILE
    fname = "dfile.yours"
    with open(fname, "r") as file:
        data = file.readlines()

    data[27] = f'{int(params["nBods"])}           NBODT      number of bodies\n'
    data[28] = f'{int(params["nBlades"])}           NBLADE     number of blades\n'
    data[35] = f'{int(params["maxiter"])}         NTIMER     number of the last time step to be performed\n'
    data[36] = f'{params["timestep"]}        DT         time step\n'
    data[55] = '4           NLEVELT    number of movement levels  ( 15 if tail rotor is considered ) \n'
    data[59] = f'{params["Uinf"][0]}       UINF(1)    the velocity at infinity\n'
    data[60] = f'{params["Uinf"][1]}       UINF(2)    .\n'
    data[61] = f'{params["Uinf"][2]}       UINF(3)    .\n'
    data[119] = f'{params["rho"]}       AIRDEN     Fluid density\n'
    data[120] = f'{params["visc"]}   VISCO      Kinematic viscosity\n'
    data[130] = f'hermes.geo   FILEGEO    the data file for the geometry of the configuration\n'

    with open(fname, "w") as file:
        file.writelines(data)

    # HERMES.GEO
    fname = "hermes.geo"
    with open(fname, "r") as file:
        data = file.readlines()

    for i, bod in enumerate(bodies):
        data[2+103 * i + 1] = f'Body Number   NB = {bod["NB"]}\n'
        data[2+103 * i + 3] = '2           NLIFT\n'
        data[2+103 * i + 6] = '20          NNBB\n'
        data[2+103 * i + 7] = '20          NCWB\n'
        data[2+103 * i + 17] = '4           LEVEL  the level of movement\n'
        # PITCH CONTROL
        data[2+103 * i + 22] = '1           IMOVEAB  type of movement\n'
        data[2+103 * i + 24] = f'-0.000001   TMOVEAB  -1  1st time step\n'
        data[2+103 * i + 25] = f'10.         TMOVEAB  -2  2nd time step\n'
        data[2+103 * i + 26] = f'0.          TMOVEAB  -3  3d  time step\n'
        data[2+103 * i + 27] = f'0.          TMOVEAB  -4  4th time step!---->omega\n'
        data[2+103 * i +
             28] = f'{airMovement["alpha_s"]}          AMOVEAB  -1  1st value of amplitude\n'
        data[2+103 * i +
             29] = f'{airMovement["alpha_s"]}          AMOVEAB  -2  2nd value of amplitude\n'
        data[2+103 * i + 30] = f'0.          AMOVEAB  -3  3d  value of amplitude\n'
        data[2+103 * i + 31] = f'0.          AMOVEAB  -4  4th value of amplitude!---->phase\n'

        # ROLL CONTROL
        data[2+103 * i + 47] = f'1           IMOVEAB  type of movement\n'
        data[2+103 * i + 49] = f'-0.000001   TMOVEAB  -1  1st time step\n'
        data[2+103 * i + 50] = f'10.         TMOVEAB  -2  2nd time step\n'
        data[2+103 * i + 51] = f'0.          TMOVEAB  -3  3d  time step\n'
        data[2+103 * i + 52] = f'0.          TMOVEAB  -4  4th time step\n'
        data[2+103 * i +
             53] = f'{airMovement["phi_s"]}         AMOVEAB  -1  1st value of amplitude\n'
        data[2+103 * i +
             54] = f'{airMovement["phi_e"]}         AMOVEAB  -2  2nd value of amplitude\n'
        data[2+103 * i + 55] = f'0.          AMOVEAB  -3  3d  value of amplitude\n'
        data[2+103 * i + 56] = f'0.          AMOVEAB  -4  4th value of amplitude\n'

        # YAW CONTROL
        data[2+103 * i + 72] = f'1           IMOVEAB  type of movement\n'
        data[2+103 * i + 74] = f'-0.000001   TMOVEAB  -1  1st time step\n'
        data[2+103 * i + 75] = f'10.         TMOVEAB  -2  2nd time step\n'
        data[2+103 * i + 76] = f'0.          TMOVEAB  -3  3d  time step\n'
        data[2+103 * i + 77] = f'0.          TMOVEAB  -4  4th time step\n'
        data[2+103 * i +
             78] = f'{airMovement["beta_s"]}          AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)\n'
        data[2+103 * i +
             79] = f'{airMovement["beta_e"]}          AMOVEAB  -2  2nd value of amplitude\n'
        data[2+103 * i + 80] = f'0.          AMOVEAB  -3  3d  value of amplitude\n'
        data[2+103 * i + 81] = f'0.          AMOVEAB  -4  4th value of amplitude\n'

        data[2+103 * i +
             99] = f'{bod["cld"]}      FLCLCD      file name wherefrom Cl, Cd are read\n'
        data[2+103 * i + 102] = f'{bod["bld"]}\n'

    with open(fname, "w") as file:
        file.writelines(data)

    # BLD FILES
    for i, bod in enumerate(bodies):
        fname = bod["bld"]
        with open(fname, "r") as file:
            data = file.readlines()
        if bod["is_right"] == True:
            step = round((bod["Root_chord"] - bod["Tip_chord"]) / (bod["y_end"]-bod["y_0"]),
                         ndigits=5)
            data[3] = f'1          {bod["NACA"]}\n'
            data[6] = f'0          0          0\n'
            data[9] = f'{bod["name"]}.FL   {bod["name"]}.DS   {bod["name"]}.WG\n'
            data[12] = f'{bod["x_0"]}        {bod["y_0"]}        {bod["z_0"]}\n'
            data[15] = f'{bod["pitch"]}        {bod["cone"]}        {bod["wngang"]}\n'
            data[18] = f'1                      0.         1.         \n'  # KSI
            data[21] = f'1                      0.         {bod["y_end"]}\n'
            data[24] = f'4                      {bod["Root_chord"]}       {-step}   0.         0.         0.         0.\n'
        else:
            step = round((bod["Root_chord"] - bod["Tip_chord"]) / (bod["y_end"]-bod["y_0"]),
                         ndigits=5)
            data[3] = f'1          {bod["NACA"]}\n'
            data[6] = f'0          0          0\n'
            data[9] = f'{bod["name"]}.FL   {bod["name"]}.DS   {bod["name"]}.WG\n'
            data[12] = f'{bod["x_0"]}        {-bod["y_end"]}        {bod["z_0"]}\n'
            data[15] = f'{bod["pitch"]}        {bod["cone"]}        {bod["wngang"]}\n'
            data[18] = f'1                      0.         1.         \n'  # KSI
            data[21] = f'1                      0.         {bod["y_end"]}\n'
            data[24] = f'4                      {bod["Tip_chord"]}      {step}    0.         0.         0.         0.\n'

        with open(fname, "w") as file:
            file.writelines(data)

    # CLD FILES
    # for airfoil, clcdData in zip(airfoils, AeroData):
    #     fname = f"{airfoil}.cld"
    #     with open(fname, "r") as file:
    #         data = file.readlines()

    #     data[4] = f'{len(clcdData)}  ! Mach numbers for which CL-CD are given\n'
    #     for i in range(0, len(clcdData)):
    #         data[5+i] = f'{i/len(clcdData)}\n'
    #     data[5+len(clcdData)] = '! Reyn numbers for which CL-CD are given\n'
    #     for i in range(0, len(clcdData)):
    #         data[6+len(clcdData)+i] = f'{ff(Reynolds[i])}\n'
    #     data[6+2*len(clcdData)] = '\n'

    #     anglenum = len(angles[:-2])

    #     for radpos in 0, 1:
    #         if radpos == 0:
    #             data[7+2*len(clcdData)] = '-10.       ! Radial Position\n'
    #         else:
    #             data[7+2*len(clcdData) + radpos*(anglenum+4)
    #                  ] = '10.       ! Radial Position\n'
    #         data[8+2*len(clcdData) + radpos*(anglenum+4)
    #              ] = f'{anglenum}         ! Number of Angles / Airfoil NACA {airfoil}\n'
    #         data[9+2*len(clcdData) + radpos*(anglenum+4)
    #              ] = f'   ALPHA   CL(M=0.0)   CD       CM    CL(M=1)   CD       CM \n'

    #         for i, ang in enumerate(angles[:-2]):
    #             string = ''
    #             for reyndict in clcdData:
    #                 try:
    #                     a = ff2(reyndict[str(ang)][0]) + "  " + ff2(
    #                         reyndict[str(ang)][1]) + "  "+ff2(reyndict[str(ang)][2]) + "  "
    #                     string = string + a
    #                 except:
    #                     string = string + string[-8:] + "  "
    #             data[10+2*len(clcdData) + radpos*(anglenum+4) +
    #                  i] = f"{ff3(ang)}   {string}\n"

    #     with open(fname, "w") as file:
    #         file.writelines(data)

    # RUN GNVP
    # os.system(\n'./gnvp < input\n')
    os.chdir(masterDir)


def removeResults(CASE):
    masterDir = os.getcwd()
    os.chdir(CASE)
    os.system('rm  strip*')
    os.system('rm  x*')
    os.system('rm YOURS*')
    os.system('rm refstate*')
    os.system('rm ing.WG')
    os.chdir(masterDir)

    return "Not implemented yet"
